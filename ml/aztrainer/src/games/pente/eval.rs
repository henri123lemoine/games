//! The strength ladder: the net (batched PUCT, no noise, argmax) against fixed
//! opponents — uniform random, a one-ply greedy [`PenteEval`] agent, and rollout
//! MCTS over [`PenteEval`] — with paired openings. All net games run in one pool
//! so the GPU sees wide batches. Pente has no external engine anchor, so this is
//! a win-rate panel rather than a calibrated Elo.

use game_core::{Agent, Eval, Game, Rng};
use pente::encode::PenteEncoder;
use pente::{Pente, PenteEval, PenteState};
use rayon::prelude::*;
use solvers::azero::{self, Gather, PuctConfig, argmax};
use solvers::mcts::Mcts;

use super::selfplay::mix;
use crate::net::{EvalRequest, EvalResult, Infer};

/// Random opening plies (after the forced center) so paired games diverge.
const OPENING_PLIES: usize = 2;
/// Rollout-MCTS playouts are truncated here and scored by [`PenteEval`].
const PLAYOUT_DEPTH: u32 = 40;

#[derive(Clone, Copy, PartialEq)]
pub enum Opponent {
    Random,
    /// One-ply greedy over [`PenteEval`].
    Greedy,
    /// Rollout MCTS (`solvers::Mcts` with [`PenteEval`]-truncated playouts).
    Mcts(u32),
}

impl Opponent {
    pub fn name(self) -> String {
        match self {
            Opponent::Random => "random".into(),
            Opponent::Greedy => "greedy".into(),
            Opponent::Mcts(sims) => format!("mcts-{sims}"),
        }
    }

    fn agent(self) -> Box<dyn Agent<Pente> + Send> {
        match self {
            Opponent::Random => Box::new(game_core::RandomAgent),
            Opponent::Greedy => Box::new(GreedyAgent),
            Opponent::Mcts(sims) => Box::new(RolloutMcts { sims }),
        }
    }
}

/// Picks the legal placement whose successor [`PenteEval`] scores highest for the
/// mover.
struct GreedyAgent;

impl Agent<Pente> for GreedyAgent {
    fn act(&self, g: &Pente, s: &PenteState, p: usize, _rng: &mut Rng) -> usize {
        let actions = g.legal_actions(s);
        let mut best = 0;
        let mut best_v = f64::NEG_INFINITY;
        for (i, &a) in actions.iter().enumerate() {
            let mut next = s.clone();
            g.apply(&mut next, a);
            let v = PenteEval.eval(g, &next, p);
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        best
    }
}

struct RolloutMcts {
    sims: u32,
}

impl Agent<Pente> for RolloutMcts {
    fn act(&self, g: &Pente, s: &PenteState, p: usize, rng: &mut Rng) -> usize {
        Mcts::with_eval(self.sims, PenteEval, PLAYOUT_DEPTH).act(g, s, p, rng)
    }
}

pub struct LadderEntry {
    pub name: String,
    pub score: f64,
    pub wins: u32,
    pub draws: u32,
    pub losses: u32,
}

struct EvalGame {
    state: PenteState,
    opponent: Opponent,
    agent: Box<dyn Agent<Pente> + Send>,
    /// 0 if the net plays seat 0 (Black, the first mover).
    net_seat: usize,
    search: azero::Search<Pente>,
    rng: Rng,
    outcome: Option<f64>,
}

impl EvalGame {
    /// Plays opponent plies and checks termination; afterwards the game is either
    /// finished or it is the net's turn.
    fn advance_to_net_turn(&mut self, game: &Pente) {
        loop {
            if self.outcome.is_some() {
                return;
            }
            if game.is_terminal(&self.state) {
                self.outcome = Some(game.returns(&self.state, self.net_seat));
                return;
            }
            let stm = self.state.to_move();
            if stm == self.net_seat {
                return;
            }
            let actions = game.legal_actions(&self.state);
            let i = self.agent.act(game, &self.state, stm, &mut self.rng);
            game.apply(&mut self.state, actions[i]);
        }
    }
}

/// Plays `pairs` paired games per opponent (net as seat 0 then seat 1 from the
/// same opening), all concurrently.
pub fn ladder(
    infer: &Infer,
    opponents: &[Opponent],
    pairs: u32,
    sims: u32,
    size: usize,
    seed: u64,
) -> Vec<LadderEntry> {
    let game = Pente::new(size);
    let enc = PenteEncoder::new(size);
    let puct = PuctConfig {
        sims,
        root_noise: 0.0,
        ..PuctConfig::default()
    };
    let mut games: Vec<EvalGame> = Vec::new();
    for (oi, &opp) in opponents.iter().enumerate() {
        for pair in 0..pairs {
            let mut rng = Rng::new(mix(seed, (oi as u64) << 32 | u64::from(pair)));
            let opening = random_opening(&game, &mut rng);
            for net_seat in 0..2 {
                games.push(EvalGame {
                    state: opening.clone(),
                    opponent: opp,
                    agent: opp.agent(),
                    net_seat,
                    search: azero::Search::new(None),
                    rng: Rng::new(mix(
                        seed,
                        (oi as u64) << 40 | u64::from(pair) << 8 | net_seat as u64,
                    )),
                    outcome: None,
                });
            }
        }
    }

    let mut results: Vec<Vec<EvalResult>> = (0..games.len()).map(|_| Vec::new()).collect();
    loop {
        let gathered: Vec<Vec<EvalRequest>> = games
            .par_iter_mut()
            .zip(results.par_iter_mut())
            .map(|(g, r)| {
                let mut pending = std::mem::take(r);
                loop {
                    g.advance_to_net_turn(&game);
                    if g.outcome.is_some() {
                        return Vec::new();
                    }
                    match g.search.advance(
                        &game,
                        &enc,
                        &g.state,
                        &puct,
                        &mut g.rng,
                        std::mem::take(&mut pending),
                        &|_| false,
                        None,
                    ) {
                        Gather::Requests(reqs) => return reqs,
                        Gather::Done => {
                            let visits = g.search.root_visits().to_vec();
                            let actions = g.search.root_actions();
                            let action = actions[argmax(&visits)];
                            game.apply(&mut g.state, action);
                            g.search = azero::Search::new(None);
                        }
                    }
                }
            })
            .collect();

        let mut flat: Vec<EvalRequest> = Vec::new();
        let mut spans: Vec<(usize, usize)> = Vec::with_capacity(gathered.len());
        for reqs in gathered {
            spans.push((flat.len(), reqs.len()));
            flat.extend(reqs);
        }
        if flat.is_empty() {
            break;
        }
        let mut outs = infer.forward_batch(&flat);
        for (i, (start, len)) in spans.into_iter().enumerate().rev() {
            results[i] = outs.split_off(start);
            debug_assert_eq!(results[i].len(), len);
        }
    }

    opponents
        .iter()
        .map(|&opp| {
            let outcomes: Vec<f64> = games
                .iter()
                .filter(|g| g.opponent == opp)
                .map(|g| g.outcome.unwrap_or(0.0))
                .collect();
            let wins = outcomes.iter().filter(|&&r| r > 0.0).count() as u32;
            let losses = outcomes.iter().filter(|&&r| r < 0.0).count() as u32;
            let n = outcomes.len() as u32;
            let draws = n - wins - losses;
            LadderEntry {
                name: opp.name(),
                score: (f64::from(wins) + 0.5 * f64::from(draws)) / f64::from(n.max(1)),
                wins,
                draws,
                losses,
            }
        })
        .collect()
}

/// Plays `pairs` paired games (each opening with net A as seat 0, then seat 1)
/// between two nets — both argmax, no root noise, at `sims`. Returns (net A wins,
/// total games): the KataGo-style relative progress signal (A = current net, B =
/// an older snapshot).
pub fn net_vs_net(
    a: &Infer,
    b: &Infer,
    pairs: u32,
    sims: u32,
    size: usize,
    seed: u64,
) -> (u32, u32) {
    let game = Pente::new(size);
    let enc = PenteEncoder::new(size);
    let puct = PuctConfig {
        sims,
        root_noise: 0.0,
        ..PuctConfig::default()
    };

    struct RateGame {
        state: PenteState,
        search: azero::Search<Pente>,
        a_seat: usize,
        rng: Rng,
        outcome: Option<f64>,
    }
    let mut games: Vec<RateGame> = Vec::new();
    for pair in 0..pairs {
        let mut rng = Rng::new(mix(seed, u64::from(pair)));
        let opening = random_opening(&game, &mut rng);
        for a_seat in 0..2 {
            games.push(RateGame {
                state: opening.clone(),
                search: azero::Search::new(None),
                a_seat,
                rng: Rng::new(mix(seed, (u64::from(pair) << 8) | a_seat as u64)),
                outcome: None,
            });
        }
    }

    let mut results: Vec<Vec<EvalResult>> = (0..games.len()).map(|_| Vec::new()).collect();
    loop {
        let gathered: Vec<(Option<bool>, Vec<EvalRequest>)> = games
            .par_iter_mut()
            .zip(results.par_iter_mut())
            .map(|(g, r)| {
                let mut pending = std::mem::take(r);
                loop {
                    if g.outcome.is_some() {
                        return (None, Vec::new());
                    }
                    if game.is_terminal(&g.state) {
                        g.outcome = Some(game.returns(&g.state, g.a_seat));
                        return (None, Vec::new());
                    }
                    let on_move = g.state.to_move();
                    match g.search.advance(
                        &game,
                        &enc,
                        &g.state,
                        &puct,
                        &mut g.rng,
                        std::mem::take(&mut pending),
                        &|_| false,
                        None,
                    ) {
                        Gather::Requests(reqs) => {
                            return (Some(on_move == g.a_seat), reqs);
                        }
                        Gather::Done => {
                            let visits = g.search.root_visits().to_vec();
                            let actions = g.search.root_actions();
                            let action = actions[argmax(&visits)];
                            game.apply(&mut g.state, action);
                            g.search = azero::Search::new(None);
                        }
                    }
                }
            })
            .collect();

        let mut a_flat: Vec<EvalRequest> = Vec::new();
        let mut b_flat: Vec<EvalRequest> = Vec::new();
        let mut route: Vec<(Option<bool>, usize)> = Vec::with_capacity(gathered.len());
        for (tag, reqs) in gathered {
            route.push((tag, reqs.len()));
            match tag {
                Some(true) => a_flat.extend(reqs),
                Some(false) => b_flat.extend(reqs),
                None => {}
            }
        }
        if a_flat.is_empty() && b_flat.is_empty() {
            break;
        }
        let mut a_out = a.forward_batch(&a_flat).into_iter();
        let mut b_out = b.forward_batch(&b_flat).into_iter();
        for (i, (tag, len)) in route.into_iter().enumerate() {
            results[i] = match tag {
                Some(true) => (0..len).filter_map(|_| a_out.next()).collect(),
                Some(false) => (0..len).filter_map(|_| b_out.next()).collect(),
                None => Vec::new(),
            };
        }
    }

    let a_wins = games
        .iter()
        .filter(|g| g.outcome.unwrap_or(0.0) > 0.0)
        .count() as u32;
    (a_wins, games.len() as u32)
}

/// An opening: the forced center, then `OPENING_PLIES` uniform random plies, so
/// paired games diverge from a shared but varied start.
fn random_opening(game: &Pente, rng: &mut Rng) -> PenteState {
    let mut s = game.initial_state();
    for _ in 0..=OPENING_PLIES {
        if game.is_terminal(&s) {
            break;
        }
        let actions = game.legal_actions(&s);
        game.apply(&mut s, actions[rng.below(actions.len())]);
    }
    s
}
