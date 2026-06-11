//! The strength ladder: the net (batched PUCT, no noise, argmax) against
//! fixed opponents — uniform random and `AlphaBeta(MaterialEval)` at depths
//! 1–3 — with paired random openings and draws scored ½. All opponents'
//! games run in one pool so the GPU sees wide batches.

use std::collections::HashMap;

use chess::{Board, Chess, ChessSpec, MaterialEval, legal_moves};
use game_core::{Agent, Game, Rng};
use rayon::prelude::*;
use solvers::AlphaBeta;

use crate::net::{EvalRequest, EvalResult, Infer};
use crate::selfplay::{argmax, mix};
use crate::uci::Uci;
use azinfer::mcts::{Gather, MctsConfig, Search};

const OPENING_PLIES: usize = 4;
const PLY_CAP: u16 = 300;

#[derive(Clone, Copy, PartialEq)]
pub enum Opponent {
    Random,
    AbMaterial(u32),
    /// Depth-1 material search playing a uniform-random move
    /// `pct_random`% of the time — fills the rating gap between `Random`
    /// and `AbMaterial(1)`.
    Diluted {
        pct_random: u32,
    },
    /// Stockfish at a calibrated `UCI_Elo`, thinking `movetime_ms` per move.
    Stockfish {
        elo: u32,
        movetime_ms: u32,
    },
}

impl Opponent {
    pub fn name(self) -> String {
        match self {
            Opponent::Random => "random".into(),
            Opponent::AbMaterial(d) => format!("ab-mat-d{d}"),
            Opponent::Diluted { pct_random } => format!("mix-{pct_random}"),
            Opponent::Stockfish { elo, .. } => format!("sf-{elo}"),
        }
    }

    fn agent(self) -> Box<dyn Agent<Chess> + Send> {
        match self {
            Opponent::Random => Box::new(game_core::RandomAgent),
            Opponent::AbMaterial(d) => Box::new(AlphaBeta::new(d, MaterialEval, ChessSpec)),
            Opponent::Diluted { pct_random } => Box::new(Diluted {
                ab: AlphaBeta::new(1, MaterialEval, ChessSpec),
                p_random: f64::from(pct_random) / 100.0,
            }),
            Opponent::Stockfish { elo, movetime_ms } => Box::new(SfAgent {
                uci: std::cell::RefCell::new(
                    Uci::spawn("stockfish", elo).expect("spawn stockfish (is it installed?)"),
                ),
                movetime_ms,
            }),
        }
    }
}

struct Diluted {
    ab: AlphaBeta<Chess, MaterialEval, ChessSpec>,
    p_random: f64,
}

impl Agent<Chess> for Diluted {
    fn act(&self, g: &Chess, s: &Board, p: usize, rng: &mut Rng) -> usize {
        if rng.unit() < self.p_random {
            rng.below(g.legal_actions(s).len())
        } else {
            self.ab.act(g, s, p, rng)
        }
    }
}

struct SfAgent {
    uci: std::cell::RefCell<Uci>,
    movetime_ms: u32,
}

impl Agent<Chess> for SfAgent {
    fn act(&self, _g: &Chess, s: &Board, _p: usize, _rng: &mut Rng) -> usize {
        let text = self
            .uci
            .borrow_mut()
            .best_move(&s.to_fen(), self.movetime_ms)
            .expect("uci best_move");
        let m: chess::Move = text.parse().expect("parse uci move");
        legal_moves(s)
            .iter()
            .position(|&x| x == m)
            .expect("stockfish move is legal")
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
    board: Board,
    opponent: Opponent,
    agent: Box<dyn Agent<Chess> + Send>,
    /// 0 if the net plays White.
    net_seat: usize,
    search: Search,
    rng: Rng,
    key_counts: HashMap<u64, u8>,
    plies: u16,
    /// Return for the net once finished.
    outcome: Option<f64>,
}

impl EvalGame {
    /// Plays opponent plies and checks termination; afterwards it is either
    /// finished or the net's turn.
    fn advance_to_net_turn(&mut self) {
        let game = &Chess;
        loop {
            if self.outcome.is_some() {
                return;
            }
            if let Some(r) = self.result_now() {
                self.outcome = Some(r);
                return;
            }
            let stm = self.board.stm.index();
            if stm == self.net_seat {
                return;
            }
            let actions = legal_moves(&self.board);
            let i = self.agent.act(game, &self.board, stm, &mut self.rng);
            self.apply(actions[i]);
        }
    }

    fn apply(&mut self, m: chess::Move) {
        self.board.apply(m);
        self.plies += 1;
        *self.key_counts.entry(self.board.key()).or_insert(0) += 1;
    }

    /// Terminal result for the net, if the game is over.
    fn result_now(&self) -> Option<f64> {
        if self.key_counts.values().any(|&c| c >= 3) || self.plies >= PLY_CAP {
            return Some(0.0);
        }
        if self.board.halfmove >= 100 || self.board.insufficient_material() {
            return Some(0.0);
        }
        if legal_moves(&self.board).is_empty() {
            if self.board.in_check(self.board.stm) {
                let loser = self.board.stm.index();
                return Some(if loser == self.net_seat { -1.0 } else { 1.0 });
            }
            return Some(0.0);
        }
        None
    }
}

/// Plays `pairs` paired games per opponent (net as White then Black from the
/// same random opening), all concurrently.
pub fn ladder(
    infer: &Infer,
    opponents: &[Opponent],
    pairs: u32,
    sims: u32,
    seed: u64,
) -> Vec<LadderEntry> {
    let mcts = MctsConfig {
        sims,
        root_noise: 0.0,
        ..MctsConfig::default()
    };
    let mut games: Vec<EvalGame> = Vec::new();
    for (oi, &opp) in opponents.iter().enumerate() {
        for pair in 0..pairs {
            let mut rng = Rng::new(mix(seed, (oi as u64) << 32 | u64::from(pair)));
            let opening = random_opening(&mut rng);
            for net_seat in 0..2 {
                let mut key_counts = HashMap::new();
                key_counts.insert(opening.key(), 1);
                games.push(EvalGame {
                    board: opening.clone(),
                    opponent: opp,
                    agent: opp.agent(),
                    net_seat,
                    search: Search::new(None),
                    rng: Rng::new(mix(
                        seed,
                        (oi as u64) << 40 | u64::from(pair) << 8 | net_seat as u64,
                    )),
                    key_counts,
                    plies: OPENING_PLIES as u16,
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
                    g.advance_to_net_turn();
                    if g.outcome.is_some() {
                        return Vec::new();
                    }
                    match g.search.advance(
                        &g.board,
                        &g.key_counts,
                        &mcts,
                        &mut g.rng,
                        std::mem::take(&mut pending),
                    ) {
                        Gather::Requests(reqs) => return reqs,
                        Gather::Done => {
                            let choice = argmax(g.search.root_visits());
                            let m = g.search.root_moves()[choice];
                            g.apply(m);
                            // No tree reuse here: the opponent moves before
                            // our next search, so the extracted subtree
                            // would be rooted one ply behind the board.
                            g.search = Search::new(None);
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

/// Plays `pairs` paired games between two fixed opponents (no net), for
/// calibrating the ladder's Elo anchors. Returns `a`'s score and the game
/// count.
pub fn duel(a: Opponent, b: Opponent, pairs: u32, seed: u64) -> (f64, u32) {
    let results: Vec<f64> = (0..pairs)
        .into_par_iter()
        .flat_map_iter(|i| {
            let mut rng = Rng::new(mix(seed, u64::from(i) + 1));
            let opening = random_opening(&mut rng);
            let as_white = fixed_game(&*a.agent(), &*b.agent(), opening.clone(), &mut rng);
            let as_black = -fixed_game(&*b.agent(), &*a.agent(), opening, &mut rng);
            [as_white, as_black]
        })
        .collect();
    let wins = results.iter().filter(|&&r| r > 0.0).count() as f64;
    let draws = results.iter().filter(|&&r| r == 0.0).count() as f64;
    let games = results.len() as u32;
    ((wins + 0.5 * draws) / f64::from(games), games)
}

/// One game between fixed agents from `opening`; returns White's result.
fn fixed_game(
    white: &dyn Agent<Chess>,
    black: &dyn Agent<Chess>,
    mut board: Board,
    rng: &mut Rng,
) -> f64 {
    let game = &Chess;
    let mut keys = HashMap::new();
    keys.insert(board.key(), 1u8);
    for _ in 0..PLY_CAP {
        let moves = legal_moves(&board);
        if moves.is_empty() {
            return if board.in_check(board.stm) {
                if board.stm == chess::Color::White {
                    -1.0
                } else {
                    1.0
                }
            } else {
                0.0
            };
        }
        if board.halfmove >= 100 || board.insufficient_material() || keys.values().any(|&c| c >= 3)
        {
            return 0.0;
        }
        let stm = board.stm.index();
        let agent = if stm == 0 { white } else { black };
        let i = agent.act(game, &board, stm, rng);
        board.apply(moves[i]);
        *keys.entry(board.key()).or_insert(0) += 1;
    }
    0.0
}

fn random_opening(rng: &mut Rng) -> Board {
    loop {
        let mut b = Board::start();
        for _ in 0..OPENING_PLIES {
            let moves = legal_moves(&b);
            if moves.is_empty() {
                break;
            }
            let i = ((rng.unit() * moves.len() as f64) as usize).min(moves.len() - 1);
            b.apply(moves[i]);
        }
        if !legal_moves(&b).is_empty() && b.halfmove < 100 && !b.insufficient_material() {
            return b;
        }
    }
}
