//! The strength ladder: the net (batched PUCT, no noise, argmax) against
//! fixed opponents — uniform random, rollout MCTS over `GoEval`, and GNU Go
//! levels over GTP — with paired random openings. Komi 7.5 means no draws;
//! scores are plain win rates. All net games run in one pool so the GPU
//! sees wide batches.

use std::cell::RefCell;

use game_core::{Agent, Game, Rng};
use go::encode::GoEncoder;
use go::{Go, GoAction, GoEval, GoState};
use rayon::prelude::*;
use solvers::azero::{self, Gather, PuctConfig, argmax};
use solvers::mcts::Mcts;

use crate::gtp::Gtp;
use crate::net::{EvalRequest, EvalResult, Infer};
use crate::selfplay::mix;

const OPENING_PLIES: usize = 2;
/// Rollout-MCTS playouts are truncated here and scored by [`GoEval`].
const PLAYOUT_DEPTH: u32 = 64;

pub fn gnugo_path() -> String {
    std::env::var("GNUGO").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_default();
        let local = format!("{home}/.local/bin/gnugo");
        if std::path::Path::new(&local).exists() {
            local
        } else {
            "gnugo".into()
        }
    })
}

#[derive(Clone, Copy, PartialEq)]
pub enum Opponent {
    Random,
    /// Rollout MCTS (`solvers::Mcts` with [`GoEval`]-truncated playouts).
    Mcts(u32),
    /// GNU Go at `--level` (1 weakest … 10 strongest/default).
    GnuGo(u32),
}

impl Opponent {
    pub fn name(self) -> String {
        match self {
            Opponent::Random => "random".into(),
            Opponent::Mcts(sims) => format!("mcts-{sims}"),
            Opponent::GnuGo(level) => format!("gnugo-l{level}"),
        }
    }

    fn agent(self, size: usize, seed: u32) -> Box<dyn Agent<Go> + Send> {
        match self {
            Opponent::Random => Box::new(game_core::RandomAgent),
            Opponent::Mcts(sims) => Box::new(RolloutMcts { sims }),
            Opponent::GnuGo(level) => Box::new(GnuGoAgent::spawn(level, seed, size)),
        }
    }
}

/// [`Mcts`] behind a `Send` wrapper: the solver boxes non-`Send` trait
/// objects, and its `act` builds a fresh tree per call anyway, so
/// constructing it per move costs nothing relative to the search.
struct RolloutMcts {
    sims: u32,
}

impl Agent<Go> for RolloutMcts {
    fn act(&self, g: &Go, s: &GoState, p: usize, rng: &mut Rng) -> usize {
        Mcts::with_eval(self.sims, GoEval, PLAYOUT_DEPTH).act(g, s, p, rng)
    }
}

/// GNU Go behind the [`Agent`] trait. GTP engines are stateful, so the agent
/// mirrors the board: at each turn it diffs the arena's state against its
/// own snapshot to find the move(s) it has not yet relayed (the opening
/// stones at the first turn, exactly one opponent move — or a pass — after
/// that), sends them, and asks for `genmove`.
pub struct GnuGoAgent {
    inner: RefCell<GnuGoInner>,
}

struct GnuGoInner {
    gtp: Option<Gtp>,
    level: u32,
    seed: u32,
    size: usize,
    /// Cells as of the last position this agent relayed to the engine.
    cells: Vec<u8>,
}

impl GnuGoAgent {
    pub fn spawn(level: u32, seed: u32, size: usize) -> GnuGoAgent {
        GnuGoAgent {
            inner: RefCell::new(GnuGoInner {
                gtp: None,
                level,
                seed,
                size,
                cells: vec![2; size * size],
            }),
        }
    }
}

fn vertex(p: usize, size: usize) -> String {
    let (r, c) = (p / size, p % size);
    let col = (b'A' + (c + usize::from(c >= 8)) as u8) as char;
    format!("{col}{}", r + 1)
}

fn parse_vertex(s: &str, size: usize) -> Option<usize> {
    let s = s.trim().to_ascii_uppercase();
    if s == "PASS" {
        return None;
    }
    let mut chars = s.chars();
    let col_letter = chars.next()?;
    let col = col_letter as usize - 'A' as usize - usize::from(col_letter > 'I');
    let row: usize = chars.as_str().parse().ok()?;
    (col < size && (1..=size).contains(&row)).then(|| (row - 1) * size + col)
}

/// The board as `0` Black / `1` White / `2` empty, for move diffing.
fn snapshot(g: &Go, s: &GoState) -> Vec<u8> {
    (0..g.size() * g.size())
        .map(|p| s.stone(p).map_or(2, |c| c as u8))
        .collect()
}

impl GnuGoInner {
    fn relay_and_genmove(&mut self, g: &Go, s: &GoState, us: usize) -> std::io::Result<String> {
        if self.gtp.is_none() {
            self.gtp = Some(Gtp::spawn_gnugo(
                &gnugo_path(),
                self.level,
                self.seed,
                self.size,
            )?);
        }
        let size = g.size();
        let board = snapshot(g, s);
        // Stones on the board that the engine has not seen yet. After the
        // first turn this is at most the opponent's last placement; at the
        // first turn it is the random opening (which is capture-free, so
        // replaying placements in black/white alternation is always legal).
        let mut new_stones: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
        for (p, &c) in board.iter().enumerate() {
            if c != 2 && self.cells[p] != c {
                new_stones[c as usize].push(p);
            }
        }
        let opp = 1 - us;
        let (mut bi, mut wi) = (0, 0);
        let (blacks, whites) = (&new_stones[0], &new_stones[1]);
        // Alternate from black unless white is owed a stone (we are black
        // and the opponent's reply is the only news).
        let mut turn = if blacks.len() >= whites.len() { 0 } else { 1 };
        let gtp = self.gtp.as_mut().expect("spawned above");
        while bi < blacks.len() || wi < whites.len() {
            let (color, p) = if turn == 0 && bi < blacks.len() {
                bi += 1;
                ("black", blacks[bi - 1])
            } else if wi < whites.len() {
                wi += 1;
                ("white", whites[wi - 1])
            } else {
                bi += 1;
                ("black", blacks[bi - 1])
            };
            gtp.cmd(&format!("play {color} {}", vertex(p, size)))?;
            turn ^= 1;
        }
        // No news and it is our move while the game is mid-stream: the
        // opponent passed (or we are opening the game as black).
        if blacks.is_empty() && whites.is_empty() && s.plies() > 0 {
            let color = if opp == 0 { "black" } else { "white" };
            gtp.cmd(&format!("play {color} pass"))?;
        }
        let color = if us == 0 { "black" } else { "white" };
        let reply = gtp.cmd(&format!("genmove {color}"))?;
        Ok(reply)
    }
}

impl Agent<Go> for GnuGoAgent {
    fn act(&self, g: &Go, s: &GoState, p: usize, _rng: &mut Rng) -> usize {
        let actions = g.legal_actions(s);
        let pass_index = actions
            .iter()
            .position(|a| matches!(a, GoAction::Pass))
            .expect("pass is always legal");
        let mut inner = self.inner.borrow_mut();
        let reply = match inner.relay_and_genmove(g, s, p) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("gnugo failed mid-game ({e}); passing from here on");
                inner.gtp = None;
                return pass_index;
            }
        };
        let choice = match parse_vertex(&reply, g.size()) {
            None => pass_index,
            Some(v) => actions
                .iter()
                .position(|a| matches!(a, GoAction::Place(q) if *q as usize == v))
                .unwrap_or(pass_index),
        };
        // Snapshot the position after our chosen move so the next diff sees
        // only the opponent's reply.
        let mut next = s.clone();
        g.apply(&mut next, actions[choice]);
        inner.cells = snapshot(g, &next);
        choice
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
    state: GoState,
    opponent: Opponent,
    agent: Box<dyn Agent<Go> + Send>,
    /// 0 if the net plays Black.
    net_seat: usize,
    search: azero::Search<Go>,
    rng: Rng,
    /// Return for the net once finished.
    outcome: Option<f64>,
}

impl EvalGame {
    /// Plays opponent plies and checks termination; afterwards it is either
    /// finished or the net's turn.
    fn advance_to_net_turn(&mut self, game: &Go) {
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

/// Plays `pairs` paired games per opponent (net as Black then White from the
/// same random opening), all concurrently.
pub fn ladder(
    infer: &Infer,
    opponents: &[Opponent],
    pairs: u32,
    sims: u32,
    size: usize,
    seed: u64,
) -> Vec<LadderEntry> {
    let game = Go::new(size);
    let enc = GoEncoder::new(size);
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
                    agent: opp.agent(
                        size,
                        mix(seed, u64::from(pair) * 2 + net_seat as u64) as u32,
                    ),
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
                    ) {
                        Gather::Requests(reqs) => return reqs,
                        Gather::Done => {
                            let mut visits = g.search.root_visits().to_vec();
                            let actions = g.search.root_actions();
                            goinfer::mask_pass_visits(&game, &g.state, actions, &mut visits);
                            let action = actions[argmax(&visits)];
                            game.apply(&mut g.state, action);
                            // No tree reuse: the opponent moves before our
                            // next search, so the extracted subtree would be
                            // rooted one ply behind the board.
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

/// Plays `pairs` paired games between two fixed opponents (no net), for
/// calibrating the ladder's Elo anchors. Returns `a`'s score and the game
/// count.
pub fn duel(a: Opponent, b: Opponent, pairs: u32, size: usize, seed: u64) -> (f64, u32) {
    let results: Vec<f64> = (0..pairs)
        .into_par_iter()
        .flat_map_iter(|i| {
            let game = Go::new(size);
            let mut rng = Rng::new(mix(seed, u64::from(i) + 1));
            let opening = random_opening(&game, &mut rng);
            let sa = mix(seed, u64::from(i) * 4 + 1) as u32;
            let sb = mix(seed, u64::from(i) * 4 + 2) as u32;
            let as_black = fixed_game(
                &game,
                &*a.agent(size, sa),
                &*b.agent(size, sb),
                opening.clone(),
                &mut rng,
            );
            let as_white = -fixed_game(
                &game,
                &*b.agent(size, sb + 1),
                &*a.agent(size, sa + 1),
                opening,
                &mut rng,
            );
            [as_black, as_white]
        })
        .collect();
    let wins = results.iter().filter(|&&r| r > 0.0).count() as f64;
    let draws = results.iter().filter(|&&r| r == 0.0).count() as f64;
    let games = results.len() as u32;
    ((wins + 0.5 * draws) / f64::from(games), games)
}

/// Plays `pairs` paired games (each random opening with net A as Black, then
/// as White) between two nets — both argmax, pass-masked, no root noise, at
/// `sims`. Returns (net A wins, total games). The KataGo-style relative
/// progress signal: A = current net, B = an older snapshot, so a win rate
/// above 0.5 means the net is still improving. Batched: every cycle the
/// parked leaves are split by whichever net is on move and sent to that net
/// in one forward pass, so it stays cheap.
pub fn net_vs_net(
    a: &Infer,
    b: &Infer,
    pairs: u32,
    sims: u32,
    size: usize,
    seed: u64,
) -> (u32, u32) {
    let game = Go::new(size);
    let enc = GoEncoder::new(size);
    let puct = PuctConfig {
        sims,
        root_noise: 0.0,
        ..PuctConfig::default()
    };

    struct RateGame {
        state: GoState,
        search: azero::Search<Go>,
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
        // Per game: (Some(net_is_a) with leaves to eval, or None when idle/done).
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
                    match g.search.advance(
                        &game,
                        &enc,
                        &g.state,
                        &puct,
                        &mut g.rng,
                        std::mem::take(&mut pending),
                        &|_| false,
                    ) {
                        Gather::Requests(reqs) => {
                            return (Some(g.state.to_move() == g.a_seat), reqs);
                        }
                        Gather::Done => {
                            let mut visits = g.search.root_visits().to_vec();
                            let actions = g.search.root_actions();
                            goinfer::mask_pass_visits(&game, &g.state, actions, &mut visits);
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

/// One game between fixed agents from `opening`; returns Black's result.
fn fixed_game(
    game: &Go,
    black: &dyn Agent<Go>,
    white: &dyn Agent<Go>,
    mut state: GoState,
    rng: &mut Rng,
) -> f64 {
    while !game.is_terminal(&state) {
        let stm = state.to_move();
        let actions = game.legal_actions(&state);
        let agent = if stm == 0 { black } else { white };
        let i = agent.act(game, &state, stm, rng);
        game.apply(&mut state, actions[i]);
    }
    game.returns(&state, 0)
}

/// A capture-free random opening: `OPENING_PLIES` uniform placements (never
/// a pass). A few plies on an empty board cannot capture, which
/// [`GnuGoAgent`] relies on when replaying the opening as a move sequence.
fn random_opening(game: &Go, rng: &mut Rng) -> GoState {
    let mut s = game.initial_state();
    for _ in 0..OPENING_PLIES {
        let placements: Vec<GoAction> = game
            .legal_actions(&s)
            .into_iter()
            .filter(|a| matches!(a, GoAction::Place(_)))
            .collect();
        game.apply(&mut s, placements[rng.below(placements.len())]);
    }
    s
}
