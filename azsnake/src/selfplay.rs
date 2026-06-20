//! Batch-across-games self-play: hundreds of concurrent games each park on
//! one pending net evaluation; every cycle, all parked leaves go to the GPU
//! as a single batch (the CPU side — legality, encoding, tree walking —
//! runs rayon-parallel over the games). Games persist across `collect`
//! calls, so an iteration boundary never abandons work.
//!
//! Snake-specific: food is a chance node. `solvers::azero::Search` resolves
//! chance nodes encountered during descent, but `Search::advance` requires
//! the *root* to be a player node — so each game resolves any pending food
//! spawn before searching its position.

use game_core::rand::sample_visits;
use game_core::{Game, PolicyValueEncoder, Rng, Turn};
use rayon::prelude::*;
use snake::encode::SnakeEncoder;
use snake::{Duel, DuelState};
use solvers::azero::{self, Gather, PuctConfig, argmax};

use crate::net::{EvalRequest, EvalResult, Infer};
use crate::train::Sample;

/// Self-play ply cap: a safety net for mutual circling. The [`Duel`] already
/// caps at `area` steps and scores on length, so this only needs to bound the
/// rare game that the engine's own cap somehow does not — a tighter bound than
/// the old `4·area` keeps self-play decisive rather than churning out long
/// circling draws that the discount would zero out anyway.
fn max_plies(size: usize) -> u16 {
    (size * size) as u16
}

/// Advances `state` through any pending food-spawn chance nodes until a
/// player is on the clock or the game is terminal.
fn resolve_chance(game: &Duel, state: &mut DuelState, rng: &mut Rng) {
    while !game.is_terminal(state) {
        match game.turn(state) {
            Turn::Chance => {
                let outs = game.chance_outcomes(state);
                let i = game_core::rand::sample_outcome(&outs, rng);
                game.apply(state, outs[i].0);
            }
            Turn::Player(_) => break,
        }
    }
}

#[derive(Clone, Copy)]
pub struct SelfPlayConfig {
    pub puct: PuctConfig,
    pub concurrent: usize,
    /// Plies played proportionally to visit counts before switching to argmax.
    pub temp_plies: u16,
    /// Resign when the mover's root Q stays below `-resign_q` for two
    /// consecutive own moves (past `resign_min_ply`). 0 disables.
    pub resign_q: f64,
    pub resign_min_ply: u16,
    /// Fraction of games that ignore resignation, keeping value targets
    /// honest about "lost" positions that turn around.
    pub resign_off: f64,
    /// Playout Cap Randomization (KataGo): when `full_prob > 0`, each move
    /// independently runs either a *full* search (`full_sims`, recorded as a
    /// policy/value target) with probability `full_prob`, or a *fast* search
    /// (`fast_sims`, played but **not** recorded).
    pub fast_sims: u32,
    pub full_sims: u32,
    pub full_prob: f64,
    /// Per-ply value discount toward 0 (`1.0` = undiscounted). A recorded
    /// position `d` plies before the end sees `gamma.powi(d)` of the terminal
    /// value, so a fast win outscores a slow one and a draw-by-circling has a
    /// vanishing target — the lever that breaks the survival equilibrium.
    pub gamma: f32,
    /// How much the length margin sharpens the terminal value: a decisive
    /// outcome is `(1-margin_w) + margin_w·tanh(len_diff/MARGIN_SCALE)` toward
    /// the winner, and a draw at the cap leans `margin_w·tanh(...)` toward the
    /// longer snake, so growing and engaging beat passive survival.
    pub margin_w: f32,
    /// Fraction of games whose seat 1 is a random past checkpoint from the
    /// opponent pool instead of the current net (Asymptotic's ~20% mixing).
    /// Only takes effect once the pool is non-empty; we keep targets on-policy
    /// by recording only the current net's (seat 0) positions in these games.
    pub pool_frac: f64,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        SelfPlayConfig {
            puct: PuctConfig {
                sims: 128,
                dirichlet_alpha: 1.0,
                ..PuctConfig::default()
            },
            concurrent: 256,
            temp_plies: 12,
            resign_q: 0.95,
            resign_min_ply: 20,
            resign_off: 0.1,
            fast_sims: 32,
            full_sims: 256,
            full_prob: 0.0,
            gamma: 0.99,
            margin_w: 0.25,
            pool_frac: 0.2,
        }
    }
}

/// Length difference at which the margin term saturates: a `tanh` over the
/// raw cell difference, so a few cells of lead already register but the term
/// stays bounded.
const MARGIN_SCALE: f32 = 8.0;

/// Seat 0's terminal value in `[-1, 1]`, sharpened by the length margin:
/// decisive games push toward the winner with a margin-scaled magnitude, and
/// a cap draw leans toward the longer snake instead of flat 0.
fn terminal_value(state: &DuelState, margin_w: f32) -> f32 {
    let len_diff = state.score(0) as f32 - state.score(1) as f32;
    let margin = (len_diff / MARGIN_SCALE).tanh();
    match state.outcome() {
        snake::Outcome::Win(0) => (1.0 - margin_w) + margin_w * margin,
        snake::Outcome::Win(_) => -((1.0 - margin_w) + margin_w * (-margin)),
        snake::Outcome::Draw | snake::Outcome::Ongoing => margin_w * margin,
    }
}

#[derive(Default, Clone, Copy)]
pub struct SelfPlayStats {
    pub games: u32,
    pub seat0_wins: u32,
    pub resigned: u32,
    pub capped: u32,
    pub plies: u64,
    pub would_resign: u32,
    pub resign_fp: u32,
    pub cpu_secs: f32,
    pub gpu_secs: f32,
    pub batches: u32,
    pub evals: u64,
}

impl SelfPlayStats {
    fn add_game(&mut self, plies: u16, z_seat0: f32, end: GameEnd, fp: Option<bool>) {
        if let Some(fp) = fp {
            self.would_resign += 1;
            self.resign_fp += u32::from(fp);
        }
        self.games += 1;
        self.plies += u64::from(plies);
        // A decisive seat-0 win carries at least `1 - margin_w` (≥ 0.5 for any
        // sane margin weight); a margin-leaning draw stays below `margin_w`.
        if z_seat0 > 0.5 {
            self.seat0_wins += 1;
        }
        match end {
            GameEnd::Resign => self.resigned += 1,
            GameEnd::PlyCap => self.capped += 1,
            GameEnd::Natural => {}
        }
    }

    pub fn avg_plies(&self) -> f32 {
        if self.games == 0 {
            0.0
        } else {
            self.plies as f32 / self.games as f32
        }
    }
}

enum GameEnd {
    Natural,
    Resign,
    PlyCap,
}

/// (unpacked planes, visit distribution, side to move, root value, ply index)
type Record = (Vec<f32>, Vec<(u16, f32)>, usize, f32, u16);

struct Worker {
    game: Duel,
    state: DuelState,
    search: azero::Search<Duel>,
    rng: Rng,
    records: Vec<Record>,
    plies: u16,
    resign_enabled: bool,
    bad_streak: [u8; 2],
    would_resign: Option<usize>,
    min_q: [f64; 2],
    cur_sims: u32,
    record_move: bool,
    /// When `Some(i)`, seat 1 is played by pool net `i` (a past checkpoint)
    /// instead of the current net — the 80/20 opponent-pool mixing that keeps
    /// self-play from collapsing onto a single cyclic strategy. `None` is a
    /// plain mirror self-play game.
    opponent: Option<usize>,
}

impl Worker {
    /// Which net should serve the search now rooted at this worker: `None` for
    /// the current net, `Some(i)` for pool net `i`. Only seat 1 of an opponent
    /// game uses a pool net; every other root is the current net.
    fn root_net(&self) -> Option<usize> {
        match (self.opponent, self.game.turn(&self.state)) {
            (Some(i), Turn::Player(1)) => Some(i),
            _ => None,
        }
    }
}

enum WorkerStep {
    Requests(Vec<EvalRequest>),
    Finished(Vec<Sample>, u16, f32, GameEnd, Option<bool>, Vec<f64>),
}

impl Worker {
    fn new(seed: u64, cfg: &SelfPlayConfig, pool_len: usize) -> Worker {
        let mut rng = Rng::new(seed);
        let resign_enabled = cfg.resign_q > 0.0 && rng.unit() >= cfg.resign_off;
        let opponent = pick_opponent(cfg, pool_len, &mut rng);
        let game = Duel::new();
        let mut state = game.initial_state();
        resolve_chance(&game, &mut state, &mut rng);
        let mut w = Worker {
            game,
            state,
            search: azero::Search::new(None),
            rng,
            records: Vec::new(),
            plies: 0,
            resign_enabled,
            bad_streak: [0, 0],
            would_resign: None,
            min_q: [1.0, 1.0],
            cur_sims: cfg.puct.sims,
            record_move: true,
            opponent,
        };
        w.roll_move(cfg);
        w
    }

    fn reset(&mut self, cfg: &SelfPlayConfig, pool_len: usize) {
        self.state = self.game.initial_state();
        resolve_chance(&self.game, &mut self.state, &mut self.rng);
        self.search = azero::Search::new(None);
        self.records.clear();
        self.plies = 0;
        self.resign_enabled = cfg.resign_q > 0.0 && self.rng.unit() >= cfg.resign_off;
        self.bad_streak = [0, 0];
        self.would_resign = None;
        self.min_q = [1.0, 1.0];
        self.opponent = pick_opponent(cfg, pool_len, &mut self.rng);
        self.roll_move(cfg);
    }

    fn roll_move(&mut self, cfg: &SelfPlayConfig) {
        if cfg.full_prob > 0.0 {
            self.record_move = self.rng.unit() < cfg.full_prob;
            self.cur_sims = if self.record_move {
                cfg.full_sims
            } else {
                cfg.fast_sims
            };
        } else {
            self.record_move = true;
            self.cur_sims = cfg.puct.sims;
        }
        // A pool opponent's reply is off-policy for the current net: play it
        // out (so the hero faces a real opponent) but never record it as a
        // training target.
        if self.root_net().is_some() {
            self.record_move = false;
        }
    }

    fn advance(&mut self, cfg: &SelfPlayConfig, mut results: Vec<EvalResult>) -> WorkerStep {
        let enc = SnakeEncoder::new();
        let puct = PuctConfig {
            sims: self.cur_sims,
            root_noise: if self.record_move {
                cfg.puct.root_noise
            } else {
                0.0
            },
            ..cfg.puct
        };
        loop {
            match self.search.advance(
                &self.game,
                &enc,
                &self.state,
                &puct,
                &mut self.rng,
                std::mem::take(&mut results),
                &|_| false,
            ) {
                Gather::Requests(reqs) => return WorkerStep::Requests(reqs),
                Gather::Done => {
                    if let Some(step) = self.play_move(cfg) {
                        return step;
                    }
                }
            }
        }
    }

    fn play_move(&mut self, cfg: &SelfPlayConfig) -> Option<WorkerStep> {
        let enc = SnakeEncoder::new();
        let visits = self.search.root_visits().to_vec();
        let actions = self.search.root_actions().to_vec();
        let stm = match self.game.turn(&self.state) {
            Turn::Player(p) => p,
            _ => unreachable!("search root is a player node"),
        };
        if self.record_move {
            let mut tvisits = visits.clone();
            let k = cfg.puct.forced_playouts_k;
            if k > 0.0 {
                let priors = self.search.root_priors();
                let total = f64::from(tvisits.iter().sum::<u32>());
                let best = argmax(&tvisits);
                for i in 0..tvisits.len() {
                    if i != best && tvisits[i] > 0 {
                        let n_forced = (f64::from(k) * f64::from(priors[i]) * total).sqrt() as u32;
                        tvisits[i] = tvisits[i].saturating_sub(n_forced);
                    }
                }
            }
            let dist: Vec<(u16, f32)> = {
                let total: u32 = tvisits.iter().sum();
                actions
                    .iter()
                    .zip(&tvisits)
                    .map(|(&a, &n)| {
                        (
                            enc.action_index(&self.game, &self.state, a) as u16,
                            n as f32 / total as f32,
                        )
                    })
                    .collect()
            };
            self.records.push((
                enc.encode_state(&self.game, &self.state),
                dist,
                stm,
                self.search.root_value() as f32,
                self.plies,
            ));
        }

        let best_q = self.search.root_q();
        if self.plies > cfg.resign_min_ply && best_q < self.min_q[stm] {
            self.min_q[stm] = best_q;
        }
        if cfg.resign_q > 0.0 && self.plies > cfg.resign_min_ply {
            if best_q < -cfg.resign_q {
                self.bad_streak[stm] += 1;
                if self.bad_streak[stm] >= 2 {
                    if self.resign_enabled {
                        // The resigning side concedes a decisive loss; there is
                        // no scored terminal, so the margin term is omitted.
                        let z_seat0 = if stm == 0 { -1.0 } else { 1.0 };
                        return Some(self.finish(cfg, z_seat0, GameEnd::Resign));
                    }
                    if self.would_resign.is_none() {
                        self.would_resign = Some(stm);
                    }
                }
            } else {
                self.bad_streak[stm] = 0;
            }
        }

        let choice = if self.plies < cfg.temp_plies {
            sample_visits(&visits, &mut self.rng)
        } else {
            argmax(&visits)
        };
        self.game.apply(&mut self.state, actions[choice]);
        self.plies += 1;
        // Tree reuse extracts the subtree under the played edge. The search
        // bakes a *sampled* food cell into the tree at expansion; resolving
        // the live game's chance node draws an independent cell, so reuse is
        // only sound when no meal happened (the successor is a player node).
        let crosses_chance = !self.game.is_terminal(&self.state)
            && matches!(self.game.turn(&self.state), Turn::Chance);
        let search = std::mem::replace(&mut self.search, azero::Search::new(None));
        let reuse = if crosses_chance {
            None
        } else {
            search.extract_child(choice)
        };
        resolve_chance(&self.game, &mut self.state, &mut self.rng);
        self.search = azero::Search::new(reuse);

        if self.game.is_terminal(&self.state) {
            let z_seat0 = terminal_value(&self.state, cfg.margin_w);
            let end = if self.plies >= max_plies(self.game.side()) {
                GameEnd::PlyCap
            } else {
                GameEnd::Natural
            };
            return Some(self.finish(cfg, z_seat0, end));
        }
        if self.plies >= max_plies(self.game.side()) {
            let z_seat0 = terminal_value(&self.state, cfg.margin_w);
            return Some(self.finish(cfg, z_seat0, GameEnd::PlyCap));
        }
        self.roll_move(cfg);
        None
    }

    /// `z_seat0` is the (continuous) terminal value from seat 0's view at the
    /// final ply. Each record discounts toward 0 by `gamma^(plies-since)` so a
    /// position far from a win carries a smaller target than one on its
    /// doorstep — rewarding decisive, fast play over circling to the cap.
    fn finish(&mut self, cfg: &SelfPlayConfig, z_seat0: f32, end: GameEnd) -> WorkerStep {
        let terminal_ply = self.plies;
        let samples = self
            .records
            .drain(..)
            .map(|(planes, policy, stm, q, ply)| {
                let discount = cfg.gamma.powi(i32::from(terminal_ply - ply));
                let z_side = if stm == 0 { z_seat0 } else { -z_seat0 };
                Sample {
                    planes,
                    policy,
                    z: discount * z_side,
                    q,
                }
            })
            .collect();
        // Resignation calibration and false-positive accounting key off the
        // sign of the outcome, not its magnitude.
        let fp = self.would_resign.map(|side| {
            let z_side = if side == 0 { z_seat0 } else { -z_seat0 };
            z_side >= 0.0
        });
        let mut calib = Vec::new();
        if !self.resign_enabled {
            for side in 0..2 {
                let z_side = if side == 0 { z_seat0 } else { -z_seat0 };
                if z_side >= 0.0 && self.min_q[side] < 1.0 {
                    calib.push(self.min_q[side]);
                }
            }
        }
        WorkerStep::Finished(samples, self.plies, z_seat0, end, fp, calib)
    }
}

/// Draws an opponent-pool index for a fresh game: with probability
/// `cfg.pool_frac` (and only when the pool is non-empty) seat 1 is a uniformly
/// random past checkpoint, else `None` for plain mirror self-play.
fn pick_opponent(cfg: &SelfPlayConfig, pool_len: usize, rng: &mut Rng) -> Option<usize> {
    (pool_len > 0 && cfg.pool_frac > 0.0 && rng.unit() < cfg.pool_frac).then(|| rng.below(pool_len))
}

/// Persistent self-play pool; call [`SelfPlay::collect`] each iteration.
pub struct SelfPlay {
    cfg: SelfPlayConfig,
    workers: Vec<Worker>,
    results: Vec<Vec<EvalResult>>,
    /// Past-checkpoint nets a fraction of games play as seat 1. Refreshed by
    /// the trainer via [`SelfPlay::set_pool`]; empty means pure self-play.
    pool: Vec<Infer>,
}

impl SelfPlay {
    pub fn new(cfg: SelfPlayConfig, seed: u64) -> SelfPlay {
        let workers = (0..cfg.concurrent)
            .map(|i| Worker::new(mix(seed, i as u64), &cfg, 0))
            .collect::<Vec<_>>();
        let results = (0..cfg.concurrent).map(|_| Vec::new()).collect();
        SelfPlay {
            cfg,
            workers,
            results,
            pool: Vec::new(),
        }
    }

    /// Replaces the opponent pool with the given past-checkpoint nets. New
    /// games (after each finishes) draw from the updated pool; in-flight games
    /// keep the net they started with.
    pub fn set_pool(&mut self, pool: Vec<Infer>) {
        self.pool = pool;
    }

    /// Runs cycles until at least `target_samples` new samples arrive from
    /// finished games. Returns samples, stats, and the resignation-calibration
    /// pool: each control game's non-losing sides' minimum searched Q.
    ///
    /// `infer` is the current net; a fraction of games route seat 1's search
    /// to a net from the opponent pool instead, so requests are split by which
    /// net should serve them and each net is forwarded as its own batch.
    pub fn collect(
        &mut self,
        infer: &Infer,
        target_samples: usize,
    ) -> (Vec<Sample>, SelfPlayStats, Vec<f64>) {
        let mut samples = Vec::with_capacity(target_samples + 4096);
        let mut stats = SelfPlayStats::default();
        let mut calib = Vec::new();
        let pool_len = self.pool.len();
        while samples.len() < target_samples {
            let cfg = self.cfg;
            let cpu_start = std::time::Instant::now();
            type Finished = (Vec<Sample>, u16, f32, GameEnd, Option<bool>, Vec<f64>);
            // `route` is the net each request batch must go to: `None` is the
            // current net, `Some(i)` is pool net `i`.
            let outcomes: Vec<(Option<Finished>, Option<usize>, Vec<EvalRequest>)> = self
                .workers
                .par_iter_mut()
                .zip(self.results.par_iter_mut())
                .map(|(w, r)| match w.advance(&cfg, std::mem::take(r)) {
                    WorkerStep::Requests(reqs) => (None, w.root_net(), reqs),
                    WorkerStep::Finished(s, plies, z, end, fp, calib) => {
                        w.reset(&cfg, pool_len);
                        let WorkerStep::Requests(reqs) = w.advance(&cfg, Vec::new()) else {
                            unreachable!("fresh game cannot finish before any eval");
                        };
                        (Some((s, plies, z, end, fp, calib)), w.root_net(), reqs)
                    }
                })
                .collect();

            // Bucket each worker's requests into the batch for its net, noting
            // (bucket, offset-within-that-batch, len) so results route back.
            // Bucket 0 is the current net; bucket `i+1` is pool net `i`.
            let mut batches: Vec<Vec<EvalRequest>> =
                (0..pool_len + 1).map(|_| Vec::new()).collect();
            let net_bucket = |route: Option<usize>| route.map_or(0, |i| i + 1);
            let mut spans: Vec<(usize, usize, usize)> = Vec::with_capacity(outcomes.len());
            for (fin, route, reqs) in outcomes {
                if let Some((s, plies, z, end, fp, cal)) = fin {
                    samples.extend(s);
                    stats.add_game(plies, z, end, fp);
                    calib.extend(cal);
                }
                let bucket = net_bucket(route);
                spans.push((bucket, batches[bucket].len(), reqs.len()));
                batches[bucket].extend(reqs);
            }
            stats.cpu_secs += cpu_start.elapsed().as_secs_f32();
            stats.batches += 1;
            stats.evals += batches.iter().map(|b| b.len() as u64).sum::<u64>();

            let gpu_start = std::time::Instant::now();
            let mut outs: Vec<Vec<EvalResult>> = batches
                .iter()
                .enumerate()
                .map(|(b, reqs)| {
                    if b == 0 {
                        infer.forward_batch(reqs)
                    } else {
                        self.pool[b - 1].forward_batch(reqs)
                    }
                })
                .collect();
            stats.gpu_secs += gpu_start.elapsed().as_secs_f32();

            // Peel each bucket's results back to its workers from the tail, so
            // `split_off` hands ownership over without cloning `EvalResult`.
            for (i, (bucket, start, len)) in spans.into_iter().enumerate().rev() {
                self.results[i] = outs[bucket].split_off(start);
                debug_assert_eq!(self.results[i].len(), len);
            }
        }
        (samples, stats, calib)
    }

    pub fn set_resign_q(&mut self, resign_q: f64) {
        self.cfg.resign_q = resign_q;
    }
}

pub fn mix(a: u64, b: u64) -> u64 {
    game_core::hash::combine(a, b)
}
