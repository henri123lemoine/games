//! Batch-across-games self-play: hundreds of concurrent games each park on
//! one pending net evaluation; every cycle, all parked leaves go to the GPU
//! as a single batch (the CPU side — movegen, tree walking, encoding — runs
//! rayon-parallel over the games). Games persist across `collect` calls, so
//! an iteration boundary never abandons work; in-flight games simply
//! continue under the next network snapshot.

use std::collections::HashMap;

use chess::encode::{PLANE_COUNT, az_move_index, encode_planes};
use chess::{Board, legal_moves};
use game_core::Rng;
use rayon::prelude::*;

use crate::net::{EvalRequest, EvalResult, Infer};
use azinfer::mcts::{Gather, MctsConfig, Search};

#[derive(Clone, Copy)]
pub struct SelfPlayConfig {
    pub mcts: MctsConfig,
    pub concurrent: usize,
    /// Plies played proportionally to visit counts before switching to
    /// argmax.
    pub temp_plies: u16,
    pub ply_cap: u16,
    /// Resign when the mover's root Q stays below `-resign_q` for two
    /// consecutive own moves (past `resign_min_ply`). 0 disables.
    pub resign_q: f64,
    pub resign_min_ply: u16,
    /// Fraction of games that ignore resignation, keeping value targets
    /// honest about "lost" positions that turn around.
    pub resign_off: f64,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        SelfPlayConfig {
            mcts: MctsConfig::default(),
            concurrent: 512,
            temp_plies: 24,
            ply_cap: 250,
            resign_q: 0.95,
            resign_min_ply: 40,
            resign_off: 0.1,
        }
    }
}

/// One training example, planes packed as bitboards (plane 17, the halfmove
/// clock, is a uniform fill reconstructed from `halfmove`).
pub struct Sample {
    pub planes: [u64; 17],
    pub halfmove: u8,
    /// Sparse visit distribution over AZ policy indices.
    pub policy: Vec<(u16, f32)>,
    /// Game outcome from the perspective of the player to move.
    pub z: f32,
    /// The search's root value at this position (player to move) — mixed
    /// into the value target to de-noise the raw outcome.
    pub q: f32,
}

#[derive(Default, Clone, Copy)]
pub struct SelfPlayStats {
    pub games: u32,
    pub decisive: u32,
    pub resigned: u32,
    pub repetition_draws: u32,
    pub capped: u32,
    pub plies: u64,
    /// Resign-disabled games where the would-resign side did NOT lose:
    /// direct measure of resignation false positives.
    pub would_resign: u32,
    pub resign_fp: u32,
    /// Wall-clock split of the collect loop, for utilization tuning.
    pub cpu_secs: f32,
    pub gpu_secs: f32,
    pub batches: u32,
    pub evals: u64,
}

impl SelfPlayStats {
    fn add_game(&mut self, plies: u16, z_white: f32, end: GameEnd, fp: Option<bool>) {
        if let Some(fp) = fp {
            self.would_resign += 1;
            self.resign_fp += u32::from(fp);
        }
        self.games += 1;
        self.plies += u64::from(plies);
        if z_white != 0.0 {
            self.decisive += 1;
        }
        match end {
            GameEnd::Resign => self.resigned += 1,
            GameEnd::Repetition => self.repetition_draws += 1,
            GameEnd::PlyCap => self.capped += 1,
            _ => {}
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
    Repetition,
    PlyCap,
}

type Record = (([u64; 17], u8), Vec<(u16, f32)>, chess::Color, f32);

struct Worker {
    board: Board,
    search: Search,
    rng: Rng,
    key_counts: HashMap<u64, u8>,
    records: Vec<Record>,
    plies: u16,
    resign_enabled: bool,
    /// Consecutive own moves with root Q below the resign bar, per color.
    bad_streak: [u8; 2],
    /// First side that hit the resign bar while resignation was disabled.
    would_resign: Option<usize>,
    /// Lowest searched best-edge Q seen by each side, for calibrating the
    /// resignation threshold from no-resign control games.
    min_q: [f64; 2],
}

enum WorkerStep {
    Requests(Vec<EvalRequest>),
    Finished(Vec<Sample>, u16, f32, GameEnd, Option<bool>, Vec<f64>),
}

impl Worker {
    fn new(seed: u64, cfg: &SelfPlayConfig) -> Worker {
        let mut rng = Rng::new(seed);
        let resign_enabled = cfg.resign_q > 0.0 && rng.unit() >= cfg.resign_off;
        let board = Board::start();
        let mut key_counts = HashMap::new();
        key_counts.insert(board.key(), 1);
        Worker {
            board,
            search: Search::new(None),
            rng,
            key_counts,
            records: Vec::new(),
            plies: 0,
            resign_enabled,
            bad_streak: [0, 0],
            would_resign: None,
            min_q: [1.0, 1.0],
        }
    }

    fn reset(&mut self, cfg: &SelfPlayConfig) {
        self.board = Board::start();
        self.search = Search::new(None);
        self.key_counts.clear();
        self.key_counts.insert(self.board.key(), 1);
        self.records.clear();
        self.plies = 0;
        self.resign_enabled = cfg.resign_q > 0.0 && self.rng.unit() >= cfg.resign_off;
        self.bad_streak = [0, 0];
        self.would_resign = None;
        self.min_q = [1.0, 1.0];
    }

    fn advance(&mut self, cfg: &SelfPlayConfig, mut results: Vec<EvalResult>) -> WorkerStep {
        loop {
            match self.search.advance(
                &self.board,
                &self.key_counts,
                &cfg.mcts,
                &mut self.rng,
                std::mem::take(&mut results),
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

    /// Plays the searched move; returns `Some(Finished)` when the game ends.
    fn play_move(&mut self, cfg: &SelfPlayConfig) -> Option<WorkerStep> {
        let visits = self.search.root_visits().to_vec();
        let moves = self.search.root_moves().to_vec();
        let stm = self.board.stm;
        let dist: Vec<(u16, f32)> = {
            let total: u32 = visits.iter().sum();
            moves
                .iter()
                .zip(&visits)
                .map(|(&m, &n)| (az_move_index(m, stm) as u16, n as f32 / total as f32))
                .collect()
        };
        self.records.push((
            compact_planes(&self.board),
            dist,
            stm,
            self.search.root_value() as f32,
        ));
        let side = stm.index();
        let best_q = self.search.root_q();
        if self.plies > cfg.resign_min_ply && best_q < self.min_q[side] {
            self.min_q[side] = best_q;
        }

        if cfg.resign_q > 0.0 && self.plies > cfg.resign_min_ply {
            if best_q < -cfg.resign_q {
                self.bad_streak[side] += 1;
                if self.bad_streak[side] >= 2 {
                    if self.resign_enabled {
                        let z_white = if side == 0 { -1.0 } else { 1.0 };
                        return Some(self.finish(z_white, GameEnd::Resign));
                    }
                    if self.would_resign.is_none() {
                        self.would_resign = Some(side);
                    }
                }
            } else {
                self.bad_streak[side] = 0;
            }
        }

        let choice = if self.plies < cfg.temp_plies {
            sample_proportional(&visits, &mut self.rng)
        } else {
            argmax(&visits)
        };
        self.board.apply(moves[choice]);
        self.plies += 1;
        let search = std::mem::replace(&mut self.search, Search::new(None));
        self.search = Search::new(search.extract_child(choice));

        let reps = self.key_counts.entry(self.board.key()).or_insert(0);
        *reps += 1;
        if *reps >= 3 {
            return Some(self.finish(0.0, GameEnd::Repetition));
        }
        if self.plies >= cfg.ply_cap {
            return Some(self.finish(0.0, GameEnd::PlyCap));
        }
        if self.board.halfmove >= 100 || self.board.insufficient_material() {
            return Some(self.finish(0.0, GameEnd::Natural));
        }
        if legal_moves(&self.board).is_empty() {
            let z_white = if self.board.in_check(self.board.stm) {
                // The side to move is mated; the previous mover won.
                if self.board.stm == chess::Color::White {
                    -1.0
                } else {
                    1.0
                }
            } else {
                0.0
            };
            return Some(self.finish(z_white, GameEnd::Natural));
        }
        None
    }

    fn finish(&mut self, z_white: f32, end: GameEnd) -> WorkerStep {
        let samples = self
            .records
            .drain(..)
            .map(|((planes, halfmove), policy, stm, q)| Sample {
                planes,
                halfmove,
                policy,
                z: if stm == chess::Color::White {
                    z_white
                } else {
                    -z_white
                },
                q,
            })
            .collect();
        let fp = self.would_resign.map(|side| {
            let z_side = if side == 0 { z_white } else { -z_white };
            z_side >= 0.0
        });
        // Non-losing sides' minimum Q from control games: the distribution
        // the resignation threshold is calibrated against.
        let mut calib = Vec::new();
        if !self.resign_enabled {
            for side in 0..2 {
                let z_side = if side == 0 { z_white } else { -z_white };
                if z_side >= 0.0 && self.min_q[side] < 1.0 {
                    calib.push(self.min_q[side]);
                }
            }
        }
        WorkerStep::Finished(samples, self.plies, z_white, end, fp, calib)
    }
}

/// Persistent self-play pool; call [`SelfPlay::collect`] each iteration.
pub struct SelfPlay {
    cfg: SelfPlayConfig,
    workers: Vec<Worker>,
    results: Vec<Vec<EvalResult>>,
}

impl SelfPlay {
    pub fn new(cfg: SelfPlayConfig, seed: u64) -> SelfPlay {
        let workers = (0..cfg.concurrent)
            .map(|i| Worker::new(mix(seed, i as u64), &cfg))
            .collect::<Vec<_>>();
        let results = (0..cfg.concurrent).map(|_| Vec::new()).collect();
        SelfPlay {
            cfg,
            workers,
            results,
        }
    }

    /// Runs cycles until at least `target_samples` new samples arrive from
    /// finished games. Unfinished games stay parked (with their pending
    /// leaf results delivered) for the next call.
    /// Returns samples, stats, and the resignation-calibration pool: each
    /// control game's non-losing sides' minimum searched Q.
    pub fn collect(
        &mut self,
        infer: &Infer,
        target_samples: usize,
    ) -> (Vec<Sample>, SelfPlayStats, Vec<f64>) {
        let mut samples = Vec::with_capacity(target_samples + 4096);
        let mut stats = SelfPlayStats::default();
        let mut calib = Vec::new();
        while samples.len() < target_samples {
            let cfg = self.cfg;
            let cpu_start = std::time::Instant::now();
            type Finished = (Vec<Sample>, u16, f32, GameEnd, Option<bool>, Vec<f64>);
            let outcomes: Vec<(Option<Finished>, Vec<EvalRequest>)> = self
                .workers
                .par_iter_mut()
                .zip(self.results.par_iter_mut())
                .map(|(w, r)| match w.advance(&cfg, std::mem::take(r)) {
                    WorkerStep::Requests(reqs) => (None, reqs),
                    WorkerStep::Finished(s, plies, z, end, fp, calib) => {
                        // Deal the next game immediately so the batch keeps
                        // its width; a fresh game always needs a root eval.
                        w.reset(&cfg);
                        let WorkerStep::Requests(reqs) = w.advance(&cfg, Vec::new()) else {
                            unreachable!("fresh game cannot finish before any eval");
                        };
                        (Some((s, plies, z, end, fp, calib)), reqs)
                    }
                })
                .collect();

            let mut flat: Vec<EvalRequest> = Vec::new();
            let mut spans: Vec<(usize, usize)> = Vec::with_capacity(outcomes.len());
            for (fin, reqs) in outcomes {
                if let Some((s, plies, z, end, fp, cal)) = fin {
                    samples.extend(s);
                    stats.add_game(plies, z, end, fp);
                    calib.extend(cal);
                }
                spans.push((flat.len(), reqs.len()));
                flat.extend(reqs);
            }
            stats.cpu_secs += cpu_start.elapsed().as_secs_f32();
            stats.batches += 1;
            stats.evals += flat.len() as u64;
            let gpu_start = std::time::Instant::now();
            let mut outs = infer.forward_batch(&flat);
            stats.gpu_secs += gpu_start.elapsed().as_secs_f32();
            for (i, (start, len)) in spans.into_iter().enumerate().rev() {
                self.results[i] = outs.split_off(start);
                debug_assert_eq!(self.results[i].len(), len);
            }
        }
        (samples, stats, calib)
    }

    /// Updates the resignation threshold (used as `Q < -resign_q`).
    pub fn set_resign_q(&mut self, resign_q: f64) {
        self.cfg.resign_q = resign_q;
    }
}

/// Bit-packs the binary planes of [`encode_planes`]; plane 17 is uniform
/// `halfmove / 100`, stored as the raw counter.
fn compact_planes(b: &Board) -> ([u64; 17], u8) {
    let x = encode_planes(b);
    let mut planes = [0u64; 17];
    for (p, plane) in planes.iter_mut().enumerate() {
        for sq in 0..64 {
            if x[p * 64 + sq] != 0.0 {
                *plane |= 1 << sq;
            }
        }
    }
    (planes, b.halfmove.min(100) as u8)
}

pub fn expand_planes(planes: &[u64; 17], halfmove: u8, out: &mut [f32]) {
    debug_assert_eq!(out.len(), PLANE_COUNT * 64);
    out.fill(0.0);
    for (p, &bits) in planes.iter().enumerate() {
        let mut b = bits;
        while b != 0 {
            let sq = b.trailing_zeros() as usize;
            out[p * 64 + sq] = 1.0;
            b &= b - 1;
        }
    }
    out[17 * 64..].fill(f32::from(halfmove) / 100.0);
}

fn sample_proportional(visits: &[u32], rng: &mut Rng) -> usize {
    let total: u32 = visits.iter().sum();
    if total == 0 {
        return 0;
    }
    let mut r = rng.unit() * f64::from(total);
    for (i, &n) in visits.iter().enumerate() {
        r -= f64::from(n);
        if r < 0.0 {
            return i;
        }
    }
    visits.len() - 1
}

pub use azinfer::argmax;

pub fn mix(a: u64, b: u64) -> u64 {
    game_core::hash::combine(a, b)
}
