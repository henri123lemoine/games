//! Batch-across-games self-play: hundreds of concurrent games each park on
//! one pending net evaluation; every cycle, all parked leaves go to the GPU
//! as a single batch (the CPU side — legality, encoding, tree walking —
//! runs rayon-parallel over the games). Games persist across `collect`
//! calls, so an iteration boundary never abandons work.
//!
//! Go specifics vs the chess harness (`azt`): no repetition machinery (the
//! game's simple ko lives in the state, and the draw-guard ply cap makes
//! every line terminate), no draws at all with komi 7.5, and a Dirichlet
//! alpha scaled for ~50 legal moves rather than chess's ~30.

use game_core::rand::sample_visits;
use game_core::{Game, PolicyValueEncoder, Rng};
use go::encode::GoEncoder;
use go::{Go, GoState};
use rayon::prelude::*;
use solvers::azero::{self, Gather, PuctConfig, argmax};

use crate::net::{EvalRequest, EvalResult, Infer};
use crate::train::{Sample, compact};

/// The go crate's draw-guard ends a game at `4·size²` plies; matching it
/// here only classifies that ending as a cap rather than a two-pass finish.
fn max_plies(size: usize) -> u16 {
    (4 * size * size) as u16
}

#[derive(Clone, Copy)]
pub struct SelfPlayConfig {
    pub puct: PuctConfig,
    pub concurrent: usize,
    /// Plies played proportionally to visit counts before switching to
    /// argmax.
    pub temp_plies: u16,
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
            puct: PuctConfig {
                sims: 192,
                dirichlet_alpha: 0.15,
                ..PuctConfig::default()
            },
            concurrent: 768,
            temp_plies: 10,
            resign_q: 0.95,
            resign_min_ply: 20,
            resign_off: 0.1,
        }
    }
}

#[derive(Default, Clone, Copy)]
pub struct SelfPlayStats {
    pub games: u32,
    pub black_wins: u32,
    pub resigned: u32,
    /// Games ended by the draw-guard ply cap rather than two passes.
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
    fn add_game(&mut self, plies: u16, z_black: f32, end: GameEnd, fp: Option<bool>) {
        if let Some(fp) = fp {
            self.would_resign += 1;
            self.resign_fp += u32::from(fp);
        }
        self.games += 1;
        self.plies += u64::from(plies);
        if z_black > 0.0 {
            self.black_wins += 1;
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

/// (packed planes, stm-is-white, visit distribution, side to move, root value)
type Record = (Box<[u64]>, bool, Vec<(u16, f32)>, usize, f32);

struct Worker {
    state: GoState,
    search: azero::Search<Go>,
    rng: Rng,
    records: Vec<Record>,
    plies: u16,
    resign_enabled: bool,
    /// Consecutive own moves with root Q below the resign bar, per player.
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
    fn new(game: &Go, seed: u64, cfg: &SelfPlayConfig) -> Worker {
        let mut rng = Rng::new(seed);
        let resign_enabled = cfg.resign_q > 0.0 && rng.unit() >= cfg.resign_off;
        Worker {
            state: game.initial_state(),
            search: azero::Search::new(None),
            rng,
            records: Vec::new(),
            plies: 0,
            resign_enabled,
            bad_streak: [0, 0],
            would_resign: None,
            min_q: [1.0, 1.0],
        }
    }

    fn reset(&mut self, game: &Go, cfg: &SelfPlayConfig) {
        self.state = game.initial_state();
        self.search = azero::Search::new(None);
        self.records.clear();
        self.plies = 0;
        self.resign_enabled = cfg.resign_q > 0.0 && self.rng.unit() >= cfg.resign_off;
        self.bad_streak = [0, 0];
        self.would_resign = None;
        self.min_q = [1.0, 1.0];
    }

    fn advance(
        &mut self,
        game: &Go,
        cfg: &SelfPlayConfig,
        mut results: Vec<EvalResult>,
    ) -> WorkerStep {
        let enc = GoEncoder::new(game.size());
        loop {
            match self.search.advance(
                game,
                &enc,
                &self.state,
                &cfg.puct,
                &mut self.rng,
                std::mem::take(&mut results),
                &|_| false,
            ) {
                Gather::Requests(reqs) => return WorkerStep::Requests(reqs),
                Gather::Done => {
                    if let Some(step) = self.play_move(game, cfg) {
                        return step;
                    }
                }
            }
        }
    }

    /// Plays the searched move; returns `Some(Finished)` when the game ends.
    fn play_move(&mut self, game: &Go, cfg: &SelfPlayConfig) -> Option<WorkerStep> {
        let enc = GoEncoder::new(game.size());
        let mut visits = self.search.root_visits().to_vec();
        let actions = self.search.root_actions().to_vec();
        // Forbid passing while productive moves remain — both for the played
        // move and the recorded policy target — so the net never learns the
        // area-scoring pass-early collapse.
        goinfer::mask_pass_visits(game, &self.state, &actions, &mut visits);
        let stm = self.state.to_move();
        let dist: Vec<(u16, f32)> = {
            let total: u32 = visits.iter().sum();
            actions
                .iter()
                .zip(&visits)
                .map(|(&a, &n)| {
                    (
                        enc.action_index(game, &self.state, a) as u16,
                        n as f32 / total as f32,
                    )
                })
                .collect()
        };
        let (planes, stm_white) = compact(&enc.encode_state(game, &self.state), game.size());
        self.records.push((
            planes,
            stm_white,
            dist,
            stm,
            self.search.root_value() as f32,
        ));

        let best_q = self.search.root_q();
        if self.plies > cfg.resign_min_ply && best_q < self.min_q[stm] {
            self.min_q[stm] = best_q;
        }
        if cfg.resign_q > 0.0 && self.plies > cfg.resign_min_ply {
            if best_q < -cfg.resign_q {
                self.bad_streak[stm] += 1;
                if self.bad_streak[stm] >= 2 {
                    if self.resign_enabled {
                        let z_black = if stm == 0 { -1.0 } else { 1.0 };
                        return Some(self.finish(game, z_black, GameEnd::Resign));
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
        game.apply(&mut self.state, actions[choice]);
        self.plies += 1;
        let search = std::mem::replace(&mut self.search, azero::Search::new(None));
        self.search = azero::Search::new(search.extract_child(choice));

        if game.is_terminal(&self.state) {
            let z_black = game.returns(&self.state, 0) as f32;
            let end = if self.plies >= max_plies(game.size()) {
                GameEnd::PlyCap
            } else {
                GameEnd::Natural
            };
            return Some(self.finish(game, z_black, end));
        }
        None
    }

    fn finish(&mut self, game: &Go, z_black: f32, end: GameEnd) -> WorkerStep {
        // The final board's ownership — the same dense territory target for
        // every position in this game.
        let ownership: Box<[i8]> = game
            .ownership(&self.state)
            .iter()
            .map(|&o| o as i8)
            .collect();
        let samples = self
            .records
            .drain(..)
            .map(|(planes, stm_white, policy, stm, q)| Sample {
                planes,
                stm_white,
                policy,
                z: if stm == 0 { z_black } else { -z_black },
                q,
                ownership: ownership.clone(),
            })
            .collect();
        let fp = self.would_resign.map(|side| {
            let z_side = if side == 0 { z_black } else { -z_black };
            z_side >= 0.0
        });
        // Non-losing sides' minimum Q from control games: the distribution
        // the resignation threshold is calibrated against.
        let mut calib = Vec::new();
        if !self.resign_enabled {
            for side in 0..2 {
                let z_side = if side == 0 { z_black } else { -z_black };
                if z_side >= 0.0 && self.min_q[side] < 1.0 {
                    calib.push(self.min_q[side]);
                }
            }
        }
        WorkerStep::Finished(samples, self.plies, z_black, end, fp, calib)
    }
}

/// Persistent self-play pool; call [`SelfPlay::collect`] each iteration.
pub struct SelfPlay {
    game: Go,
    cfg: SelfPlayConfig,
    workers: Vec<Worker>,
    results: Vec<Vec<EvalResult>>,
}

impl SelfPlay {
    pub fn new(cfg: SelfPlayConfig, size: usize, seed: u64) -> SelfPlay {
        let game = Go::new(size);
        let workers = (0..cfg.concurrent)
            .map(|i| Worker::new(&game, mix(seed, i as u64), &cfg))
            .collect::<Vec<_>>();
        let results = (0..cfg.concurrent).map(|_| Vec::new()).collect();
        SelfPlay {
            game,
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
            let game = &self.game;
            let cpu_start = std::time::Instant::now();
            type Finished = (Vec<Sample>, u16, f32, GameEnd, Option<bool>, Vec<f64>);
            let outcomes: Vec<(Option<Finished>, Vec<EvalRequest>)> = self
                .workers
                .par_iter_mut()
                .zip(self.results.par_iter_mut())
                .map(|(w, r)| match w.advance(game, &cfg, std::mem::take(r)) {
                    WorkerStep::Requests(reqs) => (None, reqs),
                    WorkerStep::Finished(s, plies, z, end, fp, calib) => {
                        // Deal the next game immediately so the batch keeps
                        // its width; a fresh game always needs a root eval.
                        w.reset(game, &cfg);
                        let WorkerStep::Requests(reqs) = w.advance(game, &cfg, Vec::new()) else {
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

pub fn mix(a: u64, b: u64) -> u64 {
    game_core::hash::combine(a, b)
}
