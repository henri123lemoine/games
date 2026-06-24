//! Batch-across-games self-play: hundreds of concurrent games each park on one
//! pending net evaluation; every cycle, all parked leaves go to the GPU as a
//! single batch (the CPU side — legality, encoding, tree walking — runs
//! rayon-parallel over the games). Games persist across `collect` calls, so an
//! iteration boundary never abandons work.
//!
//! Pente specifics vs the go harness: no chance nodes, no komi, no pass, no
//! ownership/score targets. The first move is forced to the board center (a
//! single legal action at the empty board); the search and the policy-target
//! code handle a one-edge root without a special case, and the normalization
//! guards a zero visit total just in case.

use game_core::rand::sample_visits;
use game_core::{Game, PolicyValueEncoder, Proof, Rng};
use pente::encode::PenteEncoder;
use pente::{Pente, PenteProver, PenteState, VcfConfig};
use rayon::prelude::*;
use solvers::azero::{self, Gather, PuctConfig, argmax};

use super::sample::{Sample, compact};
use crate::net::{EvalRequest, EvalResult, Infer};

/// Self-play ply cap: a hard safety net. Pente terminates by a full board at the
/// latest, so a game can never exceed `size²` placements; this only classifies a
/// game reaching that bound as a cap rather than a natural finish.
fn max_plies(size: usize) -> u16 {
    (size * size) as u16
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
    /// Fraction of games that ignore resignation, keeping value targets honest
    /// about "lost" positions that turn around.
    pub resign_off: f64,
    /// Playout Cap Randomization (KataGo): when `full_prob > 0`, each move
    /// independently runs either a *full* search (`full_sims`, recorded as a
    /// policy/value target) with probability `full_prob`, or a *fast* search
    /// (`fast_sims`, played but **not** recorded).
    pub fast_sims: u32,
    pub full_sims: u32,
    pub full_prob: f64,
    /// The *per-leaf* forcing-solver budget: the VCF+VCT prover runs at every
    /// MCTS leaf as the search's `TerminalProver`, backing up proven wins as
    /// exact ±1. The budget is deliberately small — it runs at every expanded
    /// leaf and `winning_move` clones state — so it never tanks throughput while
    /// still proving the short forcing wins that sharpen value targets.
    pub vcf: VcfConfig,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        SelfPlayConfig {
            puct: PuctConfig {
                sims: 192,
                dirichlet_alpha: 0.15,
                ..PuctConfig::default()
            },
            concurrent: 512,
            temp_plies: 10,
            resign_q: 0.95,
            resign_min_ply: 20,
            resign_off: 0.1,
            fast_sims: 100,
            full_sims: 400,
            full_prob: 0.0,
            // Conservative per-leaf budget for self-play *throughput*: the prover
            // runs at every expanded leaf across hundreds of concurrent games, so
            // it must be cheap. Iterative-deepening cost is dominated by depth
            // (each quiet leaf spends its whole budget proving no win exists), so
            // a shallow depth 5 / 250 nodes keeps self-play moving (~10× the
            // no-prover floor here, vs ~35× at the depth-7/1500 play-time budget)
            // while still proving every short forcing win — open/double fours,
            // fifth-pair captures. The richer play-time budget lives in the
            // native/wasm bots, which run one game at a time.
            vcf: VcfConfig {
                max_depth: 5,
                max_nodes: 250,
                ..VcfConfig::default()
            },
        }
    }
}

#[derive(Default, Clone, Copy)]
pub struct SelfPlayStats {
    pub games: u32,
    pub black_wins: u32,
    pub resigned: u32,
    /// Games ended by the ply cap (a full-board draw) rather than a win.
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

/// (packed planes, mover pairs, opponent pairs, visit distribution, side to
/// move, root value)
type Record = (Box<[u64]>, u8, u8, Vec<(u16, f32)>, usize, f32);

struct Worker {
    pente: Pente,
    state: PenteState,
    search: azero::Search<Pente>,
    rng: Rng,
    records: Vec<Record>,
    plies: u16,
    resign_enabled: bool,
    bad_streak: [u8; 2],
    would_resign: Option<usize>,
    min_q: [f64; 2],
    cur_sims: u32,
    record_move: bool,
}

enum WorkerStep {
    Requests(Vec<EvalRequest>),
    Finished(Vec<Sample>, u16, f32, GameEnd, Option<bool>, Vec<f64>),
}

impl Worker {
    fn new(size: usize, seed: u64, cfg: &SelfPlayConfig) -> Worker {
        let mut rng = Rng::new(seed);
        let resign_enabled = cfg.resign_q > 0.0 && rng.unit() >= cfg.resign_off;
        let pente = Pente::new(size);
        let mut w = Worker {
            state: pente.initial_state(),
            pente,
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
        };
        w.roll_move(cfg);
        w
    }

    fn reset(&mut self, cfg: &SelfPlayConfig) {
        self.state = self.pente.initial_state();
        self.search = azero::Search::new(None);
        self.records.clear();
        self.plies = 0;
        self.resign_enabled = cfg.resign_q > 0.0 && self.rng.unit() >= cfg.resign_off;
        self.bad_streak = [0, 0];
        self.would_resign = None;
        self.min_q = [1.0, 1.0];
        self.roll_move(cfg);
    }

    /// Draw this move's Playout Cap Randomization outcome: with probability
    /// `full_prob` a full, recorded search; otherwise a cheap, unrecorded one.
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
    }

    fn advance(&mut self, cfg: &SelfPlayConfig, mut results: Vec<EvalResult>) -> WorkerStep {
        let pente = self.pente;
        let enc = PenteEncoder::new(pente.size());
        let puct = PuctConfig {
            sims: self.cur_sims,
            root_noise: if self.record_move {
                cfg.puct.root_noise
            } else {
                0.0
            },
            ..cfg.puct
        };
        let prover = PenteProver { cfg: cfg.vcf };
        loop {
            match self.search.advance(
                &pente,
                &enc,
                &self.state,
                &puct,
                &mut self.rng,
                std::mem::take(&mut results),
                &|_| false,
                Some(&prover),
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
        let pente = self.pente;
        let enc = PenteEncoder::new(pente.size());
        let visits = self.search.root_visits().to_vec();
        let actions = self.search.root_actions().to_vec();
        let stm = self.state.to_move();
        if self.record_move {
            // Policy target pruning: subtract the forced-playout visits back out
            // (the played move — max visits — is never pruned). No-op when
            // `forced_playouts_k == 0`.
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
            let total: u32 = tvisits.iter().sum();
            // A one-edge root (the forced center opening) and any zero-visit edge
            // case fall back to a uniform target over the available actions.
            let dist: Vec<(u16, f32)> = if total == 0 {
                let w = 1.0 / actions.len() as f32;
                actions
                    .iter()
                    .map(|&a| (enc.action_index(&pente, &self.state, a) as u16, w))
                    .collect()
            } else {
                actions
                    .iter()
                    .zip(&tvisits)
                    .map(|(&a, &n)| {
                        (
                            enc.action_index(&pente, &self.state, a) as u16,
                            n as f32 / total as f32,
                        )
                    })
                    .collect()
            };
            let pairs = self.state.pairs();
            self.records.push((
                compact(&enc.encode_state(&pente, &self.state), pente.size()),
                pairs[stm],
                pairs[stm ^ 1],
                dist,
                stm,
                self.search.root_value() as f32,
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
                        let z_black = if stm == 0 { -1.0 } else { 1.0 };
                        return Some(self.finish(z_black, GameEnd::Resign));
                    }
                    if self.would_resign.is_none() {
                        self.would_resign = Some(stm);
                    }
                }
            } else {
                self.bad_streak[stm] = 0;
            }
        }

        // A solver-proven root win is exact — play the winning move over the
        // visit-based choice. The proof bubbled up from a winning child witnesses
        // its edge; a root the prover proves *directly* (its own leaf) carries
        // the verdict but no edge, so resolve the witnessing move from the
        // solver itself. The policy target above is still the visit distribution
        // (the net learns the search's policy, not the single forcing move), and
        // the value target benefits automatically: a proven node backs up its
        // exact ±1.
        let proven_win = (self.search.root_proof() == Some(Proof::Win))
            .then(|| {
                pente::winning_move(&pente, &self.state, cfg.vcf)
                    .and_then(|win| actions.iter().position(|&a| a == win))
                    .or_else(|| self.search.best_proven_action())
            })
            .flatten();
        let choice = match proven_win {
            Some(idx) => idx,
            None if self.plies < cfg.temp_plies => sample_visits(&visits, &mut self.rng),
            None => argmax(&visits),
        };
        pente.apply(&mut self.state, actions[choice]);
        self.plies += 1;
        let search = std::mem::replace(&mut self.search, azero::Search::new(None));
        self.search = azero::Search::new(search.extract_child(choice));

        if pente.is_terminal(&self.state) {
            let z_black = pente.returns(&self.state, 0) as f32;
            let end = if self.plies >= max_plies(pente.size()) {
                GameEnd::PlyCap
            } else {
                GameEnd::Natural
            };
            return Some(self.finish(z_black, end));
        }
        self.roll_move(cfg);
        None
    }

    fn finish(&mut self, z_black: f32, end: GameEnd) -> WorkerStep {
        let samples = self
            .records
            .drain(..)
            .map(|(planes, own_pairs, opp_pairs, policy, stm, q)| Sample {
                planes,
                policy,
                z: if stm == 0 { z_black } else { -z_black },
                q,
                own_pairs,
                opp_pairs,
                size: self.pente.size() as u8,
            })
            .collect();
        let fp = self.would_resign.map(|side| {
            let z_side = if side == 0 { z_black } else { -z_black };
            z_side >= 0.0
        });
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
    cfg: SelfPlayConfig,
    workers: Vec<Worker>,
    results: Vec<Vec<EvalResult>>,
}

impl SelfPlay {
    pub fn new(cfg: SelfPlayConfig, size: usize, seed: u64) -> SelfPlay {
        let workers = (0..cfg.concurrent)
            .map(|i| Worker::new(size, mix(seed, i as u64), &cfg))
            .collect::<Vec<_>>();
        let results = (0..cfg.concurrent).map(|_| Vec::new()).collect();
        SelfPlay {
            cfg,
            workers,
            results,
        }
    }

    /// Runs cycles until at least `target_samples` new samples arrive from
    /// finished games. Returns samples, stats, and the resignation-calibration
    /// pool: each control game's non-losing sides' minimum searched Q.
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

pub fn mix(a: u64, b: u64) -> u64 {
    game_core::hash::combine(a, b)
}
