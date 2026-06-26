//! Deploy self-play training: fit the ReBeL value net on the REAL multi-round,
//! N-player Liar's Dice.
//!
//! Each generated episode samples a round configuration, builds a
//! [`LiarsDiceAdapter`] for that round closed by a [`DeployCont`] continuation
//! (the [`DiceShareValue`] heuristic during a short warmup, then
//! [`NetContinuation`] — the net valuing its OWN round openings, the bootstrap),
//! depth-limits the subgame to `max_depth`, and runs the recursive self-play data
//! generator. The net is trained on the backed-up per-round CFR values, so one
//! generate+train pass is one fitted-value-iteration Bellman backup over the
//! per-round decomposition.
//!
//! The 2-player gate ([`DeployTrainConfig::fixed_config`]) trains a single small
//! config and measures, per continuing round, the exploitability of the
//! net-resolved policy against the EXACT continuation lattice — driven below the
//! near-Nash gate as the net learns the round-opening values.

use std::path::PathBuf;

use game_core::Rng;
use rayon::prelude::*;

use solvers::rebel_mlp::Sample;

use crate::rebel::adapter::LiarsDiceAdapter;
use crate::rebel::buffer::Reservoir;
use crate::rebel::cfr::{CfrParams, Solver};
use crate::rebel::deploy::{DeployCont, NetContinuation};
use crate::rebel::exploit::exploitability;
use crate::rebel::game::RebelGame;
use crate::rebel::hands::MAX_DICE;
use crate::rebel::leaf::TerminalLeaf;
use crate::rebel::pbs::{Belief, MAX_SEATS};
use crate::rebel::selfplay::{SelfPlayParams, generate_episode};
use crate::rebel::value_net::{MAX_REBEL_SEATS, PbsNet};
use crate::subgame::{ContinuationValue, DiceShareValue};
use crate::{FitConfig, LatticeValue, fit_two_player};

/// Largest total dice in a sampled round: `MAX_REBEL_SEATS * MAX_DICE` (the net's
/// encoding spans ≤6 seats × ≤5 dice), an explicit resample guard rather than a
/// curtailment of the flagship 5p5d6f = 25-dice regime.
const MAX_DEPLOY_TOTAL: u32 = (MAX_REBEL_SEATS * MAX_DICE) as u32;

/// One sampled round of the supported deploy family.
#[derive(Clone, Copy, Debug)]
pub struct DeployRound {
    pub players: usize,
    pub dice_per: u8,
    pub faces: u8,
    pub dice_left: [u8; MAX_SEATS],
    pub opener: usize,
    pub first_round: bool,
}

/// Sample one round configuration + dice vector + opener across the supported
/// family (players 2..=6, dice-per 2..=`MAX_DICE`, faces 2..=6), biased toward the
/// HIGH-dice-count regime the deploy cares about (flagship 5p5d6f and near-full
/// vectors). Mirrors `deepcfr::sample_round_config` / `train::sample_round` but
/// emits the [`MAX_SEATS`]-wide vector the adapter consumes and respects the
/// net's `MAX_DICE` support.
pub fn sample_deploy_round(rng: &mut Rng) -> DeployRound {
    // Mild lean toward more players (more seats ⇒ more total dice).
    let players = match rng.unit() {
        u if u < 0.13 => 2,
        u if u < 0.30 => 3,
        u if u < 0.50 => 4,
        u if u < 0.73 => 5,
        _ => 6,
    };
    // ~75% high dice-per (4..=MAX_DICE), ~25% low (2..=3), keeping some generality.
    let dice_per = if rng.unit() < 0.75 {
        (4 + rng.below(MAX_DICE - 3)) as u8
    } else {
        (2 + rng.below(2)) as u8
    };
    let faces = (2 + rng.below(5)) as u8;

    // A FULL opening (every live seat at `dice_per`) is the real game start and the
    // dominant target whenever it fits the dice budget.
    let full_total = players as u32 * dice_per as u32;
    let want_full = full_total <= MAX_DEPLOY_TOTAL && rng.unit() < 0.45;

    let mut dice = [0u8; MAX_SEATS];
    let mut ok = false;
    if want_full {
        for d in dice.iter_mut().take(players) {
            *d = dice_per;
        }
        ok = true;
    } else {
        for _ in 0..32 {
            for d in dice.iter_mut().take(players) {
                // max-of-two keeps seats HIGH; ~10% eliminated so mid-game vectors
                // with lost dice still appear.
                *d = if rng.unit() < 0.90 {
                    let a = 1 + rng.below(dice_per as usize);
                    let b = 1 + rng.below(dice_per as usize);
                    a.max(b) as u8
                } else {
                    0
                };
            }
            let total: u32 = dice[..players].iter().map(|&x| u32::from(x)).sum();
            if (0..players).filter(|&i| dice[i] > 0).count() >= 2 && total <= MAX_DEPLOY_TOTAL {
                ok = true;
                break;
            }
        }
    }
    if !ok {
        dice = [0u8; MAX_SEATS];
        for d in dice.iter_mut().take(players) {
            *d = dice_per;
        }
    }
    finish_round(rng, players, dice_per, faces, dice, 0.7)
}

/// Sample a round of a FIXED `(players, dice_per, faces)` config: every live seat
/// draws `1..=dice_per`, with `first_round_prob` controlling how often a full
/// vector is the forced first-round open. Used to densely cover one small
/// config's continuing-round lattice for the gate.
pub fn sample_fixed_round(
    rng: &mut Rng,
    players: usize,
    dice_per: u8,
    faces: u8,
    first_round_prob: f64,
) -> DeployRound {
    let mut dice = [0u8; MAX_SEATS];
    for d in dice.iter_mut().take(players) {
        *d = (1 + rng.below(dice_per as usize)) as u8;
    }
    finish_round(rng, players, dice_per, faces, dice, first_round_prob)
}

/// Pick the opener / first-round flag for a chosen dice vector: a fully-stocked
/// vector may be the forced first-round open (opener 0, the real game start);
/// otherwise a uniformly-random live seat opens a free round.
fn finish_round(
    rng: &mut Rng,
    players: usize,
    dice_per: u8,
    faces: u8,
    dice: [u8; MAX_SEATS],
    first_round_prob: f64,
) -> DeployRound {
    let all_full = (0..players).all(|i| dice[i] == dice_per);
    let first_round = all_full && rng.unit() < first_round_prob;
    let opener = if first_round {
        0
    } else {
        let live: Vec<usize> = (0..players).filter(|&i| dice[i] > 0).collect();
        live[rng.below(live.len())]
    };
    DeployRound {
        players,
        dice_per,
        faces,
        dice_left: dice,
        opener,
        first_round,
    }
}

/// Configuration for a [`DeployTrainer`] run.
#[derive(Clone, Debug)]
pub struct DeployTrainConfig {
    /// Outer generate+train steps.
    pub steps: usize,
    /// Leading outer steps that close round-ends with [`DiceShareValue`] before the
    /// continuation switches to the bootstrapped [`NetContinuation`].
    pub warmup_steps: usize,
    /// CFR iterations per self-play subgame solve.
    pub num_iters: usize,
    /// Depth limit (public plies) of each subgame.
    pub max_depth: u32,
    /// Exploring-seat random-action probability during trajectory sampling.
    pub explore_eps: f64,
    pub batch: usize,
    pub lr: f32,
    /// Halve the learning rate every this many train steps (0 disables).
    pub lr_halflife: u64,
    pub buffer_cap: usize,
    pub gen_per_step: usize,
    pub train_gen_ratio: usize,
    pub burn_in: usize,
    pub eval_every: usize,
    /// CFR iterations per subgame in the gate evaluation.
    pub eval_iters: usize,
    /// Tabular CFR iterations per round-subgame solve when fitting the exact
    /// continuation lattice the 2-player gate scores against (the fit is one-time
    /// in [`DeployTrainer::new`] but heavy — keep it accurate for the real gate,
    /// tiny for smoke runs).
    pub eval_fit_iters: u64,
    pub hidden: usize,
    pub n_layers: usize,
    pub seed: u64,
    pub outdir: PathBuf,
    pub log: bool,
    /// When set, train on this fixed `(players, dice_per, faces)` config; a
    /// 2-player setting also gates against the exact continuation lattice.
    pub fixed_config: Option<(usize, u8, u8)>,
    /// Solve the round-open node over the [`principled_open_cap`] abstraction
    /// (faithful ReBeL opening action abstraction) instead of the full
    /// `1..=total` quantity range. Enabled for production: the pruned high
    /// openings are dominated junk (lossless), and the narrower opening node is a
    /// large data-gen speedup. Disable for the lossless ablation.
    pub principled_open_cap: bool,
}

impl Default for DeployTrainConfig {
    fn default() -> Self {
        Self {
            steps: 400,
            warmup_steps: 30,
            num_iters: 96,
            max_depth: 2,
            explore_eps: 0.25,
            batch: 256,
            lr: 1e-3,
            lr_halflife: 4000,
            buffer_cap: 2_000_000,
            gen_per_step: 48,
            train_gen_ratio: 16,
            burn_in: 2048,
            eval_every: 10,
            eval_iters: 256,
            eval_fit_iters: 2000,
            hidden: 256,
            n_layers: 2,
            seed: 0,
            outdir: PathBuf::new(),
            log: false,
            fixed_config: None,
            principled_open_cap: true,
        }
    }
}

/// Result of a deploy training run.
#[derive(Clone, Debug)]
pub struct DeployReport {
    /// `(samples_generated, max per-round exploitability vs the lattice)` at each
    /// gate evaluation (empty when no 2-player gate is configured).
    pub curve: Vec<(u64, f64)>,
    pub best_exploitability: f64,
    pub samples_generated: u64,
    pub train_steps: u64,
}

/// The 2-player gate: the exact continuation lattice for the fixed config and the
/// continuing-round states to score against it.
struct DeployGate {
    faces: u8,
    lattice: LatticeValue,
    states: Vec<(u8, u8, usize)>,
    iters: usize,
}

/// One gate evaluation's worst-case numbers across the continuing-round states.
#[derive(Clone, Copy, Debug)]
struct GateMetric {
    /// Max exploitability of the net-resolved policy vs the EXACT-lattice round.
    max_exploitability: f64,
    /// Max |adapter root value (vs net continuation) − exact lattice value|.
    max_value_delta: f64,
    /// Max |NetContinuation round-opening value − exact lattice value|.
    max_direct_delta: f64,
}

/// The deploy self-play value-net trainer.
pub struct DeployTrainer {
    cfg: DeployTrainConfig,
    net: PbsNet,
    buffer: Reservoir,
    rng: Rng,
    train_steps: u64,
    samples_generated: u64,
    train_debt: f64,
    gate: Option<DeployGate>,
}

fn dice_vec(counts: &[u8]) -> [u8; MAX_SEATS] {
    let mut d = [0u8; MAX_SEATS];
    d[..counts.len()].copy_from_slice(counts);
    d
}

/// Apply the [`principled_open_cap`] opening abstraction to `ad` when `on`, so a
/// gate adapter solves the same opening node the capped data-gen produces.
fn maybe_cap<C: ContinuationValue>(
    ad: LiarsDiceAdapter<'_, C>,
    on: bool,
) -> LiarsDiceAdapter<'_, C> {
    if on {
        ad.with_principled_open_cap()
    } else {
        ad
    }
}

impl DeployTrainer {
    pub fn new(cfg: DeployTrainConfig) -> DeployTrainer {
        let mut net = PbsNet::new(cfg.hidden, cfg.n_layers, cfg.seed);
        net.net_mut().set_lr(cfg.lr);
        let buffer = Reservoir::new(cfg.buffer_cap);
        let rng = Rng::new(cfg.seed ^ 0xD1CE_5EED_0BAD_F00D);
        let fit_cfg = FitConfig {
            iters_per_solve: cfg.eval_fit_iters,
            tol: 1e-3,
            max_sweeps: 50,
            measure_exploitability: false,
        };
        let gate = cfg.fixed_config.and_then(|(players, dice_per, faces)| {
            (players == 2).then(|| {
                let lattice = fit_two_player(dice_per, faces, fit_cfg).lattice;
                let mut states = Vec::new();
                for a in 1..=dice_per {
                    for b in 1..=dice_per {
                        for opener in 0..2usize {
                            states.push((a, b, opener));
                        }
                    }
                }
                DeployGate {
                    faces,
                    lattice,
                    states,
                    iters: cfg.eval_iters,
                }
            })
        });
        DeployTrainer {
            cfg,
            net,
            buffer,
            rng,
            train_steps: 0,
            samples_generated: 0,
            train_debt: 0.0,
            gate,
        }
    }

    /// The exact 2-player continuation lattice the gate scores against, fit once
    /// in [`DeployTrainer::new`]; `None` outside a 2-player fixed-config run.
    pub fn lattice(&self) -> Option<&LatticeValue> {
        self.gate.as_ref().map(|g| &g.lattice)
    }

    pub fn net(&self) -> &PbsNet {
        &self.net
    }

    fn selfplay_params(&self) -> SelfPlayParams {
        SelfPlayParams {
            cfr: CfrParams {
                num_iters: self.cfg.num_iters,
                max_depth: self.cfg.max_depth,
                ..CfrParams::default()
            },
            explore_eps: self.cfg.explore_eps,
        }
    }

    /// Generate one outer step's episodes in parallel: each samples a config,
    /// builds the adapter closed by the current-phase continuation (heuristic in
    /// warmup, the bootstrapped net afterwards), and emits its per-round value
    /// examples. The net is read-shared and immutable during generation, so the
    /// continuation sees a stable per-step snapshot.
    fn generate(&mut self, use_net_cont: bool) -> usize {
        let base = self.rng.next_u64();
        let gen_per = self.cfg.gen_per_step;
        let sp = self.selfplay_params();
        let fixed = self.cfg.fixed_config;
        let use_cap = self.cfg.principled_open_cap;
        let net = &self.net;
        let episodes: Vec<Vec<Sample>> = (0..gen_per)
            .into_par_iter()
            .map(|e| {
                let seed = base ^ (e as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let mut rng = Rng::new(seed);
                let round = match fixed {
                    Some((p, d, f)) => sample_fixed_round(&mut rng, p, d, f, 0.0),
                    None => sample_deploy_round(&mut rng),
                };
                let cont = if use_net_cont {
                    DeployCont::Net(NetContinuation::new(net))
                } else {
                    DeployCont::Heuristic(DiceShareValue)
                };
                let adapter = LiarsDiceAdapter::new(
                    round.players,
                    round.faces,
                    round.dice_left,
                    round.opener,
                    round.first_round,
                    &cont,
                );
                let adapter = if use_cap {
                    adapter.with_principled_open_cap()
                } else {
                    adapter
                };
                generate_episode(&adapter, sp, net, &mut rng)
            })
            .collect();
        let mut count = 0;
        for ep in episodes {
            for s in ep {
                self.buffer.push(s, &mut self.rng);
                count += 1;
            }
        }
        self.samples_generated += count as u64;
        count
    }

    /// Pay down the training debt incurred by `generated` new samples.
    fn train(&mut self, generated: usize) {
        if self.buffer.len() < self.cfg.burn_in {
            return;
        }
        self.train_debt += (generated * self.cfg.train_gen_ratio) as f64;
        while self.train_debt >= self.cfg.batch as f64 {
            let batch = self.buffer.sample_batch(self.cfg.batch, &mut self.rng);
            self.net.net_mut().train_step(&batch);
            self.train_debt -= self.cfg.batch as f64;
            self.train_steps += 1;
            if self.cfg.lr_halflife > 0 && self.train_steps.is_multiple_of(self.cfg.lr_halflife) {
                let lr =
                    self.cfg.lr * 0.5f32.powi((self.train_steps / self.cfg.lr_halflife) as i32);
                self.net.net_mut().set_lr(lr);
            }
        }
    }

    /// Score the current net on the 2-player gate: per continuing round, solve the
    /// round against the net continuation and measure how exploitable the resulting
    /// policy is in the EXACT-lattice round, plus the value gaps vs the lattice.
    fn eval(&self) -> Option<GateMetric> {
        let gate = self.gate.as_ref()?;
        let cont = NetContinuation::new(&self.net);
        let params = CfrParams {
            num_iters: gate.iters,
            max_depth: u32::MAX,
            ..CfrParams::default()
        };
        let mut m = GateMetric {
            max_exploitability: 0.0,
            max_value_delta: 0.0,
            max_direct_delta: 0.0,
        };
        let use_cap = self.cfg.principled_open_cap;
        for &(a, b, opener) in &gate.states {
            let dice = dice_vec(&[a, b]);
            let exact = gate
                .lattice
                .get_two_player(&[a, b], opener)
                .expect("lattice covers every continuing 2p state");

            let ad_net = maybe_cap(
                LiarsDiceAdapter::new(2, gate.faces, dice, opener, false, &cont),
                use_cap,
            );
            let initial = Belief::uniform_prior(&ad_net.root());
            let terminal = TerminalLeaf::new(&ad_net);
            let mut solver = Solver::new(&ad_net, params, &terminal, initial.clone());
            solver.multistep();
            let v0: f64 = solver
                .root_values_mean(0)
                .iter()
                .zip(&initial.per_seat[0])
                .map(|(v, p)| v * p)
                .sum();
            let avg = solver.average_strategy().to_vec();

            let ad_exact = maybe_cap(
                LiarsDiceAdapter::new(2, gate.faces, dice, opener, false, &gate.lattice),
                use_cap,
            );
            let expl = exploitability(&ad_exact, &avg);
            let direct = (cont.value(gate.faces, &[a, b], opener, 0) - exact).abs();

            m.max_exploitability = m.max_exploitability.max(expl);
            m.max_value_delta = m.max_value_delta.max((v0 - exact).abs());
            m.max_direct_delta = m.max_direct_delta.max(direct);
        }
        Some(m)
    }

    /// Run the full loop, returning the gate exploitability curve and best score.
    pub fn run(&mut self) -> DeployReport {
        let mut curve = Vec::new();
        let mut best = f64::INFINITY;
        for step in 0..self.cfg.steps {
            let use_net_cont = step >= self.cfg.warmup_steps;
            let generated = self.generate(use_net_cont);
            self.train(generated);
            if (step + 1) % self.cfg.eval_every == 0 || step + 1 == self.cfg.steps {
                if let Some(m) = self.eval() {
                    curve.push((self.samples_generated, m.max_exploitability));
                    if m.max_exploitability < best {
                        best = m.max_exploitability;
                        self.checkpoint("best.bin");
                    }
                    if self.cfg.log {
                        eprintln!(
                            "[deploy] step={:>4} samples={:>8} train_steps={:>6} \
                             expl={:.5} best={:.5} vΔ={:.5} contΔ={:.5} cont={}",
                            step + 1,
                            self.samples_generated,
                            self.train_steps,
                            m.max_exploitability,
                            best,
                            m.max_value_delta,
                            m.max_direct_delta,
                            if use_net_cont { "net" } else { "heuristic" },
                        );
                    }
                }
                self.checkpoint("ckpt.bin");
            }
        }
        DeployReport {
            curve,
            best_exploitability: best,
            samples_generated: self.samples_generated,
            train_steps: self.train_steps,
        }
    }

    fn checkpoint(&self, name: &str) {
        if self.cfg.outdir.as_os_str().is_empty() {
            return;
        }
        let _ = self.net.save(&self.cfg.outdir.join(name));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rebel::recursive_strategy;

    #[test]
    fn sample_deploy_round_stays_in_the_supported_family() {
        let mut rng = Rng::new(42);
        let mut saw_full = false;
        let mut saw_partial = false;
        for _ in 0..2000 {
            let r = sample_deploy_round(&mut rng);
            assert!((2..=MAX_REBEL_SEATS).contains(&r.players));
            assert!((2..=MAX_DICE as u8).contains(&r.dice_per));
            assert!((2..=6).contains(&r.faces));
            let live = (0..r.players).filter(|&i| r.dice_left[i] > 0).count();
            assert!(live >= 2, "at least two live seats");
            let total: u32 = r.dice_left[..r.players].iter().map(|&x| u32::from(x)).sum();
            assert!(total <= MAX_DEPLOY_TOTAL);
            assert!(r.dice_left[r.opener] > 0, "the opener is live");
            for &d in &r.dice_left[r.players..] {
                assert_eq!(d, 0, "no dice past the player count");
            }
            if (0..r.players).all(|i| r.dice_left[i] == r.dice_per) {
                saw_full = true;
            } else {
                saw_partial = true;
            }
            if r.first_round {
                assert_eq!(r.opener, 0);
                assert!((0..r.players).all(|i| r.dice_left[i] == r.dice_per));
            }
        }
        assert!(
            saw_full && saw_partial,
            "both full and mid-game vectors appear"
        );
    }

    #[test]
    fn deploy_training_smoke_runs_and_evaluates() {
        let cfg = DeployTrainConfig {
            steps: 4,
            warmup_steps: 2,
            num_iters: 24,
            batch: 64,
            gen_per_step: 4,
            burn_in: 32,
            eval_every: 2,
            eval_iters: 48,
            eval_fit_iters: 60,
            hidden: 32,
            n_layers: 2,
            buffer_cap: 10_000,
            fixed_config: Some((2, 2, 3)),
            ..DeployTrainConfig::default()
        };
        let mut trainer = DeployTrainer::new(cfg);
        let report = trainer.run();
        assert!(!report.curve.is_empty());
        assert!(report.samples_generated > 0);
        for &(_, e) in &report.curve {
            assert!(e.is_finite());
            assert!((0.0..2.0).contains(&e));
        }
    }

    fn env_usize(key: &str, default: usize) -> usize {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }

    fn env_f64(key: &str, default: f64) -> f64 {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }

    /// DEPLOY GATE: train a net via the deploy self-play loop on a small fixed
    /// config and require the depth-limited recursive solve to be near-Nash per
    /// round — low exploitability vs the exact continuation lattice, and the
    /// adapter root value close to the lattice. Run with
    /// `cargo test -p liars-dice --release -- --ignored --nocapture gate_deploy`.
    #[test]
    #[ignore = "trains a net via deploy self-play; run explicitly"]
    fn gate_deploy_small_config() {
        let players = env_usize("PLAYERS", 2);
        let dice_per = env_usize("DICE", 2) as u8;
        let faces = env_usize("FACES", 3) as u8;
        let use_cap = env_usize("OPEN_CAP", 1) != 0;

        let cfg = DeployTrainConfig {
            steps: env_usize("STEPS", 300),
            warmup_steps: env_usize("WARMUP", 30),
            num_iters: env_usize("NUM_ITERS", 96),
            max_depth: 2,
            explore_eps: env_f64("EPS", 0.25),
            batch: env_usize("BATCH", 256),
            lr: env_f64("LR", 1e-3) as f32,
            lr_halflife: env_usize("LR_HALFLIFE", 4000) as u64,
            buffer_cap: env_usize("BUFFER", 1_000_000),
            gen_per_step: env_usize("GEN_PER", 48),
            train_gen_ratio: env_usize("TRAIN_RATIO", 16),
            burn_in: env_usize("BURN_IN", 2048),
            eval_every: env_usize("EVAL_EVERY", 10),
            eval_iters: env_usize("EVAL_ITERS", 256),
            eval_fit_iters: env_usize("FIT_ITERS", 2000) as u64,
            hidden: env_usize("HIDDEN", 256),
            n_layers: env_usize("LAYERS", 2),
            seed: env_usize("SEED", 0) as u64,
            outdir: std::env::var("OUTDIR")
                .map(PathBuf::from)
                .unwrap_or_default(),
            log: true,
            fixed_config: Some((players, dice_per, faces)),
            principled_open_cap: use_cap,
        };

        let start = std::time::Instant::now();
        let mut trainer = DeployTrainer::new(cfg);
        let report = trainer.run();
        let secs = start.elapsed().as_secs_f64();

        println!("=== {players}p{dice_per}d{faces}f deploy gate ===");
        for (samples, e) in &report.curve {
            println!("  samples={samples:>8}  max_exploitability={e:.5}");
        }

        // Final per-state numbers, including the depth-2 recursive (deploy-faithful)
        // exploitability vs the exact lattice. Reuse the lattice the trainer
        // already fit so the gate pays the heavy exact solve only once.
        let lattice = trainer
            .lattice()
            .expect("2-player gate fits a lattice")
            .clone();
        let net = trainer.net();
        let cont = NetContinuation::new(net);
        let depth2 = CfrParams {
            num_iters: env_usize("EVAL_ITERS", 256),
            max_depth: 2,
            ..CfrParams::default()
        };
        let full = CfrParams {
            num_iters: env_usize("EVAL_ITERS", 256),
            max_depth: u32::MAX,
            ..CfrParams::default()
        };

        let mut max_full_expl = 0.0f64;
        let mut max_rec_expl = 0.0f64;
        let mut max_vdelta = 0.0f64;
        let mut max_direct = 0.0f64;
        for a in 1..=dice_per {
            for b in 1..=dice_per {
                for opener in 0..2usize {
                    let dice = dice_vec(&[a, b]);
                    let exact = lattice.get_two_player(&[a, b], opener).unwrap();

                    let ad_net = maybe_cap(
                        LiarsDiceAdapter::new(2, faces, dice, opener, false, &cont),
                        use_cap,
                    );
                    let initial = Belief::uniform_prior(&ad_net.root());
                    let terminal = TerminalLeaf::new(&ad_net);
                    let mut solver = Solver::new(&ad_net, full, &terminal, initial.clone());
                    solver.multistep();
                    let v0: f64 = solver
                        .root_values_mean(0)
                        .iter()
                        .zip(&initial.per_seat[0])
                        .map(|(v, p)| v * p)
                        .sum();
                    let avg = solver.average_strategy().to_vec();
                    let rec = recursive_strategy(&ad_net, net, depth2);

                    let ad_exact = maybe_cap(
                        LiarsDiceAdapter::new(2, faces, dice, opener, false, &lattice),
                        use_cap,
                    );
                    let full_expl = exploitability(&ad_exact, &avg);
                    let rec_expl = exploitability(&ad_exact, &rec);
                    let direct = (cont.value(faces, &[a, b], opener, 0) - exact).abs();

                    println!(
                        "  [{a},{b}] op={opener}: v0={v0:+.4} exact={exact:+.4} \
                         |vΔ|={:.4} full_expl={full_expl:.4} rec2_expl={rec_expl:.4} \
                         contΔ={direct:.4}",
                        (v0 - exact).abs()
                    );
                    max_full_expl = max_full_expl.max(full_expl);
                    max_rec_expl = max_rec_expl.max(rec_expl);
                    max_vdelta = max_vdelta.max((v0 - exact).abs());
                    max_direct = max_direct.max(direct);
                }
            }
        }
        println!(
            "best_gate_expl={:.5} final: full_expl={max_full_expl:.5} rec2_expl={max_rec_expl:.5} \
             value_Δ={max_vdelta:.5} contΔ={max_direct:.5} samples={} train_steps={} wall={secs:.1}s",
            report.best_exploitability, report.samples_generated, report.train_steps
        );

        assert!(
            max_full_expl < 0.05,
            "per-round exploitability vs the exact lattice {max_full_expl:.5} >= 0.05"
        );
        assert!(
            max_vdelta < 0.05,
            "root-value gap vs the exact lattice {max_vdelta:.5} >= 0.05"
        );
    }
}
