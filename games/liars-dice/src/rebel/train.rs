//! Self-play value-net training loop and the recursive-policy exploitability
//! gate.
//!
//! Each outer step generates a batch of self-play episodes in parallel, pushes
//! their value examples into a uniform reservoir, then (once warmed) trains the
//! net with a fixed example-to-generation ratio. Periodically the policy is
//! materialized and its full-game exploitability is measured against exact
//! terminal payoffs. Two materializations are provided:
//!
//! - [`stitch_exploitability`] (the tracked gate): the Batch-2 stitch — override
//!   the top depth-limited block of the exact-Nash baseline with the net's solve
//!   there. As the net approaches the perfect oracle this approaches the
//!   depth-limited resolving floor (≈0.026 on 1x4f).
//! - [`recursive_strategy`]/[`recursive_exploitability`]: fully-recursive
//!   continual resolving — every depth-limited block resolved with the net down
//!   to terminals, intermediate plies taken from each block's own solve. This
//!   compounds resolving error so its perfect-net floor is higher (≈0.048 on
//!   1x4f); reported as a stricter secondary metric.

use std::path::PathBuf;

use game_core::Rng;
use rayon::prelude::*;

use crate::rebel::buffer::Reservoir;
use crate::rebel::cfr::{CfrParams, SMOOTHING_EPS, Solver, parent_actions, reach_probabilities};
use crate::rebel::exploit::exploitability;
use crate::rebel::game::RebelGame;
use crate::rebel::leaf::{RootedGame, TerminalLeaf};
use crate::rebel::pbs::Belief;
use crate::rebel::selfplay::{SelfPlayParams, generate_episode};
use crate::rebel::tree::Tree;
use crate::rebel::value_net::{NetLeaf, PbsNet};

/// Full-depth unroll for materializing and scoring the recursive policy.
const FULL_DEPTH: u32 = u32::MAX;

/// CFR iterations for the exact-Nash continuation baseline of the stitch gate.
const BASELINE_ITERS: usize = 1024;

/// Configuration for a [`RebelTrainer`] run.
#[derive(Clone, Debug)]
pub struct RebelTrainConfig {
    /// Outer generate+train steps.
    pub steps: usize,
    /// CFR iterations per self-play subgame solve.
    pub num_iters: usize,
    /// Depth limit (public plies) of each subgame.
    pub max_depth: u32,
    /// Exploring-seat random-action probability during trajectory sampling.
    pub explore_eps: f64,
    /// Net training minibatch size.
    pub batch: usize,
    /// Initial Adam learning rate.
    pub lr: f32,
    /// Halve the learning rate every this many train steps (0 disables).
    pub lr_halflife: u64,
    /// Reservoir capacity.
    pub buffer_cap: usize,
    /// Episodes generated per outer step.
    pub gen_per_step: usize,
    /// Net training examples consumed per generated example.
    pub train_gen_ratio: usize,
    /// Train only once the reservoir holds at least this many samples.
    pub burn_in: usize,
    /// Measure exploitability (and checkpoint) every this many outer steps.
    pub eval_every: usize,
    /// CFR iterations per subgame in the recursive-exploitability eval.
    pub eval_iters: usize,
    /// Hidden width / depth of the value net.
    pub hidden: usize,
    pub n_layers: usize,
    pub seed: u64,
    /// Checkpoint directory; empty skips checkpointing.
    pub outdir: PathBuf,
    /// Emit a progress line to stderr at each evaluation.
    pub log: bool,
}

impl Default for RebelTrainConfig {
    fn default() -> Self {
        Self {
            steps: 400,
            num_iters: 256,
            max_depth: 2,
            explore_eps: 0.25,
            batch: 256,
            lr: 3e-4,
            lr_halflife: 4000,
            buffer_cap: 2_000_000,
            gen_per_step: 32,
            train_gen_ratio: 4,
            burn_in: 512,
            eval_every: 20,
            eval_iters: 512,
            hidden: 256,
            n_layers: 2,
            seed: 0,
            outdir: PathBuf::new(),
            log: false,
        }
    }
}

/// Result of a training run.
#[derive(Clone, Debug)]
pub struct TrainReport {
    /// `(samples_generated, exploitability)` at each evaluation.
    pub curve: Vec<(u64, f64)>,
    pub best_exploitability: f64,
    pub samples_generated: u64,
    pub train_steps: u64,
}

/// A game-agnostic self-play value-net trainer.
pub struct RebelTrainer {
    cfg: RebelTrainConfig,
    net: PbsNet,
    buffer: Reservoir,
    rng: Rng,
    train_steps: u64,
    samples_generated: u64,
    train_debt: f64,
}

impl RebelTrainer {
    pub fn new(cfg: RebelTrainConfig) -> RebelTrainer {
        let mut net = PbsNet::new(cfg.hidden, cfg.n_layers, cfg.seed);
        net.net_mut().set_lr(cfg.lr);
        let buffer = Reservoir::new(cfg.buffer_cap);
        let rng = Rng::new(cfg.seed ^ 0xD1CE_5EED_0BAD_F00D);
        RebelTrainer {
            cfg,
            net,
            buffer,
            rng,
            train_steps: 0,
            samples_generated: 0,
            train_debt: 0.0,
        }
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

    fn eval_params(&self) -> CfrParams {
        CfrParams {
            num_iters: self.cfg.eval_iters,
            max_depth: self.cfg.max_depth,
            ..CfrParams::default()
        }
    }

    /// Generate one outer step's worth of episodes in parallel and push their
    /// samples into the reservoir.
    fn generate<G: RebelGame + Sync>(&mut self, game: &G) -> usize {
        let base = self.rng.next_u64();
        let gen_per = self.cfg.gen_per_step;
        let sp = self.selfplay_params();
        let episodes: Vec<Vec<_>> = {
            let net = &self.net;
            (0..gen_per)
                .into_par_iter()
                .map(|e| {
                    let seed = base ^ (e as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                    let mut rng = Rng::new(seed);
                    generate_episode(game, sp, net, &mut rng)
                })
                .collect()
        };
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

    /// Run the full loop, returning the exploitability curve and best score. The
    /// tracked metric is the Batch-2 stitch: the net's depth-limited solve at the
    /// top block over the exact-Nash continuation (floor ≈0.026 on 1x4f).
    pub fn run<G: RebelGame + Sync>(&mut self, game: &G) -> TrainReport {
        let baseline = exact_full_nash(game, BASELINE_ITERS);
        let mut curve = Vec::new();
        let mut best = f64::INFINITY;
        for step in 0..self.cfg.steps {
            let generated = self.generate(game);
            self.train(generated);
            if (step + 1) % self.cfg.eval_every == 0 || step + 1 == self.cfg.steps {
                let e = stitch_exploitability(game, &self.net, &baseline, self.eval_params());
                curve.push((self.samples_generated, e));
                if e < best {
                    best = e;
                    self.checkpoint("best.bin");
                }
                self.checkpoint("ckpt.bin");
                if self.cfg.log {
                    eprintln!(
                        "[rebel] step={:>4} samples={:>8} train_steps={:>6} stitch_exploit={:.5} best={:.5}",
                        step + 1,
                        self.samples_generated,
                        self.train_steps,
                        e,
                        best
                    );
                }
            }
        }
        TrainReport {
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

/// Exact full-game Nash strategy via full-depth vector CFR with exact terminal
/// payoffs.
pub fn exact_full_nash<G: RebelGame>(game: &G, iters: usize) -> Vec<Vec<Vec<f64>>> {
    let terminal = TerminalLeaf::new(game);
    let params = CfrParams {
        num_iters: iters,
        max_depth: FULL_DEPTH,
        ..CfrParams::default()
    };
    let mut solver = Solver::new(game, params, &terminal, Belief::uniform_prior(&game.root()));
    solver.multistep();
    solver.average_strategy().to_vec()
}

/// Batch-2 stitch gate: override the top depth-`eval.max_depth` block of the
/// exact-Nash `baseline` with the net's depth-limited solve there, then measure
/// full-game exploitability. The depth-limited tree is a BFS prefix of the
/// full-depth tree, so node indices align. As the net approaches the perfect
/// oracle this approaches the depth-limited resolving floor.
pub fn stitch_exploitability<G: RebelGame>(
    game: &G,
    net: &PbsNet,
    baseline: &[Vec<Vec<f64>>],
    eval: CfrParams,
) -> f64 {
    let mut stitched = baseline.to_vec();
    let leaf = NetLeaf::new(net, game);
    let mut solver = Solver::new(game, eval, &leaf, Belief::uniform_prior(&game.root()));
    solver.multistep();
    let top = solver.average_strategy();
    let block = Tree::build(game, eval.max_depth);
    for idx in 0..block.len() {
        if !block.nodes[idx].is_leaf {
            stitched[idx].clone_from(&top[idx]);
        }
    }
    exploitability(game, &stitched)
}

/// Materialize the fully-recursive policy (every block resolved with the net,
/// down to terminals) and measure its exploitability. Stricter than the stitch:
/// it compounds depth-limited resolving error across blocks, so its perfect-net
/// floor is higher (≈0.048 on 1x4f).
pub fn recursive_exploitability<G: RebelGame + Sync>(
    game: &G,
    net: &PbsNet,
    eval: CfrParams,
) -> f64 {
    let strategy = recursive_strategy(game, net, eval);
    exploitability(game, &strategy)
}

/// The full-depth strategy materialized by depth-limited continual resolving:
/// solve a depth-`max_depth` subgame at the root, take *every* internal node's
/// average strategy from that solve, then recurse at the subgame's non-terminal
/// leaves with their reach-propagated beliefs. Advancing a whole `max_depth`
/// block per solve keeps the intermediate-ply strategies consistent with the
/// solve that produced them — the DeepStack/ReBeL resolving scheme. (Re-solving
/// a fresh subgame at *every* node and reading only its root, by contrast, is
/// unsafe without a gadget and is far more exploitable.)
pub fn recursive_strategy<G: RebelGame + Sync>(
    game: &G,
    net: &PbsNet,
    eval: CfrParams,
) -> Vec<Vec<Vec<f64>>> {
    let full = Tree::build(game, FULL_DEPTH);
    let prior = Belief::uniform_prior(&game.root());
    let pairs = resolve_block(game, net, eval, &full, 0, prior);
    let mut strategy = vec![Vec::new(); full.len()];
    for (idx, policy) in pairs {
        strategy[idx] = policy;
    }
    strategy
}

fn resolve_block<G: RebelGame + Sync>(
    game: &G,
    net: &PbsNet,
    eval: CfrParams,
    full: &Tree,
    full_root: usize,
    belief: Belief,
) -> Vec<(usize, Vec<Vec<f64>>)> {
    if full.nodes[full_root].is_leaf {
        return Vec::new();
    }
    let players = game.players();
    let rooted = RootedGame::new(game, full.nodes[full_root].public.clone());
    let leaf = NetLeaf::new(net, game);
    let mut solver = Solver::new(&rooted, eval, &leaf, belief.clone());
    solver.multistep();

    let sub = solver.tree();
    let avg = solver.average_strategy();
    let pa = parent_actions(sub);
    let reach: Vec<Vec<Vec<f64>>> = (0..players)
        .map(|s| reach_probabilities(sub, &pa, avg, &belief.per_seat[s], s))
        .collect();

    let mut full_of = vec![usize::MAX; sub.len()];
    full_of[0] = full_root;
    let mut policies = Vec::new();
    let mut recurse_seeds = Vec::new();
    for sub_idx in 0..sub.len() {
        let full_idx = full_of[sub_idx];
        let m = &sub.nodes[sub_idx];
        if m.is_leaf {
            if !m.is_terminal {
                let per_seat = (0..players)
                    .map(|s| {
                        let r = &reach[s][sub_idx];
                        let z = r.iter().sum::<f64>().max(SMOOTHING_EPS);
                        r.iter().map(|x| x / z).collect()
                    })
                    .collect();
                recurse_seeds.push((full_idx, Belief { per_seat }));
            }
        } else {
            policies.push((full_idx, avg[sub_idx].clone()));
            for (action, &sub_child) in m.children.iter().enumerate() {
                full_of[sub_child] = full.nodes[full_idx].children[action];
            }
        }
    }

    let deeper: Vec<(usize, Vec<Vec<f64>>)> = recurse_seeds
        .into_par_iter()
        .flat_map(|(full_idx, seed)| resolve_block(game, net, eval, full, full_idx, seed))
        .collect();
    policies.extend(deeper);
    policies
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rebel::standard::StandardLiarsDice;

    #[test]
    fn training_smoke_runs_and_evaluates() {
        let game = StandardLiarsDice::new(1, 4);
        let cfg = RebelTrainConfig {
            steps: 4,
            num_iters: 32,
            batch: 64,
            gen_per_step: 4,
            burn_in: 64,
            eval_every: 2,
            eval_iters: 64,
            hidden: 32,
            n_layers: 2,
            buffer_cap: 10_000,
            ..RebelTrainConfig::default()
        };
        let mut trainer = RebelTrainer::new(cfg);
        let report = trainer.run(&game);
        assert!(!report.curve.is_empty());
        assert!(report.samples_generated > 0);
        for &(_, e) in &report.curve {
            assert!(e.is_finite());
            assert!((0.0..2.0).contains(&e));
        }
    }

    #[test]
    fn recursive_strategy_is_a_valid_profile() {
        let game = StandardLiarsDice::new(1, 4);
        let net = PbsNet::new(32, 2, 1);
        let eval = CfrParams {
            num_iters: 64,
            max_depth: 2,
            ..CfrParams::default()
        };
        let strategy = recursive_strategy(&game, &net, eval);
        let full = Tree::build(&game, FULL_DEPTH);
        for (idx, (node, policy)) in full.nodes.iter().zip(&strategy).enumerate() {
            if node.is_leaf {
                continue;
            }
            assert!(!policy.is_empty(), "node {idx} missing a policy");
            for row in policy {
                let sum: f64 = row.iter().sum();
                assert!((sum - 1.0).abs() < 1e-6);
            }
        }
        let e = exploitability(&game, &strategy);
        assert!(e.is_finite());
    }

    fn env_usize(key: &str, default: usize) -> usize {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }

    #[test]
    #[ignore = "calibration"]
    fn calibrate_costs() {
        use std::time::Instant;
        let gen_iters = env_usize("NUM_ITERS", 96);
        let eval_iters = env_usize("EVAL_ITERS", 256);
        let hidden = env_usize("HIDDEN", 256);
        for &(d, f) in &[(1u8, 4u8), (1, 5)] {
            let game = StandardLiarsDice::new(d, f);
            let full = Tree::build(&game, FULL_DEPTH);
            let nonleaf = full.nodes.iter().filter(|n| !n.is_leaf).count();
            let net = PbsNet::new(hidden, 2, 0);
            let sp = SelfPlayParams {
                cfr: CfrParams {
                    num_iters: gen_iters,
                    max_depth: 2,
                    ..CfrParams::default()
                },
                explore_eps: 0.25,
            };
            let t0 = Instant::now();
            let eps = 48usize;
            let samples: usize = (0..eps)
                .into_par_iter()
                .map(|e| {
                    let mut rng = Rng::new(e as u64 + 1);
                    generate_episode(&game, sp, &net, &mut rng).len()
                })
                .sum();
            let gen_s = t0.elapsed().as_secs_f64();

            let mut net2 = PbsNet::new(hidden, 2, 0);
            let mut rng = Rng::new(7);
            let batch: Vec<_> = {
                let mut r = Reservoir::new(10000);
                let mut g = Rng::new(9);
                for ep in 0..6 {
                    let mut er = Rng::new(ep + 100);
                    for s in generate_episode(&game, sp, &net, &mut er) {
                        r.push(s, &mut g);
                    }
                }
                r.sample_batch(256, &mut rng)
            };
            let t1 = Instant::now();
            for _ in 0..50 {
                net2.net_mut().train_step(&batch);
            }
            let train_ms = t1.elapsed().as_secs_f64() * 1000.0 / 50.0;

            let eval = CfrParams {
                num_iters: eval_iters,
                max_depth: 2,
                ..CfrParams::default()
            };
            let t2 = Instant::now();
            let e = recursive_exploitability(&game, &net, eval);
            let eval_s = t2.elapsed().as_secs_f64();
            println!(
                "{d}x{f}f nodes={} nonleaf={nonleaf} | gen {eps}ep/{gen_s:.2}s ({:.0}ms/ep) samples={samples} | train {train_ms:.1}ms/step | untrained_exploit={e:.4} eval_s={eval_s:.2}",
                full.len(),
                gen_s * 1000.0 / eps as f64,
            );
        }
    }

    fn materialize_serial(
        game: &StandardLiarsDice,
        leaf: &dyn crate::rebel::leaf::LeafValue,
        eval: CfrParams,
        full: &Tree,
        full_root: usize,
        belief: Belief,
        strategy: &mut [Vec<Vec<f64>>],
    ) {
        if full.nodes[full_root].is_leaf {
            return;
        }
        let players = game.players();
        let rooted = RootedGame::new(game, full.nodes[full_root].public.clone());
        let mut solver = Solver::new(&rooted, eval, leaf, belief.clone());
        solver.multistep();
        let sub = solver.tree();
        let avg = solver.average_strategy();
        let pa = parent_actions(sub);
        let reach: Vec<Vec<Vec<f64>>> = (0..players)
            .map(|s| reach_probabilities(sub, &pa, avg, &belief.per_seat[s], s))
            .collect();

        let mut full_of = vec![usize::MAX; sub.len()];
        full_of[0] = full_root;
        let mut seeds = Vec::new();
        for sub_idx in 0..sub.len() {
            let full_idx = full_of[sub_idx];
            let m = &sub.nodes[sub_idx];
            if m.is_leaf {
                if !m.is_terminal {
                    let per_seat = (0..players)
                        .map(|s| {
                            let r = &reach[s][sub_idx];
                            let z = r.iter().sum::<f64>().max(SMOOTHING_EPS);
                            r.iter().map(|x| x / z).collect()
                        })
                        .collect();
                    seeds.push((full_idx, Belief { per_seat }));
                }
            } else {
                strategy[full_idx] = avg[sub_idx].clone();
                for (action, &sub_child) in m.children.iter().enumerate() {
                    full_of[sub_child] = full.nodes[full_idx].children[action];
                }
            }
        }
        for (full_idx, seed) in seeds {
            materialize_serial(game, leaf, eval, full, full_idx, seed, strategy);
        }
    }

    #[test]
    #[ignore = "diagnostic; run with --ignored --nocapture"]
    fn diag_oracle_eval_floor() {
        use crate::rebel::leaf::PerfectOracleLeaf;
        let (d, f) = (env_usize("DICE", 1) as u8, env_usize("FACES", 4) as u8);
        let oracle_iters = env_usize("ORACLE_ITERS", 1024);
        let eval_iters = env_usize("EVAL_ITERS", 1024);
        let game = StandardLiarsDice::new(d, f);
        let oracle = PerfectOracleLeaf::new(&game, oracle_iters);
        let full = Tree::build(&game, FULL_DEPTH);
        let eval = CfrParams {
            num_iters: eval_iters,
            max_depth: 2,
            ..CfrParams::default()
        };
        let mut strategy = vec![Vec::new(); full.len()];
        let prior = Belief::uniform_prior(&game.root());
        materialize_serial(&game, &oracle, eval, &full, 0, prior, &mut strategy);
        let e = exploitability(&game, &strategy);
        println!(
            "=== {d}x{f}f recursive ORACLE-leaf exploitability = {e:.5} (floor for the net eval) ==="
        );
    }

    fn random_belief(public: &crate::rebel::pbs::PublicState, rng: &mut Rng) -> Belief {
        use crate::rebel::hands::hand_count;
        let players = public.players as usize;
        let per_seat = (0..players)
            .map(|s| {
                let n = hand_count(public.dice_left[s], public.faces);
                if n <= 1 {
                    return vec![1.0; n.max(1)];
                }
                let conc = if rng.unit() < 0.4 { 0.3 } else { 1.0 };
                let mut v: Vec<f64> = (0..n)
                    .map(|_| (-rng.unit().ln()).powf(1.0 / conc))
                    .collect();
                let sum: f64 = v.iter().sum::<f64>().max(1e-12);
                for x in &mut v {
                    *x /= sum;
                }
                v
            })
            .collect();
        Belief { per_seat }
    }

    /// Decisive diagnostic: train the net by supervised regression onto exact
    /// `PerfectOracleLeaf` targets over a broad belief sample at every reachable
    /// node, then measure recursive exploitability. Isolates the encoding / leaf
    /// convention / eval pipeline from the self-play bootstrap — if this reaches
    /// the oracle floor, any gate shortfall is a data/budget issue, not a bug.
    #[test]
    #[ignore = "diagnostic; run with --ignored --nocapture"]
    fn diag_oracle_supervised() {
        use crate::rebel::leaf::{LeafValue, PerfectOracleLeaf};
        let (d, f) = (env_usize("DICE", 1) as u8, env_usize("FACES", 4) as u8);
        let beliefs_per = env_usize("BELIEFS", 24);
        let steps = env_usize("STEPS", 8000);
        let hidden = env_usize("HIDDEN", 256);
        let oracle_iters = env_usize("ORACLE_ITERS", 256);
        let eval_iters = env_usize("EVAL_ITERS", 256);

        let game = StandardLiarsDice::new(d, f);
        let oracle = PerfectOracleLeaf::new(&game, oracle_iters);
        let full = Tree::build(&game, FULL_DEPTH);
        let encoder = PbsNet::new(hidden, 2, 0);
        let players = game.players();

        let mut rng = Rng::new(123);
        let mut g = Rng::new(456);
        let mut buffer = Reservoir::new(2_000_000);
        let t_data = std::time::Instant::now();
        for node in full.nodes.iter().filter(|n| !n.is_leaf) {
            let beliefs: Vec<Belief> = std::iter::once(Belief::uniform_prior(&node.public))
                .chain((0..beliefs_per).map(|_| random_belief(&node.public, &mut rng)))
                .collect();
            for belief in &beliefs {
                for seat in 0..players {
                    let target = oracle.values(&node.public, seat, belief);
                    buffer.push(
                        encoder.to_sample(&node.public, seat, belief, &target),
                        &mut g,
                    );
                }
            }
        }
        println!(
            "dataset: {} samples in {:.1}s",
            buffer.len(),
            t_data.elapsed().as_secs_f64()
        );

        let mut net = PbsNet::new(hidden, 2, 0);
        net.net_mut().set_lr(1e-3);
        let t_train = std::time::Instant::now();
        for step in 0..steps {
            if step == steps / 2 {
                net.net_mut().set_lr(5e-4);
            }
            let batch = buffer.sample_batch(256, &mut g);
            let loss = net.net_mut().train_step(&batch);
            if step % 1000 == 0 || step + 1 == steps {
                println!("  step {step:>6} loss={loss:.6}");
            }
        }
        println!(
            "trained {steps} steps in {:.1}s",
            t_train.elapsed().as_secs_f64()
        );

        let mut mae = 0.0;
        let mut cnt = 0;
        for node in full.nodes.iter().filter(|n| !n.is_leaf).take(40) {
            let belief = random_belief(&node.public, &mut rng);
            for seat in 0..players {
                let want = oracle.values(&node.public, seat, &belief);
                let got = net.evaluate(&node.public, seat, &belief);
                for (a, b) in want.iter().zip(&got) {
                    mae += (a - b).abs();
                    cnt += 1;
                }
            }
        }
        println!("net-vs-oracle MAE = {:.5}", mae / cnt as f64);

        let eval = CfrParams {
            num_iters: eval_iters,
            max_depth: 2,
            ..CfrParams::default()
        };
        let baseline = exact_full_nash(&game, 1024);
        let stitch = stitch_exploitability(&game, &net, &baseline, eval);
        let rec = recursive_exploitability(&game, &net, eval);
        println!("=== {d}x{f}f oracle-supervised: stitch={stitch:.5} full_recursive={rec:.5} ===");
    }

    fn env_f64(key: &str, default: f64) -> f64 {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }

    fn run_gate(
        num_dice: u8,
        num_faces: u8,
        default_steps: usize,
        default_ratio: usize,
    ) -> TrainReport {
        let game = StandardLiarsDice::new(num_dice, num_faces);
        let cfg = RebelTrainConfig {
            steps: env_usize("STEPS", default_steps),
            num_iters: env_usize("NUM_ITERS", 96),
            max_depth: 2,
            explore_eps: env_f64("EPS", 0.25),
            batch: env_usize("BATCH", 256),
            lr: env_f64("LR", 1e-3) as f32,
            lr_halflife: env_usize("LR_HALFLIFE", 4000) as u64,
            gen_per_step: env_usize("GEN_PER", 48),
            train_gen_ratio: env_usize("TRAIN_RATIO", default_ratio),
            burn_in: env_usize("BURN_IN", 2048),
            eval_every: env_usize("EVAL_EVERY", 10),
            eval_iters: env_usize("EVAL_ITERS", 256),
            hidden: env_usize("HIDDEN", 256),
            n_layers: env_usize("LAYERS", 2),
            buffer_cap: env_usize("BUFFER", 1_000_000),
            seed: env_usize("SEED", 0) as u64,
            log: true,
            outdir: std::env::var("OUTDIR")
                .map(PathBuf::from)
                .unwrap_or_default(),
        };
        let start = std::time::Instant::now();
        let mut trainer = RebelTrainer::new(cfg);
        let report = trainer.run(&game);
        let secs = start.elapsed().as_secs_f64();
        println!("=== {num_dice}x{num_faces}f self-play gate ===");
        for (samples, e) in &report.curve {
            println!("  samples={samples:>8}  exploitability={e:.5}");
        }
        let eval = CfrParams {
            num_iters: env_usize("EVAL_ITERS", 256),
            max_depth: 2,
            ..CfrParams::default()
        };
        let rec = recursive_exploitability(&game, trainer.net(), eval);
        println!(
            "best_stitch={:.5}  final_full_recursive={rec:.5}  samples={}  train_steps={}  wall={:.1}s",
            report.best_exploitability, report.samples_generated, report.train_steps, secs
        );
        report
    }

    #[test]
    #[ignore = "full 1x4f self-play gate; run with --ignored --nocapture"]
    fn gate_1x4f() {
        let report = run_gate(1, 4, 220, 24);
        assert!(
            report.best_exploitability <= 0.04,
            "1x4f gate not met: best stitch exploitability {:.5} > 0.04",
            report.best_exploitability
        );
    }

    #[test]
    #[ignore = "full 1x5f self-play gate; run with --ignored --nocapture"]
    fn gate_1x5f() {
        // The hard gate is 1x4f; 1x5f (4x the value-function support) is reported
        // and held to a looser, comfortably-above-random sanity bound.
        let report = run_gate(1, 5, 300, 36);
        assert!(
            report.best_exploitability <= 0.05,
            "1x5f sanity bound not met: best stitch exploitability {:.5} > 0.05",
            report.best_exploitability
        );
    }
}
