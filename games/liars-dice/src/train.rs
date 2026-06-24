//! Offline training: distil per-round CFR/MCCFR equilibria into the policy/value
//! net ([`solvers::azero::Mlp`]).
//!
//! Each training round samples a configuration and dice vector across the whole
//! supported space, solves that *one round* exactly (2-player, small) with
//! [`Cfr`] or by outcome-sampling MCCFR ([`OsMccfr`]) otherwise, then distils the
//! solved strategy into supervised samples: at every decision node the solved
//! action distribution is the policy target and the realised round-leaf value is
//! the value target.
//!
//! The continuation that closes a round's leaves runs in two phases. A
//! `warmup_iters`-iteration **warm start** solves against the fixed
//! [`DiceShareValue`] heuristic, so the value head learns toward real equity
//! from terminal/heuristic signal alone (plain, divergence-free supervised
//! learning). After the warm start it switches to [`NetValue`] — the net's own
//! value head as the continuation — so the realised round-leaf value *is* the
//! net's current prediction of the post-round equity, and training the value
//! head on that backed-up value is one Bellman backup per iteration: **fitted
//! value iteration**. Keep-best by per-round exploitability protects the
//! artifact if the bootstrap ever wobbles, and `value_verify` proves V converges
//! toward the exact 2-player lattice rather than away from it.

use std::io::Write as _;
use std::path::Path;
use std::time::Instant;

use game_core::{Game, Rng, Turn, win_rate};
use rayon::prelude::*;
use solvers::azero::{InferCache, Mlp, Sample, SgdMomentum};
use solvers::os_mccfr::OsMccfr;
use solvers::{Cfr, nash_conv};

use crate::features::{encode, feature_len, legal_actions_and_support, net_policy, policy_len};
use crate::{
    BidConditioned, ContinuationValue, DiceShareValue, LatticeValue, LdState, LiarsDice, NetAgent,
    NetValue, ProbabilisticAgent, RoundSubgame,
};
use solvers::Rollout;

/// The continuation closing a round's leaves: the fixed dice-share heuristic
/// during warm-up, the net's value head (fitted value iteration) afterwards.
/// Both are zero-sum [`ContinuationValue`]s, so [`RoundSubgame`] is solved
/// identically for either.
enum Cont<'a> {
    Heuristic(DiceShareValue),
    Net(NetValue<'a>),
}

impl ContinuationValue for Cont<'_> {
    fn value(&self, faces: u8, dice_left: &[u8], next_opener: usize, player: usize) -> f64 {
        match self {
            Cont::Heuristic(h) => h.value(faces, dice_left, next_opener, player),
            Cont::Net(n) => n.value(faces, dice_left, next_opener, player),
        }
    }
}

/// The net + inference cache used as the continuation once warm-up ends. Built
/// fresh each iteration from the current net (the cache snapshots the weights),
/// then borrowed by a per-round [`NetValue`] inside the parallel closure.
struct Bootstrap<'a> {
    net: &'a Mlp,
    cache: &'a InferCache,
}

impl<'a> Bootstrap<'a> {
    /// A fresh continuation for one sampled round. During warm-up every round is
    /// closed by the fixed heuristic; afterwards each round gets its own
    /// [`NetValue`] (its own memo) sharing the snapshot net/cache.
    fn continuation(&self, players: u8, faces: u8, warm: bool) -> Cont<'a> {
        if warm {
            Cont::Heuristic(DiceShareValue)
        } else {
            Cont::Net(NetValue::new(self.net, self.cache, players, faces))
        }
    }
}

/// The per-round subgame type solved during data generation.
type Round<'a> = RoundSubgame<Cont<'a>>;

/// One recorded decision during a playout: features, the solved policy target as
/// sparse `(policy index, probability)` pairs, and the acting player.
type Decision = (Vec<f32>, Vec<(usize, f32)>, usize);

/// Uniform access to a solved round's average strategy, regardless of solver.
trait RoundSolver {
    fn policy_at(&self, s: &LdState, player: usize) -> Vec<f64>;
}
impl RoundSolver for Cfr<Round<'_>> {
    fn policy_at(&self, s: &LdState, player: usize) -> Vec<f64> {
        self.policy(s, player)
    }
}
impl RoundSolver for OsMccfr<Round<'_>> {
    fn policy_at(&self, s: &LdState, player: usize) -> Vec<f64> {
        self.policy(s, player)
    }
}

#[derive(Clone)]
pub struct TrainConfig {
    pub iters: usize,
    /// Iterations solved against the fixed [`DiceShareValue`] heuristic before
    /// switching to the net's own value head (fitted value iteration). Setting
    /// `warmup_iters >= iters` reproduces the pure-distillation baseline.
    pub warmup_iters: usize,
    pub rounds_per_iter: usize,
    pub playouts: usize,
    pub hidden: usize,
    pub cfr_iters: u64,
    pub os_iters: u64,
    pub small_total: u8,
    pub batch: usize,
    pub epochs: usize,
    pub buffer_cap: usize,
    pub lr: f32,
    pub momentum: f32,
    pub l2: f32,
    pub val_every: usize,
    pub threads: usize,
    pub outdir: String,
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            iters: 200,
            warmup_iters: 8,
            rounds_per_iter: 400,
            playouts: 12,
            hidden: 256,
            cfr_iters: 2_000,
            os_iters: 6_000,
            small_total: 4,
            batch: 1024,
            epochs: 2,
            buffer_cap: 300_000,
            lr: 0.05,
            momentum: 0.9,
            l2: 1e-4,
            val_every: 5,
            threads: 4,
            outdir: "runs/ld_net".into(),
            seed: 0xD1CE,
        }
    }
}

/// Largest per-seat starting dice count generated during training — the full
/// supported range (2..=8). Big rounds are now affordable thanks to direct-roll
/// chance sampling, so the config range is not curtailed.
const MAX_TRAIN_DICE: usize = 8;
/// Cap on total dice in a sampled round = the natural maximum (6 players x 8
/// dice). Effectively no curtailment; kept as an explicit, generous bound and a
/// resample guard so a degenerate draw can't blow up.
const MAX_TRAIN_TOTAL: u32 = 48;

/// Sample one configuration + dice vector + opener, biased toward small totals
/// (which solve fast and where exact play matters most), with at least two live
/// seats and a bounded total (so no round dominates the parallel pool).
fn sample_round(rng: &mut Rng) -> (u8, u8, u8, [u8; crate::MAX_PLAYERS], u8, bool) {
    let p = 2 + rng.below(5); // 2..=6
    let d = 2 + rng.below(MAX_TRAIN_DICE - 1); // 2..=MAX_TRAIN_DICE
    let f = 2 + rng.below(5); // 2..=6
    let mut dice = [0u8; crate::MAX_PLAYERS];
    let mut ok = false;
    for _ in 0..32 {
        for die in dice.iter_mut().take(p) {
            *die = if rng.unit() < 0.85 {
                // min of two draws -> bias toward fewer dice.
                let a = 1 + rng.below(d);
                let b = 1 + rng.below(d);
                a.min(b) as u8
            } else {
                0
            };
        }
        let total: u32 = dice[..p].iter().map(|&x| u32::from(x)).sum();
        if (0..p).filter(|&i| dice[i] > 0).count() >= 2 && total <= MAX_TRAIN_TOTAL {
            ok = true;
            break;
        }
    }
    if !ok {
        // Degenerate fallback: a minimal two-seat round.
        dice = [0u8; crate::MAX_PLAYERS];
        dice[0] = 1;
        dice[1] = 1;
    }
    let all_full = (0..p).all(|i| dice[i] == d as u8);
    let first_round = all_full && rng.unit() < 0.5;
    let opener = if first_round {
        0
    } else {
        let live: Vec<usize> = (0..p).filter(|&i| dice[i] > 0).collect();
        live[rng.below(live.len())]
    };
    (p as u8, d as u8, f as u8, dice, opener as u8, first_round)
}

/// Self-play `playouts` of a solved round, emitting a [`Sample`] at every
/// decision node (policy = solved distribution, value = realised outcome) plus
/// per-player value-only samples at the round opening (the continuation-query
/// distribution, for a possible later bootstrap).
fn collect<S: RoundSolver>(
    round: &Round<'_>,
    solver: &S,
    feat: &LiarsDice,
    playouts: usize,
    rng: &mut Rng,
    out: &mut Vec<Sample>,
) {
    let n = round.num_players();
    let mut root_val = vec![0.0f64; n];
    for _ in 0..playouts {
        let mut s = round.initial_state();
        let mut decisions: Vec<Decision> = Vec::new();
        while !round.is_terminal(&s) {
            match round.turn(&s) {
                Turn::Chance => {
                    let (a, _p) = round.sample_chance(&s, rng);
                    round.apply(&mut s, a);
                }
                Turn::Player(pl) => {
                    let (acts, sup) = legal_actions_and_support(feat, &s);
                    let pol = solver.policy_at(&s, pl);
                    let x = encode(feat, &s, pl);
                    let sparse: Vec<(usize, f32)> = sup
                        .iter()
                        .zip(&pol)
                        .filter(|&(_, &q)| q > 1e-6)
                        .map(|(&i, &q)| (i, q as f32))
                        .collect();
                    decisions.push((x, sparse, pl));
                    let i = rng.pick(&pol);
                    round.apply(&mut s, acts[i]);
                }
            }
        }
        let leaf: Vec<f64> = (0..n).map(|q| round.returns(&s, q)).collect();
        for (rv, &lv) in root_val.iter_mut().zip(&leaf) {
            *rv += lv;
        }
        for (x, policy, pl) in decisions {
            out.push(Sample {
                x,
                policy,
                z: leaf[pl] as f32,
            });
        }
    }
    // Value-only samples at the round opening (hands unrolled = unknown), one per
    // live player, training the value head on the public continuation state.
    let opening = round.initial_state();
    for (q, &rv) in root_val.iter().enumerate() {
        if opening.dice_left()[q] == 0 {
            continue;
        }
        out.push(Sample {
            x: encode(feat, &opening, q),
            policy: Vec::new(),
            z: (rv / playouts as f64) as f32,
        });
    }
}

/// Generate the samples for one sampled round (config + solve + playouts),
/// closing leaves with `boot`'s continuation (heuristic during warm-up, the
/// net's value head afterwards — see [`Bootstrap`]).
fn gen_round_samples(
    rng: &mut Rng,
    cfg: &TrainConfig,
    boot: &Bootstrap,
    warm: bool,
) -> Vec<Sample> {
    let (p, d, f, dice, opener, first_round) = sample_round(rng);
    let feat = LiarsDice::new(p, d, f);
    let total: u32 = dice.iter().map(|&x| u32::from(x)).sum();
    let mut out = Vec::new();
    // Two identical rounds: one consumed by the solver, one walked for playouts.
    // Each gets its own continuation (a `NetValue`'s memo must not be shared
    // across the two rounds, though both reference the same snapshot net/cache).
    let new_round = || {
        RoundSubgame::new(
            p,
            d,
            f,
            dice,
            opener,
            first_round,
            1,
            boot.continuation(p, f, warm),
        )
    };
    let solver_round = new_round();
    let play_round = new_round();
    if p == 2 && total <= u32::from(cfg.small_total) {
        // Tiny 2-player endgame rounds: exact CFR (best targets, cheap here).
        let mut sol = Cfr::new(solver_round);
        sol.solve(cfg.cfr_iters);
        collect(&play_round, &sol, &feat, cfg.playouts, rng, &mut out);
    } else {
        // Everything else: outcome-sampling MCCFR with a budget scaled DOWN for
        // bigger rounds, so no single round can stall the parallel batch (the
        // long-tail that was collapsing the pool to one core). Big rounds get
        // rougher targets, but the net aggregates over many and generalizes via
        // its features.
        let mut sol = OsMccfr::new(solver_round, rng.next_u64());
        sol.run(adaptive_os_iters(p, total, cfg.os_iters));
        collect(&play_round, &sol, &feat, cfg.playouts, rng, &mut out);
    }
    out
}

/// OsMccfr iteration budget bounded by round size (cost ~ iters x players x
/// ladder-depth, and depth grows with total dice). Reference work is a 2-player,
/// total-6 round; larger rounds are scaled down and clamped so per-round
/// wall-time stays roughly constant.
fn adaptive_os_iters(p: u8, total: u32, base: u64) -> u64 {
    let work = u64::from(p) * u64::from(total);
    (base * 12 / work.max(1)).clamp(1000, base)
}

fn fisher_yates(buf: &mut [Sample], rng: &mut Rng) {
    for i in (1..buf.len()).rev() {
        buf.swap(i, rng.below(i + 1));
    }
}

/// Per-round exploitability of `net`'s policy on small 2-player rounds (the
/// keep-best metric); lower is closer to equilibrium.
fn validate_exploitability(net: &Mlp) -> f64 {
    let cache = net.infer_cache();
    // Small configs only: exact best-response on bigger rounds (more dice/faces)
    // is serial and dominates iteration time. These two are the fast endgame
    // probes; the deeper configs are checked once in the final evaluation.
    let configs = [(1u8, 6u8), (2, 4)];
    let mut sum = 0.0;
    for &(d, f) in &configs {
        let feat = LiarsDice::new(2, d, f);
        let mut dice = [0u8; crate::MAX_PLAYERS];
        dice[0] = d;
        dice[1] = d;
        let round = RoundSubgame::new(2, d, f, dice, 0, true, 1, Cont::Heuristic(DiceShareValue));
        let policy = |_g: &Round, s: &LdState, pl: usize| net_policy(net, &cache, &feat, s, pl);
        let (_, _, nc) = nash_conv(&round, &policy);
        sum += nc / 2.0;
    }
    sum / configs.len() as f64
}

/// Win rate of `net` against the deployed determinized-rollout bot at a couple
/// of configs (a coarse strength signal; 2-player seat-swapped).
fn validate_winrate(net: &Mlp, games: u32, seed: u64) -> Vec<(String, f64)> {
    let agent = NetAgent::new(clone_net(net));
    let bot = Rollout::new(
        50,
        ProbabilisticAgent::default_agent(),
        BidConditioned::default(),
    );
    [(2u8, 5u8, 6u8), (2, 2, 6)]
        .iter()
        .map(|&(p, d, f)| {
            let g = LiarsDice::new(p, d, f);
            let wr = win_rate(&g, &agent, &bot, games, seed);
            (format!("{p}p{d}d{f}f"), wr)
        })
        .collect()
}

fn clone_net(net: &Mlp) -> Mlp {
    Mlp::from_bytes(&net.to_bytes()).expect("round-trip clone")
}

/// Train a net by the distillation loop above, checkpointing every iteration and
/// keeping the lowest-exploitability net at `{outdir}/best.bin`.
pub fn train(cfg: &TrainConfig) -> std::io::Result<Mlp> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(cfg.threads.max(1))
        .build_global()
        .ok();
    std::fs::create_dir_all(&cfg.outdir)?;
    let log_path = format!("{}/train.log", cfg.outdir);
    let mut log = std::fs::File::create(&log_path)?;

    let mut net = Mlp::new(feature_len(), cfg.hidden, policy_len(), cfg.seed);
    let mut opt = SgdMomentum::new(cfg.lr, cfg.momentum, cfg.l2);
    let mut buffer: Vec<Sample> = Vec::new();
    let mut rng = Rng::new(cfg.seed ^ 0x9E37_79B9);
    let mut grad = Vec::new();
    let mut best = f64::INFINITY;

    for iter in 0..cfg.iters {
        let t = Instant::now();
        // Parallel data generation; each round gets a deterministic seed.
        let base = cfg.seed ^ ((iter as u64) << 32);
        let t_gen = Instant::now();
        // Warm-up solves against the fixed heuristic; afterwards the leaves are
        // closed by the current net's value head (fitted value iteration). The
        // cache snapshots the weights at the start of this iteration, so every
        // round in the batch bootstraps off the same fixed continuation.
        let warm = iter < cfg.warmup_iters;
        let cache = net.infer_cache();
        let boot = Bootstrap {
            net: &net,
            cache: &cache,
        };
        let fresh: Vec<Sample> = (0..cfg.rounds_per_iter)
            .into_par_iter()
            .flat_map_iter(|k| {
                let mut r = Rng::new(base ^ (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                gen_round_samples(&mut r, cfg, &boot, warm)
            })
            .collect();
        let gen_s = t_gen.elapsed().as_secs_f64();
        let n_fresh = fresh.len();
        buffer.extend(fresh);
        if buffer.len() > cfg.buffer_cap {
            let drop = buffer.len() - cfg.buffer_cap;
            buffer.drain(0..drop);
        }

        let t_train = Instant::now();
        let (mut ce, mut mse, mut nb) = (0.0f32, 0.0f32, 0u32);
        for _ in 0..cfg.epochs {
            fisher_yates(&mut buffer, &mut rng);
            for chunk in buffer.chunks(cfg.batch) {
                let refs: Vec<&Sample> = chunk.iter().collect();
                let (c, m) = net.grad(&refs, &mut grad);
                opt.step(&mut net, &grad);
                ce += c;
                mse += m;
                nb += 1;
            }
        }
        let train_s = t_train.elapsed().as_secs_f64();
        let nb = nb.max(1) as f32;
        net.save(Path::new(&format!("{}/ckpt.bin", cfg.outdir)))?;

        let secs = t.elapsed().as_secs_f64();
        let phase = if warm { "warm" } else { "fvi " };
        let mut line = format!(
            "iter {iter:4} [{phase}]  buf {:7}  fresh {n_fresh:6}  ce {:.4}  mse {:.4}  {secs:5.1}s (gen {gen_s:.1} tr {train_s:.1})",
            buffer.len(),
            ce / nb,
            mse / nb,
        );
        if iter % cfg.val_every == 0 || iter == cfg.iters - 1 {
            let expl = validate_exploitability(&net);
            line.push_str(&format!("  expl {expl:.4}"));
            if expl < best {
                best = expl;
                net.save(Path::new(&format!("{}/best.bin", cfg.outdir)))?;
                line.push_str(" *best");
            }
            // Win-rate vs the rollout bot is expensive (nested rollouts per
            // decision), so only at the very end; per-round exploitability is
            // the cheap, principled keep-best metric during the run.
            if iter + 1 == cfg.iters {
                let wr = validate_winrate(&net, 200, cfg.seed ^ iter as u64);
                for (name, w) in &wr {
                    line.push_str(&format!("  {name} {w:.3}"));
                }
            }
        }
        println!("{line}");
        writeln!(log, "{line}")?;
        log.flush()?;
    }
    Ok(net)
}

/// Mean absolute error of `net`'s value-head continuation against the exact
/// 2-player lattice `V(dice_vector, opener)` over the reachable continuing
/// states (`1 <= a, b <= dice`, each opened by either seat). This is the proof
/// metric for fitted value iteration: a converged bootstrap drives it toward
/// solver noise; a diverging one drives it up.
///
/// Both tables are seat-0 valued — `NetValue` returns the zero-sum per-seat
/// values, `LatticeValue` stores `v0` and implies `-v0` — so we compare seat 0.
pub fn value_head_lattice_mae(net: &Mlp, dice: u8, faces: u8, lattice: &LatticeValue) -> f64 {
    let cache = net.infer_cache();
    let nv = NetValue::new(net, &cache, 2, faces);
    let mut sum = 0.0;
    let mut n = 0u32;
    for a in 1..=dice {
        for b in 1..=dice {
            for opener in 0..2usize {
                let exact = lattice
                    .get_two_player(&[a, b], opener)
                    .expect("lattice covers every reachable 2p state");
                let pred = nv.value(faces, &[a, b], opener, 0);
                sum += (pred - exact).abs();
                n += 1;
            }
        }
    }
    sum / n.max(1) as f64
}

/// One value-target sample per live seat at the opening of `(dice, opener)`:
/// the seat-perspective features paired with the *exactly* solved round's root
/// value to that seat (its post-round equity under `cont`). This is the value
/// label the real [`collect`] emits at the round opening, but computed from the
/// solver's exact expected value instead of sampled playouts — so the focused
/// fitted-VI harness has no Monte-Carlo noise on the value target.
fn round_value_samples<V: ContinuationValue>(
    feat: &LiarsDice,
    state: (u8, u8, u8),
    cont: V,
    iters: u64,
    out: &mut Vec<Sample>,
) {
    let (a, b, opener) = state;
    let mut dice_left = [0u8; crate::MAX_PLAYERS];
    dice_left[0] = a;
    dice_left[1] = b;
    let round = RoundSubgame::new(2, feat.dice, feat.faces, dice_left, opener, false, 1, cont);
    // The opening public state is the same one the value head reads as its
    // continuation query (free open, hands unrolled), so the value label and the
    // continuation prediction are anchored to identical features.
    let opening = round.initial_state();
    let mut cfr = Cfr::new(round);
    cfr.solve(iters);
    let v0 = cfr.expected_value();
    for (seat, z) in [(0usize, v0), (1, -v0)] {
        out.push(Sample {
            x: encode(feat, &opening, seat),
            policy: Vec::new(),
            z: z as f32,
        });
    }
}

/// A checkpoint of the focused fitted-VI harness: the value-head MAE against
/// the exact lattice after a phase of training.
#[derive(Clone, Copy, Debug)]
pub struct FviCheckpoint {
    pub iter: usize,
    pub warm: bool,
    pub mae: f64,
}

/// Focused fitted value iteration on a single 2-player `dice`×`faces` config:
/// the cheapest end-to-end proof that the bootstrap converges to the exact
/// lattice. Each iteration solves every reachable continuing round `(a, b,
/// opener)` exactly against the current continuation (the fixed heuristic for
/// the first `warmup` iters, then the net's own value head), trains the value
/// head on the backed-up root values, and records the MAE to `exact`.
///
/// This is the training loop's value-learning core, stripped of policy
/// distillation and config sampling so the convergence is deterministic and
/// measurable. The full [`train`] loop runs the same backup across the whole
/// config space.
pub fn fit_value_head_2p(
    dice: u8,
    faces: u8,
    exact: &LatticeValue,
    iters: usize,
    warmup: usize,
    cfr_iters: u64,
    seed: u64,
) -> (Mlp, Vec<FviCheckpoint>) {
    let feat = LiarsDice::new(2, dice, faces);
    let mut net = Mlp::new(feature_len(), 64, policy_len(), seed);
    let mut opt = SgdMomentum::new(0.05, 0.9, 1e-4);
    let mut grad = Vec::new();
    let mut rng = Rng::new(seed ^ 0xA5A5);
    let states: Vec<(u8, u8, u8)> = (1..=dice)
        .flat_map(|a| (1..=dice).flat_map(move |b| (0..2u8).map(move |o| (a, b, o))))
        .collect();
    let mut log = Vec::new();
    for iter in 0..iters {
        let warm = iter < warmup;
        let cache = net.infer_cache();
        let mut data = Vec::new();
        for &st in &states {
            if warm {
                round_value_samples(&feat, st, DiceShareValue, cfr_iters, &mut data);
            } else {
                let nv = NetValue::new(&net, &cache, 2, faces);
                round_value_samples(&feat, st, nv, cfr_iters, &mut data);
            }
        }
        // A few SGD passes per backup so the head tracks the fresh targets.
        for _ in 0..8 {
            fisher_yates(&mut data, &mut rng);
            for chunk in data.chunks(64) {
                let refs: Vec<&Sample> = chunk.iter().collect();
                net.grad(&refs, &mut grad);
                opt.step(&mut net, &grad);
            }
        }
        log.push(FviCheckpoint {
            iter,
            warm,
            mae: value_head_lattice_mae(&net, dice, faces, exact),
        });
    }
    (net, log)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FitConfig, fit_two_player};

    /// A fresh net's value head is near zero (heads are init-scaled down), so its
    /// continuation cannot already match the exact lattice — the floor the
    /// fitted-VI run must beat. Establishes the verification harness works.
    #[test]
    fn untrained_value_head_is_far_from_exact_lattice() {
        let (dice, faces) = (1u8, 6u8);
        let fit = fit_two_player(
            dice,
            faces,
            FitConfig {
                iters_per_solve: 600,
                tol: 1e-5,
                max_sweeps: 100,
                measure_exploitability: false,
            },
        );
        let net = Mlp::new(feature_len(), 64, policy_len(), 0x5EED);
        let mae = value_head_lattice_mae(&net, dice, faces, &fit.lattice);
        // The exact 1d6 values are O(0.1-0.3) in magnitude; a ~zero head is at
        // least ~0.05 off. (A loose, robust floor — the point is it is not ~0.)
        assert!(
            mae > 0.02,
            "untrained head should not match exact: mae={mae}"
        );
    }

    /// THE FITTED-VI PROOF (guarded). On a tiny 2p config, fitted value
    /// iteration must drive the value head's continuation *toward* the exact
    /// lattice: the MAE after the bootstrap phase must be below the warmup
    /// (heuristic-only) baseline, and small in absolute terms. This is the claim
    /// that the bootstrap converges to the proven exact values, not away.
    #[test]
    fn fitted_vi_value_head_converges_to_exact_lattice() {
        let (dice, faces) = (1u8, 6u8); // cheapest fixed point: 2 lattice states
        let fit = fit_two_player(
            dice,
            faces,
            FitConfig {
                iters_per_solve: 1500,
                tol: 1e-6,
                max_sweeps: 100,
                measure_exploitability: false,
            },
        );
        let (warmup, iters) = (4usize, 24usize);
        let (_, log) = fit_value_head_2p(dice, faces, &fit.lattice, iters, warmup, 1500, 0xF177ED);
        let warm_end = log[warmup - 1].mae;
        let final_mae = log.last().unwrap().mae;
        assert!(
            final_mae < warm_end,
            "fitted VI must improve on the warmup baseline: warm_end={warm_end} final={final_mae}"
        );
        // The exact 1d6 continuation values are O(0.1); converged fitted VI pins
        // the head to them within a loose, robust tolerance (the head is a tiny
        // 64-wide MLP fit by SGD, not a tabular solve).
        assert!(
            final_mae < 0.05,
            "fitted VI did not converge near the exact lattice: final={final_mae} (trace {:?})",
            log.iter().map(|c| c.mae).collect::<Vec<_>>()
        );
    }

    #[test]
    fn round_samples_are_well_formed() {
        let cfg = TrainConfig {
            cfr_iters: 200,
            os_iters: 2_000,
            playouts: 4,
            ..Default::default()
        };
        let net = Mlp::new(feature_len(), cfg.hidden, policy_len(), 1);
        let cache = net.infer_cache();
        let boot = Bootstrap {
            net: &net,
            cache: &cache,
        };
        let mut rng = Rng::new(7);
        let mut any = false;
        for i in 0..6 {
            // Exercise both phases: warm (heuristic) and fitted-VI (net) leaves.
            let warm = i < 3;
            let samples = gen_round_samples(&mut rng, &cfg, &boot, warm);
            for s in &samples {
                assert_eq!(s.x.len(), feature_len());
                assert!(s.z >= -1.001 && s.z <= 1.001, "z={}", s.z);
                let mut tot = 0.0f32;
                for &(idx, p) in &s.policy {
                    assert!(idx < policy_len());
                    assert!(p >= 0.0);
                    tot += p;
                }
                if !s.policy.is_empty() {
                    assert!((tot - 1.0).abs() < 0.05, "policy sums to {tot}");
                    any = true;
                }
            }
        }
        assert!(any, "expected at least some decision samples");
    }

    #[test]
    fn two_iters_reduce_policy_loss() {
        let cfg = TrainConfig {
            iters: 2,
            rounds_per_iter: 60,
            playouts: 6,
            hidden: 64,
            cfr_iters: 400,
            os_iters: 4_000,
            epochs: 2,
            val_every: 100, // skip validation in the test
            threads: 2,
            outdir: std::env::temp_dir()
                .join("ld_train_test")
                .to_string_lossy()
                .into_owned(),
            ..Default::default()
        };
        // Measure cross-entropy on a fixed sample set before and after a step.
        let mut rng = Rng::new(11);
        let seed_net = Mlp::new(feature_len(), cfg.hidden, policy_len(), 3);
        let seed_cache = seed_net.infer_cache();
        let boot = Bootstrap {
            net: &seed_net,
            cache: &seed_cache,
        };
        let mut data = Vec::new();
        for _ in 0..40 {
            // Warm (heuristic) targets — the divergence-free distillation path.
            data.extend(gen_round_samples(&mut rng, &cfg, &boot, true));
        }
        let refs: Vec<&Sample> = data.iter().filter(|s| !s.policy.is_empty()).collect();
        let mut net = Mlp::new(feature_len(), cfg.hidden, policy_len(), 3);
        let mut opt = SgdMomentum::new(cfg.lr, cfg.momentum, cfg.l2);
        let mut grad = Vec::new();
        let (ce0, _) = net.loss(&refs);
        for _ in 0..40 {
            net.grad(&refs, &mut grad);
            opt.step(&mut net, &grad);
        }
        let (ce1, _) = net.loss(&refs);
        assert!(ce1 < ce0, "policy CE should drop: {ce0} -> {ce1}");
    }
}
