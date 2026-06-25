//! Deep CFR training for the Liar's Dice net — the second training method
//! alongside the per-round distillation in [`crate::train`].
//!
//! Where distillation *solves* each sampled round (CFR/MCCFR) and supervises the
//! net on the solved strategy, this runs the generic feature-keyed Deep CFR
//! engine ([`solvers::deepcfr`]) directly over the round subgames: per-player
//! advantage nets are fit to sampled regrets from external-sampling traversals,
//! and the deployable policy is an average-strategy net. The output is the same
//! artifact — a [`solvers::azero::Mlp`] with `input = feature_len()`, `policy =
//! policy_len()`, played by [`crate::NetAgent`] — so the two methods are a fair
//! head-to-head.
//!
//! The config family (players 2..=6, dice 2..=8, faces 2..=6, arbitrary live
//! dice vectors) is fed to ONE engine: because the engine keys on
//! [`features::encode`] (which rotates the acting seat to reference index 0 and
//! encodes the player count), a single advantage net (`adv_nets = 1`) covers
//! every seat and every config — the same generalization the distillation net
//! relies on. The round leaves are closed by a [`ContinuationValue`]:
//! [`DiceShareValue`] during a warm-up phase (divergence-free), then
//! [`NetValue`] over the average-strategy net's value head (fitted value
//! iteration, warm-started — exactly as distillation), with the value head
//! trained on the traversal-root equity the engine emits.

use std::io::Write as _;
use std::path::Path;
use std::time::Instant;

use game_core::{Agent, Game, Rng, hash, play_n, win_share};
use rayon::prelude::*;
use solvers::Rollout;
use solvers::azero::{InferCache, Mlp};
use solvers::deepcfr::{DeepCfr, DeepCfrConfig, Encoder};

use crate::features::{encode, feature_len, policy_len, support};
use crate::{
    BidConditioned, ContinuationValue, DiceShareValue, LdState, LiarsDice, NetAgent, NetValue,
    ProbabilisticAgent, RoundSubgame,
};

/// The continuation closing a round's leaves: the fixed dice-share heuristic
/// during warm-up, the average-strategy net's value head (fitted value
/// iteration) afterwards. One enum so `RoundSubgame<LdCont>` is a single type
/// the engine runs over.
pub enum LdCont<'a> {
    Heuristic(DiceShareValue),
    Net(NetValue<'a>),
}

impl ContinuationValue for LdCont<'_> {
    fn value(&self, faces: u8, dice_left: &[u8], next_opener: usize, player: usize) -> f64 {
        match self {
            LdCont::Heuristic(h) => h.value(faces, dice_left, next_opener, player),
            LdCont::Net(n) => n.value(faces, dice_left, next_opener, player),
        }
    }
}

/// The round subgame type the engine traverses.
type Round<'a> = RoundSubgame<LdCont<'a>>;

/// Config-agnostic encoder: features and legal-action support for a round
/// subgame, delegating to [`features::encode`] / [`features::support`] against
/// the round's own config. One encoder serves the whole family.
pub struct LdEncoder;

impl Encoder<Round<'_>> for LdEncoder {
    fn feature_len(&self) -> usize {
        feature_len()
    }
    fn policy_len(&self) -> usize {
        policy_len()
    }
    fn features(&self, game: &Round<'_>, state: &LdState, player: usize) -> Vec<f32> {
        encode(game.config(), state, player)
    }
    fn support(&self, game: &Round<'_>, state: &LdState) -> Vec<usize> {
        support(game.config(), state)
    }
}

#[derive(Clone)]
pub struct DeepCfrTrainConfig {
    /// Total CFR iterations (each iteration runs traversals on one sampled
    /// round, per traverser).
    pub iters: usize,
    /// Iterations between average-strategy checkpoints / continuation refreshes.
    pub block: usize,
    /// Iterations solved against the fixed [`DiceShareValue`] before switching
    /// the continuation to the net's value head.
    pub warmup_iters: usize,
    pub traversals: usize,
    /// Retrain the advantage net from scratch every `train_every` iterations
    /// (Deep CFR retrains periodically, not every iteration — the dominant cost
    /// is the retrain).
    pub train_every: usize,
    pub hidden: usize,
    pub adv_reservoir: usize,
    pub strat_reservoir: usize,
    pub adv_steps: usize,
    pub strat_steps: usize,
    pub batch: usize,
    pub lr: f32,
    pub momentum: f32,
    pub l2: f32,
    /// ε-greedy exploration weight at the sampled (opponent) traversal branches,
    /// importance-weighted to stay unbiased (see [`DeepCfrConfig::explore_eps`]).
    /// Pushes the traverser through rarely-played opponent bids so the advantage
    /// net actually practices responding to them instead of staying at the naive
    /// prior at those infosets.
    pub explore_eps: f64,
    pub threads: usize,
    pub outdir: String,
    pub seed: u64,
    /// Keep-best win-share eval budget: Rollout count for the opponent field and
    /// games played per high-dice config each block. Kept modest so the eval is a
    /// small slice of the block (the Rollout bot, not the net, is the cost).
    pub eval_rollouts: u32,
    pub eval_games: u32,
}

impl Default for DeepCfrTrainConfig {
    fn default() -> Self {
        Self {
            iters: 4000,
            block: 200,
            warmup_iters: 800,
            traversals: 4,
            train_every: 4,
            hidden: 256,
            adv_reservoir: 2_000_000,
            strat_reservoir: 4_000_000,
            adv_steps: 600,
            strat_steps: 3000,
            batch: 1024,
            lr: 0.01,
            momentum: 0.9,
            l2: 1e-4,
            explore_eps: 0.5,
            threads: 18,
            outdir: "runs/ld_deepcfr".into(),
            seed: 0xD1CE_DEEC,
            eval_rollouts: 100,
            eval_games: 40,
        }
    }
}

const MAX_TRAIN_DICE: usize = 8;
const MAX_TRAIN_TOTAL: u32 = 48;

/// One sampled round configuration from the supported family.
struct RoundCfg {
    players: u8,
    dice_per: u8,
    faces: u8,
    dice: [u8; crate::MAX_PLAYERS],
    opener: u8,
    first_round: bool,
}

/// Sample one config + dice vector + opener across the supported family,
/// RE-BIASED toward the HIGH-DICE-COUNT regime the deploy actually cares about
/// (flagship 5p5d6f = 25 dice, up to 6p8d6f = 48), away from the tiny endgame.
/// Three levers do the re-biasing:
///   * dice-per `d` lands in 5..=8 ~75% of the time (and 2..=4 the rest, so the
///     net still sees some lower-dice rounds for generality);
///   * the dice VECTOR uses a max-of-two draw (seats hold MANY dice) instead of
///     the old min-of-two, with a large fraction of FULL game openings (every
///     live seat at `d`, `first_round=true`) — the actual 5p5d6f-style start;
///   * players `p` spans 2..=6 with a mild lean toward more players, which also
///     raises the total dice in play.
///
/// A minority of mid-game vectors (some dice already lost) keeps the net handling
/// dice loss, but the tiny deep-ladder endgame is de-emphasized rather than the
/// old regime where its mass dominated. The `>=2`-live-seats and `total <=
/// MAX_TRAIN_TOTAL` guards are kept.
fn sample_round_config(rng: &mut Rng) -> RoundCfg {
    // Mild lean toward more players (more seats => more total dice). Bucket the
    // unit draw so larger `p` is somewhat more likely without dropping the small
    // tables: roughly p=2:13% 3:17% 4:20% 5:23% 6:27%.
    let p = match rng.unit() {
        u if u < 0.13 => 2,
        u if u < 0.30 => 3,
        u if u < 0.50 => 4,
        u if u < 0.73 => 5,
        _ => 6,
    };
    // ~75% of rounds use a high dice-per (5..=8), ~25% a low one (2..=4).
    let d = if rng.unit() < 0.75 {
        5 + rng.below(MAX_TRAIN_DICE - 4) // 5..=8
    } else {
        2 + rng.below(3) // 2..=4
    };
    let f = 2 + rng.below(5); // 2..=6

    // A FULL opening (every live seat at `d`, first_round) is the real game start
    // and the dominant training target whenever it fits the total-dice budget.
    let full_total = (p as u32) * (d as u32);
    let want_full = full_total <= MAX_TRAIN_TOTAL && rng.unit() < 0.45;

    let mut dice = [0u8; crate::MAX_PLAYERS];
    let mut ok = false;
    if want_full {
        for die in dice.iter_mut().take(p) {
            *die = d as u8;
        }
        ok = true;
    } else {
        for _ in 0..32 {
            for die in dice.iter_mut().take(p) {
                // max-of-two keeps seats HIGH (the opposite of the old min-of-two
                // small bias); ~10% of seats are eliminated so mid-game vectors
                // with lost dice still appear.
                *die = if rng.unit() < 0.90 {
                    let a = 1 + rng.below(d);
                    let b = 1 + rng.below(d);
                    a.max(b) as u8
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
    }
    if !ok {
        dice = [0u8; crate::MAX_PLAYERS];
        for die in dice.iter_mut().take(p) {
            *die = d as u8;
        }
    }
    let all_full = (0..p).all(|i| dice[i] == d as u8);
    let first_round = all_full && rng.unit() < 0.7;
    let opener = if first_round {
        0
    } else {
        let live: Vec<usize> = (0..p).filter(|&i| dice[i] > 0).collect();
        live[rng.below(live.len())]
    };
    RoundCfg {
        players: p as u8,
        dice_per: d as u8,
        faces: f as u8,
        dice,
        opener: opener as u8,
        first_round,
    }
}

/// Build a round subgame with the chosen continuation (heuristic during warm-up,
/// the net's value head afterwards).
fn build_round<'a>(c: &RoundCfg, warm: bool, net: &'a Mlp, cache: &'a InferCache) -> Round<'a> {
    let cont = if warm {
        LdCont::Heuristic(DiceShareValue)
    } else {
        LdCont::Net(NetValue::new(net, cache, c.players, c.faces))
    };
    RoundSubgame::new(
        c.players,
        c.dice_per,
        c.faces,
        c.dice,
        c.opener,
        c.first_round,
        1,
        cont,
    )
}

/// The high-dice configs the keep-best metric scores against — the regime the
/// deploy actually cares about (the flagship 5p5d6f start and a 4p4d6f table).
const KEEPBEST_CONFIGS: &[(u8, u8, u8)] = &[(5, 5, 6), (4, 4, 6)];

/// The deployed bot, exactly as `lab`'s registry / `ld_eval` build it for
/// `bot=rollout`, but at a reduced rollout count so the keep-best eval stays a
/// small fraction of the block (the Rollout bot is the slow part; the net is a
/// fast forward pass).
fn deployed_bot(rollouts: u32) -> Rollout<LiarsDice, ProbabilisticAgent, BidConditioned> {
    Rollout::new(
        rollouts,
        ProbabilisticAgent::default_agent(),
        BidConditioned::default(),
    )
}

/// `hero`'s win-share in a field of `field` at `(p, d, f)`: the hero is rotated
/// through every seat (the repo's win-share convention — see `ld_eval` /
/// `compare::play_one_field_game`), credit = win 1 / k-way tie `1/k` / loss 0.
/// Games run in parallel; the Rollout opponent's own per-decision rayon fan-out
/// nests under that (work-stealing keeps it correct). Fair share = `1/p`.
fn field_win_share<A, B>(p: u8, d: u8, f: u8, hero: &A, field: &B, games: u32, seed: u64) -> f64
where
    A: Agent<LiarsDice> + Sync,
    B: Agent<LiarsDice> + Sync,
{
    let game = LiarsDice::new(p, d, f);
    let n = game.num_players();
    let hero: &(dyn Agent<LiarsDice> + Sync) = hero;
    let field: &(dyn Agent<LiarsDice> + Sync) = field;
    let total: f64 = (0..games)
        .into_par_iter()
        .map(|g| {
            let mut rng = Rng::new(hash::combine(seed, u64::from(g)));
            let hero_seat = (g as usize) % n;
            let seats: Vec<&dyn Agent<LiarsDice>> = (0..n)
                .map(|seat| if seat == hero_seat { hero } else { field } as _)
                .collect();
            let terminal = play_n(&game, &seats, &mut rng);
            win_share(&game, &terminal, hero_seat)
        })
        .sum();
    total / games as f64
}

/// Keep-best metric: the MEAN win-share of `net`'s average-strategy policy
/// (played as a [`NetAgent`]) against a field of the deployed Rollout bot across
/// the high-dice [`KEEPBEST_CONFIGS`] — the user's ACTUAL goal (beat the deployed
/// bot at 5p5d6f and up), not small-config exploitability. HIGHER is better (the
/// sense flips vs the old exploitability metric, where lower won). Per-config
/// shares are returned alongside the mean for logging. A modest budget keeps this
/// a small slice of the block: the net is a fast forward pass, the Rollout bot is
/// the cost, so it runs at a reduced rollout count.
fn validate_win_share(
    net: &Mlp,
    rollouts: u32,
    games: u32,
    seed: u64,
) -> (f64, Vec<(String, f64)>) {
    let hero = NetAgent::new(clone_net(net));
    let mut per_config = Vec::with_capacity(KEEPBEST_CONFIGS.len());
    let mut sum = 0.0;
    for &(p, d, f) in KEEPBEST_CONFIGS {
        let field = deployed_bot(rollouts);
        let cfg_seed = hash::combine(
            seed,
            (u64::from(p) << 16) | (u64::from(d) << 8) | u64::from(f),
        );
        let share = field_win_share(p, d, f, &hero, &field, games, cfg_seed);
        sum += share;
        per_config.push((format!("{p}p{d}d{f}f"), share));
    }
    (sum / KEEPBEST_CONFIGS.len() as f64, per_config)
}

fn clone_net(net: &Mlp) -> Mlp {
    Mlp::from_bytes(&net.to_bytes()).expect("round-trip clone")
}

/// The engine config for a Liar's Dice run. Always ONE advantage net: the
/// features are seat-relative and encode the player count, so a single net
/// covers every seat and every config. `collect_root_value` is on for the
/// family run (the value head must learn the round-opening equity the
/// [`NetValue`] continuation reads) and off for the fixed single-round harness
/// (no continuation to learn).
fn engine_config(cfg: &DeepCfrTrainConfig, collect_root_value: bool) -> DeepCfrConfig {
    DeepCfrConfig {
        iters: cfg.iters,
        traversals: cfg.traversals,
        train_every: cfg.train_every,
        hidden: cfg.hidden,
        adv_reservoir: cfg.adv_reservoir,
        strat_reservoir: cfg.strat_reservoir,
        adv_steps: cfg.adv_steps,
        strat_steps: cfg.strat_steps,
        batch: cfg.batch,
        lr: cfg.lr,
        momentum: cfg.momentum,
        l2: cfg.l2,
        seed: cfg.seed,
        adv_nets: 1,
        collect_root_value,
        explore_eps: cfg.explore_eps,
    }
}

/// Train the Liar's Dice net by Deep CFR over the config family, checkpointing
/// the average-strategy net every block and keeping the HIGHEST-win-share net
/// (vs the deployed Rollout bot at the high-dice configs) at `{outdir}/best.bin`.
/// Returns the final average-strategy net.
pub fn train(cfg: &DeepCfrTrainConfig) -> std::io::Result<Mlp> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(cfg.threads.max(1))
        .build_global()
        .ok();
    std::fs::create_dir_all(&cfg.outdir)?;
    let log_path = format!("{}/train.log", cfg.outdir);
    let mut log = std::fs::File::create(&log_path)?;

    let enc = LdEncoder;
    // `players` only sets the default advantage-net count, which `engine_config`
    // overrides to 1; each round subgame carries its own true player count
    // (2..=6), and the single seat-relative advantage net handles all of them.
    // Blocks are driven manually so the continuation net can be refreshed each
    // block.
    let mut engine = DeepCfr::new(2, &enc, engine_config(cfg, true));

    // The current average-strategy net, refreshed each block; its value head is
    // the continuation after warm-up.
    let mut net = Mlp::new(feature_len(), cfg.hidden, policy_len(), cfg.seed ^ 0xA5A5);
    // Keep-best by win-share: HIGHER is better, so the bar starts at -inf (the
    // opposite sense of the old exploitability metric, where lower won).
    let mut best = f64::NEG_INFINITY;
    let mut done = 0usize;
    let mut sampler_rng = Rng::new(cfg.seed ^ 0x9E37_79B9);

    while done < cfg.iters {
        let t = Instant::now();
        let n = cfg.block.min(cfg.iters - done);
        let warm = done < cfg.warmup_iters;
        // Snapshot the value net + cache for this block's continuation.
        let cont_net = clone_net(&net);
        let cont_cache = cont_net.infer_cache();
        // A sub-stream for round sampling so the engine RNG drives traversals.
        let block_seed = sampler_rng.next_u64();
        let mut sample_rng = Rng::new(block_seed);
        let new_net = engine.run_family(n, &enc, |_engine_rng| {
            let c = sample_round_config(&mut sample_rng);
            build_round(&c, warm, &cont_net, &cont_cache)
        });
        net = new_net;
        done += n;

        net.save(Path::new(&format!("{}/ckpt.bin", cfg.outdir)))?;
        // Keep-best on the user's actual goal: mean win-share vs the deployed
        // Rollout bot at the high-dice configs (fair = 1/players; >fair beats it).
        let eval_seed = cfg.seed ^ (done as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let (mean_share, per_config) =
            validate_win_share(&net, cfg.eval_rollouts, cfg.eval_games, eval_seed);
        let shares: Vec<String> = per_config
            .iter()
            .map(|(name, s)| format!("{name} {s:.3}"))
            .collect();
        let secs = t.elapsed().as_secs_f64();
        let phase = if warm { "warm" } else { "fvi " };
        let mut line = format!(
            "iters {done:5} [{phase}]  strat-buf {:8}  adv-buf {:8}  {secs:6.1}s  \
             winshare {mean_share:.4}  [{}]",
            engine.strat_reservoir_len(),
            engine.advantage_reservoir_len(0),
            shares.join("  "),
        );
        if mean_share > best {
            best = mean_share;
            net.save(Path::new(&format!("{}/best.bin", cfg.outdir)))?;
            line.push_str(" *best");
        }
        println!("{line}");
        writeln!(log, "{line}")?;
        log.flush()?;
    }
    // Sanity: the produced net loads as a NetAgent (the deployable form).
    let _agent = NetAgent::new(clone_net(&net));
    Ok(net)
}

/// Run Deep CFR on a single fixed 2-player round (heuristic continuation) and
/// return the average-strategy net — the focused harness behind the small-LD
/// gate, where the same round is solvable exactly for a comparison baseline.
pub fn train_single_round_2p(dice: u8, faces: u8, cfg: &DeepCfrTrainConfig) -> Mlp {
    let enc = LdEncoder;
    let mut engine = DeepCfr::new(2, &enc, engine_config(cfg, false));
    let mut dice_left = [0u8; crate::MAX_PLAYERS];
    dice_left[0] = dice;
    dice_left[1] = dice;
    let round = RoundSubgame::new(
        2,
        dice,
        faces,
        dice_left,
        0,
        true,
        1,
        LdCont::Heuristic(DiceShareValue),
    );
    engine.run(&round, &enc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::net_policy;

    /// The sampler now targets the high-dice regime: most draws should have a
    /// large total dice count and a high dice-per, with full openings present.
    #[test]
    fn sampler_targets_high_dice_count() {
        let mut rng = Rng::new(12345);
        let n = 20000;
        let mut sum_total = 0u64;
        let mut high_total = 0; // total >= 20 (the 5p5d6f regime and up)
        let mut high_d = 0; // dice-per in 5..=8
        let mut full_open = 0;
        for _ in 0..n {
            let c = sample_round_config(&mut rng);
            let live = (0..c.players as usize).filter(|&i| c.dice[i] > 0).count();
            let total: u32 = c.dice[..c.players as usize]
                .iter()
                .map(|&x| u32::from(x))
                .sum();
            assert!(live >= 2, "at least two live seats");
            assert!(total <= MAX_TRAIN_TOTAL, "within the total-dice budget");
            sum_total += u64::from(total);
            high_total += u32::from(total >= 20);
            high_d += u32::from((5..=8).contains(&c.dice_per));
            full_open += u32::from(c.first_round);
        }
        let mean = sum_total as f64 / n as f64;
        // The old min-of-two sampler sat far lower; the regime is now high.
        assert!(mean > 16.0, "mean total dice should be high, got {mean:.2}");
        assert!(high_total as f64 / n as f64 > 0.4, "plenty of 20+ totals");
        assert!(high_d as f64 / n as f64 > 0.65, "dice-per mostly 5..=8");
        assert!(full_open > 0, "some full game openings");
    }
    use solvers::{Cfr, nash_conv};

    #[test]
    fn ld_deepcfr_smoke_produces_loadable_netagent() {
        let cfg = DeepCfrTrainConfig {
            iters: 12,
            block: 4,
            warmup_iters: 8,
            traversals: 2,
            hidden: 32,
            adv_reservoir: 50_000,
            strat_reservoir: 100_000,
            adv_steps: 30,
            strat_steps: 60,
            batch: 128,
            threads: 2,
            outdir: std::env::temp_dir()
                .join("ld_deepcfr_test")
                .to_string_lossy()
                .into_owned(),
            ..Default::default()
        };
        let net = train(&cfg).expect("deep cfr smoke trains");
        assert_eq!(net.input_len(), feature_len());
        assert_eq!(net.policy_len(), policy_len());
        // The output net is NetAgent-compatible (same form as distillation).
        let _agent = NetAgent::new(clone_net(&net));
    }

    /// Small-LD gate: Deep CFR on a fixed 2-player round drives its per-round
    /// exploitability well below the uniform-random baseline and toward the
    /// exact-CFR floor (≈0) the distillation method *distils* on the same round.
    /// Prints the Deep CFR per-round exploitability, the exact-CFR floor, and
    /// the uniform baseline. `--ignored` (slow); run with `--features parallel`.
    #[test]
    #[ignore = "slow small-LD comparison; run with --ignored --features parallel"]
    fn small_ld_deepcfr_per_round_exploitability() {
        let (dice, faces) = (1u8, 6u8); // the smallest 2p round
        let cfg = DeepCfrTrainConfig {
            iters: 2000,
            traversals: 16,
            train_every: 4,
            hidden: 64,
            adv_reservoir: 600_000,
            strat_reservoir: 1_500_000,
            adv_steps: 300,
            strat_steps: 8000,
            batch: 512,
            lr: 0.02,
            l2: 1e-5,
            ..Default::default()
        };
        let net = train_single_round_2p(dice, faces, &cfg);

        // Deep CFR's per-round exploitability via exact best response.
        let cache = net.infer_cache();
        let feat = LiarsDice::new(2, dice, faces);
        let mut dl = [0u8; crate::MAX_PLAYERS];
        dl[0] = dice;
        dl[1] = dice;
        let make_round = || {
            RoundSubgame::new(
                2,
                dice,
                faces,
                dl,
                0,
                true,
                1,
                LdCont::Heuristic(DiceShareValue),
            )
        };
        let round = make_round();
        let policy = |_g: &Round, s: &LdState, pl: usize| net_policy(&net, &cache, &feat, s, pl);
        let deep = nash_conv(&round, &policy).2 / 2.0;

        // The exact CFR floor on the same round (what distillation distils).
        let mut cfr = Cfr::new(make_round());
        cfr.solve(20_000);
        let floor = cfr.exploitability().2 / 2.0;

        // The uniform-random baseline on the same round (the high-water mark a
        // learned strategy must beat).
        let uniform_round = make_round();
        let unif = |_g: &Round, s: &LdState, _pl: usize| {
            let k = uniform_round.legal_actions(s).len();
            vec![1.0 / k as f64; k]
        };
        let uniform = nash_conv(&uniform_round, &unif).2 / 2.0;

        println!(
            "small-LD 2p{dice}d{faces}f: deep-cfr per-round expl = {deep:.4}, \
             exact-CFR floor = {floor:.4}, uniform baseline = {uniform:.4}"
        );
        // Deep CFR must learn a substantially less exploitable strategy than
        // uniform and be in the low-exploitability regime (a robust gate that
        // does not hinge on the exact SGD/sampling luck of one tiny run).
        assert!(
            deep < 0.6 * uniform && deep < 0.1,
            "Deep CFR per-round exploitability should approach the floor: \
             deep={deep} (floor {floor}, uniform {uniform})"
        );
    }
}
