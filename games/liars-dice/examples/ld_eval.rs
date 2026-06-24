//! Comparison + final-validation harness for trained Liar's Dice policy nets.
//!
//! Both the distillation trainer (`train_net`) and the Deep CFR trainer
//! (`deepcfr_train`) emit the same artifact: a [`solvers::azero::Mlp`] saved as a
//! `.bin`, played by [`liars_dice::NetAgent`]. This tool picks the head-to-head
//! winner between two such nets and serves as the final go/no-go check on one
//! net, with four lenses:
//!
//! 1. **Strength vs the deployed bot** — does the net actually beat the
//!    determinized-rollout bot the arcade ships, across a config spread?
//! 2. **Head-to-head A vs B** — the decisive who-beats-whom (only with `b=`).
//! 3. **Per-round exploitability** — equilibrium quality on small 2p configs,
//!    via exact best-response NashConv over the net's policy. A companion
//!    **multiplayer (n>=3)** section reports the profile best-response gain (the
//!    single-seat best response while all other seats follow the net) — the only
//!    well-defined exploitability for the constant-sum-but-not-zero-sum N-player
//!    game, where no Nash oracle exists.
//! 4. **Thin-bluff endgame probe** — does the net call a thin single-die bid it
//!    should be suspicious of? (The user's original complaint.)
//!
//! ```text
//! cargo run --release -p liars-dice --example ld_eval -- \
//!     a=data/ld/distill.bin b=data/ld/deepcfr.bin games=2000 rollouts=200 seed=1
//! ```
//!
//! `a=` is required; `b=` optional. Sections 2 (head-to-head) and the B columns
//! everywhere else only print when `b=` is given. The match-play sections run the
//! independent games across all cores (rayon), so the defaults finish in a couple
//! of minutes; drop `games`/`rollouts` for a quicker smoke test.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use game_core::{Agent, Game, Rng, hash, play_n, win_share};
use liars_dice::{
    Action, BidConditioned, DiceShareValue, LdState, LiarsDice, NetAgent, ProbabilisticAgent,
    RoundSubgame, net_policy,
};
use rayon::prelude::*;
use solvers::Rollout;
use solvers::azero::{InferCache, Mlp};
use solvers::exploit::{Policy, nash_conv, profile_exploitability};

/// The deployed bot, exactly as `lab`'s registry builds it for `bot=rollout`.
fn deployed_bot(rollouts: u32) -> Rollout<LiarsDice, ProbabilisticAgent, BidConditioned> {
    Rollout::new(
        rollouts,
        ProbabilisticAgent::default_agent(),
        BidConditioned::default(),
    )
}

/// A named net plus the agent that plays it. The agent borrows the net via the
/// `Mlp` it owns, so the struct keeps both alive together.
struct Net {
    label: String,
    agent: NetAgent,
}

impl Net {
    /// Load a net `.bin`, surfacing a missing/unparseable file as a clean error
    /// string (never a panic) so a bad path is a usable message, not a crash.
    fn load(label: &str, path: &str) -> Result<Self, String> {
        let p = Path::new(path);
        let agent =
            NetAgent::load(p).map_err(|e| format!("could not load net {label}={path}: {e}"))?;
        Ok(Self {
            label: format!("{label} ({path})"),
            agent,
        })
    }
}

/// Configs spanning the supported space — small heads-up, the 5p5d6f target,
/// wide tables, and a tiny 3-face board — so a strong net has to generalize.
const STRENGTH_CONFIGS: &[(u8, u8, u8)] = &[
    (2, 2, 6),
    (2, 5, 6),
    (2, 8, 6),
    (3, 3, 6),
    (4, 4, 6),
    (5, 5, 6),
    (6, 3, 6),
    (2, 3, 3),
];

/// Small 2p configs whose first-round subgame is cheap to best-respond against
/// exactly — the equilibrium-quality lens.
const EXPLOIT_CONFIGS: &[(u8, u8, u8)] = &[(2, 1, 6), (2, 2, 4), (2, 2, 6), (2, 3, 4)];

/// Small *multiplayer* (n>=3) configs whose first-round subgame is still cheap
/// enough to enumerate exactly. The N-player game is constant-sum but not
/// zero-sum, so there is no Nash oracle; the per-seat best-response gain is the
/// well-defined measure of how exploitable the profile is.
///
/// The cost is steep: the exact walk enumerates the full game DAG, which grows
/// combinatorially in *both* faces (the `Open(q, f)` grid and the per-seat hand
/// chance fan-out) and total dice (the bid-quantity ceiling). Measured net-policy
/// runtimes (memoized on infoset): 3p2d3f ~0.2s, 3p2d4f ~1.5s, 4p2d3f ~9s,
/// 3p3d3f ~15s — all kept here. Beyond that it blows past "a few seconds":
/// 3p2d6f ~50s, 4p2d4f >1min, 3p3d4f ~7.5min, 4p2d6f minutes more. Those larger
/// configs (the real 6-face target) are too expensive to enumerate exactly and
/// want a *sampled* best response instead — flagged, not run here.
const MULTI_EXPLOIT_CONFIGS: &[(u8, u8, u8)] = &[(3, 2, 3), (3, 2, 4), (3, 3, 3), (4, 2, 3)];

fn cfg_label(p: u8, d: u8, f: u8) -> String {
    format!("{p}p{d}d{f}f")
}

/// Per-config seed so the configs are independent (uncorrelated) draws.
fn config_seed(seed: u64, p: u8, d: u8, f: u8) -> u64 {
    seed ^ ((u64::from(p) << 16) | (u64::from(d) << 8) | u64::from(f))
}

/// A's score against B in `(p, d, f)`: for 2 players, A's seat-swapped win rate
/// (>0.5 = A better); for >2, A's win-share heading a uniform field of B.
///
/// This mirrors `game_core::win_rate` / `winrate_vs_field` exactly — seats
/// swapped each game (2p) or the hero rotated through every seat (field), credit
/// = win 1 / draw ½ / loss 0 — but runs the independent games in parallel, each
/// from its own `(seed, game)`-derived RNG. Rollout opponents are single-
/// threaded per decision, so without this the default `games`/`rollouts` would
/// blow the time budget on the wide configs; the per-game seed derivation keeps
/// the estimate deterministic and seat-balanced (the same pattern `lab`'s
/// parallel field runner uses).
fn score_vs<A: Agent<LiarsDice> + Sync, B: Agent<LiarsDice> + Sync>(
    p: u8,
    d: u8,
    f: u8,
    a: &A,
    b: &B,
    games: u32,
    seed: u64,
) -> f64 {
    let game = LiarsDice::new(p, d, f);
    let n = game.num_players();
    let a: &(dyn Agent<LiarsDice> + Sync) = a;
    let b: &(dyn Agent<LiarsDice> + Sync) = b;
    let total: f64 = (0..games)
        .into_par_iter()
        .map(|g| {
            let mut rng = Rng::new(hash::combine(seed, u64::from(g)));
            if n == 2 {
                // Swap seats each game to cancel first-mover bias; score A.
                let (s0, s1) = if g % 2 == 0 { (a, b) } else { (b, a) };
                let agents: [&dyn Agent<LiarsDice>; 2] = [s0, s1];
                let terminal = play_n(&game, &agents, &mut rng);
                let a_seat = if g % 2 == 0 { 0 } else { 1 };
                win_share(&game, &terminal, a_seat)
            } else {
                // Rotate the hero (A) through every seat against a field of B.
                let hero = (g as usize) % n;
                let seats: Vec<&dyn Agent<LiarsDice>> =
                    (0..n).map(|p| if p == hero { a } else { b } as _).collect();
                let terminal = play_n(&game, &seats, &mut rng);
                win_share(&game, &terminal, hero)
            }
        })
        .sum();
    total / games as f64
}

fn parse_args() -> HashMap<String, String> {
    std::env::args()
        .skip(1)
        .filter_map(|arg| {
            let (k, v) = arg.split_once('=')?;
            Some((k.to_string(), v.to_string()))
        })
        .collect()
}

/// Parse an optional numeric arg, falling back to `default` when absent and
/// surfacing `bad_msg` (verbatim) when present but unparseable.
fn parse_opt<T: std::str::FromStr>(
    args: &HashMap<String, String>,
    key: &str,
    default: T,
    bad_msg: &str,
) -> Result<T, String> {
    match args.get(key).map(|s| s.parse()) {
        Some(Ok(v)) => Ok(v),
        Some(Err(_)) => Err(bad_msg.to_string()),
        None => Ok(default),
    }
}

/// The optional run knobs: `(games, rollouts, seed)`, each defaulted when absent.
fn parse_run_args(args: &HashMap<String, String>) -> Result<(u32, u32, u64), String> {
    let games = parse_opt(args, "games", 2000, "games= must be a positive integer")?;
    let rollouts = parse_opt(
        args,
        "rollouts",
        200,
        "rollouts= must be a positive integer",
    )?;
    let seed = parse_opt(args, "seed", 0xD1CE, "seed= must be a non-negative integer")?;
    Ok((games, rollouts, seed))
}

fn main() -> ExitCode {
    let args = parse_args();
    let Some(path_a) = args.get("a") else {
        eprintln!(
            "usage: ld_eval a=<net.bin> [b=<net.bin>] [games=N] [rollouts=N] [seed=N]\n\
             a= is required (the net to validate); b= adds a second net to compare."
        );
        return ExitCode::FAILURE;
    };
    let (games, rollouts, seed) = match parse_run_args(&args) {
        Ok(values) => values,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    let net_a = match Net::load("A", path_a) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let net_b = match args.get("b") {
        Some(path_b) => match Net::load("B", path_b) {
            Ok(n) => Some(n),
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        },
        None => None,
    };

    let t0 = Instant::now();
    println!("Liar's Dice net evaluation");
    println!("  A = {}", net_a.label);
    if let Some(b) = &net_b {
        println!("  B = {}", b.label);
    }
    println!("  games/config = {games}   baseline rollouts = {rollouts}   seed = {seed}\n");

    strength_section(&net_a, net_b.as_ref(), games, rollouts, seed);
    if let Some(b) = &net_b {
        head_to_head_section(&net_a, b, games, seed);
    }
    exploitability_section(&net_a, net_b.as_ref());
    multiplayer_exploit_section(&net_a, net_b.as_ref());
    thin_bluff_section(&net_a, net_b.as_ref());

    println!("\nfinished in {:.1?}", t0.elapsed());
    ExitCode::SUCCESS
}

/// Section 1 — strength vs the deployed rollout bot. For 2p the cell is A's
/// win rate (fair = 0.5); for >2p it is A's win-share vs the fair `1/players`.
fn strength_section(net_a: &Net, net_b: Option<&Net>, games: u32, rollouts: u32, seed: u64) {
    println!("[1] STRENGTH vs the deployed rollout bot (rollouts={rollouts})");
    println!("    cell = score vs the bot; 2p: win rate (>0.5 beats it); >2p: win-share");
    if net_b.is_some() {
        println!("    {:>9} {:>6} {:>10} {:>10}", "config", "fair", "A", "B");
    } else {
        println!("    {:>9} {:>6} {:>10}", "config", "fair", "A");
    }
    println!("    {}", "-".repeat(if net_b.is_some() { 39 } else { 28 }));

    for &(p, d, f) in STRENGTH_CONFIGS {
        let bot = deployed_bot(rollouts);
        let fair = 1.0 / p as f64;
        let s = config_seed(seed, p, d, f);
        let a = score_vs(p, d, f, &net_a.agent, &bot, games, s);
        if let Some(b) = net_b {
            let bb = score_vs(p, d, f, &b.agent, &bot, games, s);
            println!(
                "    {:>9} {:>6.3} {:>10} {:>10}",
                cfg_label(p, d, f),
                fair,
                mark(a, fair),
                mark(bb, fair),
            );
        } else {
            println!(
                "    {:>9} {:>6.3} {:>10}",
                cfg_label(p, d, f),
                fair,
                mark(a, fair),
            );
        }
    }
    println!();
}

/// Section 2 — the decisive comparison: A directly against B. For 2p the cell
/// is A's win rate vs B (>0.5 = A wins the match-up); for >2p A's win-share in
/// a field of B (fair = `1/players`).
fn head_to_head_section(net_a: &Net, net_b: &Net, games: u32, seed: u64) {
    println!("[2] HEAD-TO-HEAD  A vs B  (the decisive comparison)");
    println!("    cell = A's result vs B; 2p: win rate (>0.5 = A wins); >2p: A's win-share");
    println!("    {:>9} {:>6} {:>12}", "config", "fair", "A vs B");
    println!("    {}", "-".repeat(31));

    for &(p, d, f) in STRENGTH_CONFIGS {
        let fair = 1.0 / p as f64;
        let s = config_seed(seed, p, d, f);
        let a = score_vs(p, d, f, &net_a.agent, &net_b.agent, games, s);
        println!(
            "    {:>9} {:>6.3} {:>12}",
            cfg_label(p, d, f),
            fair,
            mark(a, fair),
        );
    }
    println!();
}

/// Format a score with a `+`/`-` marker against the fair baseline, so a scan
/// down the column reads as "beats it / loses to it" at a glance.
fn mark(score: f64, fair: f64) -> String {
    let sign = if score > fair + 1e-9 {
        '+'
    } else if score < fair - 1e-9 {
        '-'
    } else {
        '='
    };
    format!("{score:.3} {sign}")
}

/// Section 3 — per-round exploitability (equilibrium quality). Builds the
/// first-round subgame (a true 2-player zero-sum game with `DiceShareValue`
/// continuation) and computes the exact best-response NashConv over each net's
/// policy; exploitability is `nashconv / 2`. Lower is closer to equilibrium.
fn exploitability_section(net_a: &Net, net_b: Option<&Net>) {
    println!("[3] PER-ROUND EXPLOITABILITY  (first-round subgame; lower = nearer equilibrium)");
    if net_b.is_some() {
        println!(
            "    {:>9} {:>14} {:>14}",
            "config", "A exploit", "B exploit"
        );
        println!("    {}", "-".repeat(39));
    } else {
        println!("    {:>9} {:>14}", "config", "A exploit");
        println!("    {}", "-".repeat(24));
    }

    for &(p, d, f) in EXPLOIT_CONFIGS {
        let expl_a = round_exploit(&net_a.agent, p, d, f);
        if let Some(b) = net_b {
            let expl_b = round_exploit(&b.agent, p, d, f);
            println!(
                "    {:>9} {:>14.5} {:>14.5}",
                cfg_label(p, d, f),
                expl_a,
                expl_b,
            );
        } else {
            println!("    {:>9} {:>14.5}", cfg_label(p, d, f), expl_a);
        }
    }
    println!();
}

/// `agent`'s net policy over a [`RoundSubgame`], memoized on infoset so each
/// distinct infoset costs one forward pass.
///
/// `net_policy` reads the policy head over the legal actions at a state — the
/// `Policy` an exact best response measures against. The policy is a pure
/// function of the (position-relative) infoset, while a best-response walk
/// queries it at every node of its passes and at states that share an infoset;
/// memoizing on `infoset_key` collapses those repeated forwards — the difference
/// between an instant config and a slow one once the bid lattice grows (e.g.
/// 2p3d4f), and it matters more for `profile_exploitability`'s 2n passes than
/// for `nash_conv`'s two. `RefCell` supplies the interior mutability the
/// `Policy` trait's `&self` signature needs.
struct MemoNetPolicy<'a> {
    net: &'a Mlp,
    cache: InferCache,
    cfg: &'a LiarsDice,
    memo: RefCell<HashMap<u64, Vec<f64>>>,
}

impl Policy<RoundSubgame<DiceShareValue>> for MemoNetPolicy<'_> {
    fn action_probs(
        &self,
        game: &RoundSubgame<DiceShareValue>,
        state: &LdState,
        player: usize,
    ) -> Vec<f64> {
        let key = game.infoset_key(state, player);
        if let Some(p) = self.memo.borrow().get(&key) {
            return p.clone();
        }
        let probs = net_policy(self.net, &self.cache, self.cfg, state, player);
        self.memo.borrow_mut().insert(key, probs.clone());
        probs
    }
}

/// Runs `measure` over the first-round subgame of `(p, d, f)` and `agent`'s
/// memoized net policy — the shared setup for both exploitability lenses. The
/// policy borrows the subgame's config, so the two cannot escape together;
/// `measure` consumes them in-scope and returns its result.
fn with_round_policy<T>(
    agent: &NetAgent,
    p: u8,
    d: u8,
    f: u8,
    measure: impl FnOnce(&RoundSubgame<DiceShareValue>, &MemoNetPolicy<'_>) -> T,
) -> T {
    let mut dice_left = [0u8; liars_dice::MAX_PLAYERS];
    for slot in dice_left.iter_mut().take(p as usize) {
        *slot = d;
    }
    let round = RoundSubgame::new(p, d, f, dice_left, 0, true, 1, DiceShareValue);
    let net = agent.net();
    let policy = MemoNetPolicy {
        net,
        cache: net.infer_cache(),
        cfg: round.config(),
        memo: RefCell::new(HashMap::new()),
    };
    measure(&round, &policy)
}

/// Exact best-response exploitability of `agent`'s policy on the first-round
/// subgame of `(p, d, f)`.
fn round_exploit(agent: &NetAgent, p: u8, d: u8, f: u8) -> f64 {
    with_round_policy(agent, p, d, f, |round, policy| {
        nash_conv(round, policy).2 / 2.0
    })
}

/// Section 3b — MULTIPLAYER EXPLOITABILITY (n>=3). The N-player game is
/// constant-sum but not zero-sum, so there is no Nash oracle; instead we report
/// the *profile* best-response exploitability: for each seat, the gain it would
/// realize by switching to its exact information-set-level best response while
/// every other seat (and chance) keeps playing the net. The mean over seats is a
/// symmetric measure of how far the whole profile is from a best-response
/// equilibrium (lower = less exploitable). Built on the first-round subgame with
/// a `DiceShareValue` continuation, exactly like the 2p lens; for 2p the mean
/// would equal the 2p exploitability above, so only n>=3 configs run here.
fn multiplayer_exploit_section(net_a: &Net, net_b: Option<&Net>) {
    println!(
        "[3b] MULTIPLAYER EXPLOITABILITY (n>=3)  (first-round subgame; mean per-seat \
         best-response gain; lower = nearer a best-response equilibrium)"
    );
    println!("     no Nash oracle exists for n>=3; this is the single-seat-BR profile measure.");
    if net_b.is_some() {
        println!("     {:>9} {:>14} {:>14}", "config", "A mean", "B mean");
        println!("     {}", "-".repeat(39));
    } else {
        println!("     {:>9} {:>14}", "config", "A mean");
        println!("     {}", "-".repeat(24));
    }

    for &(p, d, f) in MULTI_EXPLOIT_CONFIGS {
        let (a_gains, a_mean) = round_profile_exploit(&net_a.agent, p, d, f);
        if let Some(b) = net_b {
            let (b_gains, b_mean) = round_profile_exploit(&b.agent, p, d, f);
            println!(
                "     {:>9} {:>14.5} {:>14.5}",
                cfg_label(p, d, f),
                a_mean,
                b_mean,
            );
            println!("       A per-seat: {}", fmt_gains(&a_gains));
            println!("       B per-seat: {}", fmt_gains(&b_gains));
        } else {
            println!("     {:>9} {:>14.5}", cfg_label(p, d, f), a_mean);
            println!("       A per-seat: {}", fmt_gains(&a_gains));
        }
    }
    println!();
}

/// Per-seat best-response gains and their mean for `agent`'s policy on the
/// first-round subgame of `(p, d, f)`.
fn round_profile_exploit(agent: &NetAgent, p: u8, d: u8, f: u8) -> (Vec<f64>, f64) {
    with_round_policy(agent, p, d, f, |round, policy| {
        profile_exploitability(round, policy)
    })
}

/// A compact `[s0, s1, ...]` rendering of the per-seat gains.
fn fmt_gains(gains: &[f64]) -> String {
    let parts: Vec<String> = gains.iter().map(|g| format!("{g:.5}")).collect();
    format!("[{}]", parts.join(", "))
}

/// Section 4 — thin-bluff endgame probe. Builds 2p1d6f endgames where the net
/// (seat 1) holds one low die and the opponent (seat 0) has opened a single-die
/// bid of varying believability, and prints the net's action distribution,
/// spotlighting `P(CallLiar)`. A skeptical net should call a thin bid (one that
/// needs a die it does not hold) far more readily than a believable one.
fn thin_bluff_section(net_a: &Net, net_b: Option<&Net>) {
    println!("[4] THIN-BLUFF ENDGAME PROBE  (2p1d6f; net = seat 1, holds one die)");
    println!("    net should distrust a bid it cannot back from its own hand.");
    // (my die face, opponent's opened bid face). The opponent claims `1 x face`.
    // - Believable: the bid is on the net's own die face (the bid could be true
    //   from the net's hand alone), so calling liar is a mistake.
    // - Thin: the bid is on a face the net does not hold, so the whole claim
    //   rests on the opponent's single hidden die — a coin-flip the net should
    //   often call.
    let scenarios = [
        (2u8, 2u8, "believable (bid is the net's own die)"),
        (2u8, 5u8, "thin (net holds none of the bid face)"),
        (2u8, 6u8, "thin (net holds none of the bid face)"),
    ];

    for (my_face, bid_face, note) in scenarios {
        println!("\n    scenario: I hold a {my_face}; opponent opened 1x{bid_face}  [{note}]");
        report_thin_bluff(&net_a.agent, "A", my_face, bid_face);
        if let Some(b) = net_b {
            report_thin_bluff(&b.agent, "B", my_face, bid_face);
        }
    }
    println!();
}

/// Print one net's action distribution at the thin-bluff scenario, with
/// `P(CallLiar)` called out. The state is built only through the public game
/// API: a free-open round (where a single `Open(q, f)` can set any bid) lets the
/// opponent open the exact thin bid in one legal move, leaving the net to act.
fn report_thin_bluff(agent: &NetAgent, label: &str, my_face: u8, bid_face: u8) {
    // A free-open (not first-round) 2p1d6f round: both seats hold one die, seat 0
    // opens. A free open admits any `Open(q, f)`, so the opponent can place the
    // exact `1 x bid_face` directly instead of climbing +1 raises (which would
    // alternate seats and force the net to commit to intermediate bids).
    let round = RoundSubgame::new(
        2,
        1,
        6,
        [1, 1, 0, 0, 0, 0, 0, 0],
        0,
        false,
        4,
        DiceShareValue,
    );
    let game = round.config();
    let mut s = round.initial_state();

    // Drive the two chance rolls with controlled hands: seat 0 (opponent) shows a
    // 1, so its `1 x bid_face` claim on a higher face is a pure bluff; seat 1
    // (the net) shows `my_face`.
    let hands = [hand_with(1), hand_with(my_face)];
    let mut rolled = 0usize;
    while let game_core::Turn::Chance = round.turn(&s) {
        round.apply(&mut s, Action::Roll(hands[rolled]));
        rolled += 1;
    }

    // The opener (seat 0) places the thin single-die bid; the net (seat 1) acts.
    round.apply(&mut s, Action::Open(1, bid_face));
    debug_assert_eq!(s.current_bid(), (1, bid_face));
    debug_assert_eq!(round.turn(&s), game_core::Turn::Player(1), "net is to act");

    let net = agent.net();
    let cache = net.infer_cache();
    let probs = net_policy(net, &cache, game, &s, 1);
    let acts = game.legal_actions(&s);

    let mut call_liar = 0.0;
    let mut parts = Vec::new();
    for (a, pr) in acts.iter().zip(&probs) {
        parts.push(format!("{}={:.3}", game.action_label(*a), pr));
        if matches!(a, Action::CallLiar) {
            call_liar = *pr;
        }
    }
    println!(
        "      net {label}: P(call LIAR)={call_liar:.3}   [{}]",
        parts.join(", ")
    );
}

/// A one-die hand histogram with the single die showing `face` (1-based).
fn hand_with(face: u8) -> [u8; liars_dice::MAX_FACES] {
    let mut h = [0u8; liars_dice::MAX_FACES];
    h[face as usize - 1] = 1;
    h
}
