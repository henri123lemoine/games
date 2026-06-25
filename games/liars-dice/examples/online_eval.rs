//! Validation harness for the online subgame-solving agent
//! ([`liars_dice::OnlineSolveAgent`]).
//!
//! Three lenses, mirroring the build-and-validate brief:
//!
//! 1. **Correctness** — replicate `probe_check`'s 2p1d6f thin-bid scenarios
//!    through the online agent and show its `P(CallLiar)` matches the *exact*
//!    CFR equilibrium (≈0.42 on a thin `1×5`, ≈0 when the hero holds the bid
//!    face). This is the whole point: online solving fixes the raw net's
//!    over-calling.
//! 2. **Speed** — wall-clock ms/move for solving rounds of increasing size at
//!    the deploy budget, from 2p2d6f up to 6p8d6f, at representative mid-round
//!    dice vectors. Flags anything over the ~1s/move target.
//! 3. **Strength** — head-to-head vs the deployed determinized-rollout bot
//!    across a config spread, hero rotated through every seat; win rate (2p) /
//!    win share (>2p) against the fair `1/players` baseline.
//!
//! ```text
//! cargo run --release -p liars-dice --features parallel --example online_eval -- \
//!     iters=8000 restarts=3 games=200 rollouts=200 seed=1
//! ```
//!
//! All knobs are optional; defaults match the recommended deploy config. Build
//! with `--features parallel` so the match-play and inner rollouts use all cores.

use std::collections::HashMap;
use std::time::Instant;

use game_core::{Agent, Game, Rng, Turn, hash, play_n, win_share};
use liars_dice::{
    Action, BidConditioned, DiceShareValue, FitConfig, LdState, LiarsDice, MAX_FACES,
    OnlineSolveAgent, OnlineSolveConfig, ProbabilisticAgent, RoundSubgame, fit_two_player,
};
use rayon::prelude::*;
use solvers::{Cfr, Rollout};

/// The deployed bot, exactly as `lab`'s registry builds it for `bot=rollout`.
fn deployed_bot(rollouts: u32) -> Rollout<LiarsDice, ProbabilisticAgent, BidConditioned> {
    Rollout::new(
        rollouts,
        ProbabilisticAgent::default_agent(),
        BidConditioned::default(),
    )
}

fn online_agent(
    iters: u64,
    restarts: usize,
    seed: u64,
) -> OnlineSolveAgent<DiceShareValue, fn() -> DiceShareValue> {
    OnlineSolveAgent::with_config(
        || DiceShareValue,
        OnlineSolveConfig {
            iters,
            max_iters: iters,
            restarts,
            seed,
        },
    )
}

fn hand_with(face: u8) -> [u8; MAX_FACES] {
    let mut h = [0u8; MAX_FACES];
    h[face as usize - 1] = 1;
    h
}

fn cfg_label(p: u8, d: u8, f: u8) -> String {
    format!("{p}p{d}d{f}f")
}

fn call_liar_prob(probs: &[f64], acts: &[Action]) -> f64 {
    acts.iter()
        .zip(probs)
        .find(|(a, _)| matches!(a, Action::CallLiar))
        .map(|(_, &p)| p)
        .unwrap_or(0.0)
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

fn parse<T: std::str::FromStr>(args: &HashMap<String, String>, key: &str, default: T) -> T {
    args.get(key)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let args = parse_args();
    let iters: u64 = parse(&args, "iters", 8_000);
    let restarts: usize = parse(&args, "restarts", 3);
    let games: u32 = parse(&args, "games", 200);
    let rollouts: u32 = parse(&args, "rollouts", 200);
    let seed: u64 = parse(&args, "seed", 1);

    let t0 = Instant::now();
    println!("Liar's Dice — ONLINE SUBGAME-SOLVING agent validation");
    println!(
        "  solver = Mccfr (external-sampling MCCFR+), continuation = DiceShareValue\n  \
         iters/restart={iters}  restarts={restarts}  games/config={games}  \
         rollouts={rollouts}  seed={seed}\n"
    );

    correctness_section(iters, restarts, seed);
    speed_section(iters, restarts, seed);
    strength_section(iters, restarts, games, rollouts, seed);

    println!("\nfinished in {:.1?}", t0.elapsed());
}

// ----------------------------------------------------------------------------
// [1] CORRECTNESS — the online agent vs the exact 2p1d6f equilibrium.
// ----------------------------------------------------------------------------

/// Build the free-open 2p1d6f state where seat 0 holds a 1 and has opened
/// `1 × bid_face`, seat 1 holds `my_face`, and it is seat 1's turn — exactly
/// `probe_check`'s `probe_state`.
fn probe_state(my_face: u8, bid_face: u8) -> LdState {
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
    let mut s = round.initial_state();
    let hands = [hand_with(1), hand_with(my_face)];
    let mut rolled = 0;
    while let Turn::Chance = round.turn(&s) {
        round.apply(&mut s, Action::Roll(hands[rolled]));
        rolled += 1;
    }
    round.apply(&mut s, Action::Open(1, bid_face));
    s
}

fn correctness_section(iters: u64, restarts: usize, seed: u64) {
    println!("[1] CORRECTNESS — online P(call) vs the EXACT 2p1d6f equilibrium (2p1d6f)");
    println!("    online solving should fix the raw net's over-calling: ~0.42 on a thin 1x5,");
    println!("    ~0.00 when the hero holds the bid face (the bid is then guaranteed true).");

    // Exact equilibrium reference: solve the 2p1d6f free-open round with the
    // converged lattice continuation, identical to probe_check.
    let fit = fit_two_player(1, 6, FitConfig::default());
    let round = RoundSubgame::new(2, 1, 6, [1, 1, 0, 0, 0, 0, 0, 0], 0, false, 4, fit.lattice);
    let mut cfr = Cfr::new(round);
    cfr.solve(200_000);
    println!(
        "    exact-CFR exploitability (sanity): {:.5}\n",
        cfr.exploitability().2 / 2.0
    );

    let agent = online_agent(iters, restarts, seed);
    let game = LiarsDice::new(2, 1, 6);
    println!(
        "    {:<34} {:>12} {:>12} {:>8}",
        "scenario", "exact P(call)", "online P(call)", "match"
    );
    println!("    {}", "-".repeat(70));
    for (my_face, bid_face, note) in [
        (2u8, 2u8, "believable (bid = own die)"),
        (2, 5, "thin (hold none of bid face)"),
        (2, 6, "thin (hold none of bid face)"),
        (5, 5, "I HOLD the bid face (true bid)"),
    ] {
        let s = probe_state(my_face, bid_face);
        let acts = game.legal_actions(&s);
        let exact = call_liar_prob(&cfr.policy(&s, 1), &acts);
        // Read the averaged policy directly (a fixed nonce) rather than sampling,
        // so the reported call frequency is exact.
        let online = call_liar_prob(&agent.solve_policy(&game, &s, 1, 0xC0FFEE), &acts);
        let ok = (exact - online).abs() < 0.08;
        println!(
            "    hold {my_face}, face 1x{bid_face} {:<18} {:>12.3} {:>12.3} {:>8}",
            format!("[{note}]"),
            exact,
            online,
            if ok { "yes" } else { "NO" },
        );
    }
    println!();
}

// ----------------------------------------------------------------------------
// [2] SPEED — ms/move for solving rounds of increasing size.
// ----------------------------------------------------------------------------

/// A representative mid-round dice vector for `(players, dice)`: every live seat
/// near full dice, the opener having lost one — a realistic decision point.
fn mid_round_dice(players: u8, dice: u8) -> [u8; liars_dice::MAX_PLAYERS] {
    let mut v = [0u8; liars_dice::MAX_PLAYERS];
    for slot in v.iter_mut().take(players as usize) {
        *slot = dice;
    }
    if dice > 1 {
        v[0] -= 1; // the opener lost a die last round.
    }
    v
}

/// Build a live mid-round decision state: a free-open round with the given dice
/// vector, all hands rolled, the opener having placed one opening bid so the
/// *second* seat is now on a genuine raise/call decision.
fn mid_round_state(game: &LiarsDice, dice_left: [u8; liars_dice::MAX_PLAYERS]) -> LdState {
    let round = RoundSubgame::new(
        game.players,
        game.dice,
        game.faces,
        dice_left,
        0,
        false,
        4,
        DiceShareValue,
    );
    let mut s = round.initial_state();
    let mut rng = Rng::new(0x5EED);
    while let Turn::Chance = round.turn(&s) {
        let a = round.sample_chance(&s, &mut rng).0;
        round.apply(&mut s, a);
    }
    // The opener places a modest opening bid; control returns to the next seat.
    round.apply(&mut s, Action::Open(2, 3));
    s
}

fn speed_section(iters: u64, restarts: usize, seed: u64) {
    println!("[2] SPEED — ms/move at the deploy budget (representative mid-round states)");
    println!("    target: under ~1000 ms/move. iters is the per-restart BASE budget (x{restarts}");
    println!("    restarts); the agent scales it down for large rounds (reported per config).");
    println!(
        "    {:>9} {:>12} {:>12} {:>14}",
        "config", "total dice", "ms/move", "status"
    );
    println!("    {}", "-".repeat(50));

    let configs: &[(u8, u8, u8)] = &[
        (2, 2, 6),
        (2, 5, 6),
        (3, 3, 6),
        (4, 4, 6),
        (5, 5, 6),
        (6, 8, 6),
    ];
    let agent = online_agent(iters, restarts, seed);
    for &(p, d, f) in configs {
        let game = LiarsDice::new(p, d, f);
        let dice_left = mid_round_dice(p, d);
        let s = mid_round_state(&game, dice_left);
        let total: u32 = dice_left.iter().map(|&x| u32::from(x)).sum();
        // A few timed moves; report the median to shrug off scheduler jitter.
        let mut samples = Vec::new();
        let mut rng = Rng::new(seed ^ (u64::from(p) << 8));
        for _ in 0..5 {
            let t = Instant::now();
            let _ = agent.act(&game, &s, s.turn(), &mut rng);
            samples.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ms = samples[samples.len() / 2];
        let status = if ms <= 1000.0 { "ok" } else { "OVER 1s" };
        println!(
            "    {:>9} {:>12} {:>12.1} {:>14}",
            cfg_label(p, d, f),
            total,
            ms,
            status,
        );
    }
    println!();
}

// ----------------------------------------------------------------------------
// [3] STRENGTH — online agent vs the deployed rollout bot.
// ----------------------------------------------------------------------------

/// Configs spanning small/medium/large, per the brief.
const STRENGTH_CONFIGS: &[(u8, u8, u8)] = &[
    (2, 2, 6),
    (2, 5, 6),
    (3, 3, 6),
    (4, 4, 6),
    (5, 5, 6),
    (6, 3, 6),
];

/// Per-config seed so the configs are independent draws.
fn config_seed(seed: u64, p: u8, d: u8, f: u8) -> u64 {
    seed ^ ((u64::from(p) << 16) | (u64::from(d) << 8) | u64::from(f))
}

/// A's score vs B in `(p, d, f)`: 2p seat-swapped win rate; >2p the hero (A)
/// rotated through every seat against a field of B. Mirrors `ld_eval`'s
/// `score_vs` — parallel, deterministic per `(seed, game)`, seat-balanced.
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
                let (s0, s1) = if g % 2 == 0 { (a, b) } else { (b, a) };
                let agents: [&dyn Agent<LiarsDice>; 2] = [s0, s1];
                let terminal = play_n(&game, &agents, &mut rng);
                let a_seat = if g % 2 == 0 { 0 } else { 1 };
                win_share(&game, &terminal, a_seat)
            } else {
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

fn strength_section(iters: u64, restarts: usize, games: u32, rollouts: u32, seed: u64) {
    println!("[3] STRENGTH — online agent vs the deployed rollout bot (rollouts={rollouts})");
    println!("    cell = online score vs the bot; 2p: win rate (>0.5 beats it); >2p: win-share");
    println!("    {:>9} {:>6} {:>12}", "config", "fair", "online vs bot");
    println!("    {}", "-".repeat(31));

    for &(p, d, f) in STRENGTH_CONFIGS {
        let s = config_seed(seed, p, d, f);
        let agent = online_agent(iters, restarts, s);
        let bot = deployed_bot(rollouts);
        let fair = 1.0 / p as f64;
        let score = score_vs(p, d, f, &agent, &bot, games, s);
        println!(
            "    {:>9} {:>6.3} {:>12}",
            cfg_label(p, d, f),
            fair,
            mark(score, fair),
        );
    }
    println!();
}
