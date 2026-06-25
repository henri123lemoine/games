//! Decisive diagnostic for the online subgame-solving agent: is the bottleneck
//! the SOLVE BUDGET (too few MCCFR iters per round under the work ceiling) or the
//! CONTINUATION VALUE (the leaf heuristic / net)?
//!
//! TEST 1 — BUDGET SWEEP. With the DiceShareValue continuation, raise the
//! per-round budget past the deploy work-ceiling (via `flat_iters`) and measure
//! win rate vs the rollout bot as the budget grows. If win rate climbs toward /
//! past the fair line, the budget is the lever.
//!
//! TEST 2 — VALUE-HEAD QUALITY. MAE of (NetValue snapshot vs exact lattice) and
//! (DiceShareValue vs exact lattice) on the same 2p1d6f / 2p2d6f post-round
//! states. Whichever is closer is the better continuation.
//!
//! ```text
//! cargo run --release -p liars-dice --features parallel --example budget_diag -- \
//!     games=120 rollouts=200 net=runs/ld_value_snap.bin
//! ```
//!
//! Knobs (all optional): `tests=1,big,2` selects sections; `games`/`rollouts`
//! size the strength matches; `big_games=N` runs the slow 5p5d6f match (0 =
//! ms/move only); `dice=1,2` and `oracle_iters`/`oracle_tol` tune TEST 2's exact
//! lattice (the 2d6 fixed point is costly, so it is tunable). Example: re-run
//! only the 2d6 value check — `tests=2 dice=2 oracle_iters=600 oracle_tol=1e-4`.

use std::collections::HashMap;
use std::time::Instant;

use game_core::{Agent, Game, Rng, hash, play_n, win_share};
use liars_dice::train::value_head_lattice_mae;
use liars_dice::{
    BidConditioned, ContinuationValue, DiceShareValue, FitConfig, LiarsDice, OnlineSolveAgent,
    OnlineSolveConfig, ProbabilisticAgent, fit_two_player,
};
use rayon::prelude::*;
use solvers::Rollout;
use solvers::azero::Mlp;

fn deployed_bot(rollouts: u32) -> Rollout<LiarsDice, ProbabilisticAgent, BidConditioned> {
    Rollout::new(
        rollouts,
        ProbabilisticAgent::default_agent(),
        BidConditioned::default(),
    )
}

fn parse<T: std::str::FromStr>(args: &HashMap<String, String>, key: &str, default: T) -> T {
    args.get(key)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn flat_agent(
    flat_iters: u64,
    restarts: usize,
    seed: u64,
) -> OnlineSolveAgent<DiceShareValue, fn() -> DiceShareValue> {
    OnlineSolveAgent::with_config(
        || DiceShareValue,
        OnlineSolveConfig {
            iters: flat_iters,
            max_iters: flat_iters,
            restarts,
            seed,
            flat_iters: Some(flat_iters),
        },
    )
}

/// Hero (online agent) score vs a field of the rollout bot in `(p, d, f)`: 2p
/// seat-swapped win rate; >2p hero rotated through every seat (win share).
/// Mirrors `online_eval::score_vs`.
fn score_vs<A: Agent<LiarsDice> + Sync>(
    p: u8,
    d: u8,
    f: u8,
    a: &A,
    rollouts: u32,
    games: u32,
    seed: u64,
) -> f64 {
    let game = LiarsDice::new(p, d, f);
    let n = game.num_players();
    let bot = deployed_bot(rollouts);
    let a: &(dyn Agent<LiarsDice> + Sync) = a;
    let b: &(dyn Agent<LiarsDice> + Sync) = &bot;
    let total: f64 = (0..games)
        .into_par_iter()
        .map(|g| {
            let mut rng = Rng::new(hash::combine(seed, u64::from(g)));
            if n == 2 {
                let (s0, s1) = if g % 2 == 0 { (a, b) } else { (b, a) };
                let agents: [&dyn Agent<LiarsDice>; 2] = [s0, s1];
                let terminal = play_n(&game, &agents, &mut rng);
                win_share(&game, &terminal, if g % 2 == 0 { 0 } else { 1 })
            } else {
                let hero = (g as usize) % n;
                let seats: Vec<&dyn Agent<LiarsDice>> =
                    (0..n).map(|q| if q == hero { a } else { b } as _).collect();
                let terminal = play_n(&game, &seats, &mut rng);
                win_share(&game, &terminal, hero)
            }
        })
        .sum();
    total / games as f64
}

/// One representative mid-round timed move at the given config / budget: median
/// ms/move over a few solves (so the table reports cost, not just win rate).
fn ms_per_move(p: u8, d: u8, f: u8, flat_iters: u64, restarts: usize, seed: u64) -> f64 {
    let game = LiarsDice::new(p, d, f);
    let mut dice = [0u8; liars_dice::MAX_PLAYERS];
    for slot in dice.iter_mut().take(p as usize) {
        *slot = d;
    }
    if d > 1 {
        dice[0] -= 1;
    }
    // Build a live mid-round decision: free open, all hands rolled, one opening
    // bid placed so the next seat is on a genuine raise/call.
    use game_core::Turn;
    use liars_dice::{Action, RoundSubgame};
    let round = RoundSubgame::new(p, d, f, dice, 0, false, 4, DiceShareValue);
    let mut s = round.initial_state();
    let mut rng = Rng::new(0x5EED);
    while let Turn::Chance = round.turn(&s) {
        let a = round.sample_chance(&s, &mut rng).0;
        round.apply(&mut s, a);
    }
    round.apply(&mut s, Action::Open(2, 3));
    let agent = flat_agent(flat_iters, restarts, seed);
    let mut samples = Vec::new();
    let mut rng = Rng::new(seed ^ 0x99);
    for _ in 0..5 {
        let t = Instant::now();
        let _ = agent.act(&game, &s, s.turn(), &mut rng);
        samples.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn test1_budget_sweep(games: u32, rollouts: u32, seed: u64) {
    println!("================================================================");
    println!("TEST 1 — BUDGET SWEEP (DiceShareValue continuation, vs rollout bot)");
    println!("  raising per-round iters past the total_dice^2 work ceiling.");
    println!("  effective per-round iters = flat_iters * restarts (restarts=3).");
    println!("  fair line: 0.5 (2p), 0.333 (3p). games/cell={games} rollouts={rollouts}\n");

    // flat_iters per restart; effective per-round = flat * 3 restarts.
    let restarts = 3usize;
    let budgets: &[(u64, &str)] = &[
        (2_667, "~8000  (current deploy)"),
        (10_000, "~30000"),
        (33_333, "~100000"),
    ];

    println!(
        "  {:>9} {:>26} {:>10} {:>10}",
        "config", "eff per-round iters", "win rate", "ms/move"
    );
    println!("  {}", "-".repeat(60));
    for &(p, d, f) in &[(2u8, 5u8, 6u8), (3, 3, 6)] {
        let fair = 1.0 / p as f64;
        for &(flat, label) in budgets {
            let cs = seed ^ ((u64::from(p) << 16) | (u64::from(d) << 8) | u64::from(f));
            let agent = flat_agent(flat, restarts, cs);
            let t = Instant::now();
            let wr = score_vs(p, d, f, &agent, rollouts, games, cs);
            let elapsed = t.elapsed().as_secs_f64();
            let ms = ms_per_move(p, d, f, flat, restarts, cs);
            let eff = flat * restarts as u64;
            let mark = if wr > fair + 1e-9 { '+' } else { '-' };
            println!(
                "  {:>9} {:>26} {:>9.3}{} {:>9.1}   [{label}, fair {fair:.3}, run {elapsed:.0}s]",
                format!("{p}p{d}d{f}f"),
                eff,
                wr,
                mark,
                ms,
            );
        }
        println!();
    }
}

fn test1_big_config(big_games: u32, rollouts: u32, seed: u64) {
    println!("  --- 5p5d6f feasibility (ms/move at deploy vs highest budget) ---");
    let (p, d, f) = (5u8, 5u8, 6u8);
    let restarts = 3usize;
    let fair = 1.0 / p as f64;
    let cs = seed ^ ((u64::from(p) << 16) | (u64::from(d) << 8) | u64::from(f));
    // Time one move at the current deploy-equivalent (~8k) and the highest (100k)
    // budgets, so the cost of the big round is clear regardless of match length.
    for &flat in &[2_667u64, 33_333u64] {
        let ms = ms_per_move(p, d, f, flat, restarts, cs);
        println!(
            "  {:>9} eff iters {:>7}  ms/move {ms:>9.0}",
            format!("{p}p{d}d{f}f"),
            flat * restarts as u64,
        );
    }
    // Win share at the highest budget over a small match (big_games), to confirm
    // direction. 0 disables (the match is very slow at this round size).
    if big_games > 0 {
        let flat = 33_333u64;
        let agent = flat_agent(flat, restarts, cs);
        let t = Instant::now();
        let wr = score_vs(p, d, f, &agent, rollouts, big_games, cs);
        let elapsed = t.elapsed().as_secs_f64();
        let mark = if wr > fair + 1e-9 { '+' } else { '-' };
        println!(
            "  {:>9} eff iters {:>7}  win share {:.3}{} (fair {fair:.3})  [run {elapsed:.0}s, {big_games} games]",
            format!("{p}p{d}d{f}f"),
            flat * restarts as u64,
            wr,
            mark,
        );
    }
    println!();
}

fn test2_value_quality(net_path: &str, dice_list: &[u8], oracle_iters: u64, oracle_tol: f64) {
    println!("================================================================");
    println!("TEST 2 — VALUE-HEAD QUALITY: continuation MAE vs the EXACT lattice");
    println!("  states: every reachable 2p continuing (a,b,opener) for the config.");
    println!("  smaller MAE = closer to exact = better continuation.");
    println!("  oracle = fit_two_player(iters_per_solve={oracle_iters}, tol={oracle_tol:e}).\n");

    let net = match std::fs::read(net_path).and_then(|b| Mlp::from_bytes(&b)) {
        Ok(n) => Some(n),
        Err(e) => {
            println!("  could not load net '{net_path}': {e} — reporting heuristic only.\n");
            None
        }
    };

    println!(
        "  {:>9} {:>8} {:>18} {:>18} {:>10}",
        "config", "states", "DiceShare MAE", "NetValue MAE", "closer"
    );
    println!("  {}", "-".repeat(70));
    let f = 6u8;
    for &d in dice_list {
        let t = Instant::now();
        // Exact 2p continuation lattice oracle (fitted value iteration to a fixed
        // point). iters/tol are tunable so the costly 2d6 config stays runnable.
        let fit = fit_two_player(
            d,
            f,
            FitConfig {
                iters_per_solve: oracle_iters,
                tol: oracle_tol,
                max_sweeps: 200,
                measure_exploitability: false,
            },
        );
        let lattice = &fit.lattice;

        // DiceShareValue MAE over the same states (seat-0 valued, matching the
        // lattice scalar).
        let mut sum = 0.0;
        let mut n = 0u32;
        for a in 1..=d {
            for b in 1..=d {
                for opener in 0..2usize {
                    let h = DiceShareValue.value(f, &[a, b], opener, 0);
                    let e = lattice.get_two_player(&[a, b], opener).unwrap();
                    sum += (h - e).abs();
                    n += 1;
                }
            }
        }
        let heur_mae = sum / n.max(1) as f64;

        let net_mae = net
            .as_ref()
            .map(|net| value_head_lattice_mae(net, d, f, lattice));

        let (net_str, closer) = match net_mae {
            Some(m) => {
                let c = if m < heur_mae {
                    "NetValue"
                } else {
                    "DiceShare"
                };
                (format!("{m:.4}"), c)
            }
            None => ("n/a".to_string(), "DiceShare"),
        };
        println!(
            "  {:>9} {:>8} {:>18.4} {:>18} {:>10}   [oracle {} sweeps, last d {:.1e}, {:.0}s]",
            format!("2p{d}d{f}f"),
            n,
            heur_mae,
            net_str,
            closer,
            fit.sweep_deltas.len(),
            fit.sweep_deltas.last().copied().unwrap_or(f64::NAN),
            t.elapsed().as_secs_f64(),
        );
        flush();
    }
    println!();
}

fn main() {
    let args: HashMap<String, String> = std::env::args()
        .skip(1)
        .filter_map(|a| a.split_once('=').map(|(k, v)| (k.into(), v.into())))
        .collect();
    let games: u32 = parse(&args, "games", 120);
    let rollouts: u32 = parse(&args, "rollouts", 200);
    let seed: u64 = parse(&args, "seed", 1);
    // Games for the slow 5p5d6f feasibility match; 0 = ms/move only (the match is
    // very slow at this round size).
    let big_games: u32 = parse(&args, "big_games", 0);
    // Which sections to run (skip the already-done budget sweep on re-runs).
    let tests: String = args
        .get("tests")
        .cloned()
        .unwrap_or_else(|| "1,big,2".to_string());
    let net_path = args
        .get("net")
        .cloned()
        .unwrap_or_else(|| "runs/ld_value_snap.bin".to_string());
    // TEST 2 oracle controls. The 2d6 infinite-horizon fixed point is costly; the
    // dice list and per-solve iters/tol are tunable so it stays runnable.
    let dice_list: Vec<u8> = args
        .get("dice")
        .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 2]);
    let oracle_iters: u64 = parse(&args, "oracle_iters", 2_000);
    let oracle_tol: f64 = parse(&args, "oracle_tol", 1e-6);

    let t0 = Instant::now();
    if tests.contains('1') {
        test1_budget_sweep(games, rollouts, seed);
        flush();
    }
    if tests.contains("big") {
        test1_big_config(big_games, rollouts, seed);
        flush();
    }
    if tests.contains('2') {
        test2_value_quality(&net_path, &dice_list, oracle_iters, oracle_tol);
    }
    println!("done in {:.0}s", t0.elapsed().as_secs_f64());
}

fn flush() {
    use std::io::Write as _;
    let _ = std::io::stdout().flush();
}
