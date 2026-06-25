//! Net-guided search vs the rollout baseline at high dice count.
//!
//! The deployed `bot=rollout` baseline wins because it SEARCHES: determinized
//! rollouts with a hand-tuned base policy ([`ProbabilisticAgent`]). A bare net
//! ([`NetAgent`]) only forward-passes. This harness gives the net the *same*
//! search — the identical `Rollout` structure with the net as the base policy —
//! and asks whether net-guided search beats heuristic-guided search at the
//! configs that matter (5p5d6f and friends).
//!
//! It prints, per config, four rows of hero-rotated-through-every-seat win-share
//! vs the fair `1/players`:
//!   (a) net forward-pass        vs heuristic (no search either side) — the GATE:
//!       is the net a better *base policy* than the heuristic yet?
//!   (b) Rollout(net)            vs Rollout(heuristic) [the baseline] — does
//!       net-guided rollout beat the baseline?
//!   (c) Rollout(net)+v-trunc    vs the baseline — net policy + net value head.
//!   (d) Rollout(heuristic)      vs Rollout(heuristic) — sanity: peers ≈ fair.
//!
//! ```text
//! cargo run --release -p liars-dice --example net_search_eval -- \
//!     net=runs/ld_forward/ckpt.bin games=60 rollouts=80 plies=3 threads=4 seed=1
//! ```
//!
//! Budget is kept modest by default so it does not starve a concurrent training
//! run; raise `games`/`rollouts` for a tighter estimate.

use std::collections::HashMap;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use game_core::{Agent, Game, Rng, hash, play_n, win_share};
use liars_dice::{BidConditioned, LiarsDice, NetAgent, NetTruncRollout, ProbabilisticAgent};
use rayon::prelude::*;
use solvers::Rollout;
use solvers::azero::Mlp;

/// The deployed bot, exactly as `lab`'s registry builds it for `bot=rollout`.
fn deployed_bot(rollouts: u32) -> Rollout<LiarsDice, ProbabilisticAgent, BidConditioned> {
    Rollout::new(
        rollouts,
        ProbabilisticAgent::default_agent(),
        BidConditioned::default(),
    )
}

/// Net-guided rollout: the SAME baseline structure, net as the base policy.
fn net_rollout(net: Mlp, rollouts: u32) -> Rollout<LiarsDice, NetAgent, BidConditioned> {
    Rollout::new(rollouts, NetAgent::new(net), BidConditioned::default())
}

/// `hero`'s win-share in a field of `field` at `(p, d, f)`, hero rotated through
/// every seat (the repo's win-share convention). Fair share = `1/p`. Games run
/// in parallel; any per-decision rollout fan-out nests under that.
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

fn parse_args() -> HashMap<String, String> {
    std::env::args()
        .skip(1)
        .filter_map(|arg| {
            arg.split_once('=')
                .map(|(k, v)| (k.to_string(), v.to_string()))
        })
        .collect()
}

fn get<T: std::str::FromStr>(args: &HashMap<String, String>, key: &str, default: T) -> T {
    args.get(key)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Parse `configs=5x5x6:3x3x6` into `(p, d, f)` triples; default when absent.
fn parse_configs(args: &HashMap<String, String>) -> Vec<(u8, u8, u8)> {
    match args.get("configs") {
        None => vec![(5, 5, 6), (3, 3, 6)],
        Some(s) => s
            .split(':')
            .filter_map(|c| {
                let mut it = c.split('x').filter_map(|x| x.parse().ok());
                Some((it.next()?, it.next()?, it.next()?))
            })
            .collect(),
    }
}

/// Load the net bytes from `net=`, falling back ckpt.bin -> best.bin in the same
/// directory if the given path is absent, so a live training run is picked up
/// whether the current checkpoint or the kept-best is meant.
fn load_net_bytes(path: &str) -> Result<(String, Vec<u8>), String> {
    if Path::new(path).exists() {
        return std::fs::read(path)
            .map(|b| (path.to_string(), b))
            .map_err(|e| format!("{path}: {e}"));
    }
    let alt = if path.ends_with("ckpt.bin") {
        path.replace("ckpt.bin", "best.bin")
    } else {
        path.replace("best.bin", "ckpt.bin")
    };
    std::fs::read(&alt)
        .map(|b| (alt.clone(), b))
        .map_err(|_| format!("neither {path} nor {alt} is readable"))
}

fn mark(share: f64, fair: f64) -> &'static str {
    if share > fair + 1e-9 {
        "+ beats"
    } else if share < fair - 1e-9 {
        "- loses"
    } else {
        "= even"
    }
}

fn main() -> ExitCode {
    let args = parse_args();
    let net_path = args
        .get("net")
        .cloned()
        .unwrap_or_else(|| "runs/ld_forward/ckpt.bin".to_string());
    let games: u32 = get(&args, "games", 60);
    let rollouts: u32 = get(&args, "rollouts", 80);
    let plies: u32 = get(&args, "plies", 3);
    let threads: usize = get(&args, "threads", 4);
    let seed: u64 = get(&args, "seed", 0xD1CE);
    let configs = parse_configs(&args);

    let (loaded_path, bytes) = match load_net_bytes(&net_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("could not load net: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = Mlp::from_bytes(&bytes) {
        eprintln!("net {loaded_path} did not parse as an Mlp: {e}");
        return ExitCode::FAILURE;
    }

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build_global()
        .ok();

    let t0 = Instant::now();
    println!("Net-guided search vs the rollout baseline");
    println!("  net      = {loaded_path}");
    println!(
        "  games    = {games}/config   rollouts = {rollouts}   plies = {plies}   \
         threads = {threads}   seed = {seed}"
    );
    println!(
        "  rows: (a) net-fp vs heuristic  (b) Rollout(net) vs baseline  \
              (c) Rollout(net)+vtrunc vs baseline  (d) baseline vs baseline\n"
    );

    // Collected for the closing verdict: per-config (a) and (b) shares.
    let mut gate_5p: Option<(f64, f64)> = None; // (row a, row b) at the primary config

    for &(p, d, f) in &configs {
        let fair = 1.0 / p as f64;
        println!("config {p}p{d}d{f}f   (fair = {fair:.3})");
        println!("  {:<48} {:>10}  vs fair", "row", "win-share");
        println!("  {}", "-".repeat(72));

        // Independent per-row seeds so the four rows are uncorrelated draws.
        let s = |tag: u64| {
            hash::combine(
                seed ^ tag,
                (u64::from(p) << 16) | (u64::from(d) << 8) | u64::from(f),
            )
        };

        // (a) net forward-pass vs heuristic — no search either side. The GATE.
        let net_fp = NetAgent::new(Mlp::from_bytes(&bytes).unwrap());
        let heur = ProbabilisticAgent::default_agent();
        let a = field_win_share(p, d, f, &net_fp, &heur, games, s(0xA));
        println!(
            "  {:<48} {:>10.3}  {}",
            "(a) net forward-pass vs heuristic (no search)",
            a,
            mark(a, fair)
        );

        // (b) Rollout(net) vs Rollout(heuristic) = the baseline.
        let r_net = net_rollout(Mlp::from_bytes(&bytes).unwrap(), rollouts);
        let baseline = deployed_bot(rollouts);
        let b = field_win_share(p, d, f, &r_net, &baseline, games, s(0xB));
        println!(
            "  {:<48} {:>10.3}  {}",
            "(b) Rollout(net) vs Rollout(heuristic) baseline",
            b,
            mark(b, fair)
        );

        // (c) Rollout(net) + net-value truncation vs the baseline.
        let trunc = NetTruncRollout::from_bytes(&bytes, rollouts, plies).unwrap();
        let baseline_c = deployed_bot(rollouts);
        let c = field_win_share(p, d, f, &trunc, &baseline_c, games, s(0xC));
        println!(
            "  {:<48} {:>10.3}  {}",
            "(c) Rollout(net)+v-trunc vs baseline",
            c,
            mark(c, fair)
        );

        // (d) baseline vs baseline — peers, expected ≈ fair (harness sanity).
        let hero_d = deployed_bot(rollouts);
        let baseline_d = deployed_bot(rollouts);
        let dd = field_win_share(p, d, f, &hero_d, &baseline_d, games, s(0xD));
        println!(
            "  {:<48} {:>10.3}  {}",
            "(d) Rollout(heuristic) vs baseline [sanity]",
            dd,
            mark(dd, fair)
        );
        println!();

        if p == 5 {
            gate_5p = Some((a, b));
        }
    }

    print_verdict(gate_5p, &configs);
    println!("\nfinished in {:.1?}", t0.elapsed());
    ExitCode::SUCCESS
}

fn print_verdict(gate_5p: Option<(f64, f64)>, configs: &[(u8, u8, u8)]) {
    println!("VERDICT");
    match gate_5p {
        Some((a, b)) => {
            let fair = 0.2;
            if b > fair + 1e-3 {
                println!(
                    "  At 5p5d6f net-guided rollout BEATS the baseline (row b = {b:.3} > fair {fair:.3})."
                );
            } else {
                println!(
                    "  At 5p5d6f net-guided rollout does NOT beat the baseline (row b = {b:.3}, fair {fair:.3})."
                );
            }
            if a < fair - 1e-3 {
                println!(
                    "  Root cause: the net is still a WEAKER base policy than the heuristic \
                     (gate row a = {a:.3} < fair {fair:.3}). The search path is sound but waits on \
                     the net maturing — net-guided rollout can only beat heuristic-rollout once the \
                     net policy edges the heuristic (row a > fair)."
                );
            } else if a > fair + 1e-3 {
                println!(
                    "  The net already EDGES the heuristic as a base policy (gate row a = {a:.3} > \
                     fair {fair:.3}); net-guided search should now carry that edge through (see row b)."
                );
            } else {
                println!(
                    "  The net is roughly even with the heuristic as a base policy \
                     (gate row a = {a:.3} ≈ fair {fair:.3}); the search edge is marginal here."
                );
            }
        }
        None => println!(
            "  (no 5-player config in {configs:?}; read the per-config rows above for the gate.)"
        ),
    }
}
