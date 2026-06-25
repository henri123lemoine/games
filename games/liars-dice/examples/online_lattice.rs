//! Continuation-value ablation for the online subgame-solving agent.
//!
//! The main validation (`examples/online_eval.rs`) runs the agent with the
//! Stage-A [`DiceShareValue`] placeholder continuation, where it solves each
//! round to equilibrium but loses on *multi-round* games because the leaf value
//! is crude. This isolates the cause: rebuild the **2-player** agent with the
//! converged [`LatticeValue`] (the offline fitted-value-iteration table — the
//! strong continuation the codebase already produces) and re-measure strength vs
//! the deployed rollout bot. If the medium 2p configs flip from losing to
//! winning, the online-solving machinery is sound and the *continuation value*
//! is the lever — exactly the role [`NetValue`](liars_dice::NetValue) plays on
//! the full game.
//!
//! ```text
//! cargo run --release -p liars-dice --features parallel --example online_lattice -- \
//!     games=200 rollouts=80
//! ```

use std::collections::HashMap;

use game_core::{Agent, Rng, hash, play_n, win_share};
use liars_dice::{
    BidConditioned, DiceShareValue, FitConfig, LiarsDice, OnlineSolveAgent, OnlineSolveConfig,
    ProbabilisticAgent, fit_two_player,
};
use rayon::prelude::*;
use solvers::Rollout;

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

/// 2p win rate of `agent` vs the rollout bot, seats swapped each game.
fn winrate_2p<A: Agent<LiarsDice> + Sync>(
    d: u8,
    f: u8,
    agent: &A,
    rollouts: u32,
    games: u32,
    seed: u64,
) -> f64 {
    let game = LiarsDice::new(2, d, f);
    let bot = deployed_bot(rollouts);
    let a: &(dyn Agent<LiarsDice> + Sync) = agent;
    let b: &(dyn Agent<LiarsDice> + Sync) = &bot;
    let total: f64 = (0..games)
        .into_par_iter()
        .map(|g| {
            let mut rng = Rng::new(hash::combine(seed, u64::from(g)));
            let (s0, s1) = if g % 2 == 0 { (a, b) } else { (b, a) };
            let seats: [&dyn Agent<LiarsDice>; 2] = [s0, s1];
            let terminal = play_n(&game, &seats, &mut rng);
            win_share(&game, &terminal, if g % 2 == 0 { 0 } else { 1 })
        })
        .sum();
    total / games as f64
}

fn main() {
    let args: HashMap<String, String> = std::env::args()
        .skip(1)
        .filter_map(|a| a.split_once('=').map(|(k, v)| (k.into(), v.into())))
        .collect();
    let games: u32 = parse(&args, "games", 200);
    let rollouts: u32 = parse(&args, "rollouts", 80);
    let seed: u64 = parse(&args, "seed", 11);

    println!("Online-solve CONTINUATION-VALUE ablation (2p, vs the deployed rollout bot)");
    println!("  games/config={games}  rollouts={rollouts}  seed={seed}");
    println!("  win rate >0.5 beats the bot.\n");
    println!(
        "  {:>9} {:>16} {:>16}",
        "config", "DiceShareValue", "fitted Lattice"
    );
    println!("  {}", "-".repeat(45));

    for &(d, f) in &[(2u8, 6u8), (3, 6), (5, 6)] {
        let cfg = OnlineSolveConfig {
            seed,
            ..OnlineSolveConfig::default()
        };
        // Stage-A placeholder continuation.
        let share = OnlineSolveAgent::with_config(|| DiceShareValue, cfg);
        let wr_share = winrate_2p(d, f, &share, rollouts, games, seed ^ u64::from(d));

        // Converged fitted continuation (the offline value-iteration table). Fit
        // once, then the factory clones it per solve.
        let fit = fit_two_player(d, f, FitConfig::default());
        let lattice = fit.lattice;
        let lat_agent = OnlineSolveAgent::with_config(move || lattice.clone(), cfg);
        let wr_lat = winrate_2p(d, f, &lat_agent, rollouts, games, seed ^ u64::from(d));

        let mark = |w: f64| if w > 0.5 { "+" } else { "-" };
        println!(
            "  {:>9} {:>14.3} {} {:>14.3} {}",
            format!("2p{d}d{f}f"),
            wr_share,
            mark(wr_share),
            wr_lat,
            mark(wr_lat),
        );
    }
}
