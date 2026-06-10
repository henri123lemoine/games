//! Win-rate of the probabilistic agent against a field of random opponents,
//! across configurations up to the 5-player × 5-dice target.
//!
//!     cargo run --release -p liars-dice --example evaluate

use game_core::winrate_vs_field;
use liars_dice::{LiarsDice, ProbabilisticAgent, RandomAgent};

fn main() {
    let configs = [(2u8, 2u8, 6u8), (2, 5, 6), (3, 3, 6), (3, 5, 6), (5, 5, 6)];
    let games = 8000u32;
    println!(
        "{:>10} {:>16} {:>10} {:>10}",
        "config", "prob-vs-random", "fair", "lift"
    );
    println!("{}", "-".repeat(50));
    for &(p, d, f) in &configs {
        let game = LiarsDice::new(p, d, f);
        let hero = ProbabilisticAgent::default_agent();
        let base = RandomAgent;
        let wr = winrate_vs_field(&game, &hero, &base, games, 0xABCDEF);
        let fair = 1.0 / p as f64;
        println!(
            "{:>10} {:>16.3} {:>10.3} {:>9.1}x",
            format!("{p}p{d}d{f}f"),
            wr,
            fair,
            wr / fair
        );
    }
}
