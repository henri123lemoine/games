//! Does Monte-Carlo lookahead beat the raw probabilistic policy it rolls out?
//! Pits the rollout agent (hero) against a field of probabilistic agents.
//!
//!     cargo run --release -p liars-dice --example rollout_eval [players] [dice] [faces] [rollouts] [games]

use cfr_core::winrate_vs_field;
use liars_dice::{LiarsDice, ProbConfig, ProbabilisticAgent, RandomAgent, RolloutAgent};

fn arg<T: std::str::FromStr>(i: usize, d: T) -> T {
    std::env::args()
        .nth(i)
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

fn main() {
    let players: u8 = arg(1, 5);
    let dice: u8 = arg(2, 5);
    let faces: u8 = arg(3, 6);
    let rollouts: u32 = arg(4, 160);
    let games: u32 = arg(5, 1500);

    let game = LiarsDice::new(players, dice, faces);
    let fair = 1.0 / players as f64;
    let hero = RolloutAgent::new(rollouts, ProbConfig::default(), 0x5151);
    let prob = ProbabilisticAgent::default_agent();
    let rand = RandomAgent;

    println!("Rollout agent ({rollouts} rollouts) on {players}p{dice}d{faces}f  (fair {fair:.3})");
    let vs_prob = winrate_vs_field(&game, &hero, &prob, games, 0x2024);
    println!("  rollout vs probabilistic-field: {vs_prob:.3}");
    let vs_rand = winrate_vs_field(&game, &hero, &rand, games, 0x9999);
    println!("  rollout vs random-field       : {vs_rand:.3}");
}
