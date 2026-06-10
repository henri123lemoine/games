//! Isolate which rollout-agent ingredients help: opening search on/off ×
//! determinization bias settings, each scored vs a field of belief agents.
//!
//!     cargo run --release -p liars-dice --example ab [players] [dice] [faces] [rollouts] [games]

use game_core::winrate_vs_field;
use liars_dice::{BidConditioned, LiarsDice, ProbabilisticAgent};
use solvers::Rollout;

fn arg<T: std::str::FromStr>(i: usize, d: T) -> T {
    std::env::args()
        .nth(i)
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

fn main() {
    let players: u8 = arg(1, 3);
    let dice: u8 = arg(2, 3);
    let faces: u8 = arg(3, 6);
    let rollouts: u32 = arg(4, 160);
    let games: u32 = arg(5, 800);

    let game = LiarsDice::new(players, dice, faces);
    let field = ProbabilisticAgent::default_agent();
    let fair = 1.0 / players as f64;
    println!("{players}p{dice}d{faces}f, {rollouts} rollouts, {games} games/arm (fair {fair:.3})");

    let arms: [(&str, f64, f64); 4] = [
        ("bias=(0.6,0.0)  [default]", 0.6, 0.0),
        ("bias=(0.6,0.35)", 0.6, 0.35),
        ("bias=(0.8,0.2)", 0.8, 0.2),
        ("bias=(0.0,0.0)  [uniform]", 0.0, 0.0),
    ];
    for (name, bb, eb) in arms {
        let det = BidConditioned {
            bidder_bias: bb,
            endorser_bias: eb,
        };
        let hero = Rollout::new(rollouts, ProbabilisticAgent::default_agent(), det, 0x5151);
        let wr = winrate_vs_field(&game, &hero, &field, games, 0x2024);
        println!("  {name:<32} {wr:.3}");
    }
}
