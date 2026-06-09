//! Isolate which rollout-agent ingredients help: opening search on/off ×
//! determinization bias settings, each scored vs a field of belief agents.
//!
//!     cargo run --release -p liars-dice --example ab [players] [dice] [faces] [rollouts] [games]

use cfr_core::winrate_vs_field;
use liars_dice::{LiarsDice, ProbConfig, ProbabilisticAgent, RolloutAgent};

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

    let arms: [(&str, bool, f64, f64); 5] = [
        ("open=policy bias=(0.6,0.0)", false, 0.6, 0.0),
        ("open=policy bias=(0.6,0.35)", false, 0.6, 0.35),
        ("open=search bias=(0.6,0.0)", true, 0.6, 0.0),
        ("open=search bias=(0.6,0.35)", true, 0.6, 0.35),
        ("open=policy bias=(0.8,0.2)", false, 0.8, 0.2),
    ];
    for (name, open, bb, eb) in arms {
        let mut hero = RolloutAgent::new(rollouts, ProbConfig::default(), 0x5151);
        hero.search_openings = open;
        hero.bidder_bias = bb;
        hero.endorser_bias = eb;
        let wr = winrate_vs_field(&game, &hero, &field, games, 0x2024);
        println!("  {name:<32} {wr:.3}");
    }
}
