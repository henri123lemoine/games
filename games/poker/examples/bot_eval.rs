//! Measure the equity bot's edge over casual baselines, in bb/100.
//!
//! Each "match" is one hand; the hero is rotated through every seat and the
//! button rotates each hand, so position bias cancels. We report the hero's
//! mean net result in big blinds per 100 hands (the standard poker yardstick).
//!
//!   cargo run --release -p poker --example bot_eval

use game_core::{Agent, Game, Rng};
use poker::agents::AlwaysCall;
use poker::{Poker, PokerBot};

/// Hero (one seat) vs a field of `baseline`, hero rotated through seats and the
/// button rotated each hand. Returns the hero's mean net bb across `hands`.
fn bb_per_hand(
    seats: u8,
    hero: &dyn Agent<Poker>,
    baseline: &dyn Agent<Poker>,
    hands: u32,
    seed: u64,
) -> f64 {
    let mut rng = Rng::new(seed);
    let mut total = 0.0;
    for h in 0..hands {
        let game = Poker::new(seats)
            .with_blinds(1, 2)
            .with_stack(200)
            .with_button((h % seats as u32) as u8);
        let hero_seat = (h as usize) % seats as usize;
        let agents: Vec<&dyn Agent<Poker>> = (0..seats as usize)
            .map(|p| if p == hero_seat { hero } else { baseline })
            .collect();
        let terminal = game_core::play_n(&game, &agents, &mut rng);
        total += game.returns(&terminal, hero_seat);
    }
    total / hands as f64
}

fn main() {
    let hands: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000);

    let bot = PokerBot::default_bot();

    println!("No-Limit Texas Hold'em — equity bot vs baselines");
    println!("(positive = hero wins; hero rotated through all seats)\n");

    for &seats in &[2u8, 6] {
        let label = if seats == 2 { "heads-up" } else { "6-max" };
        let vs_call = bb_per_hand(seats, &bot, &AlwaysCall, hands, 0x9001);
        let vs_rand = bb_per_hand(seats, &bot, &game_core::RandomAgent, hands, 0x9002);
        println!("{label} ({seats} seats), {hands} hands each:");
        println!(
            "  vs always-call : {:+.2} bb/hand  ({:+.0} bb/100)",
            vs_call,
            vs_call * 100.0
        );
        println!(
            "  vs random      : {:+.2} bb/hand  ({:+.0} bb/100)",
            vs_rand,
            vs_rand * 100.0
        );
    }
}
