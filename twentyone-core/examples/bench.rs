//! Rough throughput benchmark for the game engine: plays full games with a
//! simple threshold policy and reports games/second and steps/second.

use std::time::Instant;
use twentyone_core::{Action, Env};

fn play_one(seed: u64) -> u64 {
    let mut env = Env::new(seed);
    let mut steps = 0u64;
    while env.hearts(0) > 0 && env.hearts(1) > 0 {
        env.start_new_round().unwrap();
        loop {
            let p = env.current_player();
            let obs = env.observation(p);
            let action = if obs.self_total < 17 {
                Action::Draw
            } else {
                Action::Stand
            };
            let res = env.step(action).unwrap();
            steps += 1;
            if res.round_over || res.game_over {
                break;
            }
        }
    }
    steps
}

fn main() {
    let games = 1_000_000u64;
    let t = Instant::now();
    let mut steps = 0u64;
    for i in 0..games {
        steps += play_one(0x9E3779B97F4A7C15 ^ i);
    }
    let dt = t.elapsed();
    let secs = dt.as_secs_f64();
    eprintln!(
        "{games} games, {steps} steps in {dt:.2?} -> {:.2}M games/s, {:.2}M steps/s",
        games as f64 / secs / 1e6,
        steps as f64 / secs / 1e6
    );
}
