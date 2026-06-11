//! Eval-truncated MCTS vs vanilla MCTS on 9x9, color-paired games.
//!
//! Both sides run 2000 simulations per move; the hero truncates playouts at
//! size*size moves and scores the leaf with [`GoEval`], the baseline plays
//! random playouts to the end. Pairs run in parallel, one thread per core.
//!
//!     cargo run --release -p go --example mcts_eval [pairs]

use std::sync::atomic::{AtomicU32, Ordering};

use game_core::{Rng, play};
use go::{Go, GoEval};
use solvers::mcts::Mcts;

fn main() {
    let pairs: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let sims = 2000;
    let game = Go::new(9);
    let cap = (game.size() * game.size()) as u32;
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let next = AtomicU32::new(0);
    let wins = AtomicU32::new(0);
    let start = std::time::Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= pairs {
                        break;
                    }
                    for swap in 0..2u64 {
                        let seed = 1_000 + u64::from(i) * 4 + swap * 2;
                        let hero = Mcts::with_eval(sims, GoEval, cap);
                        let base = Mcts::new(sims);
                        let mut rng = Rng::new(seed ^ 0x5EED);
                        let hero_won = if swap == 0 {
                            play(&game, &hero, &base, &mut rng) > 0.0
                        } else {
                            play(&game, &base, &hero, &mut rng) < 0.0
                        };
                        if hero_won {
                            wins.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            });
        }
    });
    let games = pairs * 2;
    let w = wins.load(Ordering::Relaxed);
    println!(
        "GoEval-truncated MCTS vs vanilla MCTS (9x9, {sims} sims): {w}/{games} = {:.1}% in {:.0}s",
        100.0 * f64::from(w) / f64::from(games),
        start.elapsed().as_secs_f64(),
    );
}
