//! Score benchmark for the registered Snake bot (`lab play snake bot=mcts-eval`
//! defaults: sims=200, depth=12, 10x10 board). Usage:
//!
//!     cargo run --release -p snake --example bench [episodes] [sims] [depth] [seed]

use game_core::{Agent, Game, Rng, Turn};
use snake::{Snake, SnakeEval, SnakeState};
use solvers::mcts::Mcts;

fn episode(game: &Snake, agent: &dyn Agent<Snake>, seed: u64) -> SnakeState {
    let mut rng = Rng::new(seed);
    let mut s = game.initial_state();
    while !game.is_terminal(&s) {
        match game.turn(&s) {
            Turn::Chance => {
                let outs = game.chance_outcomes(&s);
                let r = rng.unit();
                let mut acc = 0.0;
                let mut chosen = outs[outs.len() - 1].0;
                for (a, p) in &outs {
                    acc += *p;
                    if r < acc {
                        chosen = *a;
                        break;
                    }
                }
                game.apply(&mut s, chosen);
            }
            Turn::Player(_) => {
                let actions = game.legal_actions(&s);
                let i = agent.act(game, &s, 0, &mut rng);
                game.apply(&mut s, actions[i]);
            }
        }
    }
    s
}

fn main() {
    let mut args = std::env::args().skip(1);
    let episodes: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(50);
    let sims: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(200);
    let depth: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(12);
    let seed: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);

    let game = Snake::new(10, 10);
    let mut lengths = Vec::new();
    for e in 0..episodes {
        let agent = Mcts::with_eval(sims, SnakeEval, depth);
        let end = episode(&game, &agent, seed.wrapping_add(2 * e));
        lengths.push(end.len() as u64);
        println!(
            "episode {e:>3}  length {:>3}  score {:.3}",
            end.len(),
            game.score(&end)
        );
    }
    lengths.sort_unstable();
    let mean = lengths.iter().sum::<u64>() as f64 / lengths.len() as f64;
    let median = lengths[lengths.len() / 2];
    println!(
        "episodes {}  sims {}  depth {}  seed {}  board 10x10 (area 100)",
        episodes, sims, depth, seed
    );
    println!(
        "length mean {:.1}  median {}  min {}  max {}",
        mean,
        median,
        lengths[0],
        lengths[lengths.len() - 1]
    );
}
