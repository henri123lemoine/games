//! Score benchmark for the registered 2048 bot (`lab play 2048 bot=mcts-eval`
//! defaults: sims=200, depth=8). Usage:
//!
//!     cargo run --release -p g2048 --example bench [episodes] [sims] [depth] [seed]

use g2048::{G2048, Heuristic2048};
use game_core::{Agent, Game, Rng, Turn};
use solvers::mcts::Mcts;

fn episode(game: &G2048, agent: &dyn Agent<G2048>, seed: u64) -> (u64, u32) {
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
                let i = agent.act(game, &s, 0, rng.unit());
                game.apply(&mut s, actions[i]);
            }
        }
    }
    (s.score(), s.max_tile())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let episodes: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(50);
    let sims: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(200);
    let depth: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);
    let seed: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);

    let game = G2048;
    let mut scores = Vec::new();
    let mut tiles = Vec::new();
    for e in 0..episodes {
        let agent = Mcts::with_eval(
            sims,
            Heuristic2048,
            depth,
            seed.wrapping_add(2 * e) ^ 0x2048,
        );
        let (score, tile) = episode(&game, &agent, seed.wrapping_add(2 * e));
        scores.push(score);
        tiles.push(tile);
        println!("episode {e:>3}  score {score:>6}  max tile {tile}");
    }
    scores.sort_unstable();
    let mean = scores.iter().sum::<u64>() as f64 / scores.len() as f64;
    let median = scores[scores.len() / 2];
    let best_tile = tiles.iter().max().copied().unwrap_or(0);
    let reach = |t: u32| tiles.iter().filter(|&&x| x >= t).count();
    println!(
        "episodes {}  sims {}  depth {}  seed {}",
        episodes, sims, depth, seed
    );
    println!(
        "score mean {:.0}  median {}  min {}  max {}",
        mean,
        median,
        scores[0],
        scores[scores.len() - 1]
    );
    println!(
        "max tile {}  reached>=1024: {}/{}  reached>=2048: {}/{}",
        best_tile,
        reach(1024),
        episodes,
        reach(2048),
        episodes
    );
}
