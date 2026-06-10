//! Eval-truncated MCTS should comfortably out-score uniform random play.

use g2048::{G2048, G2048State, Heuristic2048};
use game_core::{Agent, Game, Rng, Turn};
use solvers::mcts::Mcts;

fn play_score(game: &G2048, agent: &dyn Agent<G2048>, seed: u64) -> (u64, u32) {
    let mut rng = Rng::new(seed);
    let mut s: G2048State = game.initial_state();
    while !game.is_terminal(&s) {
        match game.turn(&s) {
            Turn::Chance => {
                let outs = game.chance_outcomes(&s);
                let r = rng.unit();
                let mut acc = 0.0;
                let mut chosen = outs[outs.len() - 1].0;
                for &(a, p) in &outs {
                    acc += p;
                    if r < acc {
                        chosen = a;
                        break;
                    }
                }
                game.apply(&mut s, chosen);
            }
            Turn::Player(p) => {
                let acts = game.legal_actions(&s);
                let i = agent.act(game, &s, p, rng.unit());
                game.apply(&mut s, acts[i]);
            }
        }
    }
    (s.score(), s.max_tile())
}

fn mean_score(game: &G2048, agent: &dyn Agent<G2048>, games: u64) -> (f64, u32) {
    let mut total = 0u64;
    let mut best = 0u32;
    for g in 0..games {
        let (score, max) = play_score(game, agent, g * 977 + 13);
        total += score;
        best = best.max(max);
    }
    (total as f64 / games as f64, best)
}

#[test]
fn mcts_with_eval_beats_random() {
    let game = G2048;
    let random = |g: &G2048, s: &G2048State, _p: usize, r: f64| -> usize {
        let n = g.legal_actions(s).len();
        ((r * n as f64) as usize).min(n - 1)
    };
    let mcts = Mcts::with_eval(100, Heuristic2048, 8, 42);

    let (rand_avg, rand_best) = mean_score(&game, &random, 10);
    let (mcts_avg, mcts_best) = mean_score(&game, &mcts, 10);
    println!(
        "random: avg score {rand_avg:.0} best tile {rand_best} | \
         mcts(100 sims, depth 8): avg score {mcts_avg:.0} best tile {mcts_best}"
    );
    assert!(
        mcts_avg > 2.0 * rand_avg,
        "mcts {mcts_avg} vs random {rand_avg}"
    );
}
