//! Mean final length on the default 10x10 board: uniform-random actions vs
//! greedy 1-ply lookahead on [`SnakeEval`].

use game_core::{Eval, Game, Rng, Turn};
use snake::{Snake, SnakeEval, SnakeState};

fn step_chance(g: &Snake, s: &mut SnakeState, rng: &mut Rng) {
    let outs = g.chance_outcomes(s);
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
    g.apply(s, chosen);
}

fn run(g: &Snake, games: u32, seed: u64, pick: impl Fn(&Snake, &SnakeState, f64) -> usize) -> f64 {
    let mut rng = Rng::new(seed);
    let mut total = 0usize;
    for _ in 0..games {
        let mut s = g.initial_state();
        while !g.is_terminal(&s) {
            match g.turn(&s) {
                Turn::Chance => step_chance(g, &mut s, &mut rng),
                Turn::Player(_) => {
                    let acts = g.legal_actions(&s);
                    let i = pick(g, &s, rng.unit());
                    g.apply(&mut s, acts[i]);
                }
            }
        }
        total += s.len();
    }
    total as f64 / games as f64
}

fn main() {
    let g = Snake::default();
    let games = 10_000;

    let random = run(&g, games, 1, |_, _, r| ((r * 3.0) as usize).min(2));

    let eval = SnakeEval;
    let greedy = run(&g, games, 2, |g, s, _| {
        let mut best = (0, f64::NEG_INFINITY);
        for (i, &a) in g.legal_actions(s).iter().enumerate() {
            let mut next = s.clone();
            g.apply(&mut next, a);
            let v = eval.eval(g, &next, 0);
            if v > best.1 {
                best = (i, v);
            }
        }
        best.0
    });

    println!("{games} games on 10x10, mean final length:");
    println!("  random        {random:.2}");
    println!("  greedy 1-ply  {greedy:.2}");
}
