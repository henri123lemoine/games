//! Sanity bench for the 1v1 [`Duel`] bot: MCTS-eval (the registered baseline)
//! vs uniform-random over a handful of games, reporting the win split and a
//! few outcome stats. Usage:
//!
//!     cargo run --release -p snake --example duel_bench [games] [sims] [depth] [seed]

use game_core::{Agent, Game, RandomAgent, Rng, Turn};
use snake::DuelEval;
use snake::duel::{Duel, DuelState, Outcome};
use solvers::mcts::Mcts;

fn play(game: &Duel, agents: [&dyn Agent<Duel>; 2], seed: u64) -> DuelState {
    let mut rng = Rng::new(seed);
    let mut s = game.initial_state();
    while !game.is_terminal(&s) {
        match game.turn(&s) {
            Turn::Chance => {
                let outs = game.chance_outcomes(&s);
                let i = game_core::rand::sample_outcome(&outs, &mut rng);
                game.apply(&mut s, outs[i].0);
            }
            Turn::Player(p) => {
                let actions = game.legal_actions(&s);
                let i = agents[p].act(game, &s, p, &mut rng);
                game.apply(&mut s, actions[i]);
            }
        }
    }
    s
}

fn main() {
    let mut args = std::env::args().skip(1);
    let games: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(20);
    let sims: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(200);
    let depth: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(16);
    let seed: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);

    let game = Duel::new();
    let mut bot_wins = 0u64;
    let mut rand_wins = 0u64;
    let mut draws = 0u64;
    let mut steps_total = 0u64;
    for e in 0..games {
        // Rotate the bot's seat each game so first-mover bias is averaged out.
        let bot_seat = (e % 2) as usize;
        let bot = Mcts::with_eval(sims, DuelEval, depth);
        let rnd = RandomAgent;
        let agents: [&dyn Agent<Duel>; 2] = if bot_seat == 0 {
            [&bot, &rnd]
        } else {
            [&rnd, &bot]
        };
        let end = play(&game, agents, seed.wrapping_add(e));
        steps_total += end.steps() as u64;
        match end.outcome() {
            Outcome::Win(w) if w == bot_seat => bot_wins += 1,
            Outcome::Win(_) => rand_wins += 1,
            _ => draws += 1,
        }
    }
    println!(
        "duel MCTS-eval(sims={sims},depth={depth}) vs random over {games} games (seats rotated):"
    );
    println!("  bot wins {bot_wins}  random wins {rand_wins}  draws {draws}");
    println!("  mean ticks {:.1}", steps_total as f64 / games as f64);
}
