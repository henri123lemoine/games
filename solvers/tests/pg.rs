//! REINFORCE tests on an inline tic-tac-toe: self-play training beats a
//! random player, with the mean return vs random improving across training.

use game_core::Game;
use game_core::{Agent, RandomAgent, Rng, Turn, play, win_rate};
use solvers::pg::{PgConfig, Reinforce};

mod common;
use common::{Ttt, TttEnc};

/// Mean return vs random over `games`, seats swapped to cancel X's edge.
fn mean_return_vs_random(agent: &impl Agent<Ttt>, games: u32, seed: u64) -> f64 {
    let mut rng = Rng::new(seed);
    let mut total = 0.0;
    for g in 0..games {
        total += if g % 2 == 0 {
            play(&Ttt, agent, &RandomAgent, &mut rng)
        } else {
            -play(&Ttt, &RandomAgent, agent, &mut rng)
        };
    }
    total / f64::from(games)
}

#[test]
fn reinforce_learns_tictactoe() {
    let mut tr = Reinforce::new(&Ttt, &TttEnc, PgConfig::default(), 42);

    let first = tr.train_episodes(3_000);
    let initial_return = mean_return_vs_random(&tr.agent(), 200, 99);
    eprintln!(
        "initial window: vs-random return {initial_return:.3} self-play return {:.3} entropy {:.3} mse {:.3}",
        first.mean_return, first.mean_entropy, first.value_mse
    );

    let mut last = first;
    for _ in 0..9 {
        last = tr.train_episodes(3_000);
    }
    let final_return = mean_return_vs_random(&tr.agent(), 200, 99);
    eprintln!(
        "final window:   vs-random return {final_return:.3} self-play return {:.3} entropy {:.3} mse {:.3}",
        last.mean_return, last.mean_entropy, last.value_mse
    );

    assert!(last.mean_return.is_finite() && last.value_mse.is_finite());
    assert!(
        final_return > initial_return,
        "mean return vs random did not improve: first {initial_return:.3} last {final_return:.3}"
    );

    let score = win_rate(&Ttt, &tr.greedy_agent(), &RandomAgent, 200, 7);
    eprintln!("greedy vs random over 200 games: {score:.3}");
    assert!(
        score >= 0.75,
        "greedy policy scored only {score:.3} vs random"
    );
}

#[test]
fn agents_return_legal_indices_untrained() {
    let tr = Reinforce::new(&Ttt, &TttEnc, PgConfig::default(), 5);
    let game = &Ttt;
    for greedy in [false, true] {
        let agent = if greedy {
            tr.greedy_agent()
        } else {
            tr.agent()
        };
        let mut s = game.initial_state();
        let mut rng = game_core::Rng::new(11);
        while !game.is_terminal(&s) {
            let actions = game.legal_actions(&s);
            let Turn::Player(p) = game.turn(&s) else {
                unreachable!()
            };
            let i = game_core::Agent::act(&agent, game, &s, p, &mut rng);
            assert!(i < actions.len(), "illegal index {i} of {}", actions.len());
            game.apply(&mut s, actions[i]);
        }
    }
}
