//! Depth-7 alpha-beta with the heuristic eval must crush a uniform-random
//! opponent. Run with `--release`; debug-mode search is painfully slow.

use connect4::{Connect4, Connect4Eval, Connect4State};
use game_core::{Game, NoSpec, win_rate};
use solvers::AlphaBeta;

fn random_agent(game: &Connect4, state: &Connect4State, _player: usize, r: f64) -> usize {
    let n = game.legal_actions(state).len();
    ((r * n as f64) as usize).min(n - 1)
}

#[test]
fn alphabeta_beats_random() {
    let game = Connect4;
    let ab = AlphaBeta::new(7, Connect4Eval, NoSpec);
    let wr = win_rate(&game, &ab, &random_agent, 40, 0xC4C4);
    assert!(wr >= 0.95, "alpha-beta win rate vs random: {wr}");
}
