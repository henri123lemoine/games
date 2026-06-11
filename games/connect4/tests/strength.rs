//! Depth-7 alpha-beta with the heuristic eval must crush a uniform-random
//! opponent. Run with `--release`; debug-mode search is painfully slow.

use connect4::{Connect4, Connect4Eval};
use game_core::{NoSpec, RandomAgent, win_rate};
use solvers::AlphaBeta;

#[test]
fn alphabeta_beats_random() {
    let game = Connect4;
    let ab = AlphaBeta::new(7, Connect4Eval, NoSpec);
    let wr = win_rate(&game, &ab, &RandomAgent, 40, 0xC4C4);
    assert!(wr >= 0.95, "alpha-beta win rate vs random: {wr}");
}
