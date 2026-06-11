use game_core::{RandomAgent, win_rate};
use othello::{Othello, OthelloEval, OthelloSpec};
use solvers::AlphaBeta;

#[test]
fn alphabeta_crushes_random() {
    let game = Othello;
    let engine = AlphaBeta::new(5, OthelloEval, OthelloSpec);
    let rate = win_rate(&game, &engine, &RandomAgent, 30, 0xD15C5);
    assert!(rate >= 0.9, "win rate vs random was only {rate}");
}
