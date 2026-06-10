use game_core::{Agent, Game, win_rate};
use othello::{Board, Othello, OthelloEval, OthelloSpec};
use solvers::AlphaBeta;

fn random_agent() -> impl Agent<Othello> {
    |game: &Othello, state: &Board, _player: usize, r: f64| -> usize {
        let n = game.legal_actions(state).len();
        ((r * n as f64) as usize).min(n - 1)
    }
}

#[test]
fn alphabeta_crushes_random() {
    let game = Othello;
    let engine = AlphaBeta::new(5, OthelloEval, OthelloSpec);
    let rate = win_rate(&game, &engine, &random_agent(), 30, 0xD15C5);
    assert!(rate >= 0.9, "win rate vs random was only {rate}");
}
