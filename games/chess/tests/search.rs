use chess::{Board, Chess, ChessSpec, MaterialEval};
use game_core::{Agent, Game, RandomAgent, Rng, win_rate};
use solvers::AlphaBeta;

#[test]
fn crushes_random() {
    let game = Chess;
    let engine = AlphaBeta::new(3, MaterialEval, ChessSpec);
    let rate = win_rate(&game, &engine, &RandomAgent, 30, 0xC0FFEE);
    assert!(rate >= 0.9, "win rate vs random was only {rate}");
}

#[test]
fn plays_mate_in_one() {
    let game = Chess;
    let board = Board::from_fen("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1").expect("valid FEN");
    let engine = AlphaBeta::new(3, MaterialEval, ChessSpec);

    let actions = game.legal_actions(&board);
    let chosen = actions[engine.act(&game, &board, 0, &mut Rng::new(1))];
    assert_eq!(chosen.to_string(), "a1a8");

    let mut next = board.clone();
    game.apply(&mut next, chosen);
    assert!(game.is_terminal(&next));
    assert_eq!(game.returns(&next, 0), 1.0);
    assert_eq!(game.returns(&next, 1), -1.0);
}
