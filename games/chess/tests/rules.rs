//! Position-key and adjudication edge cases the search stack depends on.

use chess::{Adjudication, Board, Color, Move, adjudicate};

#[test]
fn key_ignores_the_halfmove_clock() {
    // A genuine repetition recurs at ever-higher clocks; the position key
    // must still collide or threefold detection never fires.
    let mut b = Board::start();
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] {
        b.apply(uci.parse().unwrap());
    }
    assert_eq!(b.halfmove, 4);
    assert_eq!(b.key(), Board::start().key());
}

#[test]
fn ep_square_keys_only_when_capturable() {
    let dead_ep = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
    let no_ep = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1";
    assert_eq!(
        Board::from_fen(dead_ep).unwrap().key(),
        Board::from_fen(no_ep).unwrap().key(),
        "an uncapturable ep square must not split the position identity"
    );

    let live_ep = "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3";
    let live_no_ep = "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 3";
    assert_ne!(
        Board::from_fen(live_ep).unwrap().key(),
        Board::from_fen(live_no_ep).unwrap().key(),
        "a capturable ep square is part of the position"
    );
}

#[test]
fn mate_on_the_fifty_move_boundary_outranks_the_draw() {
    let mut b = Board::from_fen("6k1/5ppp/8/8/8/8/8/4R2K w - - 99 80").unwrap();
    b.apply("e1e8".parse().unwrap());
    assert_eq!(b.halfmove, 100);
    assert_eq!(
        adjudicate(&b, 1),
        Some(Adjudication::Checkmate {
            winner: Color::White
        })
    );
}

#[test]
fn malformed_move_text_errors_instead_of_panicking() {
    for text in ["e2é", "♔xe5", "e2e4é", ""] {
        assert!(text.parse::<Move>().is_err(), "{text:?} must not parse");
    }
}
