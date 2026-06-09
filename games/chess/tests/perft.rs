//! Perft correctness gate: exact legal-move-tree leaf counts for the standard
//! reference positions (startpos, Kiwipete, and CPW positions 3-5).
//! Run with `cargo test --release -p chess` — debug perft is too slow.

use chess::{Board, START_FEN, perft};

fn assert_perft(fen: &str, expected: &[u64]) {
    let board = Board::from_fen(fen).expect("valid FEN");
    for (i, &want) in expected.iter().enumerate() {
        let depth = (i + 1) as u32;
        let got = perft(&board, depth);
        assert_eq!(got, want, "perft({depth}) mismatch for '{fen}'");
    }
}

#[test]
fn perft_startpos() {
    assert_perft(START_FEN, &[20, 400, 8902, 197_281, 4_865_609]);
}

#[test]
fn perft_kiwipete() {
    assert_perft(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        &[48, 2039, 97_862, 4_085_603],
    );
}

#[test]
fn perft_position_3() {
    assert_perft(
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        &[14, 191, 2812, 43_238, 674_624],
    );
}

#[test]
fn perft_position_4() {
    assert_perft(
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        &[6, 264, 9467, 422_333],
    );
}

#[test]
fn perft_position_5() {
    assert_perft(
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        &[44, 1486, 62_379, 2_103_487],
    );
}

#[test]
fn fen_round_trip() {
    for fen in [
        START_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    ] {
        let board = Board::from_fen(fen).expect("valid FEN");
        assert_eq!(board.to_fen(), fen);
    }
}
