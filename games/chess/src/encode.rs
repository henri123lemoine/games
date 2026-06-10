//! Flat `f32` board features and a dense move index for policy/value networks
//! (e.g. AlphaZero-style solvers). Solver crates adapt these functions to
//! their own encoder traits, keeping this crate free of solver dependencies.
//!
//! Features (`INPUT_LEN` = 781): 12×64 one-hot piece planes (color-major,
//! a1 = 0), side to move, the four castling rights, en-passant file one-hot.
//! Move index (`POLICY_LEN` = 64·64·5 = 20480): `from·320 + to·5 + promo`
//! where promo is 0 = none, 1..=4 = knight/bishop/rook/queen — injective over
//! the legal moves of any position.

use crate::board::{CASTLE_BK, CASTLE_BQ, CASTLE_WK, CASTLE_WQ};
use crate::{Board, Color, Move, Piece};

pub const INPUT_LEN: usize = 12 * 64 + 1 + 4 + 8;
pub const POLICY_LEN: usize = 64 * 64 * 5;

pub fn encode_board(b: &Board) -> Vec<f32> {
    let mut x = vec![0.0; INPUT_LEN];
    for (sq, cell) in b.squares.iter().enumerate() {
        if let Some((c, p)) = cell {
            x[(c.index() * 6 + *p as usize) * 64 + sq] = 1.0;
        }
    }
    if b.stm == Color::Black {
        x[768] = 1.0;
    }
    for (i, bit) in [CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ]
        .into_iter()
        .enumerate()
    {
        if b.castling & bit != 0 {
            x[769 + i] = 1.0;
        }
    }
    if let Some(ep) = b.ep {
        x[773 + (ep % 8) as usize] = 1.0;
    }
    x
}

pub fn move_index(m: Move) -> usize {
    let promo = match m.promo {
        None => 0,
        Some(Piece::Knight) => 1,
        Some(Piece::Bishop) => 2,
        Some(Piece::Rook) => 3,
        Some(Piece::Queen) => 4,
        Some(p) => unreachable!("illegal promotion piece {p:?}"),
    };
    m.from as usize * 320 + m.to as usize * 5 + promo
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::START_FEN;
    use crate::movegen::legal_moves;
    use std::collections::HashSet;

    #[test]
    fn start_position_features() {
        let x = encode_board(&Board::start());
        assert_eq!(x.len(), INPUT_LEN);
        assert_eq!(x[..768].iter().sum::<f32>(), 32.0);
        assert_eq!(x[768], 0.0);
        assert!(x[769..773].iter().all(|&v| v == 1.0));
        assert!(x[773..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn side_and_ep_file_are_encoded() {
        let mut b = Board::start();
        b.apply("e2e4".parse().unwrap());
        let x = encode_board(&b);
        assert_eq!(x[768], 1.0);
        assert_eq!(x[773 + 4], 1.0);
        assert_eq!(x[773..].iter().sum::<f32>(), 1.0);
    }

    #[test]
    fn move_index_injective_and_in_range() {
        for fen in [
            START_FEN,
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        ] {
            let b = Board::from_fen(fen).unwrap();
            let moves = legal_moves(&b);
            let indices: HashSet<usize> = moves.iter().map(|&m| move_index(m)).collect();
            assert_eq!(indices.len(), moves.len(), "collision in {fen}");
            assert!(indices.into_iter().all(|i| i < POLICY_LEN));
        }
    }
}
