//! Flat `f32` board features and dense move indices for policy/value
//! networks (e.g. AlphaZero-style solvers). The flat encoding is bound to
//! game-core's [`PolicyValueEncoder`] capability as [`FlatEncoder`] (MLPs)
//! and [`PlanesEncoder`] (conv nets — azt, azinfer and the browser bind
//! this one).
//!
//! Two encodings live here:
//!
//! * Flat (for MLPs): `INPUT_LEN` = 781 features (12×64 one-hot piece
//!   planes, color-major, a1 = 0; side to move; the four castling rights;
//!   en-passant file one-hot) and `POLICY_LEN` = 64·64·5 = 20480 move
//!   indices (`from·320 + to·5 + promo`).
//! * Planes (for conv nets): [`encode_planes`] gives `PLANE_COUNT` = 18
//!   board planes from the side to move's perspective, and [`az_move_index`]
//!   the AlphaZero 64×73 move-plane policy index (`AZ_POLICY_LEN` = 4672).
//!   Both orient the board so "forward" is always +1 rank.
//!
//! All move indices are injective over the legal moves of any one position.

use game_core::PolicyValueEncoder;

use crate::board::{CASTLE_BK, CASTLE_BQ, CASTLE_WK, CASTLE_WQ};
use crate::{Board, Chess, Color, Move, Piece};

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

/// The flat encoding declared as the [`PolicyValueEncoder`] capability, so
/// every consumer binds the same adapter instead of writing its own.
pub struct FlatEncoder;

impl PolicyValueEncoder<Chess> for FlatEncoder {
    fn input_len(&self) -> usize {
        INPUT_LEN
    }
    fn policy_len(&self) -> usize {
        POLICY_LEN
    }
    fn encode_state(&self, _g: &Chess, s: &Board) -> Vec<f32> {
        encode_board(s)
    }
    fn action_index(&self, _g: &Chess, _s: &Board, m: Move) -> usize {
        move_index(m)
    }
}

/// Number of 8×8 feature planes in [`encode_planes`].
pub const PLANE_COUNT: usize = 18;
/// AlphaZero-style move-plane policy space: 64 from-squares × 73 planes.
pub const AZ_POLICY_LEN: usize = 64 * 73;

/// The plane encoding as a [`PolicyValueEncoder`]: [`encode_planes`]
/// features (flat `[plane · 64 + square]`, consumers reshape) and
/// [`az_move_index`] policy indices. This is what conv-net stacks bind.
pub struct PlanesEncoder;

impl PolicyValueEncoder<Chess> for PlanesEncoder {
    fn input_len(&self) -> usize {
        PLANE_COUNT * 64
    }
    fn policy_len(&self) -> usize {
        AZ_POLICY_LEN
    }
    fn encode_state(&self, _g: &Chess, s: &Board) -> Vec<f32> {
        encode_planes(s)
    }
    fn action_index(&self, _g: &Chess, s: &Board, m: Move) -> usize {
        az_move_index(m, s.stm)
    }
}

/// Clockwise from "forward" (the side to move's +1 rank direction).
const DIRS: [(isize, isize); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];
const KNIGHT_DELTAS: [(isize, isize); 8] = [
    (2, 1),
    (1, 2),
    (-1, 2),
    (-2, 1),
    (-2, -1),
    (-1, -2),
    (1, -2),
    (2, -1),
];

fn orient(sq: u8, stm: Color) -> usize {
    match stm {
        Color::White => usize::from(sq),
        Color::Black => usize::from(sq ^ 56),
    }
}

/// 18 board planes from the side to move's perspective (ranks flipped for
/// Black, so the net always sees itself moving "up"): our 6 piece types,
/// their 6, our two castling rights, theirs, the en-passant file, and the
/// halfmove clock scaled to [0, 1]. Layout: `plane · 64 + oriented square`.
pub fn encode_planes(b: &Board) -> Vec<f32> {
    let mut x = vec![0.0f32; PLANE_COUNT * 64];
    for (sq, cell) in b.squares.iter().enumerate() {
        if let Some((c, p)) = cell {
            let side = usize::from(*c != b.stm);
            x[(side * 6 + *p as usize) * 64 + orient(sq as u8, b.stm)] = 1.0;
        }
    }
    let rights = match b.stm {
        Color::White => [CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ],
        Color::Black => [CASTLE_BK, CASTLE_BQ, CASTLE_WK, CASTLE_WQ],
    };
    for (i, bit) in rights.into_iter().enumerate() {
        if b.castling & bit != 0 {
            x[(12 + i) * 64..(13 + i) * 64].fill(1.0);
        }
    }
    if let Some(ep) = b.ep {
        let file = usize::from(ep % 8);
        for rank in 0..8 {
            x[16 * 64 + rank * 8 + file] = 1.0;
        }
    }
    x[17 * 64..18 * 64].fill((f32::from(b.halfmove) / 100.0).min(1.0));
    x
}

/// AlphaZero move-plane policy index in `0..AZ_POLICY_LEN`: oriented
/// from-square × 73 movement planes — 56 queen-like (8 directions × 7
/// distances; includes king, castling, and queen-promotion moves), 8 knight
/// deltas, 9 underpromotions (3 directions × knight/bishop/rook).
pub fn az_move_index(m: Move, stm: Color) -> usize {
    let from = orient(m.from, stm);
    let to = orient(m.to, stm);
    let dr = to as isize / 8 - from as isize / 8;
    let df = to as isize % 8 - from as isize % 8;
    let plane = match m.promo {
        Some(p @ (Piece::Knight | Piece::Bishop | Piece::Rook)) => {
            debug_assert_eq!(dr, 1, "underpromotion must advance one rank");
            let piece = match p {
                Piece::Knight => 0,
                Piece::Bishop => 1,
                _ => 2,
            };
            64 + (df + 1) as usize * 3 + piece
        }
        _ if dr.abs().min(df.abs()) == 1 && dr.abs().max(df.abs()) == 2 => {
            56 + KNIGHT_DELTAS
                .iter()
                .position(|&d| d == (dr, df))
                .expect("knight delta")
        }
        _ => {
            let dir = DIRS
                .iter()
                .position(|&d| d == (dr.signum(), df.signum()))
                .expect("queen-like direction");
            let dist = dr.abs().max(df.abs()) as usize;
            debug_assert!((1..=7).contains(&dist));
            dir * 7 + dist - 1
        }
    };
    from * 73 + plane
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

    #[test]
    fn planes_start_position() {
        let x = encode_planes(&Board::start());
        assert_eq!(x.len(), PLANE_COUNT * 64);
        // Our pawns on the oriented second rank, theirs mirrored onto the
        // seventh; 16 pieces per side; all four castling planes set.
        assert!((8..16).all(|sq| x[sq] == 1.0));
        assert!((48..56).all(|sq| x[6 * 64 + sq] == 1.0));
        assert_eq!(x[..6 * 64].iter().sum::<f32>(), 16.0);
        assert_eq!(x[6 * 64..12 * 64].iter().sum::<f32>(), 16.0);
        assert!(x[12 * 64..16 * 64].iter().all(|&v| v == 1.0));
        assert!(x[16 * 64..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn planes_are_color_symmetric() {
        // After 1. e4 e5 Black's view of the board equals White's at start
        // shifted by the symmetric pawn pushes; spot-check the e-pawns and
        // the en-passant file plane.
        let mut b = Board::start();
        b.apply("e2e4".parse().unwrap());
        let x = encode_planes(&b);
        // Black to move: its own e7 pawn appears at oriented e2 (sq 12).
        assert_eq!(x[12], 1.0);
        // White's e4 pawn is "their" pawn at oriented e5 (sq 36).
        assert_eq!(x[6 * 64 + 36], 1.0);
        // En-passant file e: the whole file-4 column.
        assert_eq!(x[16 * 64..17 * 64].iter().sum::<f32>(), 8.0);
        assert_eq!(x[16 * 64 + 4], 1.0);
    }

    #[test]
    fn az_index_orients_mirrored_moves_identically() {
        let mut b = Board::start();
        let e2e4 = "e2e4".parse().unwrap();
        let white = az_move_index(e2e4, b.stm);
        // e2 = sq 12, two squares forward = plane 1.
        assert_eq!(white, 12 * 73 + 1);
        b.apply(e2e4);
        let e7e5 = "e7e5".parse().unwrap();
        assert_eq!(az_move_index(e7e5, b.stm), white);
    }

    #[test]
    fn az_index_injective_and_in_range() {
        let fens = [
            START_FEN,
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R b KQ - 1 8",
            "n1n5/1P6/8/8/8/8/6p1/k1K5 w - - 0 1",
            "n1n5/1P6/8/8/8/8/6p1/k1K5 b - - 0 1",
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        ];
        for fen in fens {
            let b = Board::from_fen(fen).unwrap();
            let moves = legal_moves(&b);
            let indices: HashSet<usize> = moves.iter().map(|&m| az_move_index(m, b.stm)).collect();
            assert_eq!(indices.len(), moves.len(), "collision in {fen}");
            assert!(indices.into_iter().all(|i| i < AZ_POLICY_LEN));
        }
    }

    #[test]
    fn az_index_underpromotions_distinct() {
        // White pawn b7 can push or capture onto the eighth rank with all
        // four promotion pieces: 12 promotion moves, all distinct indices.
        let b = Board::from_fen("n1n5/1P6/8/8/8/8/8/k2K4 w - - 0 1").unwrap();
        let promos: Vec<Move> = legal_moves(&b)
            .into_iter()
            .filter(|m| m.promo.is_some())
            .collect();
        assert_eq!(promos.len(), 12);
        let indices: HashSet<usize> = promos.iter().map(|&m| az_move_index(m, b.stm)).collect();
        assert_eq!(indices.len(), 12);
        // Underpromotions land in the 64..73 plane block, queen promotions
        // in the queen-like block.
        for m in promos {
            let plane = az_move_index(m, b.stm) % 73;
            if m.promo == Some(Piece::Queen) {
                assert!(plane < 56, "queen promo in queen-like planes");
            } else {
                assert!((64..73).contains(&plane), "underpromo plane {plane}");
            }
        }
    }
}
