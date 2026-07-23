//! Absolute-seat policy/value encoding for four-player FFA chess.
//!
//! The policy is the ordinary AlphaZero chess move-plane layout widened to a
//! 14×14 board: 8 ray directions × 13 distances plus 8 knight jumps, for 112
//! planes per origin square. Promotion is automatic, so no underpromotion
//! planes are needed. The value head has four absolute seat logits.

use game_core::PolicyValueEncoder;

use crate::board::{CELLS, Color, Move, NONE_SQUARE, SIDE, State, xy};
use crate::{FourPlayerChess, castle_bit};

pub const MOVE_PLANES: usize = 8 * 13 + 8;
pub const POLICY_LEN: usize = CELLS * MOVE_PLANES;

// Piece occupancy: 4 seats × 6 kinds.
const PIECES: usize = 0;
// Promoted one-point queen markers, one plane per owner.
const PROMOTED: usize = PIECES + 24;
const ACTIVE: usize = PROMOTED + 4;
const TO_MOVE: usize = ACTIVE + 4;
const CASTLING: usize = TO_MOVE + 4;
// Sparse en-passant pawn destinations, one plane per owner.
const EN_PASSANT: usize = CASTLING + 8;
// One-hot checker credit for each victim: victim × credited army. This keeps
// delayed checkmate attribution Markov when another player moves in between.
const CHECK_CREDIT: usize = EN_PASSANT + 4;
// Uniform score planes, one per seat.
const SCORES: usize = CHECK_CREDIT + 16;
const HALFMOVE: usize = SCORES + 4;
const PLY: usize = HALFMOVE + 1;
const VALID: usize = PLY + 1;
pub const PLANE_COUNT: usize = VALID + 1;
pub const INPUT_LEN: usize = PLANE_COUNT * CELLS;

const DIRS: [(i8, i8); 8] = [
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
];
const KNIGHTS: [(i8, i8); 8] = [
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
];

#[derive(Debug, Default, Clone, Copy)]
pub struct FourPlayerChessEncoder;

impl PolicyValueEncoder<FourPlayerChess> for FourPlayerChessEncoder {
    fn input_len(&self) -> usize {
        INPUT_LEN
    }

    fn policy_len(&self) -> usize {
        POLICY_LEN
    }

    fn encode_state(&self, game: &FourPlayerChess, state: &State) -> Vec<f32> {
        let _ = game;
        encode_planes(state)
    }

    fn action_index(&self, _game: &FourPlayerChess, _state: &State, action: Move) -> usize {
        move_index(action)
    }
}

pub fn encode_planes(state: &State) -> Vec<f32> {
    let mut x = vec![0.0; INPUT_LEN];
    for (square, &piece) in state.board.iter().enumerate() {
        if piece.is_empty() {
            continue;
        }
        let kind = piece.kind() as usize - 1;
        x[(PIECES + piece.color().index() * 6 + kind) * CELLS + square] = 1.0;
        if piece.promoted() {
            x[(PROMOTED + piece.color().index()) * CELLS + square] = 1.0;
        }
    }
    for color in Color::ALL {
        let seat = color.index();
        if state.is_active(color) {
            x[(ACTIVE + seat) * CELLS..(ACTIVE + seat + 1) * CELLS].fill(1.0);
        }
        if state.to_move == color {
            x[(TO_MOVE + seat) * CELLS..(TO_MOVE + seat + 1) * CELLS].fill(1.0);
        }
        for king_side in [true, false] {
            if state.castling & castle_bit(color, king_side) != 0 {
                let offset = CASTLING + seat * 2 + usize::from(!king_side);
                x[offset * CELLS..(offset + 1) * CELLS].fill(1.0);
            }
        }
        let ep = state.en_passant[seat];
        if ep != NONE_SQUARE {
            x[(EN_PASSANT + seat) * CELLS + ep as usize] = 1.0;
        }
        let credit = state.check_credit[seat];
        if credit != NONE_SQUARE {
            let offset = CHECK_CREDIT + seat * 4 + credit as usize;
            x[offset * CELLS..(offset + 1) * CELLS].fill(1.0);
        }
        x[(SCORES + seat) * CELLS..(SCORES + seat + 1) * CELLS]
            .fill(f32::from(state.scores[seat]) / 100.0);
    }
    x[HALFMOVE * CELLS..(HALFMOVE + 1) * CELLS].fill(f32::from(state.halfmove.min(200)) / 200.0);
    x[PLY * CELLS..(PLY + 1) * CELLS].fill(f32::from(state.ply) / 600.0);
    for square in 0..CELLS {
        let sx = (square % SIDE) as i8;
        let sy = (square / SIDE) as i8;
        x[VALID * CELLS + square] = f32::from(crate::is_valid_xy(sx, sy));
    }
    x
}

pub fn move_index(action: Move) -> usize {
    let (from_x, from_y) = xy(action.from);
    let (to_x, to_y) = xy(action.to);
    let dx = to_x - from_x;
    let dy = to_y - from_y;
    let plane = if matches!((dx.abs(), dy.abs()), (1, 2) | (2, 1)) {
        8 * 13
            + KNIGHTS
                .iter()
                .position(|&delta| delta == (dx, dy))
                .expect("knight delta")
    } else {
        let direction = (dx.signum(), dy.signum());
        let dir = DIRS
            .iter()
            .position(|&candidate| candidate == direction)
            .expect("ray direction");
        let distance = dx.abs().max(dy.abs()) as usize;
        debug_assert!((1..=13).contains(&distance));
        dir * 13 + distance - 1
    };
    action.from as usize * MOVE_PLANES + plane
}

/// Maps four win-share probabilities to the zero-sum return convention used
/// by the game: sole winner +1, fair 25% expectation 0, zero share -1/3.
pub fn shares_to_returns(shares: &[f32]) -> [f32; 8] {
    let total: f32 = shares[..4].iter().sum();
    let mut values = [0.0; 8];
    for seat in 0..4 {
        let share = if total > 0.0 {
            shares[seat] / total
        } else {
            0.25
        };
        values[seat] = (4.0 * share - 1.0) / 3.0;
    }
    values
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use game_core::{Game, PolicyValueEncoder};

    use super::*;

    #[test]
    fn start_features_have_all_armies_and_valid_mask() {
        let state = State::standard();
        let x = encode_planes(&state);
        assert_eq!(x.len(), INPUT_LEN);
        assert_eq!(x[..24 * CELLS].iter().sum::<f32>(), 64.0);
        assert_eq!(x[VALID * CELLS..].iter().sum::<f32>(), 160.0);
        assert_eq!(
            x[ACTIVE * CELLS..(ACTIVE + 4) * CELLS].iter().sum::<f32>(),
            4.0 * CELLS as f32
        );
    }

    #[test]
    fn legal_move_indices_are_unique() {
        let game = FourPlayerChess::default();
        let state = game.initial_state();
        let moves = game.legal_actions(&state);
        let indices: HashSet<_> = moves.iter().copied().map(move_index).collect();
        assert_eq!(indices.len(), moves.len());
        assert!(indices.into_iter().all(|index| index < POLICY_LEN));
        assert_eq!(FourPlayerChessEncoder.policy_len(), POLICY_LEN);
    }
}
