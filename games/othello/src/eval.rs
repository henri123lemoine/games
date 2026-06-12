//! Othello search *knowledge*: the classic weighted-square table (corners
//! huge, X/C-squares negative) plus a mobility term, and a search spec that
//! orders moves by square weight so corners are searched first. The search
//! algorithm itself is the generic `solvers::AlphaBeta`.

use game_core::{Eval, SearchSpec};

use crate::{Board, Move, Othello, placements};

#[rustfmt::skip]
pub(crate) const WEIGHTS: [i32; 64] = [
    100, -20,  10,   5,   5,  10, -20, 100,
    -20, -50,  -2,  -2,  -2,  -2, -50, -20,
     10,  -2,  -1,  -1,  -1,  -1,  -2,  10,
      5,  -2,  -1,  -1,  -1,  -1,  -2,   5,
      5,  -2,  -1,  -1,  -1,  -1,  -2,   5,
     10,  -2,  -1,  -1,  -1,  -1,  -2,  10,
    -20, -50,  -2,  -2,  -2,  -2, -50, -20,
    100, -20,  10,   5,   5,  10, -20, 100,
];

const MOBILITY_WEIGHT: i32 = 8;

fn weighted_sum(mut discs: u64) -> i32 {
    let mut sum = 0;
    while discs != 0 {
        sum += WEIGHTS[discs.trailing_zeros() as usize];
        discs &= discs - 1;
    }
    sum
}

/// [`Eval`] for Othello: weighted-square table difference plus a mobility
/// (legal-placement count) difference, from `player`'s perspective, squashed
/// onto the `(-1, 1)` returns scale the [`Eval`] contract requires
/// (`(2/π)·atan(score/100)` — strictly monotone, so alpha-beta's move choice
/// is unchanged).
pub struct OthelloEval;

impl Eval<Othello> for OthelloEval {
    fn eval(&self, _game: &Othello, state: &Board, player: usize) -> f64 {
        let (mine, theirs) = (state.bb(player), state.bb(player ^ 1));
        let squares = weighted_sum(mine) - weighted_sum(theirs);
        let mobility = placements(mine, theirs).count_ones() as i32
            - placements(theirs, mine).count_ones() as i32;
        let score = (squares + MOBILITY_WEIGHT * mobility) as f64;
        game_core::eval_squash(score, 100.0)
    }
}

/// [`SearchSpec`] for Othello: order placements by square weight, so corners
/// are searched first and X/C-squares last.
pub struct OthelloSpec;

impl SearchSpec<Othello> for OthelloSpec {
    fn order_hint(&self, _game: &Othello, _state: &Board, action: Move) -> i64 {
        match action {
            Move::Place(sq) => WEIGHTS[sq as usize] as i64,
            Move::Pass => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Board;

    fn board(black: u64, white: u64) -> Board {
        Board {
            black,
            white,
            to_move: 0,
        }
    }

    #[test]
    fn weights_have_the_full_square_symmetry() {
        for r in 0..8usize {
            for c in 0..8usize {
                let w = WEIGHTS[r * 8 + c];
                assert_eq!(w, WEIGHTS[r * 8 + (7 - c)], "mirror at ({r},{c})");
                assert_eq!(w, WEIGHTS[(7 - r) * 8 + c], "flip at ({r},{c})");
                assert_eq!(w, WEIGHTS[c * 8 + r], "transpose at ({r},{c})");
            }
        }
    }

    #[test]
    fn eval_is_antisymmetric_between_the_players() {
        let positions = [
            Board::start(),
            board(1 | (1 << 9), (1 << 27) | (1 << 36)),
            board(0xFF, 0xFF00),
            board((1 << 63) | (1 << 7), (1 << 28) | (1 << 35) | (1 << 42)),
        ];
        for (i, s) in positions.iter().enumerate() {
            let e0 = OthelloEval.eval(&Othello, s, 0);
            let e1 = OthelloEval.eval(&Othello, s, 1);
            assert!(
                (e0 + e1).abs() < 1e-12,
                "position {i}: eval(p0)={e0}, eval(p1)={e1}"
            );
        }
    }

    #[test]
    fn corners_outvalue_x_squares() {
        let neutral_white = 1 << 27;
        let corner = OthelloEval.eval(&Othello, &board(1, neutral_white), 0);
        let x_square = OthelloEval.eval(&Othello, &board(1 << 9, neutral_white), 0);
        assert!(
            corner > x_square,
            "corner ({corner}) must beat X-square ({x_square})"
        );
        assert!(x_square < 0.0, "an X-square next to nothing is a liability");
    }

    #[test]
    fn eval_stays_strictly_inside_the_returns_scale() {
        let extremes = [
            board(u64::MAX, 0),
            board(0, u64::MAX),
            board(0x8100_0000_0000_0081, 0), // all four corners
            Board::start(),
        ];
        for s in &extremes {
            for p in 0..2 {
                let e = OthelloEval.eval(&Othello, s, p);
                assert!(e.abs() < 1.0, "eval {e} escaped (-1, 1)");
            }
        }
    }
}
