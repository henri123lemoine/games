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
/// (legal-placement count) difference, from `player`'s perspective.
pub struct OthelloEval;

impl Eval<Othello> for OthelloEval {
    fn eval(&self, _game: &Othello, state: &Board, player: usize) -> f64 {
        let (mine, theirs) = (state.bb(player), state.bb(player ^ 1));
        let squares = weighted_sum(mine) - weighted_sum(theirs);
        let mobility = placements(mine, theirs).count_ones() as i32
            - placements(theirs, mine).count_ones() as i32;
        (squares + MOBILITY_WEIGHT * mobility) as f64
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
