//! Chess search *knowledge*: a material + piece-square evaluation and a
//! search spec (captures/promotions are noisy; MVV-LVA ordering). The search
//! algorithm itself is the generic `solvers::AlphaBeta`.

use crate::Chess;
use crate::board::{Board, Color, Move, Piece};
use game_core::{Eval, SearchSpec};

#[rustfmt::skip]
const PAWN_PST: [i32; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
];
#[rustfmt::skip]
const KNIGHT_PST: [i32; 64] = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
];
#[rustfmt::skip]
const BISHOP_PST: [i32; 64] = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
];
#[rustfmt::skip]
const ROOK_PST: [i32; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
];
#[rustfmt::skip]
const QUEEN_PST: [i32; 64] = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
];
#[rustfmt::skip]
const KING_MID_PST: [i32; 64] = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
];
#[rustfmt::skip]
const KING_END_PST: [i32; 64] = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
];

fn pst(piece: Piece, color: Color, sq: usize, endgame: bool) -> i32 {
    let idx = match color {
        Color::White => sq ^ 56,
        Color::Black => sq,
    };
    match piece {
        Piece::Pawn => PAWN_PST[idx],
        Piece::Knight => KNIGHT_PST[idx],
        Piece::Bishop => BISHOP_PST[idx],
        Piece::Rook => ROOK_PST[idx],
        Piece::Queen => QUEEN_PST[idx],
        Piece::King => {
            if endgame {
                KING_END_PST[idx]
            } else {
                KING_MID_PST[idx]
            }
        }
    }
}

fn center_distance(sq: u8) -> i32 {
    let f = (sq % 8) as i32;
    let r = (sq / 8) as i32;
    let df = if f <= 3 { 3 - f } else { f - 4 };
    let dr = if r <= 3 { 3 - r } else { r - 4 };
    df + dr
}

fn king_distance(a: u8, b: u8) -> i32 {
    let df = ((a % 8) as i32 - (b % 8) as i32).abs();
    let dr = ((a / 8) as i32 - (b / 8) as i32).abs();
    df + dr
}

/// Static evaluation in centipawns from the side-to-move's perspective.
pub fn evaluate(board: &Board) -> i32 {
    let mut material = [0i32; 2];
    let mut non_pawn = [0i32; 2];
    for cell in board.squares.iter().flatten() {
        let (c, p) = *cell;
        material[c.index()] += p.value();
        if p != Piece::Pawn {
            non_pawn[c.index()] += p.value();
        }
    }
    let endgame = non_pawn[0] + non_pawn[1] <= 2600;

    let mut score = 0i32;
    for (sq, cell) in board.squares.iter().enumerate() {
        if let Some((c, p)) = cell {
            let v = p.value() + pst(*p, *c, sq, endgame);
            score += if *c == Color::White { v } else { -v };
        }
    }

    for loser in [Color::White, Color::Black] {
        let winner = loser.flip();
        if material[loser.index()] == 0 && material[winner.index()] >= Piece::Rook.value() {
            let mopup = 10 * center_distance(board.kings[loser.index()])
                + 4 * (14 - king_distance(board.kings[0], board.kings[1]));
            score += if winner == Color::White {
                mopup
            } else {
                -mopup
            };
        }
    }

    match board.stm {
        Color::White => score,
        Color::Black => -score,
    }
}

fn move_order_score(board: &Board, m: &Move) -> i32 {
    let mut s = 0;
    if let Some((_, victim)) = board.squares[m.to as usize] {
        let (_, attacker) = board.squares[m.from as usize].expect("move from empty square");
        s = 10_000 + 10 * victim.value() - attacker.value();
    } else if board.ep == Some(m.to)
        && m.from % 8 != m.to % 8
        && matches!(board.squares[m.from as usize], Some((_, Piece::Pawn)))
    {
        s = 10_000 + 9 * Piece::Pawn.value();
    }
    if let Some(p) = m.promo {
        s += p.value();
    }
    s
}

/// [`Eval`] for chess: material + piece-square tables (+ endgame king
/// tables and a bare-king mop-up term), in centipawns.
pub struct MaterialEval;

impl Eval<Chess> for MaterialEval {
    fn eval(&self, _game: &Chess, state: &Board, player: usize) -> f64 {
        let stm_score = evaluate(state) as f64;
        if state.stm.index() == player {
            stm_score
        } else {
            -stm_score
        }
    }
}

/// [`SearchSpec`] for chess: captures (incl. en passant) and promotions are
/// noisy; MVV-LVA ordering.
pub struct ChessSpec;

impl SearchSpec<Chess> for ChessSpec {
    fn is_noisy(&self, _game: &Chess, state: &Board, m: Move) -> bool {
        state.squares[m.to as usize].is_some()
            || m.promo.is_some()
            || (state.ep == Some(m.to)
                && m.from % 8 != m.to % 8
                && matches!(state.squares[m.from as usize], Some((_, Piece::Pawn))))
    }

    fn order_hint(&self, _game: &Chess, state: &Board, m: Move) -> i64 {
        move_order_score(state, &m) as i64
    }
}
