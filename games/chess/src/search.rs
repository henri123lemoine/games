//! Fixed-depth iterative-deepening negamax with alpha-beta, quiescence on
//! captures, MVV-LVA ordering, and a material + piece-square evaluation.

use crate::Chess;
use crate::board::{Board, Color, Move, Piece};
use crate::movegen::{legal_moves, pseudo_captures, pseudo_moves};
use cfr_core::Agent;

const INF: i32 = 1_000_000;
const MATE: i32 = 100_000;

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

fn order_moves(board: &Board, moves: &mut [Move]) {
    moves.sort_by_key(|m| -move_order_score(board, m));
}

fn quiesce(board: &Board, mut alpha: i32, beta: i32) -> i32 {
    let stand = evaluate(board);
    if stand >= beta {
        return stand;
    }
    if stand > alpha {
        alpha = stand;
    }

    let us = board.stm;
    let mut captures = pseudo_captures(board);
    order_moves(board, &mut captures);

    let mut best = stand;
    for m in captures {
        let mut child = board.clone();
        child.apply(m);
        if child.is_attacked(child.kings[us.index()], child.stm) {
            continue;
        }
        let score = -quiesce(&child, -beta, -alpha);
        if score > best {
            best = score;
        }
        if score > alpha {
            alpha = score;
        }
        if alpha >= beta {
            break;
        }
    }
    best
}

fn negamax(board: &Board, depth: u32, mut alpha: i32, beta: i32, ply: i32) -> i32 {
    if board.halfmove >= 100 || board.insufficient_material() {
        return 0;
    }
    if depth == 0 {
        return quiesce(board, alpha, beta);
    }

    let us = board.stm;
    let mut moves = pseudo_moves(board);
    order_moves(board, &mut moves);

    let mut best = -INF;
    let mut any_legal = false;
    for m in moves {
        let mut child = board.clone();
        child.apply(m);
        if child.is_attacked(child.kings[us.index()], child.stm) {
            continue;
        }
        any_legal = true;
        let score = -negamax(&child, depth - 1, -beta, -alpha, ply + 1);
        if score > best {
            best = score;
        }
        if score > alpha {
            alpha = score;
        }
        if alpha >= beta {
            break;
        }
    }

    if !any_legal {
        return if board.in_check(us) { -MATE + ply } else { 0 };
    }
    best
}

/// Deterministic fixed-depth alpha-beta searcher. The arena's tie-break `r`
/// is ignored; the first best-scoring move in search order is played.
pub struct AlphaBetaAgent {
    depth: u32,
}

impl AlphaBetaAgent {
    pub fn new(depth: u32) -> Self {
        assert!(depth >= 1, "search depth must be at least 1");
        Self { depth }
    }

    pub fn best_move(&self, board: &Board) -> Option<Move> {
        let mut moves = legal_moves(board);
        if moves.is_empty() {
            return None;
        }
        order_moves(board, &mut moves);

        let mut best = moves[0];
        for depth in 1..=self.depth {
            if let Some(pos) = moves.iter().position(|&m| m == best) {
                moves[..=pos].rotate_right(1);
            }
            let mut alpha = -INF;
            let mut best_this = moves[0];
            for &m in &moves {
                let mut child = board.clone();
                child.apply(m);
                let score = -negamax(&child, depth - 1, -INF, -alpha, 1);
                if score > alpha {
                    alpha = score;
                    best_this = m;
                }
            }
            best = best_this;
        }
        Some(best)
    }
}

impl Agent<Chess> for AlphaBetaAgent {
    fn act(&self, game: &Chess, state: &Board, _player: usize, _r: f64) -> usize {
        use cfr_core::Game;
        let actions = game.legal_actions(state);
        let best = self.best_move(state).expect("act called on terminal state");
        actions
            .iter()
            .position(|&m| m == best)
            .expect("best move is legal")
    }
}
