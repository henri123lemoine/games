//! Pseudo-legal move generation with incremental legality filtering, plus
//! perft. Legality of a pseudo-legal move is decided without applying it in
//! the common cases: a non-king move made while not in check is illegal only
//! if it uncovers a slider ray to the mover's king, and a king move only if
//! its destination is already attacked. En-passant captures and moves made
//! while in check fall back to copy-make probing.

use crate::board::{
    Board, CASTLE_BK, CASTLE_BQ, CASTLE_WK, CASTLE_WQ, Color, DIR_DELTAS, KING_DELTAS,
    KNIGHT_DELTAS, Move, Piece, RAY_LEN, dir, square_at,
};

const ORTHO_RAYS: [usize; 4] = [dir::E, dir::W, dir::N, dir::S];
const DIAG_RAYS: [usize; 4] = [dir::NE, dir::SE, dir::NW, dir::SW];
const PROMO_PIECES: [Piece; 4] = [Piece::Queen, Piece::Rook, Piece::Bishop, Piece::Knight];

#[derive(Clone, Copy)]
struct Targets {
    len: u8,
    sq: [u8; 8],
}

const fn leaper_targets(deltas: [(i8, i8); 8]) -> [Targets; 64] {
    let mut table = [Targets { len: 0, sq: [0; 8] }; 64];
    let mut from = 0;
    while from < 64 {
        let mut i = 0;
        while i < 8 {
            let f = (from % 8) as i8 + deltas[i].0;
            let r = (from / 8) as i8 + deltas[i].1;
            if f >= 0 && f < 8 && r >= 0 && r < 8 {
                let n = table[from].len as usize;
                table[from].sq[n] = (r * 8 + f) as u8;
                table[from].len += 1;
            }
            i += 1;
        }
        from += 1;
    }
    table
}

const KNIGHT_TARGETS: [Targets; 64] = leaper_targets(KNIGHT_DELTAS);
const KING_TARGETS: [Targets; 64] = leaper_targets(KING_DELTAS);

pub fn pseudo_moves(board: &Board) -> Vec<Move> {
    generate(board, false)
}

pub fn legal_moves(board: &Board) -> Vec<Move> {
    let in_check = board.in_check(board.stm);
    pseudo_moves(board)
        .into_iter()
        .filter(|&m| is_legal(board, m, in_check))
        .collect()
}

/// Early-exit mate/stalemate probe: stops at the first legal move instead of
/// generating them all.
pub fn has_legal_move(board: &Board) -> bool {
    let in_check = board.in_check(board.stm);
    visit(board, false, &mut |m| is_legal(board, m, in_check))
}

fn is_legal(board: &Board, m: Move, in_check: bool) -> bool {
    let us = board.stm;
    let (_, piece) = board.squares[m.from as usize].expect("move from empty square");
    let is_ep = piece == Piece::Pawn && board.ep == Some(m.to) && m.from % 8 != m.to % 8;
    if in_check || is_ep {
        let mut child = board.clone();
        child.apply(m);
        return !child.is_attacked(child.kings[us.index()], child.stm);
    }
    if piece == Piece::King {
        return !board.is_attacked(m.to, us.flip());
    }
    !uncovers_king(board, m)
}

fn ray_dir_index(from: u8, to: u8) -> Option<usize> {
    let df = (to % 8) as i8 - (from % 8) as i8;
    let dr = (to / 8) as i8 - (from / 8) as i8;
    if df != 0 && dr != 0 && df.abs() != dr.abs() {
        return None;
    }
    Some(match (df.signum(), dr.signum()) {
        (0, 1) => dir::N,
        (0, -1) => dir::S,
        (1, 0) => dir::E,
        (-1, 0) => dir::W,
        (1, 1) => dir::NE,
        (-1, 1) => dir::NW,
        (1, -1) => dir::SE,
        _ => dir::SW,
    })
}

/// Whether a non-king, non-en-passant move made while not in check exposes
/// the mover's king to an enemy slider through the vacated from-square.
fn uncovers_king(board: &Board, m: Move) -> bool {
    let king = board.kings[board.stm.index()];
    let Some(d) = ray_dir_index(king, m.from) else {
        return false;
    };
    let them = board.stm.flip();
    let slider = if d < 4 { Piece::Rook } else { Piece::Bishop };
    let mut s = king as i32;
    for _ in 0..RAY_LEN[king as usize][d] {
        s += DIR_DELTAS[d] as i32;
        let sq = s as u8;
        if sq == m.from {
            continue;
        }
        if sq == m.to {
            return false;
        }
        if let Some((c, p)) = board.squares[s as usize] {
            return c == them && (p == slider || p == Piece::Queen);
        }
    }
    false
}

fn generate(board: &Board, captures_only: bool) -> Vec<Move> {
    let mut moves = Vec::with_capacity(48);
    visit(board, captures_only, &mut |m| {
        moves.push(m);
        false
    });
    moves
}

/// Feeds every pseudo-legal move (in the stable generation order) to `emit`
/// until it returns `true`; returns whether emission was stopped.
fn visit(board: &Board, captures_only: bool, emit: &mut impl FnMut(Move) -> bool) -> bool {
    let us = board.stm;
    for from in 0..64u8 {
        let Some((color, piece)) = board.squares[from as usize] else {
            continue;
        };
        if color != us {
            continue;
        }
        let stopped = match piece {
            Piece::Pawn => gen_pawn(board, from, us, captures_only, emit),
            Piece::Knight => gen_leaper(
                board,
                us,
                &KNIGHT_TARGETS[from as usize],
                from,
                captures_only,
                emit,
            ),
            Piece::King => {
                gen_leaper(
                    board,
                    us,
                    &KING_TARGETS[from as usize],
                    from,
                    captures_only,
                    emit,
                ) || (!captures_only && gen_castles(board, us, emit))
            }
            Piece::Bishop => gen_slider(board, from, us, &DIAG_RAYS, captures_only, emit),
            Piece::Rook => gen_slider(board, from, us, &ORTHO_RAYS, captures_only, emit),
            Piece::Queen => {
                gen_slider(board, from, us, &ORTHO_RAYS, captures_only, emit)
                    || gen_slider(board, from, us, &DIAG_RAYS, captures_only, emit)
            }
        };
        if stopped {
            return true;
        }
    }
    false
}

fn push_pawn_move(from: u8, to: u8, promotes: bool, emit: &mut impl FnMut(Move) -> bool) -> bool {
    if promotes {
        PROMO_PIECES.iter().any(|&p| {
            emit(Move {
                from,
                to,
                promo: Some(p),
            })
        })
    } else {
        emit(Move {
            from,
            to,
            promo: None,
        })
    }
}

fn gen_pawn(
    board: &Board,
    from: u8,
    us: Color,
    captures_only: bool,
    emit: &mut impl FnMut(Move) -> bool,
) -> bool {
    let f = (from % 8) as i8;
    let r = (from / 8) as i8;
    let (dir, start_rank, promo_rank): (i8, i8, i8) = match us {
        Color::White => (1, 1, 7),
        Color::Black => (-1, 6, 0),
    };

    if let Some(one) = square_at(f, r + dir)
        && board.squares[one as usize].is_none()
    {
        let promotes = r + dir == promo_rank;
        if (promotes || !captures_only) && push_pawn_move(from, one, promotes, emit) {
            return true;
        }
        if !captures_only
            && r == start_rank
            && let Some(two) = square_at(f, r + 2 * dir)
            && board.squares[two as usize].is_none()
            && emit(Move {
                from,
                to: two,
                promo: None,
            })
        {
            return true;
        }
    }

    for df in [-1, 1] {
        if let Some(to) = square_at(f + df, r + dir) {
            let stopped = match board.squares[to as usize] {
                Some((c, _)) if c != us => push_pawn_move(from, to, r + dir == promo_rank, emit),
                None if board.ep == Some(to) => emit(Move {
                    from,
                    to,
                    promo: None,
                }),
                _ => false,
            };
            if stopped {
                return true;
            }
        }
    }
    false
}

fn gen_leaper(
    board: &Board,
    us: Color,
    targets: &Targets,
    from: u8,
    captures_only: bool,
    emit: &mut impl FnMut(Move) -> bool,
) -> bool {
    for &to in &targets.sq[..targets.len as usize] {
        let movable = match board.squares[to as usize] {
            Some((c, _)) => c != us,
            None => !captures_only,
        };
        if movable
            && emit(Move {
                from,
                to,
                promo: None,
            })
        {
            return true;
        }
    }
    false
}

fn gen_slider(
    board: &Board,
    from: u8,
    us: Color,
    rays: &[usize; 4],
    captures_only: bool,
    emit: &mut impl FnMut(Move) -> bool,
) -> bool {
    for &d in rays {
        let mut s = from as i32;
        for _ in 0..RAY_LEN[from as usize][d] {
            s += DIR_DELTAS[d] as i32;
            let to = s as u8;
            match board.squares[s as usize] {
                Some((c, _)) => {
                    if c != us
                        && emit(Move {
                            from,
                            to,
                            promo: None,
                        })
                    {
                        return true;
                    }
                    break;
                }
                None => {
                    if !captures_only
                        && emit(Move {
                            from,
                            to,
                            promo: None,
                        })
                    {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn gen_castles(board: &Board, us: Color, emit: &mut impl FnMut(Move) -> bool) -> bool {
    let (king_sq, ks_right, qs_right) = match us {
        Color::White => (4u8, CASTLE_WK, CASTLE_WQ),
        Color::Black => (60u8, CASTLE_BK, CASTLE_BQ),
    };
    let them = us.flip();
    let base = king_sq as usize;

    if board.castling & ks_right != 0
        && board.squares[base + 1].is_none()
        && board.squares[base + 2].is_none()
        && board.squares[base + 3] == Some((us, Piece::Rook))
        && !board.is_attacked(king_sq, them)
        && !board.is_attacked(king_sq + 1, them)
        && !board.is_attacked(king_sq + 2, them)
        && emit(Move {
            from: king_sq,
            to: king_sq + 2,
            promo: None,
        })
    {
        return true;
    }

    if board.castling & qs_right != 0
        && board.squares[base - 1].is_none()
        && board.squares[base - 2].is_none()
        && board.squares[base - 3].is_none()
        && board.squares[base - 4] == Some((us, Piece::Rook))
        && !board.is_attacked(king_sq, them)
        && !board.is_attacked(king_sq - 1, them)
        && !board.is_attacked(king_sq - 2, them)
        && emit(Move {
            from: king_sq,
            to: king_sq - 2,
            promo: None,
        })
    {
        return true;
    }
    false
}

/// Leaf-node count of the legal move tree to `depth` — the standard
/// move-generator correctness metric.
pub fn perft(board: &Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let in_check = board.in_check(board.stm);
    let mut nodes = 0;
    for m in pseudo_moves(board) {
        if !is_legal(board, m, in_check) {
            continue;
        }
        nodes += if depth == 1 {
            1
        } else {
            let mut child = board.clone();
            child.apply(m);
            perft(&child, depth - 1)
        };
    }
    nodes
}

/// Per-root-move subtree counts, for drilling into perft mismatches.
pub fn perft_divide(board: &Board, depth: u32) -> Vec<(Move, u64)> {
    assert!(depth >= 1);
    legal_moves(board)
        .into_iter()
        .map(|m| {
            let mut child = board.clone();
            child.apply(m);
            (m, perft(&child, depth - 1))
        })
        .collect()
}
