//! Pseudo-legal move generation with copy-make legality filtering, plus perft.

use crate::board::{
    Board, CASTLE_BK, CASTLE_BQ, CASTLE_WK, CASTLE_WQ, Color, Move, Piece, square_at,
};

const KNIGHT_DELTAS: [(i8, i8); 8] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
];
const KING_DELTAS: [(i8, i8); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];
const ORTHO_DIRS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const DIAG_DIRS: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
const PROMO_PIECES: [Piece; 4] = [Piece::Queen, Piece::Rook, Piece::Bishop, Piece::Knight];

pub fn pseudo_moves(board: &Board) -> Vec<Move> {
    generate(board, false)
}

pub fn legal_moves(board: &Board) -> Vec<Move> {
    let us = board.stm;
    pseudo_moves(board)
        .into_iter()
        .filter(|&m| {
            let mut child = board.clone();
            child.apply(m);
            !child.is_attacked(child.kings[us.index()], child.stm)
        })
        .collect()
}

fn generate(board: &Board, captures_only: bool) -> Vec<Move> {
    let mut moves = Vec::with_capacity(48);
    let us = board.stm;
    for from in 0..64u8 {
        let Some((color, piece)) = board.squares[from as usize] else {
            continue;
        };
        if color != us {
            continue;
        }
        match piece {
            Piece::Pawn => gen_pawn(board, from, us, captures_only, &mut moves),
            Piece::Knight => gen_leaper(board, from, us, &KNIGHT_DELTAS, captures_only, &mut moves),
            Piece::King => {
                gen_leaper(board, from, us, &KING_DELTAS, captures_only, &mut moves);
                if !captures_only {
                    gen_castles(board, us, &mut moves);
                }
            }
            Piece::Bishop => gen_slider(board, from, us, &DIAG_DIRS, captures_only, &mut moves),
            Piece::Rook => gen_slider(board, from, us, &ORTHO_DIRS, captures_only, &mut moves),
            Piece::Queen => {
                gen_slider(board, from, us, &ORTHO_DIRS, captures_only, &mut moves);
                gen_slider(board, from, us, &DIAG_DIRS, captures_only, &mut moves);
            }
        }
    }
    moves
}

fn push_pawn_move(from: u8, to: u8, promotes: bool, moves: &mut Vec<Move>) {
    if promotes {
        for p in PROMO_PIECES {
            moves.push(Move {
                from,
                to,
                promo: Some(p),
            });
        }
    } else {
        moves.push(Move {
            from,
            to,
            promo: None,
        });
    }
}

fn gen_pawn(board: &Board, from: u8, us: Color, captures_only: bool, moves: &mut Vec<Move>) {
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
        if promotes || !captures_only {
            push_pawn_move(from, one, promotes, moves);
        }
        if !captures_only
            && r == start_rank
            && let Some(two) = square_at(f, r + 2 * dir)
            && board.squares[two as usize].is_none()
        {
            moves.push(Move {
                from,
                to: two,
                promo: None,
            });
        }
    }

    for df in [-1, 1] {
        if let Some(to) = square_at(f + df, r + dir) {
            match board.squares[to as usize] {
                Some((c, _)) if c != us => {
                    push_pawn_move(from, to, r + dir == promo_rank, moves);
                }
                None if board.ep == Some(to) => {
                    moves.push(Move {
                        from,
                        to,
                        promo: None,
                    });
                }
                _ => {}
            }
        }
    }
}

fn gen_leaper(
    board: &Board,
    from: u8,
    us: Color,
    deltas: &[(i8, i8)],
    captures_only: bool,
    moves: &mut Vec<Move>,
) {
    let f = (from % 8) as i8;
    let r = (from / 8) as i8;
    for &(df, dr) in deltas {
        if let Some(to) = square_at(f + df, r + dr) {
            match board.squares[to as usize] {
                Some((c, _)) if c == us => {}
                Some(_) => moves.push(Move {
                    from,
                    to,
                    promo: None,
                }),
                None if !captures_only => moves.push(Move {
                    from,
                    to,
                    promo: None,
                }),
                None => {}
            }
        }
    }
}

fn gen_slider(
    board: &Board,
    from: u8,
    us: Color,
    dirs: &[(i8, i8)],
    captures_only: bool,
    moves: &mut Vec<Move>,
) {
    let f = (from % 8) as i8;
    let r = (from / 8) as i8;
    for &(df, dr) in dirs {
        let mut step = 1;
        while let Some(to) = square_at(f + df * step, r + dr * step) {
            match board.squares[to as usize] {
                Some((c, _)) => {
                    if c != us {
                        moves.push(Move {
                            from,
                            to,
                            promo: None,
                        });
                    }
                    break;
                }
                None => {
                    if !captures_only {
                        moves.push(Move {
                            from,
                            to,
                            promo: None,
                        });
                    }
                }
            }
            step += 1;
        }
    }
}

fn gen_castles(board: &Board, us: Color, moves: &mut Vec<Move>) {
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
    {
        moves.push(Move {
            from: king_sq,
            to: king_sq + 2,
            promo: None,
        });
    }

    if board.castling & qs_right != 0
        && board.squares[base - 1].is_none()
        && board.squares[base - 2].is_none()
        && board.squares[base - 3].is_none()
        && board.squares[base - 4] == Some((us, Piece::Rook))
        && !board.is_attacked(king_sq, them)
        && !board.is_attacked(king_sq - 1, them)
        && !board.is_attacked(king_sq - 2, them)
    {
        moves.push(Move {
            from: king_sq,
            to: king_sq - 2,
            promo: None,
        });
    }
}

/// Leaf-node count of the legal move tree to `depth` — the standard
/// move-generator correctness metric.
pub fn perft(board: &Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let us = board.stm;
    let mut nodes = 0;
    for m in pseudo_moves(board) {
        let mut child = board.clone();
        child.apply(m);
        if child.is_attacked(child.kings[us.index()], child.stm) {
            continue;
        }
        nodes += if depth == 1 {
            1
        } else {
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
