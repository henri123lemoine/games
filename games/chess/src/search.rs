//! Chess search *knowledge*: evaluations (material + piece-square baseline,
//! and a richer tapered evaluation) and a search spec (captures/promotions are
//! noisy; MVV-LVA ordering). The search algorithm itself is the generic
//! `solvers::AlphaBeta`.

use crate::Chess;
use crate::board::{Board, Color, KNIGHT_DELTAS, Move, Piece, square_at};
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
/// tables and a bare-king mop-up term). Computed in centipawns
/// ([`evaluate`]), reported on the returns scale.
pub struct MaterialEval;

impl Eval<Chess> for MaterialEval {
    fn eval(&self, _game: &Chess, state: &Board, player: usize) -> f64 {
        let stm_score = evaluate(state) as f64;
        let cp = if state.stm.index() == player {
            stm_score
        } else {
            -stm_score
        };
        game_core::eval_squash(cp, 400.0)
    }
}

const PHASE_MAX: i32 = 24;

const PASSED_MG: [i32; 8] = [0, 5, 10, 20, 35, 60, 90, 0];
const PASSED_EG: [i32; 8] = [0, 15, 25, 40, 65, 110, 170, 0];
const DOUBLED_MG: i32 = 15;
const DOUBLED_EG: i32 = 20;
const ISOLATED_MG: i32 = 12;
const ISOLATED_EG: i32 = 8;
const ROOK_OPEN_MG: i32 = 25;
const ROOK_OPEN_EG: i32 = 15;
const ROOK_SEMI_MG: i32 = 12;
const ROOK_SEMI_EG: i32 = 8;
const BISHOP_PAIR_MG: i32 = 30;
const BISHOP_PAIR_EG: i32 = 45;
const SHIELD_NEAR: i32 = 12;
const SHIELD_FAR: i32 = 6;
const TEMPO_MG: i32 = 10;

fn piece_phase(p: Piece) -> i32 {
    match p {
        Piece::Knight | Piece::Bishop => 1,
        Piece::Rook => 2,
        Piece::Queen => 4,
        Piece::Pawn | Piece::King => 0,
    }
}

fn knight_mobility(board: &Board, sq: usize, color: Color) -> i32 {
    let f = (sq % 8) as i8;
    let r = (sq / 8) as i8;
    KNIGHT_DELTAS
        .iter()
        .filter(|(df, dr)| {
            square_at(f + df, r + dr).is_some_and(|t| match board.squares[t as usize] {
                None => true,
                Some((c, _)) => c != color,
            })
        })
        .count() as i32
}

fn slider_mobility(board: &Board, sq: usize, dirs: std::ops::Range<usize>, color: Color) -> i32 {
    use crate::board::{DIR_DELTAS, RAY_LEN};
    let mut n = 0;
    for d in dirs {
        let mut s = sq as i32;
        for _ in 0..RAY_LEN[sq][d] {
            s += DIR_DELTAS[d] as i32;
            match board.squares[s as usize] {
                None => n += 1,
                Some((c, _)) => {
                    if c != color {
                        n += 1;
                    }
                    break;
                }
            }
        }
    }
    n
}

/// Middlegame king-safety bonus: friendly pawns shielding a king that sits on
/// its back two ranks, on the king's file and adjacent files.
fn pawn_shield(board: &Board, color: Color) -> i32 {
    let k = board.kings[color.index()];
    let kf = (k % 8) as i8;
    let kr = (k / 8) as i8;
    let (rel_rank, dr) = match color {
        Color::White => (kr, 1i8),
        Color::Black => (7 - kr, -1i8),
    };
    if rel_rank > 1 {
        return 0;
    }
    let mut s = 0;
    for f in kf - 1..=kf + 1 {
        for (steps, bonus) in [(1, SHIELD_NEAR), (2, SHIELD_FAR)] {
            if let Some(sq) = square_at(f, kr + steps * dr)
                && board.squares[sq as usize] == Some((color, Piece::Pawn))
            {
                s += bonus;
            }
        }
    }
    s
}

/// A richer static evaluation in centipawns from the side-to-move's
/// perspective: tapered middlegame/endgame interpolation by material phase
/// over the same piece-square tables, plus pawn structure (doubled, isolated,
/// rank-scaled passed pawns), rooks on open/semi-open files, the bishop pair,
/// a middlegame pawn-shield king-safety term, cheap mobility, and a tempo bonus.
pub fn rich_evaluate(board: &Board) -> i32 {
    let mut file_pawns = [[0u8; 8]; 2];
    let mut max_pawn_rank = [[-1i8; 8]; 2];
    let mut min_pawn_rank = [[8i8; 8]; 2];
    let mut bishops = [0i32; 2];
    let mut material = [0i32; 2];
    let mut phase = 0i32;
    for (sq, cell) in board.squares.iter().enumerate() {
        let Some((c, p)) = *cell else { continue };
        let ci = c.index();
        material[ci] += p.value();
        phase += piece_phase(p);
        match p {
            Piece::Pawn => {
                let f = sq % 8;
                let r = (sq / 8) as i8;
                file_pawns[ci][f] += 1;
                max_pawn_rank[ci][f] = max_pawn_rank[ci][f].max(r);
                min_pawn_rank[ci][f] = min_pawn_rank[ci][f].min(r);
            }
            Piece::Bishop => bishops[ci] += 1,
            _ => {}
        }
    }
    let phase = phase.min(PHASE_MAX);

    let mut mg = 0i32;
    let mut eg = 0i32;
    for (sq, cell) in board.squares.iter().enumerate() {
        let Some((c, p)) = *cell else { continue };
        let ci = c.index();
        let sign = if c == Color::White { 1 } else { -1 };
        mg += sign * (p.value() + pst(p, c, sq, false));
        eg += sign * (p.value() + pst(p, c, sq, true));
        let f = (sq % 8) as i8;
        let r = (sq / 8) as i8;
        match p {
            Piece::Pawn => {
                let friends_on = |g: i8| -> u8 {
                    if (0..8).contains(&g) {
                        file_pawns[ci][g as usize]
                    } else {
                        0
                    }
                };
                if friends_on(f - 1) == 0 && friends_on(f + 1) == 0 {
                    mg -= sign * ISOLATED_MG;
                    eg -= sign * ISOLATED_EG;
                }
                let enemy = 1 - ci;
                let passed = [f - 1, f, f + 1]
                    .into_iter()
                    .filter(|g| (0..8).contains(g))
                    .all(|g| match c {
                        Color::White => max_pawn_rank[enemy][g as usize] <= r,
                        Color::Black => min_pawn_rank[enemy][g as usize] >= r,
                    });
                if passed {
                    let rr = match c {
                        Color::White => r,
                        Color::Black => 7 - r,
                    } as usize;
                    mg += sign * PASSED_MG[rr];
                    eg += sign * PASSED_EG[rr];
                }
            }
            Piece::Knight => {
                let m = knight_mobility(board, sq, c);
                mg += sign * m * 4;
                eg += sign * m * 4;
            }
            Piece::Bishop => {
                let m = slider_mobility(board, sq, 4..8, c);
                mg += sign * m * 4;
                eg += sign * m * 5;
            }
            Piece::Rook => {
                let m = slider_mobility(board, sq, 0..4, c);
                mg += sign * m * 2;
                eg += sign * m * 4;
                let fu = f as usize;
                if file_pawns[ci][fu] == 0 {
                    if file_pawns[1 - ci][fu] == 0 {
                        mg += sign * ROOK_OPEN_MG;
                        eg += sign * ROOK_OPEN_EG;
                    } else {
                        mg += sign * ROOK_SEMI_MG;
                        eg += sign * ROOK_SEMI_EG;
                    }
                }
            }
            Piece::Queen => {
                let m = slider_mobility(board, sq, 0..8, c);
                mg += sign * m;
                eg += sign * m * 2;
            }
            Piece::King => {}
        }
    }

    for c in [Color::White, Color::Black] {
        let ci = c.index();
        let sign = if c == Color::White { 1 } else { -1 };
        if bishops[ci] >= 2 {
            mg += sign * BISHOP_PAIR_MG;
            eg += sign * BISHOP_PAIR_EG;
        }
        for count in file_pawns[ci] {
            let extra = i32::from(count.saturating_sub(1));
            mg -= sign * extra * DOUBLED_MG;
            eg -= sign * extra * DOUBLED_EG;
        }
        mg += sign * pawn_shield(board, c);
    }

    let mut white = (mg * phase + eg * (PHASE_MAX - phase)) / PHASE_MAX;

    for loser in [Color::White, Color::Black] {
        let winner = loser.flip();
        if material[loser.index()] == 0 && material[winner.index()] >= Piece::Rook.value() {
            let mopup = 10 * center_distance(board.kings[loser.index()])
                + 4 * (14 - king_distance(board.kings[0], board.kings[1]));
            white += if winner == Color::White {
                mopup
            } else {
                -mopup
            };
        }
    }

    let stm = match board.stm {
        Color::White => white,
        Color::Black => -white,
    };
    stm + TEMPO_MG * phase / PHASE_MAX
}

/// [`Eval`] for chess: the tapered [`rich_evaluate`] (pawn structure, rook
/// files, bishop pair, king shield, mobility). Computed in centipawns,
/// reported on the returns scale.
pub struct RichEval;

impl Eval<Chess> for RichEval {
    fn eval(&self, _game: &Chess, state: &Board, player: usize) -> f64 {
        let stm_score = rich_evaluate(state) as f64;
        let cp = if state.stm.index() == player {
            stm_score
        } else {
            -stm_score
        };
        game_core::eval_squash(cp, 400.0)
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
