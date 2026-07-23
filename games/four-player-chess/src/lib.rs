//! Chess.com-style four-player chess, standard free-for-all scoring.
//!
//! The external rules contract is the Chess.com Help Center article
//! “4 Player Chess (4PC)”, dated 2025-10-10. The implementation is the
//! standard 14×14 / 160-square FFA game: Red, Blue, Yellow, Green move in that
//! order; pawns promote automatically to one-point queens on their eighth
//! rank; active captures and multi-checks score; checkmate, self-stalemate,
//! and king capture eliminate an army; dead pieces remain as inert blockers;
//! final placement is by points. See the crate README for the exact rules and
//! the deliberately out-of-band clock/resignation boundary.

mod agents;
mod board;
pub mod encode;
mod ui;

pub use agents::{GreedyAgent, MobilityAgent};
pub use board::{
    CELLS, Color, EndReason, Move, NONE_SQUARE, Piece, PieceKind, SIDE, State, is_valid_xy,
    parse_square, square, square_name, xy,
};

use game_core::{Game, Turn};

use board::{add, castle_bit, castle_step, home_king, home_rook};

/// Exact site rules have no arbitrary turn cap. Training/fuzzing can opt into
/// one to keep episodes bounded; capped games are ranked by their current score.
#[derive(Debug, Clone, Copy)]
pub struct FourPlayerChess {
    pub ply_cap: u16,
}

impl Default for FourPlayerChess {
    fn default() -> Self {
        FourPlayerChess { ply_cap: u16::MAX }
    }
}

impl FourPlayerChess {
    pub const fn with_ply_cap(ply_cap: u16) -> FourPlayerChess {
        FourPlayerChess { ply_cap }
    }

    pub fn legal_moves(&self, state: &State) -> Vec<Move> {
        legal_moves_for(state, state.to_move)
    }

    pub fn in_check(&self, state: &State, color: Color) -> bool {
        state
            .king_square(color)
            .is_some_and(|king| is_attacked(state, king, color))
    }

    /// Applies a known-legal move and returns points gained by the mover.
    pub fn apply_move(&self, state: &mut State, action: Move) -> i16 {
        debug_assert!(self.legal_moves(state).contains(&action));
        let actor = state.to_move;
        let before_score = state.scores[actor.index()];
        let outcome = apply_board_move(state, action);

        // Active captured pieces score; dead pieces are inert zero-point blockers.
        for captured in outcome.captured.into_iter().flatten() {
            if state.is_active(captured.color()) {
                state.scores[actor.index()] += captured.kind().score(captured.promoted());
                if captured.kind() == PieceKind::King {
                    eliminate(state, captured.color());
                }
            }
        }
        update_check_credit(state, actor);

        // A move that leaves two/three opponents simultaneously checked by
        // this army earns the documented bonus. Include discovered checks and
        // the rook moved by castling, not only attacks from `action.to`.
        let checked = Color::ALL
            .into_iter()
            .filter(|&color| color != actor && state.is_active(color))
            .filter(|&color| {
                state
                    .king_square(color)
                    .is_some_and(|king| is_attacked_by(state, king, actor))
            })
            .count();
        if checked >= 2 {
            let queen = outcome.moved.kind() == PieceKind::Queen;
            state.scores[actor.index()] += match (checked, queen) {
                (2, true) => 1,
                (2, false) => 5,
                (_, true) => 5,
                (_, false) => 20,
            };
        }

        state.ply = state.ply.saturating_add(1);
        state.last_move = Some(action);
        state.halfmove = if outcome.irreversible {
            0
        } else {
            state.halfmove.saturating_add(1)
        };

        advance_and_eliminate(state, actor);
        if state.active.count_ones() <= 1 {
            state.end = EndReason::LastArmy;
        }

        // Mate/stalemate resolution outranks draw rules, as in ordinary chess.
        if state.end == EndReason::Ongoing {
            let repetitions = state.record_position(outcome.irreversible);
            if repetitions >= 3 {
                award_draw_points(state, EndReason::Repetition);
            } else if state.halfmove >= 200 {
                // 50 complete four-player rounds = 200 individual plies.
                award_draw_points(state, EndReason::FiftyMove);
            } else if insufficient_material(state) {
                award_draw_points(state, EndReason::InsufficientMaterial);
            } else if state.ply >= self.ply_cap {
                state.end = EndReason::PlyCap;
            }
        }

        state.scores[actor.index()] - before_score
    }
}

impl Game for FourPlayerChess {
    type State = State;
    type Action = Move;

    fn num_players(&self) -> usize {
        4
    }

    fn initial_state(&self) -> State {
        State::standard()
    }

    fn turn(&self, state: &State) -> Turn {
        Turn::Player(state.to_move.index())
    }

    fn is_terminal(&self, state: &State) -> bool {
        state.end != EndReason::Ongoing
    }

    fn returns(&self, state: &State, player: usize) -> f64 {
        debug_assert!(self.is_terminal(state));
        let best = *state.scores.iter().max().expect("four scores");
        let leaders = state.scores.iter().filter(|&&score| score == best).count();
        let win_share = if state.scores[player] == best {
            1.0 / leaders as f64
        } else {
            0.0
        };
        // The value head predicts a categorical first-place share. Affinely
        // center that distribution at the fair 25% baseline: sole winner +1,
        // no share -1/3, with tied first place split before centering.
        (4.0 * win_share - 1.0) / 3.0
    }

    fn legal_actions(&self, state: &State) -> Vec<Move> {
        self.legal_moves(state)
    }

    fn chance_outcomes(&self, _state: &State) -> Vec<(Move, f64)> {
        Vec::new()
    }

    fn apply(&self, state: &mut State, action: Move) {
        self.apply_move(state, action);
    }

    fn infoset_key(&self, state: &State, _player: usize) -> u64 {
        state.state_key()
    }

    fn state_key(&self, state: &State) -> Option<u64> {
        Some(state.state_key())
    }

    fn repetition_key(&self, state: &State) -> Option<u64> {
        Some(state.repetition_key())
    }

    fn action_id(&self, action: &Move) -> u64 {
        u64::from(action.from) * CELLS as u64 + u64::from(action.to)
    }
}

impl game_core::ScoreShare for FourPlayerChess {
    fn score_share(&self, state: &State, player: usize) -> f64 {
        debug_assert!(self.is_terminal(state));
        let total: i32 = state.scores.iter().map(|&score| i32::from(score)).sum();
        if total > 0 {
            f64::from(state.scores[player]) / f64::from(total)
        } else {
            0.25
        }
    }
}

#[derive(Clone, Copy)]
struct MoveOutcome {
    moved: Piece,
    captured: [Option<Piece>; 2],
    irreversible: bool,
}

fn legal_moves_for(state: &State, color: Color) -> Vec<Move> {
    if !state.is_active(color) || state.end != EndReason::Ongoing {
        return Vec::new();
    }
    let mut pseudo = Vec::with_capacity(96);
    for square in 0..CELLS as u8 {
        let piece = state.board[square as usize];
        if piece.is_empty() || piece.color() != color {
            continue;
        }
        pseudo_moves(state, square, piece, &mut pseudo);
    }
    pseudo
        .into_iter()
        .filter(|&action| {
            let mut next = state.clone();
            let outcome = apply_board_move(&mut next, action);
            // Capturing a live king eliminates its whole army immediately.
            // Its remaining pieces are therefore inert and cannot make the
            // capturing move look self-checking in this legality probe.
            for captured in outcome.captured.into_iter().flatten() {
                if captured.kind() == PieceKind::King {
                    eliminate(&mut next, captured.color());
                }
            }
            next.king_square(color)
                .is_some_and(|king| !is_attacked(&next, king, color))
        })
        .collect()
}

fn pseudo_moves(state: &State, from: u8, piece: Piece, moves: &mut Vec<Move>) {
    match piece.kind() {
        PieceKind::Pawn => pawn_moves(state, from, piece, moves),
        PieceKind::Knight => {
            for (dx, dy) in [
                (-2, -1),
                (-2, 1),
                (-1, -2),
                (-1, 2),
                (1, -2),
                (1, 2),
                (2, -1),
                (2, 1),
            ] {
                push_step(state, from, piece.color(), dx, dy, moves);
            }
        }
        PieceKind::Bishop => slide_moves(
            state,
            from,
            piece.color(),
            &[(-1, -1), (-1, 1), (1, -1), (1, 1)],
            moves,
        ),
        PieceKind::Rook => slide_moves(
            state,
            from,
            piece.color(),
            &[(-1, 0), (1, 0), (0, -1), (0, 1)],
            moves,
        ),
        PieceKind::Queen => slide_moves(
            state,
            from,
            piece.color(),
            &[
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
            ],
            moves,
        ),
        PieceKind::King => {
            for dx in -1..=1 {
                for dy in -1..=1 {
                    if dx != 0 || dy != 0 {
                        push_step(state, from, piece.color(), dx, dy, moves);
                    }
                }
            }
            castle_moves(state, piece.color(), moves);
        }
    }
}

fn pawn_moves(state: &State, from: u8, piece: Piece, moves: &mut Vec<Move>) {
    let color = piece.color();
    let (fx, fy) = color.forward();
    let (rx, ry) = color.right();
    if let Some(to) = add(from, fx, fy)
        && state.board[to as usize].is_empty()
    {
        moves.push(Move::new(from, to));
        if pawn_on_start(from, color)
            && let Some(two) = add(to, fx, fy)
            && state.board[two as usize].is_empty()
        {
            moves.push(Move::new(from, two));
        }
    }
    for side in [-1, 1] {
        let dx = fx + side * rx;
        let dy = fy + side * ry;
        if let Some(to) = add(from, dx, dy) {
            let target = state.board[to as usize];
            if !target.is_empty() && target.color() != color {
                moves.push(Move::new(from, to));
            }
        }
    }

    // Four-player en passant: a perpendicular enemy double-push may finish
    // immediately in front of this pawn. Every opponent gets their next turn
    // to take it, and the transit square may meanwhile contain another enemy.
    let Some(front) = add(from, fx, fy) else {
        return;
    };
    let front_piece = state.board[front as usize];
    if front_piece.is_empty()
        || front_piece.kind() != PieceKind::Pawn
        || front_piece.color() == color
        || state.en_passant[front_piece.color().index()] != front
    {
        return;
    }
    let (ofx, ofy) = front_piece.color().forward();
    let Some(transit) = add(front, -ofx, -ofy) else {
        return;
    };
    let (from_x, from_y) = xy(from);
    let (to_x, to_y) = xy(transit);
    let delta = (to_x - from_x, to_y - from_y);
    if (delta == (fx + rx, fy + ry) || delta == (fx - rx, fy - ry))
        && state.board[transit as usize]
            .color_if_present()
            .is_none_or(|owner| owner != color)
    {
        moves.push(Move::new(from, transit));
    }
}

trait PieceExt {
    fn color_if_present(self) -> Option<Color>;
}

impl PieceExt for Piece {
    fn color_if_present(self) -> Option<Color> {
        (!self.is_empty()).then(|| self.color())
    }
}

fn pawn_on_start(square: u8, color: Color) -> bool {
    let (x, y) = xy(square);
    match color {
        Color::Red => y == 1,
        Color::Blue => x == 1,
        Color::Yellow => y == 12,
        Color::Green => x == 12,
    }
}

fn pawn_promotes(square: u8, color: Color) -> bool {
    let (x, y) = xy(square);
    match color {
        Color::Red => y >= 7,
        Color::Blue => x >= 7,
        Color::Yellow => y <= 6,
        Color::Green => x <= 6,
    }
}

fn push_step(state: &State, from: u8, color: Color, dx: i8, dy: i8, moves: &mut Vec<Move>) {
    let Some(to) = add(from, dx, dy) else {
        return;
    };
    let target = state.board[to as usize];
    if target.is_empty() || target.color() != color {
        moves.push(Move::new(from, to));
    }
}

fn slide_moves(state: &State, from: u8, color: Color, dirs: &[(i8, i8)], moves: &mut Vec<Move>) {
    for &(dx, dy) in dirs {
        let mut at = from;
        while let Some(to) = add(at, dx, dy) {
            let target = state.board[to as usize];
            if target.is_empty() {
                moves.push(Move::new(from, to));
                at = to;
            } else {
                if target.color() != color {
                    moves.push(Move::new(from, to));
                }
                break;
            }
        }
    }
}

fn castle_moves(state: &State, color: Color, moves: &mut Vec<Move>) {
    let king = home_king(color);
    if state.board[king as usize] != Piece::new(color, PieceKind::King)
        || is_attacked(state, king, color)
    {
        return;
    }
    for king_side in [true, false] {
        if state.castling & castle_bit(color, king_side) == 0 {
            continue;
        }
        let (step_x, step_y) = castle_step(color, king_side);
        let rook = home_rook(color, king_side);
        if state.board[rook as usize] != Piece::new(color, PieceKind::Rook) {
            continue;
        }
        let distance = if king_side { 3 } else { 4 };
        let mut clear = true;
        for step in 1..distance {
            let sq = add(king, step * step_x, step * step_y).expect("home rank");
            if !state.board[sq as usize].is_empty() {
                clear = false;
                break;
            }
        }
        if !clear {
            continue;
        }
        let through = add(king, step_x, step_y).expect("castle through");
        let to = add(king, 2 * step_x, 2 * step_y).expect("castle to");
        if !is_attacked(state, through, color) && !is_attacked(state, to, color) {
            moves.push(Move::new(king, to));
        }
    }
}

fn apply_board_move(state: &mut State, action: Move) -> MoveOutcome {
    let actor = state.to_move;
    let mut moved = state.board[action.from as usize];
    debug_assert!(!moved.is_empty() && moved.color() == actor);
    let was_pawn = moved.kind() == PieceKind::Pawn;
    let mut captured = [None, None];
    let target = state.board[action.to as usize];
    if !target.is_empty() {
        captured[0] = Some(target);
    }

    state.en_passant[actor.index()] = NONE_SQUARE;
    state.board[action.from as usize] = Piece::EMPTY;
    state.board[action.to as usize] = moved;

    if moved.kind() == PieceKind::Pawn {
        let (fx, fy) = actor.forward();
        if let Some(front) = add(action.from, fx, fy) {
            let front_piece = state.board[front as usize];
            if !front_piece.is_empty()
                && front_piece.kind() == PieceKind::Pawn
                && front_piece.color() != actor
                && state.en_passant[front_piece.color().index()] == front
            {
                let (ofx, ofy) = front_piece.color().forward();
                if add(front, -ofx, -ofy) == Some(action.to) {
                    captured[1] = Some(front_piece);
                    state.board[front as usize] = Piece::EMPTY;
                }
            }
        }
        if add(action.from, 2 * fx, 2 * fy) == Some(action.to) {
            state.en_passant[actor.index()] = action.to;
        }
        if pawn_promotes(action.to, actor) {
            moved = Piece::promoted_queen(actor);
            state.board[action.to as usize] = moved;
        }
    }

    if moved.kind() == PieceKind::King {
        state.castling &= !(castle_bit(actor, true) | castle_bit(actor, false));
        let (king_x, king_y) = castle_step(actor, true);
        let (queen_x, queen_y) = castle_step(actor, false);
        let king_side = add(action.from, 2 * king_x, 2 * king_y) == Some(action.to);
        let queen_side = add(action.from, 2 * queen_x, 2 * queen_y) == Some(action.to);
        if action.from == home_king(actor) && (king_side || queen_side) {
            let rook_from = home_rook(actor, king_side);
            let (step_x, step_y) = castle_step(actor, king_side);
            let rook_to = add(action.from, step_x, step_y).expect("rook destination");
            state.board[rook_to as usize] = state.board[rook_from as usize];
            state.board[rook_from as usize] = Piece::EMPTY;
        }
    } else if moved.kind() == PieceKind::Rook {
        for king_side in [true, false] {
            if action.from == home_rook(actor, king_side) {
                state.castling &= !castle_bit(actor, king_side);
            }
        }
    }
    for victim in Color::ALL {
        for king_side in [true, false] {
            if action.to == home_rook(victim, king_side)
                && captured
                    .into_iter()
                    .flatten()
                    .any(|piece| piece.color() == victim && piece.kind() == PieceKind::Rook)
            {
                state.castling &= !castle_bit(victim, king_side);
            }
        }
    }

    MoveOutcome {
        moved,
        captured,
        // `moved` may now be a promoted queen, but the pawn move still resets
        // the 50-round counter and repetition window.
        irreversible: was_pawn || captured.iter().any(Option::is_some),
    }
}

fn eliminate(state: &mut State, color: Color) {
    state.active &= !(1 << color.index());
    state.castling &= !(castle_bit(color, true) | castle_bit(color, false));
    state.en_passant[color.index()] = NONE_SQUARE;
}

fn advance_and_eliminate(state: &mut State, scorer: Color) {
    let mut candidate = (state.to_move.index() + 1) % 4;
    for _ in 0..4 {
        let color = Color::from_index(candidate);
        if !state.is_active(color) {
            candidate = (candidate + 1) % 4;
            continue;
        }
        state.to_move = color;
        let moves = legal_moves_for(state, color);
        if !moves.is_empty() {
            return;
        }
        if state
            .king_square(color)
            .is_some_and(|king| is_attacked(state, king, color))
        {
            let credit = state.check_credit[color.index()];
            let checkmating = (credit != NONE_SQUARE)
                .then(|| Color::from_index(credit as usize))
                .filter(|&owner| state.is_active(owner))
                .unwrap_or(scorer);
            state.scores[checkmating.index()] += 20;
        } else {
            state.scores[color.index()] += 20;
        }
        eliminate(state, color);
        update_check_credit(state, scorer);
        if state.active.count_ones() <= 1 {
            return;
        }
        candidate = (candidate + 1) % 4;
    }
}

fn update_check_credit(state: &mut State, actor: Color) {
    for victim in Color::ALL {
        if !state.is_active(victim) {
            state.check_credit[victim.index()] = NONE_SQUARE;
            continue;
        }
        let Some(king) = state.king_square(victim) else {
            state.check_credit[victim.index()] = NONE_SQUARE;
            continue;
        };
        let mut attackers = [false; 4];
        for (from, &piece) in state.board.iter().enumerate() {
            if !piece.is_empty()
                && piece.color() != victim
                && state.is_active(piece.color())
                && piece_attacks(state, from as u8, piece, king)
            {
                attackers[piece.color().index()] = true;
            }
        }
        let previous = state.check_credit[victim.index()];
        let credit = if attackers[actor.index()] {
            actor.index()
        } else if previous != NONE_SQUARE && attackers[previous as usize] {
            previous as usize
        } else if let Some(owner) = attackers.iter().position(|&checking| checking) {
            owner
        } else {
            state.check_credit[victim.index()] = NONE_SQUARE;
            continue;
        };
        state.check_credit[victim.index()] = credit as u8;
    }
}

fn award_draw_points(state: &mut State, reason: EndReason) {
    for color in Color::ALL {
        if state.is_active(color) {
            state.scores[color.index()] += 10;
        }
    }
    state.end = reason;
}

fn insufficient_material(state: &State) -> bool {
    !state.board.iter().any(|&piece| {
        !piece.is_empty()
            && state.is_active(piece.color())
            && !matches!(piece.kind(), PieceKind::King)
    })
}

fn is_attacked(state: &State, target: u8, defender: Color) -> bool {
    state.board.iter().enumerate().any(|(from, &piece)| {
        !piece.is_empty()
            && piece.color() != defender
            && state.is_active(piece.color())
            && piece_attacks(state, from as u8, piece, target)
    })
}

fn is_attacked_by(state: &State, target: u8, attacker: Color) -> bool {
    state.board.iter().enumerate().any(|(from, &piece)| {
        !piece.is_empty()
            && piece.color() == attacker
            && state.is_active(attacker)
            && piece_attacks(state, from as u8, piece, target)
    })
}

fn piece_attacks(state: &State, from: u8, piece: Piece, target: u8) -> bool {
    let (from_x, from_y) = xy(from);
    let (to_x, to_y) = xy(target);
    let dx = to_x - from_x;
    let dy = to_y - from_y;
    match piece.kind() {
        PieceKind::Pawn => {
            let (fx, fy) = piece.color().forward();
            let (rx, ry) = piece.color().right();
            (dx, dy) == (fx + rx, fy + ry) || (dx, dy) == (fx - rx, fy - ry)
        }
        PieceKind::Knight => matches!((dx.abs(), dy.abs()), (1, 2) | (2, 1)),
        PieceKind::King => dx.abs().max(dy.abs()) == 1,
        PieceKind::Bishop => dx.abs() == dy.abs() && ray_clear(state, from, target, dx, dy),
        PieceKind::Rook => (dx == 0 || dy == 0) && ray_clear(state, from, target, dx, dy),
        PieceKind::Queen => {
            (dx == 0 || dy == 0 || dx.abs() == dy.abs()) && ray_clear(state, from, target, dx, dy)
        }
    }
}

fn ray_clear(state: &State, from: u8, target: u8, dx: i8, dy: i8) -> bool {
    let step_x = dx.signum();
    let step_y = dy.signum();
    let mut at = from;
    while let Some(next) = add(at, step_x, step_y) {
        if next == target {
            return true;
        }
        if !state.board[next as usize].is_empty() {
            return false;
        }
        at = next;
    }
    false
}

#[cfg(test)]
mod tests;
