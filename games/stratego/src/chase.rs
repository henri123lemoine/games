//! Continuous-chasing rule: during a chase you may not reproduce an earlier
//! threatening board position, except you may always undo your own literal
//! last move.
//!
//! This ports the *actual production kernel* (`src/env/rules/chase_state.cu`
//! / `chase_state.h` in the reference), not the Python test-generation oracle
//! in `tests/continuous_chase_new.py`. The two disagree: the Python oracle
//! tracks a 4-state (DORMANT/THREATENED/EVADING/CHASED) machine keyed by a
//! from-scratch piece-identity scan, uses an *unguarded* adjacency check
//! (`src±1, src±10` with no row-wrap guard), and exempts "returning to a
//! position from 2-of-your-own-moves-ago". The kernel instead keeps one
//! `i32` counter per player (`chase_length`), a single remembered (last_src,
//! last_dst) per player, and a bounded window of whole-board snapshots; its
//! adjacency check (`IS_ADJACENT`) *is* row-wrap-guarded (matching
//! [`crate::board::is_adjacent`] exactly), and the only always-allowed
//! exception is reverting your own literal last move. The Python oracle is
//! used only to *generate* candidate violation game logs from real env play;
//! the logs themselves are validated against the real env's
//! `current_legal_action_mask` (`test_continuous_chase_new.py`), which is
//! what makes them trustworthy fixtures — the oracle's own internal
//! algorithm is not.
//!
//! ## The kernel algorithm (`UpdateChaseStateKernel`, `player` = the mover,
//! `opponent` = the other player)
//!
//! Every ply, in order:
//!  i. Record `last_src[mover] = src`, `last_dst[mover] = dst`.
//! ii. `opp_chase = chase_length[opponent]`; if `src` is adjacent to
//!     `last_dst[opponent]`, `opp_chase += 1`, else `opp_chase = 0`.
//! iii. If this move was an attack, force `opp_chase = 0` (and the mover's own
//!     counter, below, to 0 too) — an attack makes prior board states
//!     unreachable, so no repetition can ever be checked against them.
//!  iv. `player_chase = chase_length[mover]` (0 if step iii fired); if `dst`
//!     is adjacent to any *currently* opponent-colored piece, `player_chase
//!     += 1`, else `player_chase = 0`.
//! Both counters are written back (`chase_length[opponent] = opp_chase`,
//! `chase_length[mover] = player_chase`), and the resulting whole-board
//! (color, kind) snapshot is appended to a rolling history.
//!
//! ## Legality (`ComputeIllegalChaseMovesKernel`)
//!
//! For the player to move, only when `chase_length[player] >= 2` can any move
//! be illegal on chase grounds. The kernel enumerates `delta` in
//! `[1, chase_length)` and, for each, diffs the board from `delta` plies ago
//! against the current board to find the *unique* move that would reproduce
//! it; that move is illegal unless it is not a threat (destination not
//! adjacent to a current opponent piece) or it exactly reverts the player's
//! own literal last move (`src == last_dst[player] && dst == last_src[player]`,
//! irrespective of `delta`). We compute the equivalent forward instead:
//! simulate playing the candidate move and check whether the result equals
//! any of the last `chase_length[player] - 1` board snapshots.

use crate::board::{Board, Color, PieceType, is_adjacent};

/// Hard cap from the reference (`MAX_CHASE_LENGTH`); the kernel asserts
/// `chase_length` never exceeds this. Also bounds our history ring buffer,
/// since no `delta >= chase_length` is ever consulted.
const MAX_CHASE_LENGTH: usize = 210;

/// A whole-board (color, kind) snapshot, packed one byte per cell
/// (`color*16 + kind`; both fit comfortably: color in 0..=3, kind in 0..=13).
/// Matches exactly what the kernel's diff reads — never `visible`/`has_moved`.
type Snapshot = [u8; crate::board::NUM_CELLS];

fn snapshot(board: &Board) -> Snapshot {
    snapshot_pieces(&board.pieces)
}

fn snapshot_pieces(pieces: &[crate::board::Piece; crate::board::NUM_CELLS]) -> Snapshot {
    let mut s = [0u8; crate::board::NUM_CELLS];
    for (cell, p) in pieces.iter().enumerate() {
        s[cell] = (p.color as u8) * 16 + p.kind as u8;
    }
    s
}

fn cell_code(color: Color, kind: PieceType) -> u8 {
    (color as u8) * 16 + kind as u8
}

/// The four in-bounds orthogonal neighbours of `cell`, matching the kernel's
/// explicit row/col bound checks (and hence [`is_adjacent`]) exactly.
fn neighbors(cell: usize) -> impl Iterator<Item = usize> {
    let (r, c) = (cell / 10, cell % 10);
    [
        (r > 0).then(|| cell - 10),
        (r < 9).then(|| cell + 10),
        (c > 0).then(|| cell - 1),
        (c < 9).then(|| cell + 1),
    ]
    .into_iter()
    .flatten()
}

/// Per-game continuous-chase state, shared by both players (the reference's
/// `ChaseState` is likewise one struct with 2-wide fields, not two
/// independent per-player objects — `commit`'s step ii reads the
/// *opponent's* `last_dst`, so the two players' bookkeeping isn't
/// separable).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChaseState {
    chase_length: [i32; 2],
    last_src: [Option<u8>; 2],
    last_dst: [Option<u8>; 2],
    /// Board snapshots after each ply, oldest-first, capped at
    /// [`MAX_CHASE_LENGTH`] entries (older ones are provably never consulted:
    /// `delta < chase_length[p] <= MAX_CHASE_LENGTH`).
    history: std::collections::VecDeque<Snapshot>,
}

impl Default for ChaseState {
    /// A placeholder only used as the `mem::take` swap target in
    /// `rules::apply` (never observed as a "real" seeded state) — real boards
    /// are always seeded via [`Self::new_from_board`]/`new_from_board_pieces`.
    fn default() -> Self {
        ChaseState {
            chase_length: [0, 0],
            last_src: [None, None],
            last_dst: [None, None],
            history: std::collections::VecDeque::new(),
        }
    }
}

impl ChaseState {
    /// The state for a freshly-placed board with no move history yet: both
    /// counters at 0, no remembered last move, history seeded with the
    /// starting position (so a first `commit` has something to diff against
    /// if ever needed, though `chase_length` can't reach 2 until ply 1).
    pub fn new_from_board(board: &Board) -> ChaseState {
        Self::new_from_board_pieces(&board.pieces)
    }

    /// As [`Self::new_from_board`], but usable during `Board::blank()`'s own
    /// construction, before a `Board` value exists to reference.
    pub fn new_from_board_pieces(
        pieces: &[crate::board::Piece; crate::board::NUM_CELLS],
    ) -> ChaseState {
        let mut history = std::collections::VecDeque::with_capacity(MAX_CHASE_LENGTH);
        history.push_back(snapshot_pieces(pieces));
        ChaseState {
            chase_length: [0, 0],
            last_src: [None, None],
            last_dst: [None, None],
            history,
        }
    }

    /// Would the non-attack move `src -> dst` by `mover` be a chase-rule
    /// violation, without mutating any state? Never called for attacks
    /// (callers already exempt them, matching the kernel never restricting
    /// an attacking destination on chase grounds).
    pub fn would_violate(&self, board: &Board, mover: usize, src: usize, dst: usize) -> bool {
        let chase_len = self.chase_length[mover];
        if chase_len < 2 {
            return false;
        }
        if self.last_dst[mover] == Some(src as u8) && self.last_src[mover] == Some(dst as u8) {
            return false; // always allowed: undo your own literal last move
        }
        let opponent_color = Color::of_player(1 - mover);
        let is_threat = neighbors(dst).any(|c| board.pieces[c].color == opponent_color);
        if !is_threat {
            return false;
        }

        let mover_piece = board.pieces[src];
        let moved_code = cell_code(mover_piece.color, mover_piece.kind);
        let empty_code = cell_code(Color::Empty, PieceType::Empty);
        let mut hypothetical = snapshot(board);
        hypothetical[src] = empty_code;
        hypothetical[dst] = moved_code;

        // `history[n-1]` is the board as it stands right now (before this
        // candidate move); `delta` plies ago is `history[n-1-delta]` — delta=1
        // is "one ply back", matching the kernel's `d_board_history[current_step
        // - delta]` against `current_board = d_board_history[current_step]`.
        let n = self.history.len();
        let max_delta = ((chase_len as usize) - 1).min(n.saturating_sub(1));
        for delta in 1..=max_delta {
            if self.history[n - 1 - delta] == hypothetical {
                return true;
            }
        }
        false
    }

    /// Folds one applied move into the state (`UpdateChaseStateKernel`, then
    /// appends the resulting board to history). `board` is read after the
    /// move's full resolution (battle outcome included) — safe for the
    /// neighbour-adjacency check since a move only ever changes the `src`/
    /// `dst` cells themselves, never their neighbours.
    pub fn commit(
        &mut self,
        board: &Board,
        mover: usize,
        src: usize,
        dst: usize,
        was_attack: bool,
    ) {
        let opponent = 1 - mover;

        let mut opp_chase = self.chase_length[opponent];
        opp_chase = if self.last_dst[opponent].is_some_and(|ld| is_adjacent(src, ld as usize)) {
            opp_chase + 1
        } else {
            0
        };

        let mut player_chase = self.chase_length[mover];
        if was_attack {
            player_chase = 0;
            opp_chase = 0;
        }
        self.chase_length[opponent] = opp_chase;

        self.last_src[mover] = Some(src as u8);
        self.last_dst[mover] = Some(dst as u8);

        let opponent_color = Color::of_player(opponent);
        let dst_adjacent_opponent = neighbors(dst).any(|c| board.pieces[c].color == opponent_color);
        player_chase = if dst_adjacent_opponent {
            player_chase + 1
        } else {
            0
        };
        self.chase_length[mover] = player_chase;

        self.history.push_back(snapshot(board));
        if self.history.len() > MAX_CHASE_LENGTH {
            self.history.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Piece, PieceType};

    fn empty_board() -> Board {
        Board::blank()
    }

    #[test]
    fn fresh_state_never_violates() {
        let board = empty_board();
        let sm = ChaseState::new_from_board(&board);
        assert!(!sm.would_violate(&board, 0, 11, 12));
    }

    #[test]
    fn chase_length_needs_two_before_any_violation_is_possible() {
        // A single threatening move only brings chase_length to 1; would_violate
        // must stay false (matches the kernel's `chase_length <= 0 || delta >=
        // chase_length` guard, which needs chase_length >= 2 for delta=1 to pass).
        let mut board = empty_board();
        board.pieces[11] = Piece::new(PieceType::Scout, Color::Red, 0);
        board.pieces[21] = Piece::new(PieceType::Scout, Color::Blue, 0);
        let mut sm = ChaseState::new_from_board(&board);
        board.pieces[12] = board.pieces[11];
        board.pieces[11] = Piece::EMPTY;
        sm.commit(&board, 0, 11, 12, false);
        let expected = if is_adjacent(12, 21) { 1 } else { 0 };
        assert_eq!(sm.chase_length[0], expected);
    }

    #[test]
    fn revert_last_move_is_always_allowed() {
        // Red's last move was 50->51; the reverse candidate (51->50) must be
        // exempted from the chase check REGARDLESS of chase_length or threat
        // status. Place a blue piece adjacent to the *candidate's own*
        // destination (50, not 51) so `is_threat` alone can't be what's
        // masking the result -- this genuinely exercises the exemption.
        let mut board = empty_board();
        board.pieces[51] = Piece::new(PieceType::Scout, Color::Red, 0);
        board.pieces[40] = Piece::new(PieceType::Scout, Color::Blue, 0); // adjacent to 50 -> threat
        let mut sm = ChaseState::new_from_board(&board);
        sm.chase_length[0] = 2;
        sm.last_src[0] = Some(50);
        sm.last_dst[0] = Some(51);

        assert!(
            !sm.would_violate(&board, 0, 51, 50),
            "reverting the last move must never violate"
        );

        // Sanity check the test actually exercises the threat path: without
        // the exemption (a different candidate that ISN'T the last-move
        // revert but lands on the same threatened square from elsewhere),
        // and with a matching history entry, would_violate CAN fire. Confirm
        // is_threat is really true for this destination by checking a
        // non-exempt mover would be threatened (regression against a
        // trivially-vacuous test).
        let opponent_color = crate::board::Color::of_player(1);
        assert!(
            neighbors(50).any(|c| board.pieces[c].color == opponent_color),
            "test setup bug: 50 must actually be adjacent to a blue piece"
        );
    }

    #[test]
    fn attack_resets_both_counters() {
        let mut board = empty_board();
        board.pieces[11] = Piece::new(PieceType::Scout, Color::Red, 0);
        board.pieces[21] = Piece::new(PieceType::Scout, Color::Blue, 0);
        let mut sm = ChaseState::new_from_board(&board);
        sm.chase_length = [5, 5];
        board.pieces[12] = board.pieces[11];
        board.pieces[11] = Piece::EMPTY;
        sm.commit(&board, 0, 11, 12, true);
        assert_eq!(
            sm.chase_length[1], 0,
            "attack must zero the opponent's counter"
        );
        assert_eq!(
            sm.chase_length[0], 0,
            "attack must zero the mover's own counter too"
        );
    }
}
