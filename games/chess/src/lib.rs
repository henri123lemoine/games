//! Chess as a [`game_core::Game`].
//!
//! Perfect information, no chance nodes: `turn` is always a player decision
//! and `infoset_key` equals the canonical position key. Move generation is
//! pseudo-legal generation plus copy-make legality filtering (clone, apply,
//! reject if the mover's king is attacked), which handles pins, castling
//! through check, and the discovered-check en-passant pitfall uniformly. It is
//! validated against the standard perft suite in `tests/perft.rs`.
//!
//! Draw rules covered: stalemate, the 50-move rule, and insufficient material
//! for K vs K and K + single minor vs K. Intentionally skipped for simplicity:
//! threefold repetition and richer dead-position cases (e.g. K+B vs K+B with
//! same-colored bishops).

mod board;
mod movegen;
mod search;
mod ui;

pub use board::{Board, Color, Move, Piece, START_FEN};
pub use movegen::{legal_moves, perft, perft_divide};
pub use search::{ChessSpec, MaterialEval, evaluate};

use game_core::{Game, Turn};

/// White is player 0, Black is player 1.
pub struct Chess;

impl Game for Chess {
    type State = Board;
    type Action = Move;

    fn initial_state(&self) -> Board {
        Board::start()
    }

    fn turn(&self, state: &Board) -> Turn {
        Turn::Player(state.stm.index())
    }

    fn is_terminal(&self, state: &Board) -> bool {
        state.halfmove >= 100 || state.insufficient_material() || legal_moves(state).is_empty()
    }

    fn returns(&self, state: &Board, player: usize) -> f64 {
        if legal_moves(state).is_empty() && state.in_check(state.stm) {
            let loser = state.stm.index();
            if player == loser { -1.0 } else { 1.0 }
        } else {
            0.0
        }
    }

    fn legal_actions(&self, state: &Board) -> Vec<Move> {
        legal_moves(state)
    }

    fn chance_outcomes(&self, _state: &Board) -> Vec<(Move, f64)> {
        vec![]
    }

    fn apply(&self, state: &mut Board, action: Move) {
        state.apply(action);
    }

    fn infoset_key(&self, state: &Board, _player: usize) -> u64 {
        state.key()
    }

    fn state_key(&self, state: &Board) -> Option<u64> {
        Some(state.key())
    }
}
