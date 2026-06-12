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
pub mod encode;
mod movegen;
mod search;
mod ui;

pub use board::{Board, Color, Move, Piece, START_FEN};
pub use movegen::{has_legal_move, legal_moves, perft, perft_divide};
pub use search::{ChessSpec, MaterialEval, RichEval, evaluate, rich_evaluate};

use game_core::hash::splitmix64;
use game_core::{Game, Turn};

/// Why a game is over. Checkmate outranks the draw rules: a mating move that
/// also reaches the 50-move boundary or a third repetition wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Adjudication {
    Checkmate { winner: Color },
    Stalemate,
    Repetition,
    FiftyMove,
    InsufficientMaterial,
}

impl Adjudication {
    /// Game score from White's perspective.
    pub fn white_score(self) -> f64 {
        match self {
            Adjudication::Checkmate {
                winner: Color::White,
            } => 1.0,
            Adjudication::Checkmate {
                winner: Color::Black,
            } => -1.0,
            _ => 0.0,
        }
    }
}

/// The one chess game-over rule: `repetitions` is how many times the current
/// position has occurred in the game (≥ 1; pass 1 when history is untracked).
/// Returns `None` while the game continues.
pub fn adjudicate(board: &Board, repetitions: u8) -> Option<Adjudication> {
    if !has_legal_move(board) {
        return Some(if board.in_check(board.stm) {
            Adjudication::Checkmate {
                winner: board.stm.flip(),
            }
        } else {
            Adjudication::Stalemate
        });
    }
    if repetitions >= 3 {
        Some(Adjudication::Repetition)
    } else if board.halfmove >= 100 {
        Some(Adjudication::FiftyMove)
    } else if board.insufficient_material() {
        Some(Adjudication::InsufficientMaterial)
    } else {
        None
    }
}

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
        adjudicate(state, 1).is_some()
    }

    fn returns(&self, state: &Board, player: usize) -> f64 {
        let score = adjudicate(state, 1).map_or(0.0, Adjudication::white_score);
        if player == 0 { score } else { -score }
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
        self.state_key(state).expect("chess has a state key")
    }

    /// [`Board::key`] plus the halfmove clock: value-equivalence for the
    /// search TT, where 50-move proximity changes the game value.
    fn state_key(&self, state: &Board) -> Option<u64> {
        Some(state.key() ^ splitmix64(0x4000 + u64::from(state.halfmove)))
    }

    fn repetition_key(&self, state: &Board) -> Option<u64> {
        Some(state.key())
    }

    fn action_id(&self, action: &Move) -> u64 {
        encode::move_index(*action) as u64
    }
}
