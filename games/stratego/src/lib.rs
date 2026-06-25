//! Stratego as a [`game_core::Game`] — a from-scratch, Ataraxos-faithful rules
//! engine (the classic 10x10 variant).
//!
//! The state is a cheap-to-clone packed [`Board`](board::Board) carrying every
//! field the reference rule kernels touch. Play has two phases: a serialized
//! per-square *deployment* (one piece-type placement per empty home square, in
//! row-major order, matching the setup net), then the alternating *move* phase
//! with the exact battle table, scout slides, reveal-on-attack, and the three
//! restriction/termination state machines (continuous-chasing, two-square,
//! k-move) plus the flag-capture / wipe-out / stuck / timeout reward.
//!
//! Milestone 1 implements correct, tested, playable rules. Deferred to
//! milestone 2 (fields/hooks present, marked `TODO(m2)`): the 643-channel
//! encoder tensor and the threat/evade/protection bitset update geometry +
//! death-reason classification that feed the neural nets.
//!
//! ## Public surface for the encoder (milestone 2)
//! * [`board`] — `Board`, `Piece`, `PieceType`, `Color`, `DeathStatus`,
//!   `DeathReason`, the lake/count constants, `MoveSummary`, `is_adjacent`.
//! * [`action`] — `Action` and the 1800-slot encode/decode (`to_abs`/`from_abs`).
//! * [`arrangement`] — the A-M bijection, `Arrangement`, `DeploymentState`,
//!   `board_from_arrangements`, `is_terminal`.
//! * [`rules`] — `defender_wins`/`resolve`/`apply`, `legal_mask`,
//!   `has_legal_movement`, `is_terminal`, `reward_pl0`, the move limits.
//! * [`chase`], [`twosquare`] — the restriction state machines.
//! * [`game`] — the `Stratego` `Game`/`GameUi` impl, `State`, `Move`.

pub mod action;
pub mod arrangement;
pub mod board;
pub mod chase;
pub mod game;
pub mod rules;
pub mod twosquare;

pub use action::{Action, NUM_ACTIONS};
pub use arrangement::{Arrangement, DeploymentState};
pub use board::{Board, Color, DeathReason, DeathStatus, Piece, PieceType};
pub use game::{Move, State, Stratego};

#[cfg(test)]
mod tests;
