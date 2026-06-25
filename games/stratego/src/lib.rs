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
//! Milestone 1 implements correct, tested, playable rules; milestone 2 adds the
//! in-step threat/evade/protection bitset geometry, the death-reason
//! classification, and the byte-exact [`encode`] tensor that feeds the neural
//! nets.
//!
//! ## Public surface
//! * [`board`] — `Board`, `Piece`, `PieceType`, `Color`, `DeathStatus`,
//!   `DeathReason`, the lake/count constants, `MoveSummary`, `is_adjacent`.
//! * [`action`] — `Action` and the 1800-slot encode/decode (`to_abs`/`from_abs`).
//! * [`arrangement`] — the A-M bijection, `Arrangement`, `DeploymentState`,
//!   `board_from_arrangements`, `is_terminal`.
//! * [`rules`] — `defender_wins`/`resolve`/`apply`, `legal_mask`,
//!   `has_legal_movement`, `is_terminal`, `reward_pl0`, the move limits.
//! * [`encode`] — the 355 board + 32 history + 256 piece-id `(92, 643)` move-net
//!   input (`EncoderConfig`, `encode_infostate`, `encode_tokens`).
//! * [`chase`], [`twosquare`] — the restriction state machines.
//! * [`game`] — the `Stratego` `Game`/`GameUi` impl, `State`, `Move`,
//!   `random_play_state`/`random_arrangement` (the move-phase start).
//! * [`bots`] — `HeuristicBot`, the competent material+belief baseline agent.

pub mod action;
pub mod arrangement;
pub mod board;
pub mod bots;
pub mod buffer;
pub mod chase;
pub mod encode;
pub mod evaluator;
pub mod game;
pub mod rules;
pub mod sim;
pub mod twosquare;

pub use action::{Action, NUM_ACTIONS};
pub use arrangement::{Arrangement, DeploymentState};
pub use board::{Board, Color, DeathReason, DeathStatus, Piece, PieceType};
pub use bots::HeuristicBot;
pub use buffer::{EncodedView, ReplayBuffer, SetupGame, Snapshot, Targets, Transition};
pub use encode::{
    EncoderConfig, NUM_BOARD_STATE_CHANNELS, NUM_OCCUPIABLE_CELLS, NUM_PIECE_ID, encode_infostate,
    encode_tokens, encode_tokens_batch,
};
pub use evaluator::{Decision, Evaluation, Evaluator, Phase, UniformEvaluator};
pub use game::{Move, State, Stratego};
pub use sim::{Arena, Collected, CommitResult, EnvDecision, RunStats, Simulator};

#[cfg(test)]
mod encode_tests;
#[cfg(test)]
mod sim_tests;
#[cfg(test)]
mod tests;
