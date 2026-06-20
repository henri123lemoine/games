//! Generic AlphaZero trainer: one config-driven tch resnet plus a shared
//! self-play / replay / optimizer / run-dir harness, parameterized by a
//! `(Game, PolicyValueEncoder, NetConfig)` triple. Unifies the `azt` (chess),
//! `azgo` (go), and `azsnake` (snake) trainers — the same algorithm written
//! once, the game knowledge plugged in per binary (`src/bin/{chess,go,snake}`).
//!
//! Standalone (empty `[workspace]`) on purpose: keeps libtorch off the main
//! workspace's `cargo test`.

pub mod export;
pub mod net;
pub mod train;
pub mod verify;

pub use net::{EvalRequest, EvalResult, Infer, Net, NetConfig};
pub use nn_infer::{HeadFlags, HeadKind};
pub use train::{Batch, OptConfig, Replay, TrainSample, Trainer};
