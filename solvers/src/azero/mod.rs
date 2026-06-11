//! AlphaZero-style self-play learning, CPU-only and dependency-free (rayon
//! aside): a tiny pure-Rust MLP with manual backprop, PUCT search guided by
//! it, and a self-play / train loop.
//!
//! The pieces, and what a game must provide:
//!
//! * [`PolicyValueEncoder`] — game knowledge: a flat `f32` encoding of states
//!   and a dense, injective index for actions in a fixed policy space.
//! * [`Mlp`] — input → two ReLU hidden layers → policy logits (softmax over
//!   the *legal* subset only) + scalar tanh value. He init, [`SgdMomentum`],
//!   versioned binary save/load with atomic rename.
//! * [`Puct`] — one net evaluation per expanded node, optional Dirichlet root
//!   noise; [`PuctAgent`] adapts it to the arena's [`game_core::Agent`].
//! * [`SelfPlayTrainer`] — plays games in parallel, stores
//!   (encoding, visit distribution, outcome) [`Sample`]s in a replay buffer,
//!   trains with policy cross-entropy + value MSE + L2.
//!
//! Two-player zero-sum games only — the scalar value head assumes it.

mod mlp;
mod puct;
mod rand;
#[cfg(feature = "parallel")]
mod train;

pub use mlp::{InferCache, Mlp, Sample, SgdMomentum};
pub use puct::{PolicyValueEncoder, Puct, PuctAgent};
#[cfg(feature = "parallel")]
pub use train::{AzeroConfig, IterStats, SelfPlayTrainer};
