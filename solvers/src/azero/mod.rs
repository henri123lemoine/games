//! AlphaZero-style policy-value search and self-play learning.
//!
//! The pieces, and what a game must provide:
//!
//! * [`PolicyValueEncoder`] — game knowledge: a flat `f32` encoding of states
//!   and a dense, injective index for actions in a fixed policy space.
//! * [`Search`] — the one PUCT implementation: batched park/resume descent
//!   with virtual loss, FPU, optional root Dirichlet noise, cycle draws and
//!   subtree reuse. The caller owns the evaluator (GPU batch, CPU net,
//!   browser WebGPU); everything below drives this search.
//! * [`Mlp`] — input → two ReLU hidden layers → policy logits (softmax over
//!   the *legal* subset only) + scalar tanh value. He init, [`SgdMomentum`],
//!   versioned binary save/load with atomic rename.
//! * [`Puct`] — [`Search`] driven synchronously by an [`Mlp`], one leaf at a
//!   time; [`PuctAgent`] adapts it to the arena's [`game_core::Agent`].
//! * [`SelfPlayTrainer`] — plays games in parallel, stores
//!   (encoding, visit distribution, outcome) [`Sample`]s in a replay buffer,
//!   trains with policy cross-entropy + value MSE + L2.
//!
//! Two-player zero-sum games only — the scalar value head assumes it.

mod mlp;
mod puct;
mod rand;
mod search;
#[cfg(feature = "parallel")]
mod train;

pub use mlp::{InferCache, Mlp, Sample, SgdMomentum};
pub use puct::{PolicyValueEncoder, Puct, PuctAgent};
pub use search::{EvalRequest, EvalResult, Gather, Node, PuctConfig, Search, Tree, argmax};
#[cfg(feature = "parallel")]
pub use train::{AzeroConfig, IterStats, SelfPlayTrainer};
