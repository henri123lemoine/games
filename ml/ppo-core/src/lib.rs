//! Reusable PPO math, factored out of `slither-ppo` so every PPO substrate trains
//! through one implementation of the algorithm rather than a per-game copy.
//!
//! The core owns the math: GAE(λ) advantage estimation ([`gae`]), the clipped
//! surrogate update with (optional) value clipping, entropy bonus, advantage
//! normalization, gradient clipping, and the minibatch/epoch loop ([`update`]).
//! The divergences the design found between substrates — value-clip on/off, where
//! advantages are normalized, shuffle-vs-BPTT minibatching, an optional BC anchor —
//! are config choices on [`PpoConfig`], not forks.
//!
//! The core never names an env's `Obs`/`Action`. It speaks two languages only:
//! plain `f32` per-transition `(reward, value, done, log_prob)` for GAE, and
//! transition *indices* handed to a [`Policy`] that owns its own obs/action tensors
//! and re-evaluates the indexed rows. The collector — how rollouts are gathered,
//! opponents assigned, arenas reset — stays env-specific (slither = N parallel
//! arenas; doom = a singleton over the C substrate). The [`Env`]/[`ObsEncoder`]/
//! [`ActionHead`] traits pin the per-step boundary so a new substrate is an adapter,
//! not a fork.
//!
//! Standalone (empty `[workspace]`) on purpose: keeps libtorch out of the repo's
//! main cargo workspace.

mod config;
mod env;
mod gae;
mod policy;
mod update;

pub use config::{AdvNorm, BcAnchor, Minibatch, PpoConfig};
pub use env::{ActionHead, Env, ObsEncoder, ObsShape, StepResult};
pub use gae::{Step, gae};
pub use policy::{Eval, Policy};
pub use update::{UpdateStats, update};
