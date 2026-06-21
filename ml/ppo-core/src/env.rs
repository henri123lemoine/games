//! The env adapter boundary: `Env`, `ObsEncoder`, `ActionHead`.
//!
//! These are the contract a substrate implements so a PPO collector can drive it,
//! with `Obs` and `Action` as associated types the PPO *update* never sees
//! concretely (it speaks only in transition indices via [`crate::Policy`]). The
//! shape was fixed against two substrates: slither (N pure-Rust parallel arenas)
//! and doom (a singleton wrapping C via FFI). The collector — how rollouts are
//! gathered, opponents assigned, arenas reset — stays env-specific; this is only
//! the minimal per-step interface the design pinned down.

/// The shape of an observation, enough for the net trunk to size itself:
/// `planes·size² + scalars`. Matches the unified-format arch dims (a board-size-
/// agnostic global-pool net runs at any `size`; flat nets use `size = 1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObsShape {
    /// Input feature channels.
    pub planes: usize,
    /// Board side (1 for flat/non-spatial obs).
    pub size: usize,
    /// Side-input scalar count (0 if none).
    pub scalars: usize,
}

impl ObsShape {
    /// Total spatial element count `planes·size²` (excludes scalars).
    pub fn grid_len(&self) -> usize {
        self.planes * self.size * self.size
    }
}

/// One actor's transition out of a step: the next observation, the reward for the
/// action just taken, and whether the actor terminated this step (the env is
/// expected to auto-reset a terminated actor so the rollout stays a fixed block).
pub struct StepResult<O> {
    pub obs: O,
    pub reward: f32,
    pub done: bool,
}

/// A vectorized environment a PPO collector drives. `Obs` / `Action` are the env's
/// own types; the PPO update never names them. Many actors step in lockstep
/// (slither = N arenas; doom = one actor, `num_actors() == 1`).
pub trait Env: Sync {
    type Obs;
    type Action;

    /// (Re)seed and reset every actor to a fresh episode start.
    fn reset(&mut self, seed: u64);

    /// Step every actor with its chosen action; return one [`StepResult`] per actor
    /// in actor order. A terminated actor is auto-reset for the next step.
    fn step(&mut self, actions: &[Self::Action]) -> Vec<StepResult<Self::Obs>>;

    /// Current observation for one actor (e.g. the rollout's last obs, or the
    /// bootstrap obs after the final step).
    fn obs(&self, actor: usize) -> Self::Obs;

    fn num_actors(&self) -> usize;
    fn action_space(&self) -> usize;
    fn obs_shape(&self) -> ObsShape;
}

/// Encodes one env `Obs` into the flat `(grid, scalars)` f32 the net trunk packs.
/// Game/env knowledge — stays with the substrate, mirrors the deploy-side encoder.
pub trait ObsEncoder {
    type Obs;
    fn encode(&self, obs: &Self::Obs) -> (Vec<f32>, Vec<f32>);
    fn shape(&self) -> ObsShape;
}

/// Maps the net's policy-head outputs to/from an env `Action` and supplies the
/// per-action log-prob and entropy the PPO surrogate needs. The head structure
/// (a single categorical, slither's factored turn+boost, a multi-discrete vector)
/// is env knowledge; the core only consumes log-prob/entropy through [`Eval`].
///
/// [`Eval`]: crate::Eval
pub trait ActionHead {
    type Action;
    /// Number of independent sub-actions the head factors into (1 for a single
    /// categorical). Informational for adapters; the core does not branch on it.
    fn num_components(&self) -> usize;
}
