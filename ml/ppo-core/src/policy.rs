//! The `Policy` trait: the seam between the env-agnostic PPO update and a concrete
//! tch net. The core hands the policy a list of transition indices (a minibatch)
//! and gets back the per-transition log-prob, entropy, and value under the current
//! params — exactly what the clipped surrogate needs. The policy owns its obs and
//! action storage and packs the right rows itself, so the core never sees the env's
//! `Obs`/`Action` types.

use tch::Tensor;

/// What a forward pass returns for one minibatch: the per-transition action
/// log-prob, the joint entropy, and the value estimate. All are 1-D tensors of
/// length `idx.len()`, on the policy's device, with a graph attached (the core
/// calls `.backward()` on a loss built from them).
pub struct Eval {
    pub log_prob: Tensor,
    pub entropy: Tensor,
    pub value: Tensor,
}

/// A tch policy/value net the PPO core can update. The net stores the rollout's
/// observations and sampled actions (whatever shape its env produced) and exposes
/// them by index — the core only ever speaks in transition indices.
pub trait Policy {
    /// Re-evaluate the stored actions at `idx` under the current params, building
    /// the graph. `idx` is a 1-D `Int64` tensor of transition indices on the
    /// policy's device. Returns log-prob, entropy, value (each length `idx.len()`).
    fn evaluate(&self, idx: &Tensor) -> Eval;

    /// Optional behavior-cloning term for the minibatch at `idx`: a scalar tensor
    /// (with graph) to be added, scaled by [`crate::BcAnchor::coef`], to the loss.
    /// Default `None` — slither has no BC anchor. A policy that wants one (e.g. a
    /// reference-net KL) overrides this.
    fn bc_term(&self, _idx: &Tensor) -> Option<Tensor> {
        None
    }

    /// Optional auxiliary supervised term for the minibatch at `idx`: a scalar
    /// tensor (with graph) to be added, scaled by [`crate::PpoConfig::aux_coef`],
    /// to the loss. Default `None`. Distinct from [`Self::bc_term`] (which anchors
    /// the policy to a reference net's action distribution): this is for a policy
    /// that trains an unrelated supervised head off the trunk — e.g. Liar's Dice's
    /// opponent-hand belief head, trained against ground-truth hands the collector
    /// has access to during self-play but the deployed policy never sees.
    fn aux_term(&self, _idx: &Tensor) -> Option<Tensor> {
        None
    }
}
