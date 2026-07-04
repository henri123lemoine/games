//! PPO update configuration.
//!
//! The divergences the design found between the slither and doom trainers are
//! *config*, not forks: value-clip on/off, where advantage normalization happens,
//! how minibatches are drawn (shuffle vs BPTT windows), and an optional behavior-
//! cloning anchor. They live here as enum/flag choices so one `update` covers both.

/// Where advantage normalization is applied. cleanrl's single-file PPO (slither's
/// reference) normalizes per minibatch; some setups normalize once over the whole
/// rollout. The choice changes the scale the surrogate sees, so it is explicit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdvNorm {
    /// Recompute mean/std on each minibatch (cleanrl default).
    PerMinibatch,
    /// Normalize once over the full rollout before the epoch loop.
    PerRollout,
    /// No normalization.
    Off,
}

/// How minibatches are drawn each epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Minibatch {
    /// Random permutation of all transitions, split into `count` contiguous
    /// chunks (cleanrl default; the order is i.i.d. across transitions).
    Shuffled { count: usize },
    /// BPTT-style windows: keep each actor's time order intact and draw whole
    /// `window`-length segments. For recurrent / sequence policies that must see
    /// contiguous time. (Designed-for; doom's adapter may select it.)
    BpttWindows { window: usize },
}

/// Optional behavior-cloning anchor: add `coef * KL(reference || policy)` (or a
/// surrogate) to keep the policy near a fixed reference net. Off for slither.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BcAnchor {
    pub coef: f64,
}

/// The full PPO update config. Hyperparameters plus the four divergence knobs.
#[derive(Clone, Debug)]
pub struct PpoConfig {
    pub gamma: f32,
    pub lambda: f32,
    pub clip: f64,
    pub value_coef: f64,
    pub entropy_coef: f64,
    pub max_grad_norm: f64,
    pub epochs: usize,
    /// `T`: rollout length (steps per actor). `N` = buffer_len / T is the actor
    /// count; the adapter guarantees the buffer is exactly `T*N` step-major.
    pub steps: usize,

    /// Whether the value loss uses cleanrl's clipped form (max of clipped and
    /// unclipped squared error) or the plain MSE.
    pub value_clip: bool,
    /// Advantage-normalization scope.
    pub adv_norm: AdvNorm,
    /// Minibatch sampling strategy.
    pub minibatch: Minibatch,
    /// Optional behavior-cloning anchor; `None` to disable.
    pub bc_anchor: Option<BcAnchor>,
    /// Coefficient for [`crate::Policy::aux_term`]; `0.0` disables it (the default
    /// `None` return is never called for, but a nonzero coef with a policy that
    /// still returns `None` is silently inert too).
    pub aux_coef: f64,
}

impl PpoConfig {
    /// The cleanrl-faithful configuration slither trains with: clipped value loss,
    /// per-minibatch advantage normalization, shuffled minibatches, no BC anchor.
    /// `steps` is the rollout length and `minibatches` the shuffle chunk count.
    pub fn cleanrl(steps: usize, minibatches: usize) -> Self {
        Self {
            gamma: 0.99,
            lambda: 0.95,
            clip: 0.2,
            value_coef: 0.5,
            entropy_coef: 0.01,
            max_grad_norm: 0.5,
            epochs: 4,
            steps,
            value_clip: true,
            adv_norm: AdvNorm::PerMinibatch,
            minibatch: Minibatch::Shuffled { count: minibatches },
            bc_anchor: None,
            aux_coef: 0.0,
        }
    }
}
