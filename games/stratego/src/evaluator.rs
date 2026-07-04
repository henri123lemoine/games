//! The neural-net seam: an [`Evaluator`] turns a batch of decision states into
//! per-state action logits over the legal set plus a value estimate. The real
//! move/setup transformers (milestone 3) implement this trait; the sim and
//! buffer are written against it so the whole self-play pipeline is buildable
//! and testable now with the [`UniformEvaluator`] reference.
//!
//! Both game phases flow through one trait. A [`Decision`] is either a
//! deployment placement (the setup-policy head: logits over the legal piece
//! types) or a move (the move head: logits over the legal action indices). The
//! evaluator never sees illegal actions — the request carries exactly the legal
//! option list, and the returned `logits` are parallel to it.

use crate::board::PieceType;

/// The three value categories the categorical value head aggregates over (loss,
/// tie, win), matching `stratego_nets.spec.CATEGORICAL_AGGREGATION`.
pub const VALUE_CATEGORIES: [f32; 3] = [-1.0, 0.0, 1.0];

/// Two-hot encodes a scalar in `[-1, 1]` over [`VALUE_CATEGORIES`]: the unique
/// distribution over the three anchors whose expectation is `v` (`onehot(-1)`,
/// `onehot(1)` exactly when `v` lands on an anchor, e.g. a terminal reward).
/// Used wherever only a scalar value is available and a categorical stand-in is
/// needed (the uniform/fixed-value test evaluators, terminal-reward targets).
pub fn two_hot(v: f32) -> [f32; 3] {
    let v = v.clamp(VALUE_CATEGORIES[0], VALUE_CATEGORIES[2]);
    if v <= VALUE_CATEGORIES[1] {
        let w = v - VALUE_CATEGORIES[0]; // in [0, 1]
        [1.0 - w, w, 0.0]
    } else {
        let w = v - VALUE_CATEGORIES[1]; // in [0, 1]
        [0.0, 1.0 - w, w]
    }
}

/// Which head a [`Decision`] addresses. The setup net scores piece-type
/// placements; the move net scores board actions. The sim tags every request so
/// a combined evaluator can route to the right head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Deploy,
    Move,
}

/// One decision to score. Carries the encoded observation the net consumes and
/// the legal option set; `logits` returned by the evaluator are parallel to
/// `legal`.
///
/// The observation is the per-token feature matrix from
/// [`encode_tokens`](crate::encode::encode_tokens) for a move decision. For a
/// deployment decision there is no board yet; `obs` carries the partial
/// deployment encoding the setup net keys on (the placed-so-far one-hots), and
/// `legal` is the list of legal [`PieceType`]s as `u16` indices.
#[derive(Debug, Clone)]
pub struct Decision<'a> {
    pub phase: Phase,
    /// Flat observation features for this state (move: the `(92, 643)` token
    /// matrix flattened row-major; deploy: the setup-net feature vector).
    pub obs: &'a [f32],
    /// The legal options, as action indices (move: 1800-space indices; deploy:
    /// `PieceType as u16`). `logits` in the returned [`Evaluation`] are parallel
    /// to this slice.
    pub legal: &'a [u16],
    /// Acting player (0 = red, 1 = blue).
    pub player: usize,
}

/// The evaluator's output for one [`Decision`]: a logit per legal option (same
/// order as `Decision::legal`), a scalar value in `[-1, 1]` (the win/lose/tie
/// expectation for the acting player), and the categorical distribution that
/// scalar is the expectation of.
#[derive(Debug, Clone)]
pub struct Evaluation {
    /// One logit per legal option, parallel to `Decision::legal`.
    pub logits: Vec<f32>,
    /// Scalar value for the acting player (`+1` win, `-1` loss).
    pub value: f32,
    /// The value head's categorical distribution over [`VALUE_CATEGORIES`]
    /// (`[P(loss), P(tie), P(win)]`), acting-player POV. `value == value_probs
    /// @ VALUE_CATEGORIES`. The replay buffer's move-RL value target is
    /// bootstrapped from this distribution directly (ATARAXOS_SPEC §4.1's
    /// categorical λ-return), not from a two-hot projection of `value` — the two
    /// differ whenever the bootstrap value isn't a bare category anchor.
    pub value_probs: [f32; 3],
}

/// The neural-net seam. One batched call scores every decision in the batch —
/// the single point where a real GPU forward will run (one-GPU-thread
/// discipline). Implementors must return one [`Evaluation`] per [`Decision`],
/// in order, each with `logits.len() == decision.legal.len()`.
pub trait Evaluator: Sync {
    fn evaluate_batch(&self, batch: &[Decision]) -> Vec<Evaluation>;
}

/// Reference evaluator: uniform logits over the legal set (so sampling is
/// uniform-random) and value `0`. Lets the whole sim/buffer pipeline run and be
/// tested before the real nets land.
#[derive(Debug, Clone, Copy, Default)]
pub struct UniformEvaluator;

impl Evaluator for UniformEvaluator {
    fn evaluate_batch(&self, batch: &[Decision]) -> Vec<Evaluation> {
        batch
            .iter()
            .map(|d| Evaluation {
                logits: vec![0.0; d.legal.len()],
                value: 0.0,
                value_probs: two_hot(0.0),
            })
            .collect()
    }
}

/// Convenience: the [`PieceType`] a deploy-phase legal index encodes.
#[inline]
pub fn deploy_index_type(index: u16) -> PieceType {
    PieceType::from_u8(index as u8)
}
