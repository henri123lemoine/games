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
/// order as `Decision::legal`) and a scalar value in `[-1, 1]` (the win/lose/tie
/// expectation for the acting player).
#[derive(Debug, Clone)]
pub struct Evaluation {
    /// One logit per legal option, parallel to `Decision::legal`.
    pub logits: Vec<f32>,
    /// Scalar value for the acting player (`+1` win, `-1` loss).
    pub value: f32,
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
            })
            .collect()
    }
}

/// Convenience: the [`PieceType`] a deploy-phase legal index encodes.
#[inline]
pub fn deploy_index_type(index: u16) -> PieceType {
    PieceType::from_u8(index as u8)
}
