//! Torch-free azero go inference, shared by the native trainer (`azgo/`) and
//! the browser. [`model`] parses the `AZWEBGO1` export and runs a reference
//! fp32 forward pass on the CPU — the ground truth the WebGPU kernels are
//! tested against. [`mcts`] re-exports `solvers::azero`'s batched park/resume
//! PUCT search for go.

pub mod model;

use go::{Go, GoAction, GoState};

pub use solvers::azero::{self, EvalRequest, EvalResult, Gather, PuctConfig, Search, argmax};

/// Zeroes the pass action's visits when the mover still has a productive move
/// (see [`Go::has_productive_move`]), so move selection and the recorded
/// policy target never favor passing early — the guard against area scoring's
/// degenerate "pass for the komi win" equilibrium. A no-op once only
/// eye-filling moves remain, so finished games still end by passing. Leaves
/// visits untouched if pass is the only visited action.
pub fn mask_pass_visits(game: &Go, state: &GoState, actions: &[GoAction], visits: &mut [u32]) {
    if !game.has_productive_move(state) {
        return;
    }
    let Some(pass_i) = actions.iter().position(|a| matches!(a, GoAction::Pass)) else {
        return;
    };
    let others: u32 = visits
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != pass_i)
        .map(|(_, &v)| v)
        .sum();
    if others > 0 {
        visits[pass_i] = 0;
    }
}

/// In-place softmax: logits → distribution.
pub fn softmax(logits: &mut [f32]) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for q in logits.iter_mut() {
        *q = (*q - max).exp();
        sum += *q;
    }
    for q in logits.iter_mut() {
        *q /= sum;
    }
}
