//! Torch-free azero snake inference, shared by the native trainer (`azsnake/`)
//! and the browser. [`model`] parses the `AZSNK1` export and runs a reference
//! fp32 forward pass on the CPU — the net the browser bot plays through, and the
//! ground truth the tch export is validated against (`azsnake verify-export`).
//! The batched park/resume PUCT search comes from `solvers::azero`, instantiated
//! over snake's [`Duel`].

pub mod model;

pub use solvers::azero::{self, EvalRequest, EvalResult, Gather, PuctConfig, Search, argmax};

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
