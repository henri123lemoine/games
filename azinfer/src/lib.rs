//! Torch-free azero chess inference, shared by the native trainer (`azt/`)
//! and the browser. [`mcts`] instantiates `solvers::azero`'s batched
//! park/resume PUCT search for chess; [`model`] parses the `AZWEB001` export
//! and runs a reference fp32 forward pass on the CPU — the ground truth the
//! WebGPU kernels are tested against.

pub mod mcts;
pub mod model;

pub use solvers::azero::{EvalRequest, EvalResult, argmax};

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
