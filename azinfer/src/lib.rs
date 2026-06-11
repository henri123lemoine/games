//! Torch-free azero chess inference, shared by the native trainer (`azt/`)
//! and the browser. [`mcts`] is the batched park/resume PUCT search;
//! [`model`] parses the `AZWEB001` export and runs a reference fp32 forward
//! pass on the CPU — the ground truth the WebGPU kernels are tested against.

pub mod mcts;
pub mod model;

/// One evaluation request: encoded planes plus the legal policy indices.
pub struct EvalRequest {
    pub planes: Vec<f32>,
    pub support: Vec<u16>,
}

/// Priors over `support` (softmax restricted to the legal subset) and the
/// value, both from the side to move's perspective.
pub struct EvalResult {
    pub priors: Vec<f32>,
    pub value: f32,
}

pub fn argmax(visits: &[u32]) -> usize {
    visits
        .iter()
        .enumerate()
        .max_by_key(|&(_, &n)| n)
        .map_or(0, |(i, _)| i)
}
