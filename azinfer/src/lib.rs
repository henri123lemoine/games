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

/// CPU↔GPU calibration (the CI half). The no-GPU browser path plays the same
/// `.azweb` net through this reference forward, so it must reproduce the
/// committed fixtures the WebGPU kernels are checked against in the browser
/// (`/azero-test.html`). That pins **CPU ≡ fixtures ≡ GPU**: change the forward
/// and these fail until fixtures are regenerated, and the browser re-validates
/// the GPU against the new fixtures. The live GPU-vs-CPU comparison runs in the
/// browser harness (it needs WebGPU); this locks the reference end.
#[cfg(test)]
mod calibration {
    use crate::mcts::{MctsConfig, Search};
    use crate::model::Model;
    use crate::{EvalRequest, argmax};
    use chess::Board;
    use game_core::Rng;
    use std::collections::HashMap;

    const NET: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../web/app/public/azero/azero-chess.azweb"
    );
    const FIXTURES: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../web/app/public/azero/fixtures.json"
    );

    fn net() -> Model {
        Model::parse(&std::fs::read(NET).expect("committed chess net")).expect("parse net")
    }

    fn f32s(v: &serde_json::Value) -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    }

    #[test]
    fn reference_forward_reproduces_committed_fixtures() {
        let model = net();
        let fixtures: serde_json::Value =
            serde_json::from_slice(&std::fs::read(FIXTURES).expect("fixtures")).unwrap();
        let arr = fixtures.as_array().unwrap();
        assert!(arr.len() >= 6, "fixtures present ({} found)", arr.len());
        let (mut max_dp, mut max_dv) = (0f32, 0f32);
        for fx in arr {
            let req = EvalRequest {
                features: f32s(&fx["planes"]),
                support: fx["support"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as u16)
                    .collect(),
            };
            let res = &model.eval(std::slice::from_ref(&req))[0];
            let expect = f32s(&fx["priors"]);
            assert_eq!(res.priors.len(), expect.len());
            for (a, b) in res.priors.iter().zip(&expect) {
                max_dp = max_dp.max((a - b).abs());
            }
            max_dv = max_dv.max((res.value - fx["value"].as_f64().unwrap() as f32).abs());
        }
        assert!(
            max_dp < 1e-4,
            "max |Δprior| vs committed chess fixtures = {max_dp}"
        );
        assert!(
            max_dv < 1e-4,
            "max |Δvalue| vs committed chess fixtures = {max_dv}"
        );
    }

    #[test]
    fn trivial_search_is_deterministic_and_legal() {
        let model = net();
        let board = Board::start();
        let history = HashMap::new();
        let pick = |seed: u64| {
            let cfg = MctsConfig {
                sims: 1,
                max_leaves: 8,
                root_noise: 0.0,
                ..MctsConfig::default()
            };
            let mut rng = Rng::new(seed);
            let mut search = Search::new(None);
            crate::mcts::run_to_done(&mut search, &board, &history, &cfg, &mut rng, |reqs| {
                model.eval(reqs)
            });
            search.root_moves()[argmax(search.root_visits())].to_string()
        };
        let first = pick(1);
        assert_eq!(first, pick(99), "no root noise → deterministic move");
        assert!(
            chess::legal_moves(&board)
                .iter()
                .any(|m| m.to_string() == first),
            "trivial move {first} is legal from the start position",
        );
    }
}
