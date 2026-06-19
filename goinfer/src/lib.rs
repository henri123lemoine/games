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

/// CPU↔GPU calibration (the CI half). The no-GPU browser path plays the same
/// `.azweb` net through this reference forward, so it must reproduce the
/// committed fixtures the WebGPU kernels are checked against in the browser
/// (`/go-azero-test.html`). That pins **CPU ≡ fixtures ≡ GPU**: change the
/// forward and these fail until fixtures are regenerated, and the browser
/// re-validates the GPU against the new fixtures — neither backend can drift
/// from the other without a red test. The live GPU-vs-CPU comparison runs in
/// the browser harness (it needs WebGPU); this locks the reference end.
#[cfg(test)]
mod calibration {
    use crate::model::Model;
    use crate::{EvalRequest, Gather, PuctConfig, Search, argmax};
    use game_core::{Game, GameUi, Rng};
    use go::Go;
    use go::encode::{GoEncoder, PLANES};

    const NET: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../web/app/public/azero/azero-go.azweb"
    );
    const FIXTURES: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../web/app/public/azero/go-fixtures.json"
    );

    fn net() -> Model {
        Model::parse(&std::fs::read(NET).expect("committed go net")).expect("parse net")
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
            "max |Δprior| vs committed go fixtures = {max_dp}"
        );
        assert!(
            max_dv < 1e-4,
            "max |Δvalue| vs committed go fixtures = {max_dv}"
        );
    }

    #[test]
    fn forward_is_size_agnostic() {
        let model = net();
        let big = model.size;
        let planes = vec![0.25f32; PLANES * big * big];
        assert_eq!(model.forward(&planes), model.forward_at(&planes, big));
        let (logits9, value9) = model.forward_at(&vec![0.1f32; PLANES * 81], 9);
        assert_eq!(logits9.len(), 82, "9×9 policy = 81 points + pass");
        assert!(logits9.iter().all(|x| x.is_finite()) && value9.is_finite());
    }

    #[test]
    fn trivial_search_is_deterministic_and_sane() {
        let model = net();
        let game = Go::new(9);
        let enc = GoEncoder::new(9);
        let pick = |seed: u64| {
            let cfg = PuctConfig {
                sims: 1,
                max_leaves: 8,
                root_noise: 0.0,
                ..PuctConfig::default()
            };
            let mut rng = Rng::new(seed);
            let mut search = Search::new(None);
            let state = game.initial_state();
            let mut results = Vec::new();
            while let Gather::Requests(reqs) = search.advance(
                &game,
                &enc,
                &state,
                &cfg,
                &mut rng,
                std::mem::take(&mut results),
                &|_| false,
            ) {
                results = model.eval(&reqs);
            }
            let mut visits = search.root_visits().to_vec();
            let actions = search.root_actions();
            crate::mask_pass_visits(&game, &state, actions, &mut visits);
            game.action_label(&state, actions[argmax(&visits)])
        };
        let first = pick(1);
        assert_eq!(first, pick(99), "no root noise → deterministic move");
        assert_ne!(first, "pass", "the trivial budget still plays a stone");
    }
}
