//! Writes WebGPU validation fixtures for go: positions (planes + legal
//! indices) with the reference model's expected priors and values.
//!
//! ```text
//! cargo run --release -p goinfer --example gen_fixtures -- \
//!     <export.azweb> <out.json> [positions]
//! ```

use game_core::{Game, PolicyValueEncoder, Rng};
use go::Go;
use go::encode::GoEncoder;
use goinfer::EvalRequest;
use goinfer::model::Model;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let export = args
        .first()
        .expect("usage: gen_fixtures <export.azweb> <out.json> [n]");
    let out = args
        .get(1)
        .expect("usage: gen_fixtures <export.azweb> <out.json> [n]");
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(6);

    let data = std::fs::read(export).expect("read export");
    let model = Model::parse(&data).expect("parse export");
    let size = model.size;
    let game = Go::new(size);
    let enc = GoEncoder::new(size);

    let mut rng = Rng::new(20_260_612);
    let mut state = game.initial_state();
    let mut items = Vec::new();
    let mut plies = 0;
    while items.len() < n {
        if game.is_terminal(&state) || plies > 2 * size * size {
            state = game.initial_state();
            plies = 0;
            continue;
        }
        let actions = game.legal_actions(&state);
        // Sample positions at varying depths.
        if plies % 13 == (items.len() * 5) % 13 {
            let req = EvalRequest {
                features: enc.encode_state(&game, &state),
                support: actions
                    .iter()
                    .map(|&a| enc.action_index(&game, &state, a) as u16)
                    .collect(),
            };
            let res = &model.eval(std::slice::from_ref(&req))[0];
            let join = |v: Vec<String>| v.join(",");
            items.push(format!(
                r#"{{"size":{},"plies":{},"planes":[{}],"support":[{}],"priors":[{}],"value":{}}}"#,
                size,
                plies,
                join(req.features.iter().map(|x| format!("{x}")).collect()),
                join(req.support.iter().map(|s| s.to_string()).collect()),
                join(res.priors.iter().map(|p| format!("{p}")).collect()),
                res.value,
            ));
        }
        let i = ((rng.unit() * actions.len() as f64) as usize).min(actions.len() - 1);
        game.apply(&mut state, actions[i]);
        plies += 1;
    }
    std::fs::write(out, format!("[{}]", items.join(",\n"))).expect("write fixtures");
    println!("wrote {} fixtures (size {size}) to {out}", items.len());
}
