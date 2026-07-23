//! Regenerates the committed Go AlphaZero conformance fixtures
//! (`web/app/public/azero/go-fixtures.json`) the `/go-azero-test.html` page
//! validates the WebGPU evaluator against.
//!
//! Each fixture is a self-play position snapshotted at a spread of plies,
//! weighted toward mid/late game, with the engine's own encoder output
//! (`planes`), the legal-action policy indices (`support`), and the reference
//! forward's softmax-over-support priors and side-to-move value. Generating
//! them through the same Rust code paths the runtime bot uses guarantees the
//! golden matches what the live WebGPU and wasm-CPU forwards produce.
//!
//! Deterministic (fixed per-game seeds, no entropy): rerunning writes a
//! byte-identical file.
//!
//! ```text
//! cargo run --release -p web-engine --example gen_go_fixtures
//! ```
//!
//! Optional args override the defaults:
//! ```text
//! cargo run --release -p web-engine --example gen_go_fixtures -- <weights.azweb> <out.json>
//! ```

use game_core::{Game, PolicyValueEncoder};
use go::encode::GoEncoder;
use go::{Go, GoAction};
use nn_infer::Net;
use serde_json::{Value, json};

/// Logical asset path, fetched from the arcade-assets bucket via
/// tools/fetch-asset.sh when no explicit weights path is given.
const DEFAULT_WEIGHTS_ASSET: &str = "azero/azero-go.azweb";
const DEFAULT_OUT: &str = "web/app/public/azero/go-fixtures.json";

/// One board size and the move depths to snapshot at, weighted toward the
/// midgame and endgame where eval divergence matters. `restarts` seeds extra
/// self-play games so the deep snapshots stay filled when a game ends early.
struct Plan {
    size: usize,
    plies: &'static [u32],
    restarts: u64,
}

const PLANS: &[Plan] = &[
    Plan {
        size: 9,
        plies: &[
            0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 80, 90, 100, 110,
        ],
        restarts: 8,
    },
    Plan {
        size: 13,
        plies: &[
            0, 10, 20, 30, 44, 58, 72, 86, 100, 114, 128, 142, 156, 170, 190, 210, 230,
        ],
        restarts: 8,
    },
    Plan {
        size: 19,
        plies: &[
            0, 18, 36, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 400, 440, 480,
        ],
        restarts: 8,
    },
];

/// A reproducible move chooser: takes the `spread`-th best move (by reference
/// policy) among the legal `eligible` actions, so a fixed rotation of `spread`
/// makes successive games diverge with no entropy. `eligible` lets the caller
/// drop pass while productive moves remain.
fn pick_move(
    net: &Net,
    planes: &[f32],
    support: &[u16],
    size: usize,
    eligible: &[usize],
    spread: usize,
) -> usize {
    let logits = net.forward_at(planes, &[], size).policy;
    let mut ranked = eligible.to_vec();
    ranked.sort_by(|&a, &b| {
        logits[usize::from(support[b])].total_cmp(&logits[usize::from(support[a])])
    });
    ranked[spread.min(ranked.len() - 1)]
}

fn fixture(net: &Net, enc: &GoEncoder, game: &Go, state: &go::GoState, plies: u32) -> Value {
    let planes = enc.encode_state(game, state);
    let actions = game.legal_actions(state);
    let support: Vec<u16> = actions
        .iter()
        .map(|&a| enc.action_index(game, state, a) as u16)
        .collect();
    let (priors, value) = net.forward_support(&planes, &[], &support);
    json!({
        "size": game.size(),
        "plies": plies,
        "planes": planes,
        "support": support,
        "priors": priors,
        "value": value,
    })
}

fn generate(net: &Net) -> Vec<Value> {
    let mut fixtures = Vec::new();
    for plan in PLANS {
        let game = Go::new(plan.size);
        let enc = GoEncoder::new(plan.size);
        let mut wanted: Vec<u32> = plan.plies.to_vec();
        let max_wanted = *wanted.iter().max().unwrap();

        'plan: for restart in 0..plan.restarts {
            if wanted.is_empty() {
                break;
            }
            let mut state = game.initial_state();
            let mut plies = 0u32;
            loop {
                if let Some(pos) = wanted.iter().position(|&w| w == plies) {
                    fixtures.push(fixture(net, &enc, &game, &state, plies));
                    wanted.remove(pos);
                    if wanted.is_empty() {
                        break 'plan;
                    }
                }
                if game.is_terminal(&state) || plies > max_wanted {
                    break;
                }
                let actions = game.legal_actions(&state);
                let planes = enc.encode_state(&game, &state);
                let support: Vec<u16> = actions
                    .iter()
                    .map(|&a| enc.action_index(&game, &state, a) as u16)
                    .collect();
                let drop_pass = game.has_productive_move(&state);
                let eligible: Vec<usize> = (0..actions.len())
                    .filter(|&i| !(drop_pass && matches!(actions[i], GoAction::Pass)))
                    .collect();
                let spread = ((restart as usize) + (plies as usize) / 7) % 3;
                let choice = pick_move(net, &planes, &support, plan.size, &eligible, spread);
                game.apply(&mut state, actions[choice]);
                plies += 1;
            }
        }
    }
    fixtures
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let weights_path = args.first().cloned().unwrap_or_else(|| {
        let script =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/fetch-asset.sh");
        let out = std::process::Command::new(&script)
            .arg(DEFAULT_WEIGHTS_ASSET)
            .output()
            .expect("run tools/fetch-asset.sh");
        assert!(
            out.status.success(),
            "fetch-asset: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout)
            .expect("utf8 path")
            .trim()
            .to_string()
    });
    let out_path = args.get(1).map(String::as_str).unwrap_or(DEFAULT_OUT);

    let data =
        std::fs::read(&weights_path).unwrap_or_else(|e| panic!("read weights {weights_path}: {e}"));
    let net = Net::parse(&data).expect("parse AZNET1 go weights");

    let fixtures = generate(&net);
    let json = serde_json::to_string(&fixtures).expect("serialize fixtures");
    std::fs::write(out_path, json).unwrap_or_else(|e| panic!("write {out_path}: {e}"));

    let mut by_size: std::collections::BTreeMap<u64, Vec<u64>> = std::collections::BTreeMap::new();
    for fx in &fixtures {
        let size = fx["size"].as_u64().unwrap();
        let plies = fx["plies"].as_u64().unwrap();
        by_size.entry(size).or_default().push(plies);
    }
    println!("wrote {} fixtures to {out_path}", fixtures.len());
    for (size, plies) in &by_size {
        println!(
            "  {size}x{size}: {} positions @ plies {plies:?}",
            plies.len()
        );
    }
}
