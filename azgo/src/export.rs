//! Checkpoint export to the portable `AZWEBGO1` browser format (every
//! BatchNorm folded into its conv), and the tch-vs-goinfer agreement check
//! that guards the folding and layout.

use std::path::PathBuf;

use tch::{Device, Kind, Tensor};

use crate::net::{self, Infer};
use crate::{arg, net_config_for};

/// Exports a checkpoint as the portable browser format: magic, dims, then
/// fp32 tensors in fixed order with every BatchNorm folded into its conv
/// (`w' = w·γ/√(σ²+ε)`, `b' = β − μ·γ/√(σ²+ε)`), so a runtime needs only
/// conv+bias, linear, relu, tanh.
///
/// Tensor order: stem, then each block's (c1, c2), then the policy head
/// (p1 conv, pf linear) and the value head (v1 conv, vf1, vf2 linears).
pub fn export(args: &[String]) {
    let net_path: PathBuf = arg(args, "--net", PathBuf::from("../data/azgo/run19/latest.ot"));
    let out: PathBuf = arg(
        args,
        "--out",
        PathBuf::from("../data/azgo/run19/azero-go.azweb"),
    );
    let cfg = net_config_for(args, &net_path);

    let mut vs = tch::nn::VarStore::new(Device::Cpu);
    let _net = net::Net::new(&vs.root(), cfg);
    vs.load(&net_path).unwrap_or_else(|e| {
        eprintln!("failed to load {}: {e}", net_path.display());
        std::process::exit(1);
    });
    let vars = vs.variables();
    let get = |name: &str| -> Tensor {
        vars.get(name)
            .unwrap_or_else(|| panic!("missing tensor {name}"))
            .to_kind(Kind::Float)
            .to_device(Device::Cpu)
    };
    let folded = |conv: &str, bn: &str| -> (Vec<f32>, Vec<f32>) {
        let w = get(&format!("{conv}.weight"));
        let gamma = get(&format!("{bn}.weight"));
        let beta = get(&format!("{bn}.bias"));
        let mean = get(&format!("{bn}.running_mean"));
        let var = get(&format!("{bn}.running_var"));
        let scale = &gamma / (&var + 1e-5).sqrt();
        let wf = &w * &scale.reshape([-1, 1, 1, 1]);
        let bf = &beta - &mean * &scale;
        (
            Vec::<f32>::try_from(wf.flatten(0, -1)).unwrap(),
            Vec::<f32>::try_from(bf.flatten(0, -1)).unwrap(),
        )
    };
    let plain =
        |name: &str| -> Vec<f32> { Vec::<f32>::try_from(get(name).flatten(0, -1)).unwrap() };

    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"AZWEBGO1");
    buf.extend_from_slice(&(cfg.blocks as u32).to_le_bytes());
    buf.extend_from_slice(&(cfg.channels as u32).to_le_bytes());
    buf.extend_from_slice(&(cfg.size as u32).to_le_bytes());
    let mut push = |v: &[f32]| {
        for x in v {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    };
    let (w, b) = folded("stem_c", "stem_b");
    push(&w);
    push(&b);
    for i in 0..cfg.blocks {
        for half in ["c1", "c2"] {
            let bn = if half == "c1" { "b1" } else { "b2" };
            let (w, b) = folded(&format!("block{i}.{half}"), &format!("block{i}.{bn}"));
            push(&w);
            push(&b);
        }
    }
    let (w, b) = folded("p1", "pb");
    push(&w);
    push(&b);
    push(&plain("pf.weight"));
    push(&plain("pf.bias"));
    let (w, b) = folded("v1", "vb");
    push(&w);
    push(&b);
    push(&plain("vf1.weight"));
    push(&plain("vf1.bias"));
    push(&plain("vf2.weight"));
    push(&plain("vf2.bias"));

    std::fs::write(&out, &buf).expect("write export");
    println!(
        "exported {}x{} size-{} net: {} ({:.1} MB) from {}",
        cfg.blocks,
        cfg.channels,
        cfg.size,
        out.display(),
        buf.len() as f64 / 1e6,
        net_path.display()
    );
}

/// Compares the tch forward pass with goinfer's reference forward on the
/// exported file over random positions — guards the BN folding and layout.
pub fn verify_export(args: &[String]) {
    use game_core::{Game, PolicyValueEncoder};
    use go::Go;
    use go::encode::GoEncoder;

    let net_path: PathBuf = arg(args, "--net", PathBuf::from("../data/azgo/run19/latest.ot"));
    let export_path: PathBuf = arg(
        args,
        "--export",
        PathBuf::from("../data/azgo/run19/azero-go.azweb"),
    );
    let cfg = net_config_for(args, &net_path);
    let infer = Infer::load(&net_path, cfg, Device::Cpu, Kind::Float).expect("load checkpoint");
    let data = std::fs::read(&export_path).expect("read export");
    let model = goinfer::model::Model::parse(&data).expect("parse export");

    let game = Go::new(cfg.size as usize);
    let enc = GoEncoder::new(cfg.size as usize);
    let mut rng = game_core::Rng::new(7);
    let mut state = game.initial_state();
    let (mut max_dp, mut max_dv) = (0.0f32, 0.0f32);
    for _ in 0..120 {
        if game.is_terminal(&state) {
            state = game.initial_state();
            continue;
        }
        let actions = game.legal_actions(&state);
        let req = goinfer::EvalRequest {
            features: enc.encode_state(&game, &state),
            support: actions
                .iter()
                .map(|&a| enc.action_index(&game, &state, a) as u16)
                .collect(),
        };
        let a = &infer.forward_batch(std::slice::from_ref(&req))[0];
        let b = &model.eval(std::slice::from_ref(&req))[0];
        for (pa, pb) in a.priors.iter().zip(&b.priors) {
            max_dp = max_dp.max((pa - pb).abs());
        }
        max_dv = max_dv.max((a.value - b.value).abs());
        let i = rng.below(actions.len());
        game.apply(&mut state, actions[i]);
    }
    println!("max |prior diff| {max_dp:.2e}, max |value diff| {max_dv:.2e} over 120 positions");
    assert!(max_dp < 1e-3 && max_dv < 1e-3, "export does not match tch");
    println!("export verified");
}
