//! Checkpoint export to the portable `AZSNK1` browser format (every BatchNorm
//! folded into its conv), and the tch-vs-snakeinfer agreement check that guards
//! the folding and layout.
//!
//! `AZSNK1` matches the global-pooling snake net: a conv stem + residual tower,
//! a policy head (1×1 conv → global pool → MLP to the four heading logits) and
//! a value head (1×1 conv → global pool → MLP). Global pooling collapses
//! `[C,H,W]` to `[3C]` (mean, size-scaled mean, max) — the runtime implements
//! that; the file is only weights.

use std::path::PathBuf;

use tch::{Device, Kind, Tensor};

use crate::net::{self, Infer};
use crate::{arg, net_config_for};

/// Exports a checkpoint as the portable browser format: magic, dims, then fp32
/// tensors in fixed order with every BatchNorm folded into its conv
/// (`w' = w·γ/√(σ²+ε)`, `b' = β − μ·γ/√(σ²+ε)`), so a runtime needs only
/// conv+bias, linear, global-pool, relu, tanh.
///
/// Tensor order: stem, each block's (c1, c2); policy head (p1 conv, pf1 linear
/// 3C→C, pf2 linear C→4); value head (v1 conv, vf1 linear 3C→128, vf2 linear
/// 128→1). `pf1`/`vf1` take the `3·channels` global-pool vector.
pub fn export(args: &[String]) {
    let net_path: PathBuf = arg(args, "--net", PathBuf::from("../data/azsnake/run3/latest.ot"));
    let out: PathBuf = arg(
        args,
        "--out",
        PathBuf::from("../data/azsnake/run3/azero-snake.azweb"),
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
    buf.extend_from_slice(net::MAGIC);
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
    // Policy head: p1 conv (+BN) → global pool → MLP (pf1 3C→C, pf2 C→4).
    let (w, b) = folded("p1", "pb");
    push(&w);
    push(&b);
    push(&plain("pf1.weight"));
    push(&plain("pf1.bias"));
    push(&plain("pf2.weight"));
    push(&plain("pf2.bias"));
    // Value head: v1 conv (+BN) → global pool → MLP (vf1 3C→128, vf2 128→1).
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

/// Compares the tch forward pass with snakeinfer's reference forward on the
/// exported file over random positions — guards the BN folding and layout.
pub fn verify_export(args: &[String]) {
    use game_core::{Game, PolicyValueEncoder};
    use snake::Duel;
    use snake::encode::SnakeEncoder;

    let net_path: PathBuf = arg(args, "--net", PathBuf::from("../data/azsnake/run3/latest.ot"));
    let export_path: PathBuf = arg(
        args,
        "--export",
        PathBuf::from("../data/azsnake/run3/azero-snake.azweb"),
    );
    let cfg = net_config_for(args, &net_path);
    let infer = Infer::load(&net_path, cfg, Device::Cpu, Kind::Float).expect("load checkpoint");
    let data = std::fs::read(&export_path).expect("read export");
    let model = snakeinfer::model::Model::parse(&data).expect("parse export");

    let game = Duel::new();
    let enc = SnakeEncoder::new();
    let mut rng = game_core::Rng::new(7);
    let mut state = game.initial_state();
    let (mut max_dp, mut max_dv) = (0.0f32, 0.0f32);
    let mut positions = 0;
    while positions < 120 {
        if game.is_terminal(&state) {
            state = game.initial_state();
            continue;
        }
        if matches!(game.turn(&state), game_core::Turn::Chance) {
            let outs = game.chance_outcomes(&state);
            let j = game_core::rand::sample_outcome(&outs, &mut rng);
            game.apply(&mut state, outs[j].0);
            continue;
        }
        let actions = game.legal_actions(&state);
        let req = snakeinfer::EvalRequest {
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
        positions += 1;
        let i = rng.below(actions.len());
        game.apply(&mut state, actions[i]);
    }
    println!("max |prior diff| {max_dp:.2e}, max |value diff| {max_dv:.2e} over 120 positions");
    assert!(max_dp < 1e-3 && max_dv < 1e-3, "export does not match tch");
    println!("export verified");
}
