//! Checkpoint export to the portable AZWEB001 browser format (every
//! BatchNorm folded into its conv), and the tch-vs-azinfer agreement check
//! that guards the folding and layout.

use std::path::PathBuf;

use game_core::Rng;
use tch::{Device, Kind};

use crate::net::{self, Infer};
use crate::{arg, net_config_for};

/// Exports a checkpoint as the portable browser format: magic, dims, then
/// fp32 tensors in fixed order with every BatchNorm folded into its conv
/// (`w' = w·γ/√(σ²+ε)`, `b' = β − μ·γ/√(σ²+ε)`), so a runtime needs only
/// conv+bias, linear, relu, tanh.
pub fn export(args: &[String]) {
    use tch::Tensor;

    let net_path: PathBuf = arg(args, "--net", PathBuf::from("../data/azt/run2/latest.ot"));
    let out: PathBuf = arg(
        args,
        "--out",
        PathBuf::from("../data/azt/run2/azero-chess.azweb"),
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
    buf.extend_from_slice(b"AZWEB001");
    buf.extend_from_slice(&(cfg.blocks as u32).to_le_bytes());
    buf.extend_from_slice(&(cfg.channels as u32).to_le_bytes());
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
    push(&plain("p2.weight"));
    push(&vec![0.0; 73]);
    let (w, b) = folded("v1", "vb");
    push(&w);
    push(&b);
    push(&plain("vf1.weight"));
    push(&plain("vf1.bias"));
    push(&plain("vf2.weight"));
    push(&plain("vf2.bias"));

    std::fs::write(&out, &buf).expect("write export");
    println!(
        "exported {}x{} net: {} ({:.1} MB) from {}",
        cfg.blocks,
        cfg.channels,
        out.display(),
        buf.len() as f64 / 1e6,
        net_path.display()
    );
}

/// Compares the tch forward pass with azinfer's reference forward on the
/// exported file over random positions — guards the BN folding and layout.
pub fn verify_export(args: &[String]) {
    let net_path: PathBuf = arg(args, "--net", PathBuf::from("../data/azt/run2/latest.ot"));
    let export_path: PathBuf = arg(
        args,
        "--export",
        PathBuf::from("../data/azt/run2/azero-chess.azweb"),
    );
    let cfg = net_config_for(args, &net_path);
    let infer = Infer::load(&net_path, cfg, Device::Cpu, Kind::Float).expect("load checkpoint");
    let data = std::fs::read(&export_path).expect("read export");
    let model = azinfer::model::Model::parse(&data).expect("parse export");

    let mut rng = Rng::new(7);
    let mut board = chess::Board::start();
    let (mut max_dp, mut max_dv) = (0.0f32, 0.0f32);
    for ply in 0..120 {
        let moves = chess::legal_moves(&board);
        if moves.is_empty() || board.halfmove >= 100 {
            board = chess::Board::start();
            continue;
        }
        let req = azinfer::EvalRequest {
            planes: chess::encode::encode_planes(&board),
            support: moves
                .iter()
                .map(|&m| chess::encode::az_move_index(m, board.stm) as u16)
                .collect(),
        };
        let a = &infer.forward_batch(std::slice::from_ref(&req))[0];
        let b = &model.eval(std::slice::from_ref(&req))[0];
        for (pa, pb) in a.priors.iter().zip(&b.priors) {
            max_dp = max_dp.max((pa - pb).abs());
        }
        max_dv = max_dv.max((a.value - b.value).abs());
        let i = ((rng.unit() * moves.len() as f64) as usize).min(moves.len() - 1);
        board.apply(moves[i]);
        let _ = ply;
    }
    println!("max |prior diff| {max_dp:.2e}, max |value diff| {max_dv:.2e} over 120 positions");
    assert!(max_dp < 1e-3 && max_dv < 1e-3, "export does not match tch");
    println!("export verified");
}
