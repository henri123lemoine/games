//! Checkpoint export with every BatchNorm folded into its conv, written in the
//! unified `AZNET1` container. The BN-folded weight stream (the file's body) is
//! built once and written under the `AZNET1` header.
//!
//! Fold: `w' = w·γ/√(σ²+ε)`, `b' = β − μ·γ/√(σ²+ε)`, so a runtime needs only
//! conv+bias, linear, global-pool, relu, tanh.

use std::path::Path;

use nn_infer::{Arch, HeadFlags, HeadKind};
use tch::{Device, Kind, Tensor, nn};

use crate::net::{Net, NetConfig};

/// The `AZNET1` header for this architecture.
fn aznet1_header(cfg: &NetConfig, has_ownership: bool) -> Vec<u8> {
    let policy_len = match cfg.head {
        HeadKind::GlobalPoolSpatial => 0,
        HeadKind::FlatConv | HeadKind::GlobalPoolDense => cfg.policy_len as usize,
    };
    Arch {
        blocks: cfg.blocks,
        channels: cfg.channels as usize,
        planes: cfg.planes as usize,
        size: cfg.size as usize,
        scalars: 0,
        head: cfg.head,
        policy_len,
        flags: HeadFlags(if has_ownership {
            HeadFlags::OWNERSHIP
        } else {
            0
        }),
    }
    .header_bytes()
}

/// Builds the BN-folded weight body (everything after the header), in the fixed
/// `AZNET1` layer order. Returns the body and whether the net carries an
/// ownership head. `vs` holds the loaded weights; `cfg` names the architecture.
fn build_body(vs: &nn::VarStore, cfg: &NetConfig) -> (Vec<u8>, bool) {
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

    // Policy head, by kind.
    let (w, b) = folded("p1", "pb");
    push(&w);
    push(&b);
    match cfg.head {
        HeadKind::FlatConv => {
            // p2 is bias-free in tch; AZNET1/legacy carry a zero bias of width
            // `move_planes` (chess: 73 — the reserved pass slot it never uses).
            push(&plain("p2.weight"));
            push(&vec![0.0; cfg.move_planes() as usize]);
        }
        HeadKind::GlobalPoolSpatial => {
            push(&plain("pgb.weight"));
            push(&plain("pgb.bias"));
            // pfc is the bias-less placement conv (weight only).
            push(&plain("pfc.weight"));
            push(&plain("ppass.weight"));
            push(&plain("ppass.bias"));
        }
        HeadKind::GlobalPoolDense => {
            push(&plain("pf1.weight"));
            push(&plain("pf1.bias"));
            push(&plain("pf2.weight"));
            push(&plain("pf2.bias"));
        }
    }

    // Value head, by kind.
    let (w, b) = folded("v1", "vb");
    push(&w);
    push(&b);
    push(&plain("vf1.weight"));
    push(&plain("vf1.bias"));
    push(&plain("vf2.weight"));
    push(&plain("vf2.bias"));

    // Ownership head (go GO3 / AZNET1 OWNERSHIP flag): the bias-less 1×1 `o1`
    // conv. The score head (`sf`) is training-only and never exported.
    let has_ownership = vars.contains_key("o1.weight");
    if has_ownership {
        push(&plain("o1.weight"));
    }

    (buf, has_ownership)
}

/// Loads `net_path` into a fresh VarStore for `cfg`, builds the BN-folded body,
/// and writes the `AZNET1` file (`out`). Returns the byte length of the body for
/// logging.
pub fn export(net_path: &Path, cfg: NetConfig, out: &Path) -> Result<usize, String> {
    let mut vs = nn::VarStore::new(Device::Cpu);
    let _net = Net::new(&vs.root(), cfg);
    crate::net::load_inference_weights(&mut vs, net_path)
        .map_err(|e| format!("load {}: {e}", net_path.display()))?;

    let (body, has_ownership) = build_body(&vs, &cfg);

    let mut aznet1 = aznet1_header(&cfg, has_ownership);
    aznet1.extend_from_slice(&body);
    std::fs::write(out, &aznet1).map_err(|e| format!("write aznet1: {e}"))?;

    Ok(body.len())
}
