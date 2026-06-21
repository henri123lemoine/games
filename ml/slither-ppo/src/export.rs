//! Checkpoint export to the portable `SLNET1` browser format, and the
//! tch-vs-`slitherinfer` agreement check that guards the layout.
//!
//! The slither policy net has no BatchNorm, so export is a plain dump of the
//! three convs and four linears in a fixed order — no folding. A torch-free
//! runtime (`slitherinfer`) needs only conv+bias, relu, linear, and the
//! flatten/concat-scalars wiring to reproduce the heads.
//!
//! Tensor order (matches [`crate::net::Policy::features`]/`heads`):
//!   c1 conv, c2 conv, c3 conv (each weight `[c_out,c_in,3,3]` then bias),
//!   trunk linear (weight `[H, conv_flat+scalars]` then bias),
//!   turn linear, boost linear, value linear (each weight `[out,H]` then bias).
//! Strides (1,2,2) and padding (same) are fixed by the architecture, so the
//! file carries only the magic, the channel/grid/scalar dims, and the weights.

use std::path::PathBuf;

use slither_rl::env::SHAPES;
use tch::nn::{self};
use tch::{Device, Kind, Tensor};

use crate::net::{self, Policy};

const MAGIC: &[u8; 6] = b"SLNET1";

fn arg<T: std::str::FromStr>(args: &[String], key: &str, default: T) -> T {
    args.iter()
        .find_map(|a| a.strip_prefix(&format!("{key}=")))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Loads the checkpoint, then writes magic + dims + the conv/linear tensors in
/// the fixed order above as little-endian fp32. `--quiet` suppresses the
/// variable listing.
pub fn export(args: &[String]) {
    let net_path: PathBuf = arg(
        args,
        "net",
        PathBuf::from("../../data/slither/slither-best-sym4-i130.ot"),
    );
    let out: PathBuf = arg(
        args,
        "out",
        PathBuf::from("../../web/app/public/slither/slither.weights"),
    );

    let mut vs = nn::VarStore::new(Device::Cpu);
    let _policy = Policy::new(&vs.root());
    vs.load(&net_path).unwrap_or_else(|e| {
        eprintln!("failed to load {}: {e}", net_path.display());
        std::process::exit(1);
    });

    let vars = vs.variables();
    let get = |name: &str| -> Vec<f32> {
        let t = vars
            .get(name)
            .unwrap_or_else(|| panic!("missing tensor {name}"))
            .to_kind(Kind::Float)
            .to_device(Device::Cpu)
            .flatten(0, -1);
        Vec::<f32>::try_from(t).unwrap()
    };

    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&(net::CHANNELS as u32).to_le_bytes());
    buf.extend_from_slice(&(net::GRID as u32).to_le_bytes());
    buf.extend_from_slice(&(net::SCALARS as u32).to_le_bytes());
    let mut push = |v: &[f32]| {
        for x in v {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    };
    for name in net::EXPORT_ORDER {
        push(&get(name));
    }

    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&out, &buf).expect("write export");
    println!(
        "exported slither net: {} ({:.2} MB) from {}  [grid {}x{}x{}, {} scalars, {} turn buckets]",
        out.display(),
        buf.len() as f64 / 1e6,
        net_path.display(),
        net::CHANNELS,
        net::GRID,
        net::GRID,
        net::SCALARS,
        SHAPES.turn_buckets,
    );

    // HARD RELEASE GATE: the exported blob must reproduce the tch forward via the
    // torch-free `slitherinfer` runtime the browser uses, or the deployed net is
    // unverified. Check parity now and, on failure, delete the bad blob and exit
    // non-zero — so `export` can never leave an unverified blob at `out`. (Skip
    // with `--no-verify` only for a deliberate, throwaway export.)
    if args.iter().any(|a| a == "--no-verify") {
        eprintln!("WARNING: --no-verify set, skipping the parity gate");
        return;
    }
    let (max_dt, max_db, max_dv) = parity_check(&net_path, &out);
    if max_dt < PARITY_TOL && max_db < PARITY_TOL && max_dv < PARITY_TOL {
        println!(
            "parity gate PASSED: max |Δturn| {max_dt:.2e} |Δboost| {max_db:.2e} |Δvalue| {max_dv:.2e} (tol {PARITY_TOL:.0e})"
        );
    } else {
        std::fs::remove_file(&out).ok();
        eprintln!(
            "parity gate FAILED: max |Δturn| {max_dt:.2e} |Δboost| {max_db:.2e} |Δvalue| {max_dv:.2e} >= tol {PARITY_TOL:.0e} — deleted {} (NOT deployed)",
            out.display()
        );
        std::process::exit(1);
    }
}

/// The parity tolerance the `slitherinfer` reference forward must match the tch
/// forward to — the train↔deploy contract. An export above this is a broken blob.
const PARITY_TOL: f32 = 1e-3;

/// Compares the tch forward with `slitherinfer`'s reference forward on `export_path`
/// over 64 random observations, returning the max head deviations. Guards the
/// layout and the flatten/concat wiring. Shared by `verify_export` and the
/// post-export gate in `export` so the deployed blob is always checked.
fn parity_check(net_path: &PathBuf, export_path: &PathBuf) -> (f32, f32, f32) {
    let mut vs = nn::VarStore::new(Device::Cpu);
    let policy = Policy::new(&vs.root());
    vs.load(net_path).expect("load checkpoint");

    let data = std::fs::read(export_path).expect("read export");
    let model = slitherinfer::Model::parse(&data).expect("parse export");

    let c = net::CHANNELS;
    let g = net::GRID;
    let s = net::SCALARS;
    let grid_len = (c * g * g) as usize;

    let mut seed: u64 = 0x1234_5678;
    let mut rand = || {
        // SplitMix64, centered to roughly [-1, 1] — the obs grids are sparse 0/1
        // in practice, but random inputs exercise every weight path.
        seed = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        ((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };

    let (mut max_dt, mut max_db, mut max_dv) = (0.0f32, 0.0f32, 0.0f32);
    for _ in 0..64 {
        let grid: Vec<f32> = (0..grid_len).map(|_| rand()).collect();
        let scalars: Vec<f32> = (0..s as usize).map(|_| rand()).collect();

        let gt = Tensor::from_slice(&grid).reshape([1, c, g, g]);
        let st = Tensor::from_slice(&scalars).reshape([1, s]);
        let (turn_t, boost_t, value_t) = policy.raw_heads(&gt, &st);
        let turn_tch: Vec<f32> = (&turn_t.reshape([-1])).try_into().unwrap();
        let boost_tch = f32::try_from(&boost_t.reshape([-1])).unwrap();
        let value_tch = f32::try_from(&value_t.reshape([-1])).unwrap();

        let out = model.forward(&grid, &scalars);
        for (a, b) in out.turn_logits.iter().zip(&turn_tch) {
            max_dt = max_dt.max((a - b).abs());
        }
        max_db = max_db.max((out.boost_logit - boost_tch).abs());
        max_dv = max_dv.max((out.value - value_tch).abs());
    }
    (max_dt, max_db, max_dv)
}

/// Compares the tch forward with `slitherinfer`'s reference forward on the
/// exported file over random observations — guards the layout and the
/// flatten/concat wiring. Asserts every head agrees to `PARITY_TOL`.
pub fn verify_export(args: &[String]) {
    let net_path: PathBuf = arg(
        args,
        "net",
        PathBuf::from("../../data/slither/slither-best-sym4-i130.ot"),
    );
    let export_path: PathBuf = arg(
        args,
        "export",
        PathBuf::from("../../web/app/public/slither/slither.weights"),
    );

    let (max_dt, max_db, max_dv) = parity_check(&net_path, &export_path);
    println!(
        "max |Δturn| {max_dt:.2e}, max |Δboost| {max_db:.2e}, max |Δvalue| {max_dv:.2e} over 64 inputs"
    );
    assert!(
        max_dt < PARITY_TOL && max_db < PARITY_TOL && max_dv < PARITY_TOL,
        "export does not match tch"
    );
    println!("export verified");
}
