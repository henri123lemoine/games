//! Train the Liar's Dice policy/value net by distilling per-round CFR/MCCFR
//! equilibria (see [`liars_dice::train`]). CPU only.
//!
//! `warmup_iters` solves against the fixed dice-share heuristic; afterwards the
//! net's own value head closes the round leaves (fitted value iteration). Set
//! `warmup_iters >= iters` to reproduce the pure-distillation baseline.
//!
//!     # quick smoke run (~1-2 min)
//!     cargo run --release -p liars-dice --example train_net -- \
//!         iters=3 rounds_per_iter=80 hidden=128 threads=4 outdir=runs/ld_smoke
//!
//!     # overnight-scale run
//!     cargo run --release -p liars-dice --example train_net -- \
//!         iters=400 warmup_iters=8 rounds_per_iter=600 hidden=256 threads=4 outdir=runs/ld_net

use liars_dice::train::{TrainConfig, train};

fn main() -> std::io::Result<()> {
    let mut cfg = TrainConfig::default();
    for arg in std::env::args().skip(1) {
        let Some((k, v)) = arg.split_once('=') else {
            eprintln!("ignoring arg without '=': {arg}");
            continue;
        };
        match k {
            "iters" => cfg.iters = v.parse().unwrap(),
            "warmup_iters" => cfg.warmup_iters = v.parse().unwrap(),
            "rounds_per_iter" => cfg.rounds_per_iter = v.parse().unwrap(),
            "playouts" => cfg.playouts = v.parse().unwrap(),
            "hidden" => cfg.hidden = v.parse().unwrap(),
            "cfr_iters" => cfg.cfr_iters = v.parse().unwrap(),
            "os_iters" => cfg.os_iters = v.parse().unwrap(),
            "small_total" => cfg.small_total = v.parse().unwrap(),
            "batch" => cfg.batch = v.parse().unwrap(),
            "epochs" => cfg.epochs = v.parse().unwrap(),
            "buffer_cap" => cfg.buffer_cap = v.parse().unwrap(),
            "lr" => cfg.lr = v.parse().unwrap(),
            "momentum" => cfg.momentum = v.parse().unwrap(),
            "l2" => cfg.l2 = v.parse().unwrap(),
            "val_every" => cfg.val_every = v.parse().unwrap(),
            "threads" => cfg.threads = v.parse().unwrap(),
            "outdir" => cfg.outdir = v.to_string(),
            "seed" => cfg.seed = v.parse().unwrap(),
            other => eprintln!("unknown arg: {other}"),
        }
    }
    println!(
        "training: iters={} warmup={} rounds/iter={} hidden={} threads={} -> {}",
        cfg.iters, cfg.warmup_iters, cfg.rounds_per_iter, cfg.hidden, cfg.threads, cfg.outdir
    );
    train(&cfg)?;
    println!("done. best net at {}/best.bin", cfg.outdir);
    Ok(())
}
