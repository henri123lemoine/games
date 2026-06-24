//! Train the Liar's Dice policy net by Deep CFR (Brown et al. 2019) over the
//! config family — the second training method alongside the distillation in
//! `train_net.rs`. The output is the same artifact (a `solvers::azero::Mlp`
//! played by `liars_dice::NetAgent`), saved to `{outdir}/best.bin`.
//!
//! Build with the `parallel` feature so the net-gradient kernels use rayon
//! (the trainer otherwise runs single-threaded):
//!
//!     # quick smoke run over the family (~minutes)
//!     cargo run --release -p liars-dice --features parallel \
//!         --example deepcfr_train -- \
//!         iters=600 block=100 warmup_iters=300 hidden=128 threads=18 \
//!         outdir=runs/ld_deepcfr_smoke
//!
//!     # longer run
//!     cargo run --release -p liars-dice --features parallel \
//!         --example deepcfr_train -- \
//!         iters=8000 block=200 warmup_iters=1500 hidden=256 threads=18 \
//!         outdir=runs/ld_deepcfr

use liars_dice::deepcfr::{DeepCfrTrainConfig, train};

fn main() -> std::io::Result<()> {
    let mut cfg = DeepCfrTrainConfig::default();
    for arg in std::env::args().skip(1) {
        let Some((k, v)) = arg.split_once('=') else {
            eprintln!("ignoring arg without '=': {arg}");
            continue;
        };
        match k {
            "iters" => cfg.iters = v.parse().unwrap(),
            "block" => cfg.block = v.parse().unwrap(),
            "warmup_iters" => cfg.warmup_iters = v.parse().unwrap(),
            "traversals" => cfg.traversals = v.parse().unwrap(),
            "hidden" => cfg.hidden = v.parse().unwrap(),
            "adv_reservoir" => cfg.adv_reservoir = v.parse().unwrap(),
            "strat_reservoir" => cfg.strat_reservoir = v.parse().unwrap(),
            "adv_steps" => cfg.adv_steps = v.parse().unwrap(),
            "strat_steps" => cfg.strat_steps = v.parse().unwrap(),
            "batch" => cfg.batch = v.parse().unwrap(),
            "lr" => cfg.lr = v.parse().unwrap(),
            "momentum" => cfg.momentum = v.parse().unwrap(),
            "l2" => cfg.l2 = v.parse().unwrap(),
            "threads" => cfg.threads = v.parse().unwrap(),
            "outdir" => cfg.outdir = v.to_string(),
            "seed" => cfg.seed = v.parse().unwrap(),
            other => eprintln!("unknown arg: {other}"),
        }
    }
    println!(
        "deep cfr: iters={} block={} warmup={} traversals={} hidden={} threads={} -> {}",
        cfg.iters, cfg.block, cfg.warmup_iters, cfg.traversals, cfg.hidden, cfg.threads, cfg.outdir
    );
    #[cfg(not(feature = "parallel"))]
    eprintln!(
        "note: built WITHOUT the `parallel` feature — net training is single-threaded. \
         Rebuild with `--features parallel` to use `threads`."
    );
    train(&cfg)?;
    println!("done. best net at {}/best.bin", cfg.outdir);
    Ok(())
}
