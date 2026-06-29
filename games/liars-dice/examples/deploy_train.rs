//! Production deploy training for the ReBeL Liar's Dice bot — the long (e.g. 24h)
//! run. Trains the single config-invariant PBS value net by bootstrapped
//! self-play on the real multi-round game (the mixed config sampler, biased to
//! the 5p5d6f flagship), checkpointing `ckpt.bin` into OUTDIR (default
//! `runs/ld_rebel`) for explicit ReBeL evaluation or experiments.
//!
//!     cargo run --release -p liars-dice --features parallel --example deploy_train
//!
//! Env overrides: STEPS HIDDEN NUM_ITERS GEN_PER WARMUP EVAL_EVERY TRAIN_RATIO
//! BUFFER OUTDIR SEED. Defaults target the flagship; raise STEPS to fill the
//! wall-clock budget (each step generates GEN_PER episodes, checkpointing every
//! EVAL_EVERY steps — interrupt any time and use the latest `ckpt.bin`).

use std::path::PathBuf;

use liars_dice::rebel::{DeployTrainConfig, DeployTrainer};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let threads = rayon::current_num_threads();
    let steps = env_usize("STEPS", 100_000);
    let hidden = env_usize("HIDDEN", 512);
    let num_iters = env_usize("NUM_ITERS", 256);
    let gen_per = env_usize("GEN_PER", threads);
    let warmup = env_usize("WARMUP", 200);
    let eval_every = env_usize("EVAL_EVERY", 50);
    let train_ratio = env_usize("TRAIN_RATIO", 8);
    let buffer = env_usize("BUFFER", 2_000_000);
    let seed = env_usize("SEED", 0) as u64;
    let outdir = PathBuf::from(std::env::var("OUTDIR").unwrap_or_else(|_| "runs/ld_rebel".into()));
    std::fs::create_dir_all(&outdir).unwrap();

    let cfg = DeployTrainConfig {
        steps,
        warmup_steps: warmup,
        num_iters,
        max_depth: 2,
        batch: 512,
        lr: 3e-4,
        gen_per_step: gen_per,
        train_gen_ratio: train_ratio,
        burn_in: 2048,
        eval_every,
        eval_iters: 256,
        eval_fit_iters: 400,
        hidden,
        n_layers: 2,
        buffer_cap: buffer,
        seed,
        log: true,
        outdir: outdir.clone(),
        fixed_config: None,
        ..DeployTrainConfig::default()
    };

    println!(
        "=== ReBeL deploy training (mixed sampler, 5p5d6f-biased) ===\n\
         hidden={hidden} num_iters={num_iters} depth=2 gen_per={gen_per} train_ratio={train_ratio} \
         buffer={buffer} threads={threads}\n\
         steps={steps} warmup={warmup} eval_every={eval_every} outdir={}",
        outdir.display()
    );
    let mut trainer = DeployTrainer::new(cfg);
    let report = trainer.run();
    println!(
        "=== done: samples={} train_steps={} — latest net at {}/ckpt.bin ===",
        report.samples_generated,
        report.train_steps,
        outdir.display()
    );
}
