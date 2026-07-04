//! Train the Liar's Dice policy/value net by distilling per-round CFR/MCCFR
//! equilibria (see [`liars_dice::train`]). CPU only.
//!
//! `warmup_iters` solves against the fixed dice-share heuristic; afterwards the
//! net's own value head closes the round leaves (fitted value iteration). Set
//! `warmup_iters >= iters` to reproduce the pure-distillation baseline.
//!
//!     # quick smoke run (~1-2 min)
//!     cargo run --release -p liars-dice --example train_net -- \
//!         iters=3 rounds_per_iter=80 hidden=128 threads=4 eval_games=20 outdir=runs/ld_smoke
//!     # emits runs/ld_smoke/metrics.jsonl plus durable ckpt_N.bin curve points
//!
//!     # overnight-scale run
//!     cargo run --release -p liars-dice --example train_net -- \
//!         iters=400 warmup_iters=8 rounds_per_iter=600 hidden=256 threads=4 outdir=runs/ld_net

use std::io;

use liars_dice::train::{TrainConfig, train};

fn main() -> io::Result<()> {
    let cfg = parse_args(std::env::args().skip(1)).map_err(invalid_input)?;
    println!(
        "training: iters={} warmup={} rounds/iter={} hidden={} threads={} eval={}x{} keep_ckpts={} -> {}",
        cfg.iters,
        cfg.warmup_iters,
        cfg.rounds_per_iter,
        cfg.hidden,
        cfg.threads,
        cfg.eval_rollouts,
        cfg.eval_games,
        cfg.keep_checkpoints,
        cfg.outdir
    );
    train(&cfg)?;
    println!("done. best net at {}/best.bin", cfg.outdir);
    Ok(())
}

fn parse_args<I, S>(raw_args: I) -> Result<TrainConfig, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut cfg = TrainConfig::default();
    for raw in raw_args {
        let arg = raw.as_ref();
        let Some((k, v)) = arg.split_once('=') else {
            return Err(format!("argument must be key=value, got '{arg}'"));
        };
        match k {
            "iters" => cfg.iters = parse_num(v, k)?,
            "warmup_iters" => cfg.warmup_iters = parse_num(v, k)?,
            "rounds_per_iter" => cfg.rounds_per_iter = parse_num(v, k)?,
            "playouts" => cfg.playouts = parse_num(v, k)?,
            "hidden" => cfg.hidden = parse_num(v, k)?,
            "cfr_iters" => cfg.cfr_iters = parse_num(v, k)?,
            "es_iters" => cfg.es_iters = parse_num(v, k)?,
            "small_total" => cfg.small_total = parse_num(v, k)?,
            "batch" => cfg.batch = parse_num(v, k)?,
            "epochs" => cfg.epochs = parse_num(v, k)?,
            "buffer_cap" => cfg.buffer_cap = parse_num(v, k)?,
            "lr" => cfg.lr = parse_num(v, k)?,
            "momentum" => cfg.momentum = parse_num(v, k)?,
            "l2" => cfg.l2 = parse_num(v, k)?,
            "val_every" => cfg.val_every = parse_num(v, k)?,
            "eval_rollouts" => cfg.eval_rollouts = parse_num(v, k)?,
            "eval_games" => cfg.eval_games = parse_num(v, k)?,
            "keep_checkpoints" => cfg.keep_checkpoints = parse_bool(v, k)?,
            "threads" => cfg.threads = parse_num(v, k)?,
            "outdir" => cfg.outdir = v.to_string(),
            "seed" => cfg.seed = parse_num(v, k)?,
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok(cfg)
}

fn invalid_input(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

fn parse_num<T>(value: &str, key: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse()
        .map_err(|e| format!("failed to parse {key}='{value}': {e}"))
}

fn parse_bool(value: &str, key: &str) -> Result<bool, String> {
    match value {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(format!(
            "failed to parse {key}='{value}' as boolean (use 1/0, true/false, yes/no, on/off)"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::parse_args;

    #[test]
    fn rejects_unknown_args() {
        assert!(parse_args(["unknown=1"]).is_err());
    }

    #[test]
    fn rejects_malformed_args() {
        assert!(parse_args(["iters"]).is_err());
    }

    #[test]
    fn rejects_bad_values() {
        assert!(parse_args(["iters=nope"]).is_err());
        assert!(parse_args(["keep_checkpoints=maybe"]).is_err());
    }
}
