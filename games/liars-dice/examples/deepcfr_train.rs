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
//!         eval_rollouts=20 eval_games=10 outdir=runs/ld_deepcfr_smoke
//!
//!     # longer run
//!     cargo run --release -p liars-dice --features parallel \
//!         --example deepcfr_train -- \
//!         iters=8000 block=200 warmup_iters=1500 hidden=256 threads=18 \
//!         outdir=runs/ld_deepcfr

use std::io;

use liars_dice::deepcfr::{DeepCfrTrainConfig, train};

fn main() -> io::Result<()> {
    let cfg = parse_args(std::env::args().skip(1)).map_err(invalid_input)?;
    println!(
        "deep cfr: train={} iters={} block={} warmup={} traversals={} hidden={} \
         explore_eps={} threads={} eval={}x{} keep_ckpts={} -> {}",
        if cfg.mixed {
            "mixed-family".to_string()
        } else {
            format!("{}p{}d{}f", cfg.players, cfg.dice, cfg.faces)
        },
        cfg.iters,
        cfg.block,
        cfg.warmup_iters,
        cfg.traversals,
        cfg.hidden,
        cfg.explore_eps,
        cfg.threads,
        cfg.eval_rollouts,
        cfg.eval_games,
        cfg.keep_checkpoints,
        cfg.outdir
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

fn parse_args<I, S>(raw_args: I) -> Result<DeepCfrTrainConfig, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut cfg = DeepCfrTrainConfig::default();
    for raw in raw_args {
        let arg = raw.as_ref();
        let Some((k, v)) = arg.split_once('=') else {
            return Err(format!("argument must be key=value, got '{arg}'"));
        };
        match k {
            "mixed" => cfg.mixed = parse_bool(v, k)?,
            "players" => cfg.players = parse_num(v, k)?,
            "dice" => cfg.dice = parse_num(v, k)?,
            "faces" => cfg.faces = parse_num(v, k)?,
            "iters" => cfg.iters = parse_num(v, k)?,
            "block" => cfg.block = parse_num(v, k)?,
            "warmup_iters" => cfg.warmup_iters = parse_num(v, k)?,
            "traversals" => cfg.traversals = parse_num(v, k)?,
            "train_every" => cfg.train_every = parse_num(v, k)?,
            "hidden" => cfg.hidden = parse_num(v, k)?,
            "adv_reservoir" => cfg.adv_reservoir = parse_num(v, k)?,
            "strat_reservoir" => cfg.strat_reservoir = parse_num(v, k)?,
            "adv_steps" => cfg.adv_steps = parse_num(v, k)?,
            "strat_steps" => cfg.strat_steps = parse_num(v, k)?,
            "batch" => cfg.batch = parse_num(v, k)?,
            "lr" => cfg.lr = parse_num(v, k)?,
            "momentum" => cfg.momentum = parse_num(v, k)?,
            "l2" => cfg.l2 = parse_num(v, k)?,
            "explore_eps" => cfg.explore_eps = parse_num(v, k)?,
            "threads" => cfg.threads = parse_num(v, k)?,
            "outdir" => cfg.outdir = v.to_string(),
            "seed" => cfg.seed = parse_num(v, k)?,
            "eval_rollouts" => cfg.eval_rollouts = parse_num(v, k)?,
            "eval_games" => cfg.eval_games = parse_num(v, k)?,
            "keep_checkpoints" => cfg.keep_checkpoints = parse_bool(v, k)?,
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
