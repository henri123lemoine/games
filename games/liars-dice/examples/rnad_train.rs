//! Train the Liar's Dice regularized actor-critic self-play contender.
//!
//! This emits `metrics.jsonl`, `ckpt.bin`, durable `ckpt_N.bin` curve points,
//! and `best.bin`, all in the same MLP artifact format consumed by
//! `examples/tournament -- nets=...`.
//!
//!     cargo run --release -p liars-dice --example rnad_train -- \
//!         players=2 dice=1 faces=2 iters=2 episodes_per_iter=8 hidden=16 \
//!         eval_games=0 eval_exploitability=0 mixed=0 outdir=runs/ld_rnad_smoke
//!
//!     cargo run --release -p liars-dice --example rnad_train -- \
//!         mixed=1 max_players=5 max_dice=8 iters=400 episodes_per_iter=512 \
//!         hidden=256 eval_games=200 \
//!         outdir=runs/ld_rnad
//!
//!     cargo run --release -p liars-dice --example rnad_train -- \
//!         architecture=history mixed=1 max_players=5 max_dice=8 iters=400 \
//!         episodes_per_iter=512 hidden=256 outdir=runs/ld_history

use std::io;

use liars_dice::pg_train::{PgArchitecture, PgTrainConfig, train};

fn main() -> io::Result<()> {
    let cfg = parse_args(std::env::args().skip(1)).map_err(invalid_input)?;
    println!(
        "rnad training: target {}p{}d{}f arch={} train={} iters={} episodes/iter={} hidden={} eval={}x{} expl={} keep_ckpts={} -> {}",
        cfg.players,
        cfg.dice,
        cfg.faces,
        cfg.architecture.as_str(),
        train_range(&cfg),
        cfg.iters,
        cfg.episodes_per_iter,
        cfg.hidden,
        cfg.eval_rollouts,
        cfg.eval_games,
        cfg.eval_exploitability,
        cfg.keep_checkpoints,
        cfg.outdir
    );
    train(&cfg)?;
    println!("done. best net at {}/best.bin", cfg.outdir);
    Ok(())
}

fn parse_args<I, S>(raw_args: I) -> Result<PgTrainConfig, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut cfg = PgTrainConfig::default();
    for raw in raw_args {
        let arg = raw.as_ref();
        let Some((k, v)) = arg.split_once('=') else {
            return Err(format!("argument must be key=value, got '{arg}'"));
        };
        match k {
            "players" => cfg.players = parse_num(v, k)?,
            "dice" => cfg.dice = parse_num(v, k)?,
            "faces" => cfg.faces = parse_num(v, k)?,
            "architecture" | "arch" => {
                cfg.architecture = PgArchitecture::parse(v)
                    .ok_or_else(|| format!("unknown architecture '{v}'"))?;
            }
            "mixed" => cfg.mixed = parse_bool(v, k)?,
            "min_players" => cfg.min_players = parse_num(v, k)?,
            "max_players" => cfg.max_players = parse_num(v, k)?,
            "min_dice" => cfg.min_dice = parse_num(v, k)?,
            "max_dice" => cfg.max_dice = parse_num(v, k)?,
            "min_faces" => cfg.min_faces = parse_num(v, k)?,
            "max_faces" => cfg.max_faces = parse_num(v, k)?,
            "iters" => cfg.iters = parse_num(v, k)?,
            "episodes_per_iter" | "episodes" => cfg.episodes_per_iter = parse_num(v, k)?,
            "max_episode_len" => cfg.max_episode_len = parse_num(v, k)?,
            "hidden" => cfg.hidden = parse_num(v, k)?,
            "batch" => cfg.batch = parse_num(v, k)?,
            "epochs" => cfg.epochs = parse_num(v, k)?,
            "lr" => cfg.lr = parse_num(v, k)?,
            "momentum" => cfg.momentum = parse_num(v, k)?,
            "l2" => cfg.l2 = parse_num(v, k)?,
            "entropy" => cfg.entropy = parse_num(v, k)?,
            "anchor" => cfg.anchor = parse_num(v, k)?,
            "anchor_update" => cfg.anchor_update = parse_num(v, k)?,
            "adv_clip" => cfg.adv_clip = parse_num(v, k)?,
            "val_every" => cfg.val_every = parse_num(v, k)?,
            "eval_games" => cfg.eval_games = parse_num(v, k)?,
            "eval_rollouts" => cfg.eval_rollouts = parse_num(v, k)?,
            "eval_exploitability" => cfg.eval_exploitability = parse_bool(v, k)?,
            "keep_checkpoints" => cfg.keep_checkpoints = parse_bool(v, k)?,
            "outdir" => cfg.outdir = v.to_string(),
            "seed" => cfg.seed = parse_num(v, k)?,
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok(cfg)
}

fn train_range(cfg: &PgTrainConfig) -> String {
    if cfg.mixed {
        format!(
            "{}-{}p{}-{}d{}-{}f",
            cfg.min_players,
            cfg.max_players,
            cfg.min_dice,
            cfg.max_dice,
            cfg.min_faces,
            cfg.max_faces
        )
    } else {
        format!("{}p{}d{}f", cfg.players, cfg.dice, cfg.faces)
    }
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
        assert!(parse_args(["architecture=nope"]).is_err());
    }
}
