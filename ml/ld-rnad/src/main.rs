use liars_dice::pg_train::{PgArchitecture, PgTrainConfig, train};
use std::io;

fn main() -> io::Result<()> {
    let cfg = parse_args(std::env::args().skip(1)).map_err(invalid_input)?;
    println!(
        "ld-rnad: target {}p{}d{}f arch={} train={} iters={} episodes/iter={} hidden={} eval={}x{} expl={} keep_ckpts={} -> {}",
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

fn parse_args<I, S>(raw_args: I) -> Result<PgTrainConfig, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut cfg = PgTrainConfig::default();
    for raw in raw_args {
        let arg = raw.as_ref();
        let Some((key, value)) = arg.split_once('=') else {
            return Err(format!("argument must be key=value, got '{arg}'"));
        };
        match key {
            "players" => cfg.players = parse_num(value, key)?,
            "dice" => cfg.dice = parse_num(value, key)?,
            "faces" => cfg.faces = parse_num(value, key)?,
            "architecture" | "arch" => {
                cfg.architecture = PgArchitecture::parse(value)
                    .ok_or_else(|| format!("unknown architecture '{value}'"))?;
            }
            "mixed" => cfg.mixed = parse_bool(value, key)?,
            "min_players" => cfg.min_players = parse_num(value, key)?,
            "max_players" => cfg.max_players = parse_num(value, key)?,
            "min_dice" => cfg.min_dice = parse_num(value, key)?,
            "max_dice" => cfg.max_dice = parse_num(value, key)?,
            "min_faces" => cfg.min_faces = parse_num(value, key)?,
            "max_faces" => cfg.max_faces = parse_num(value, key)?,
            "iters" => cfg.iters = parse_num(value, key)?,
            "episodes_per_iter" | "episodes" => cfg.episodes_per_iter = parse_num(value, key)?,
            "max_episode_len" => cfg.max_episode_len = parse_num(value, key)?,
            "hidden" => cfg.hidden = parse_num(value, key)?,
            "batch" => cfg.batch = parse_num(value, key)?,
            "epochs" => cfg.epochs = parse_num(value, key)?,
            "lr" => cfg.lr = parse_num(value, key)?,
            "momentum" => cfg.momentum = parse_num(value, key)?,
            "l2" => cfg.l2 = parse_num(value, key)?,
            "entropy" => cfg.entropy = parse_num(value, key)?,
            "anchor" => cfg.anchor = parse_num(value, key)?,
            "anchor_update" => cfg.anchor_update = parse_num(value, key)?,
            "adv_clip" => cfg.adv_clip = parse_num(value, key)?,
            "val_every" => cfg.val_every = parse_num(value, key)?,
            "eval_games" => cfg.eval_games = parse_num(value, key)?,
            "eval_rollouts" => cfg.eval_rollouts = parse_num(value, key)?,
            "eval_exploitability" => cfg.eval_exploitability = parse_bool(value, key)?,
            "keep_checkpoints" => cfg.keep_checkpoints = parse_bool(value, key)?,
            "outdir" => cfg.outdir = value.to_string(),
            "seed" => cfg.seed = parse_num(value, key)?,
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
        assert!(parse_args(["architecture=nope"]).is_err());
    }
}
