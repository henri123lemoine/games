//! Checkpoint progress and absolute-strength gauges for simultaneous Battlesnake.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use tch::Kind;

use super::eval::{
    Opponent, field_ladder, ladder, net_vs_net, net_vs_net_field, net_vs_net_split, score_interval,
};
use super::run::{device_for, net_config_for, parse_arg as arg};
use super::sim_selfplay::{BackupMethod, SolveConfig};
use crate::net::Infer;
use crate::rundir::{append_line, epoch_secs};

fn solve(args: &[String]) -> SolveConfig {
    SolveConfig {
        method: arg(args, "--method", BackupMethod::Logit),
        rationality: arg(args, "--rationality", 8.0),
        solve_iters: arg(args, "--solve-iters", 32),
        damping: arg(args, "--damping", 0.5),
    }
}

fn solve_method(
    args: &[String],
    method_name: &str,
    rationality_name: &str,
    iterations_name: &str,
    damping_name: &str,
    default: BackupMethod,
) -> SolveConfig {
    let global = solve(args);
    SolveConfig {
        method: arg(args, method_name, default),
        rationality: arg(args, rationality_name, global.rationality),
        solve_iters: arg(args, iterations_name, global.solve_iters),
        damping: arg(args, damping_name, global.damping),
    }
}

fn oldest_snapshot(dir: &Path) -> Option<(PathBuf, u64)> {
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let iteration = path
                .file_name()?
                .to_str()?
                .strip_prefix("ckpt-")?
                .strip_suffix(".ot")?
                .parse()
                .ok()?;
            Some((path, iteration))
        })
        .min_by_key(|&(_, iteration)| iteration)
}

pub fn rate(args: &[String]) {
    let dir: PathBuf = arg(args, "--dir", PathBuf::from("runs/battlesnake/logit-p2"));
    let pairs: u32 = arg(args, "--pairs", 16);
    let seed: u64 = arg(args, "--seed", 0x4A7E_u64);
    let watch_minutes: f64 = arg(args, "--watch", 0.0);
    let latest = dir.join("latest.ot");
    let metrics = dir.join("metrics.jsonl");
    let dev = device_for();
    let cfg = net_config_for(args, &latest);
    let kind = if dev == tch::Device::Cpu {
        Kind::Float
    } else {
        Kind::Half
    };
    loop {
        let latest_iter = crate::rundir::last_iter(&metrics);
        match oldest_snapshot(&dir) {
            Some((reference, reference_iter)) if reference_iter < latest_iter => {
                let first = Infer::load(&latest, cfg, dev, kind);
                let second = Infer::load(&reference, cfg, dev, kind);
                match (first, second) {
                    (Ok(first), Ok(second)) => {
                        let started = Instant::now();
                        let (wins, draws, losses) =
                            net_vs_net(&first, solve(args), &second, solve(args), pairs, seed);
                        let (score, ci_low, ci_high) = score_interval(wins, draws, losses, 0.5);
                        println!(
                            "rate: iter {latest_iter} vs {reference_iter}: {score:.3} \
                             (95% CI {ci_low:.3}..{ci_high:.3}, {wins}-{draws}-{losses}, {:.1}s)",
                            started.elapsed().as_secs_f32()
                        );
                        append_line(
                            &metrics,
                            &serde_json::json!({
                                "event": "rate", "time": epoch_secs(), "iter": latest_iter,
                                "ref_iter": reference_iter, "score": score,
                                "ci_low": ci_low, "ci_high": ci_high,
                                "wins": wins, "draws": draws, "losses": losses,
                            })
                            .to_string(),
                        );
                    }
                    _ => eprintln!("rate: checkpoint load failed"),
                }
            }
            _ => println!("rate: waiting for an older snapshot"),
        }
        if watch_minutes <= 0.0 {
            break;
        }
        std::thread::sleep(Duration::from_secs_f64(watch_minutes * 60.0));
    }
}

pub fn elo_gauge(args: &[String]) {
    let net_path: PathBuf = arg(
        args,
        "--net",
        PathBuf::from("runs/battlesnake/logit-p2/latest.ot"),
    );
    let pairs: u32 = arg(args, "--pairs", 16);
    let seed: u64 = arg(args, "--seed", 0x510_u64);
    let watch_minutes: f64 = arg(args, "--watch", 0.0);
    let dev = device_for();
    let cfg = net_config_for(args, &net_path);
    let kind = if dev == tch::Device::Cpu {
        Kind::Float
    } else {
        Kind::Half
    };
    let metrics = net_path
        .parent()
        .unwrap_or(Path::new("."))
        .join("metrics.jsonl");
    let opponents = [
        Opponent::Random,
        Opponent::Search {
            millis: 1,
            depth: 4,
        },
        Opponent::Search {
            millis: 5,
            depth: 8,
        },
    ];
    loop {
        let infer = Infer::load(&net_path, cfg, dev, kind)
            .unwrap_or_else(|error| panic!("failed to load {}: {error}", net_path.display()));
        let started = Instant::now();
        let entries = ladder(&infer, solve(args), &opponents, pairs, seed);
        let detail = entries
            .iter()
            .map(|entry| {
                format!(
                    "{}:{:.3}[{:.3},{:.3}]({}-{}-{})",
                    entry.name,
                    entry.score,
                    entry.ci_low,
                    entry.ci_high,
                    entry.wins,
                    entry.draws,
                    entry.losses
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        println!(
            "strength: {detail} ({:.1}s)",
            started.elapsed().as_secs_f32()
        );
        append_line(
            &metrics,
            &serde_json::json!({
                "event": "strength", "time": epoch_secs(), "detail": detail,
            })
            .to_string(),
        );
        if watch_minutes <= 0.0 {
            break;
        }
        std::thread::sleep(Duration::from_secs_f64(watch_minutes * 60.0));
    }
}

pub fn field_gauge(args: &[String]) {
    let net_path: PathBuf = arg(
        args,
        "--net",
        PathBuf::from("runs/battlesnake/logit-p4/latest.ot"),
    );
    let sets: u32 = arg(args, "--sets", 8);
    let seed: u64 = arg(args, "--seed", 0xF1E1_D400_u64);
    let dev = device_for();
    let cfg = net_config_for(args, &net_path);
    let kind = if dev == tch::Device::Cpu {
        Kind::Float
    } else {
        Kind::Half
    };
    let infer = Infer::load(&net_path, cfg, dev, kind)
        .unwrap_or_else(|error| panic!("failed to load {}: {error}", net_path.display()));
    let opponents = [
        Opponent::Random,
        Opponent::Search {
            millis: 1,
            depth: 4,
        },
        Opponent::Search {
            millis: 5,
            depth: 8,
        },
    ];
    let started = Instant::now();
    let entries = field_ladder(&infer, solve(args), &opponents, sets, seed);
    let detail = entries
        .iter()
        .map(|entry| {
            format!(
                "{}:{:.3}[{:.3},{:.3}]({}-{}-{})",
                entry.name,
                entry.score,
                entry.ci_low,
                entry.ci_high,
                entry.wins,
                entry.draws,
                entry.losses
            )
        })
        .collect::<Vec<_>>()
        .join(" ");
    println!(
        "field win share (fair=0.250): {detail} ({:.1}s, {} seat-rotated games/opponent)",
        started.elapsed().as_secs_f32(),
        sets * 4,
    );
}

/// Deterministic paired-seat comparison for two independently trained
/// checkpoints. Each network retains its own simultaneous backup rule; the
/// opponent never observes the other network's current action.
pub fn compare(args: &[String]) {
    let first_path: PathBuf = arg(
        args,
        "--first",
        PathBuf::from("runs/battlesnake/logit-p2/latest.ot"),
    );
    let second_path: PathBuf = arg(
        args,
        "--second",
        PathBuf::from("runs/battlesnake/maximin-p2/latest.ot"),
    );
    let first_solve = solve_method(
        args,
        "--first-method",
        "--first-rationality",
        "--first-solve-iters",
        "--first-damping",
        BackupMethod::Logit,
    );
    let second_solve = solve_method(
        args,
        "--second-method",
        "--second-rationality",
        "--second-solve-iters",
        "--second-damping",
        BackupMethod::Maximin,
    );
    let pairs: u32 = arg(args, "--pairs", 32);
    let seed: u64 = arg(args, "--seed", 0xBA_CE_0F_F0_u64);
    let dev = device_for();
    let kind = if dev == tch::Device::Cpu {
        Kind::Float
    } else {
        Kind::Half
    };
    let first_cfg = net_config_for(args, &first_path);
    let second_cfg = net_config_for(args, &second_path);
    let first = Infer::load(&first_path, first_cfg, dev, kind)
        .unwrap_or_else(|error| panic!("failed to load {}: {error}", first_path.display()));
    let second = Infer::load(&second_path, second_cfg, dev, kind)
        .unwrap_or_else(|error| panic!("failed to load {}: {error}", second_path.display()));
    let started = Instant::now();
    let (wins, draws, losses) = net_vs_net(&first, first_solve, &second, second_solve, pairs, seed);
    let (score, ci_low, ci_high) = score_interval(wins, draws, losses, 0.5);
    println!(
        "compare: {} ({} r={:.1} i={} d={:.2}) vs {} ({} r={:.1} i={} d={:.2}): {score:.3} \
         (95% CI {ci_low:.3}..{ci_high:.3}, {wins}-{draws}-{losses}, {:.1}s, seed {seed})",
        first_path.display(),
        first_solve.method.name(),
        first_solve.rationality,
        first_solve.solve_iters,
        first_solve.damping,
        second_path.display(),
        second_solve.method.name(),
        second_solve.rationality,
        second_solve.solve_iters,
        second_solve.damping,
        started.elapsed().as_secs_f32(),
    );
}

pub fn field_compare(args: &[String]) {
    let first_path: PathBuf = arg(
        args,
        "--first",
        PathBuf::from("runs/battlesnake/logit-p4-a/latest.ot"),
    );
    let second_path: PathBuf = arg(
        args,
        "--second",
        PathBuf::from("runs/battlesnake/logit-p4-b/latest.ot"),
    );
    let first_solve = solve_method(
        args,
        "--first-method",
        "--first-rationality",
        "--first-solve-iters",
        "--first-damping",
        BackupMethod::Logit,
    );
    let second_solve = solve_method(
        args,
        "--second-method",
        "--second-rationality",
        "--second-solve-iters",
        "--second-damping",
        BackupMethod::Logit,
    );
    let sets: u32 = arg(args, "--sets", 8);
    let seed: u64 = arg(args, "--seed", 0xF1E1_DA7A_u64);
    let dev = device_for();
    let kind = if dev == tch::Device::Cpu {
        Kind::Float
    } else {
        Kind::Half
    };
    let first = Infer::load(&first_path, net_config_for(args, &first_path), dev, kind)
        .unwrap_or_else(|error| panic!("failed to load {}: {error}", first_path.display()));
    let second = Infer::load(&second_path, net_config_for(args, &second_path), dev, kind)
        .unwrap_or_else(|error| panic!("failed to load {}: {error}", second_path.display()));
    let started = Instant::now();
    let first_field = net_vs_net_field(&first, first_solve, &second, second_solve, sets, seed);
    eprintln!(
        "field compare progress: first composition complete ({:.1}s)",
        started.elapsed().as_secs_f32()
    );
    let second_field = net_vs_net_field(&second, second_solve, &first, first_solve, sets, seed);
    let first_score = score_interval(first_field.0, first_field.1, first_field.2, 0.0);
    let second_score = score_interval(second_field.0, second_field.1, second_field.2, 0.0);
    println!(
        "field compare (fair=0.250, {} games/composition, seed {seed}):\n  {} hero vs 3x {}: \
         {:.3} (95% CI {:.3}..{:.3}, {}-{}-{})\n  {} hero vs 3x {}: {:.3} \
         (95% CI {:.3}..{:.3}, {}-{}-{}); {:.1}s",
        sets * 4,
        first_path.display(),
        second_path.display(),
        first_score.0,
        first_score.1,
        first_score.2,
        first_field.0,
        first_field.1,
        first_field.2,
        second_path.display(),
        first_path.display(),
        second_score.0,
        second_score.1,
        second_score.2,
        second_field.0,
        second_field.1,
        second_field.2,
        started.elapsed().as_secs_f32(),
    );
}

pub fn split_compare(args: &[String]) {
    let first_path: PathBuf = arg(
        args,
        "--first",
        PathBuf::from("runs/battlesnake/logit-p4-a/latest.ot"),
    );
    let second_path: PathBuf = arg(
        args,
        "--second",
        PathBuf::from("runs/battlesnake/logit-p4-b/latest.ot"),
    );
    let first_solve = solve_method(
        args,
        "--first-method",
        "--first-rationality",
        "--first-solve-iters",
        "--first-damping",
        BackupMethod::Logit,
    );
    let second_solve = solve_method(
        args,
        "--second-method",
        "--second-rationality",
        "--second-solve-iters",
        "--second-damping",
        BackupMethod::Logit,
    );
    let sets: u32 = arg(args, "--sets", 8);
    let seed: u64 = arg(args, "--seed", 0x5A11_7DA7A_u64);
    let dev = device_for();
    let kind = if dev == tch::Device::Cpu {
        Kind::Float
    } else {
        Kind::Half
    };
    let first = Infer::load(&first_path, net_config_for(args, &first_path), dev, kind)
        .unwrap_or_else(|error| panic!("failed to load {}: {error}", first_path.display()));
    let second = Infer::load(&second_path, net_config_for(args, &second_path), dev, kind)
        .unwrap_or_else(|error| panic!("failed to load {}: {error}", second_path.display()));
    let started = Instant::now();
    let result = net_vs_net_split(&first, first_solve, &second, second_solve, sets, seed);
    let score = score_interval(result.0, result.1, result.2, 0.5);
    println!(
        "split compare (fair=0.500, {} games, seed {seed}):\n  {} (2 snakes) vs {} (2 snakes): \
         {:.3} (95% CI {:.3}..{:.3}, {}-{}-{}); {:.1}s",
        sets * 6,
        first_path.display(),
        second_path.display(),
        score.0,
        score.1,
        score.2,
        result.0,
        result.1,
        result.2,
        started.elapsed().as_secs_f32(),
    );
}
