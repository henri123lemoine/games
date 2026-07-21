//! Canonical simultaneous Battlesnake training entry and preflight benchmark.

use std::path::{Path, PathBuf};
use std::time::Instant;

use game_core::Rng;
use nn_infer::HeadKind;
use tch::{Device, Kind};

use super::sim_selfplay::{BackupMethod, SelfPlay, SelfPlayConfig, mix};
use crate::net::{Infer, NetConfig};
use crate::rundir::{append_line, device, epoch_secs, save_with_retry};
use crate::train::{OptConfig, Replay, Trainer};

const ACTIONS: i64 = 4;
const BOARD_SIZE: i64 = snake::battlesnake::SIDE as i64;

pub fn config(blocks: usize, channels: i64, size: i64) -> NetConfig {
    assert_eq!(size, BOARD_SIZE, "canonical Battlesnake is 11x11");
    NetConfig {
        blocks,
        channels,
        planes: snake::battlesnake_encode::PLANES as i64,
        size,
        head: HeadKind::GlobalPoolDense,
        policy_len: ACTIONS,
        go_aux: false,
        seats: 1,
    }
}

pub fn parse_arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    arg_opt(args, name).unwrap_or(default)
}

fn arg_opt<T: std::str::FromStr>(args: &[String], name: &str) -> Option<T> {
    args.windows(2)
        .find(|window| window[0] == name)
        .and_then(|window| window[1].parse().ok())
}

pub fn net_config_for(args: &[String], net_path: &Path) -> NetConfig {
    let (blocks, channels, size) = crate::rundir::resolve_arch(
        arg_opt(args, "--blocks"),
        arg_opt(args, "--ch"),
        arg_opt(args, "--size"),
        net_path,
        (4, 64, BOARD_SIZE),
    );
    config(blocks, channels, size)
}

pub fn device_for() -> Device {
    device()
}

#[allow(clippy::too_many_lines)]
pub fn run(args: &[String]) {
    let players: usize = parse_arg(args, "--players", 2);
    let method: BackupMethod = parse_arg(args, "--method", BackupMethod::Logit);
    let default_dir = PathBuf::from(format!("runs/battlesnake/{}-p{players}", method.name()));
    let dir: PathBuf = parse_arg(args, "--dir", default_dir);
    let hours: f64 = parse_arg(args, "--hours", 0.5);
    let default_concurrent = if players == 4 { 2 } else { 128 };
    let concurrent: usize = parse_arg(args, "--concurrent", default_concurrent);
    let samples_per_iter: usize = parse_arg(args, "--samples-per-iter", 4096);
    let rationality: f32 = parse_arg(args, "--rationality", 8.0);
    let solve_iters: usize = parse_arg(args, "--solve-iters", 32);
    let damping: f32 = parse_arg(args, "--damping", 0.5);
    let root_noise: f32 = parse_arg(args, "--root-noise", 0.15);
    let dirichlet_alpha: f64 = parse_arg(args, "--alpha", 1.0);
    let sample_turns: u16 = parse_arg(args, "--sample-turns", 20);
    let gamma: f32 = parse_arg(args, "--gamma", 0.997);
    let max_turns: u16 = parse_arg(args, "--max-turns", 750);
    let batch: usize = parse_arg(args, "--batch", 256);
    let reuse: f64 = parse_arg(args, "--reuse", 1.5);
    let replay_cap: usize = parse_arg(args, "--replay", 50_000);
    let lr: f64 = parse_arg(args, "--lr", 1e-3);
    let value_mix: f32 = parse_arg(args, "--value-mix", 0.25);
    let weight_decay: f64 = parse_arg(args, "--wd", 1e-4);
    let grad_clip: f64 = parse_arg(args, "--grad-clip", 1.0);
    let swa_decay: f64 = parse_arg(args, "--swa-decay", 0.0);
    let snapshot_every: u64 = parse_arg(args, "--snapshot-every", 10);
    let max_iters: u64 = parse_arg(args, "--max-iters", 0);
    let seed: u64 = parse_arg(args, "--seed", 0xBA77_1E5A);

    if !(2..=4).contains(&players) {
        panic!("--players must be 2, 3, or 4");
    }
    if method == BackupMethod::Maximin && players != 2 {
        panic!("--method maximin is defined only for two-player zero-sum games");
    }
    let run_args = RunArgs {
        args,
        method,
        dir,
        hours,
        concurrent,
        samples_per_iter,
        rationality,
        solve_iters,
        damping,
        root_noise,
        dirichlet_alpha,
        sample_turns,
        gamma,
        max_turns,
        batch,
        reuse,
        replay_cap,
        lr,
        value_mix,
        weight_decay,
        grad_clip,
        swa_decay,
        snapshot_every,
        max_iters,
        seed,
    };
    match players {
        2 => run_players::<2>(run_args),
        3 => run_players::<3>(run_args),
        4 => run_players::<4>(run_args),
        _ => unreachable!(),
    }
}

struct RunArgs<'a> {
    args: &'a [String],
    method: BackupMethod,
    dir: PathBuf,
    hours: f64,
    concurrent: usize,
    samples_per_iter: usize,
    rationality: f32,
    solve_iters: usize,
    damping: f32,
    root_noise: f32,
    dirichlet_alpha: f64,
    sample_turns: u16,
    gamma: f32,
    max_turns: u16,
    batch: usize,
    reuse: f64,
    replay_cap: usize,
    lr: f64,
    value_mix: f32,
    weight_decay: f64,
    grad_clip: f64,
    swa_decay: f64,
    snapshot_every: u64,
    max_iters: u64,
    seed: u64,
}

fn run_players<const N: usize>(a: RunArgs<'_>) {
    std::fs::create_dir_all(&a.dir).expect("create durable run directory");
    let latest = a.dir.join("latest.ot");
    let metrics = a.dir.join("metrics.jsonl");
    let stop = a.dir.join("STOP");
    if stop.exists() {
        std::fs::remove_file(&stop).expect("clear stale STOP file");
    }

    let net_cfg = net_config_for(a.args, &latest);
    let dev = device();
    let mut trainer = Trainer::new(
        dev,
        net_cfg,
        a.lr,
        a.value_mix,
        a.swa_decay,
        OptConfig {
            sgd: false,
            momentum: 0.9,
            weight_decay: a.weight_decay,
            grad_clip: a.grad_clip,
        },
    );
    let mut iter = crate::rundir::last_iter(&metrics);
    let continuity;
    if latest.exists() {
        trainer
            .load(&latest)
            .unwrap_or_else(|error| panic!("failed to resume {}: {error}", latest.display()));
        println!(
            "checkpoint continuation: {} at iter {iter} (weights restored; optimizer and replay fresh)",
            latest.display()
        );
        continuity = "weights-only-continuation";
    } else if let Some(seed_path) = arg_opt::<PathBuf>(a.args, "--init-from") {
        trainer
            .init_from(&seed_path)
            .unwrap_or_else(|error| panic!("failed to seed from {}: {error}", seed_path.display()));
        println!("fresh run seeded from {}", seed_path.display());
        continuity = "fresh-dir-seeded-weights";
    } else {
        println!("fresh scratch run");
        continuity = "fresh-scratch";
    }

    let sp_cfg = SelfPlayConfig {
        concurrent: a.concurrent,
        method: a.method,
        rationality: a.rationality,
        solve_iters: a.solve_iters,
        damping: a.damping,
        root_noise: a.root_noise,
        dirichlet_alpha: a.dirichlet_alpha,
        sample_turns: a.sample_turns,
        gamma: a.gamma,
        max_turns: a.max_turns,
        safety_mask: true,
    };
    let mut selfplay = SelfPlay::<N>::new(sp_cfg, mix(a.seed, iter));
    let mut replay = Replay::new(a.replay_cap);
    let mut train_rng = Rng::new(mix(a.seed, 0x7A11_0000));
    let inference_kind = if dev == Device::Cpu {
        Kind::Float
    } else {
        Kind::Half
    };

    append_line(
        &metrics,
        &serde_json::json!({
            "event": "start", "time": epoch_secs(), "iter": iter,
            "continuity": continuity,
            "rules": "canonical-simultaneous-11x11", "players": N,
            "method": a.method.name(), "joint_actions": 4usize.pow(N as u32),
            "hours": a.hours, "seed": a.seed, "max_iters": a.max_iters,
            "blocks": net_cfg.blocks, "channels": net_cfg.channels,
            "planes": net_cfg.planes, "size": net_cfg.size,
            "concurrent": a.concurrent, "samples_per_iter": a.samples_per_iter,
            "rationality": a.rationality, "solve_iters": a.solve_iters,
            "damping": a.damping, "root_noise": a.root_noise,
            "dirichlet_alpha": a.dirichlet_alpha, "sample_turns": a.sample_turns,
            "safety_mask": true,
            "gamma": a.gamma, "max_turns": a.max_turns,
            "batch": a.batch, "reuse": a.reuse, "replay": a.replay_cap,
            "lr": a.lr, "value_mix": a.value_mix,
            "weight_decay": a.weight_decay, "grad_clip": a.grad_clip,
            "swa_decay": a.swa_decay, "snapshot_every": a.snapshot_every,
            "device": format!("{dev:?}"),
        })
        .to_string(),
    );
    println!(
        "run: {:.2}h {} p{N}, {}x{} net, {} concurrent, dir {} on {dev:?}",
        a.hours,
        a.method.name(),
        net_cfg.blocks,
        net_cfg.channels,
        a.concurrent,
        a.dir.display(),
    );

    let started = Instant::now();
    loop {
        if stop.exists() || (a.max_iters > 0 && iter >= a.max_iters) {
            break;
        }
        if started.elapsed().as_secs_f64() >= a.hours * 3600.0 {
            break;
        }
        iter += 1;
        let infer = Infer::snapshot(trainer.infer_vs(), net_cfg, inference_kind);
        let selfplay_started = Instant::now();
        let (samples, stats) = selfplay.collect(&infer, a.samples_per_iter.max(1));
        let selfplay_secs = selfplay_started.elapsed().as_secs_f32();
        let new_samples = samples.len();
        replay.extend(samples);

        let steps = ((new_samples as f64 * a.reuse) / a.batch as f64)
            .ceil()
            .max(1.0) as usize;
        let train_started = Instant::now();
        let (policy_loss, value_loss) = trainer.train(&replay, steps, a.batch, &mut train_rng);
        trainer.update_swa();
        let train_secs = train_started.elapsed().as_secs_f32();
        save_with_retry(&trainer, &latest);
        if a.snapshot_every > 0 && iter.is_multiple_of(a.snapshot_every) {
            save_with_retry(&trainer, &a.dir.join(format!("ckpt-{iter:06}.ot")));
        }
        let samples_per_sec = new_samples as f32 / selfplay_secs.max(1e-6);
        append_line(
            &metrics,
            &serde_json::json!({
                "event": "train", "time": epoch_secs(), "iter": iter,
                "method": a.method.name(), "players": N,
                "samples": new_samples, "replay_size": replay.len(), "steps": steps,
                "games": stats.games, "avg_turns": stats.avg_turns(),
                "draws": stats.draws, "capped": stats.capped,
                "root_evals": stats.root_evals, "leaf_evals": stats.leaf_evals,
                "selfplay_secs": selfplay_secs, "train_secs": train_secs,
                "samples_per_sec": samples_per_sec,
                "policy_loss": policy_loss, "value_loss": value_loss,
                "elapsed_hours": started.elapsed().as_secs_f64() / 3600.0,
            })
            .to_string(),
        );
        println!(
            "iter {iter}: {new_samples} samples ({samples_per_sec:.0}/s), {} games x {:.1} turns, \
             replay {}, train {steps} steps, loss p={policy_loss:.4} v={value_loss:.4}",
            stats.games,
            stats.avg_turns(),
            replay.len(),
        );
    }
    save_with_retry(&trainer, &latest);
    append_line(
        &metrics,
        &serde_json::json!({
            "event": "stop", "time": epoch_secs(), "iter": iter,
            "elapsed_hours": started.elapsed().as_secs_f64() / 3600.0,
        })
        .to_string(),
    );
    println!("stopped cleanly at iter {iter}: {}", latest.display());
}

pub fn bench(args: &[String]) {
    let players: usize = parse_arg(args, "--players", 2);
    let method: BackupMethod = parse_arg(args, "--method", BackupMethod::Logit);
    let samples: usize = parse_arg(args, "--samples", 512);
    let concurrent: usize = parse_arg(args, "--concurrent", if players == 4 { 2 } else { 32 });
    let blocks: usize = parse_arg(args, "--blocks", 4);
    let channels: i64 = parse_arg(args, "--ch", 64);
    let rationality: f32 = parse_arg(args, "--rationality", 8.0);
    let solve_iters: usize = parse_arg(args, "--solve-iters", 32);
    let damping: f32 = parse_arg(args, "--damping", 0.5);
    let safety_mask: bool = parse_arg(args, "--safety-mask", true);
    tch::manual_seed(7);
    let dev = device();
    let cfg = config(blocks, channels, BOARD_SIZE);
    let mut trainer = Trainer::new(
        dev,
        cfg,
        1e-3,
        0.25,
        0.0,
        OptConfig {
            sgd: false,
            momentum: 0.9,
            weight_decay: 1e-4,
            grad_clip: 1.0,
        },
    );
    let kind = if dev == Device::Cpu {
        Kind::Float
    } else {
        Kind::Half
    };
    let infer = Infer::snapshot(trainer.infer_vs(), cfg, kind);
    let sp_cfg = SelfPlayConfig {
        concurrent,
        method,
        rationality,
        solve_iters,
        damping,
        safety_mask,
        ..SelfPlayConfig::default()
    };
    let started = Instant::now();
    let (generated, stats) = match players {
        2 => SelfPlay::<2>::new(sp_cfg, 7).collect(&infer, samples),
        3 => SelfPlay::<3>::new(sp_cfg, 7).collect(&infer, samples),
        4 => SelfPlay::<4>::new(sp_cfg, 7).collect(&infer, samples),
        _ => panic!("--players must be 2, 3, or 4"),
    };
    let elapsed = started.elapsed().as_secs_f32();
    let generated_count = generated.len();
    drop(infer);
    let mut replay = Replay::new(generated.len());
    replay.extend(generated);
    let train_started = Instant::now();
    let mut rng = Rng::new(9);
    let train_batch = samples.clamp(1, 128);
    let (policy_loss, value_loss) = trainer.train(&replay, 2, train_batch, &mut rng);
    let train_elapsed = train_started.elapsed().as_secs_f32();
    println!(
        "preflight: {} p{players} mask={safety_mask}, {generated_count} samples (target {samples}) \
         in {elapsed:.2}s ({:.0}/s), \
         {} root + {} leaf evals, {} games, {:.1} avg turns, {} caps; \
         two train steps in {train_elapsed:.2}s, loss p={policy_loss:.4} v={value_loss:.4}",
        method.name(),
        generated_count as f32 / elapsed.max(1e-6),
        stats.root_evals,
        stats.leaf_evals,
        stats.games,
        stats.avg_turns(),
        stats.capped,
    );
}
