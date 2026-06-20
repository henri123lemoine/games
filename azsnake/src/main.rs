//! AlphaZero 1v1 Snake (20×20) on Metal: batched self-play + tch training.
//!
//! ```text
//! cargo run --release -- run --dir ../data/azsnake/run1 --hours 5
//! cargo run --release -- bench
//! cargo run --release -- rate --dir ../data/azsnake/run1 --watch 20
//! cargo run --release -- elo --net ../data/azsnake/run1/latest.ot
//! ```
//!
//! One JSON line per iteration in `<dir>/metrics.jsonl`, `latest.ot`
//! checkpoints (auto-resume), periodic `ckpt-NNNNNN.ot` snapshots,
//! `dashboard.html` for the live view, and a `STOP` file for graceful
//! shutdown.

mod eval;
mod gauge;
mod net;
mod selfplay;
mod train;

use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{Instant, SystemTime};

use game_core::Rng;
use tch::{Device, Kind};

use eval::{Opponent, ladder};
use net::{Infer, NetConfig};
use selfplay::{SelfPlay, SelfPlayConfig, mix};
use solvers::azero::PuctConfig;
use train::{OptConfig, Replay, Trainer};

const DASHBOARD: &str = include_str!("../../assets/azsnake_dashboard.html");

pub(crate) fn arg<T: FromStr>(args: &[String], name: &str, default: T) -> T {
    arg_opt(args, name).unwrap_or(default)
}

pub(crate) fn arg_opt<T: FromStr>(args: &[String], name: &str) -> Option<T> {
    args.windows(2)
        .find(|w| w[0] == name)
        .and_then(|w| w[1].parse().ok())
}

/// Net architecture for a checkpoint: explicit `--blocks`/`--ch`/`--size`
/// flags win, then the checkpoint's own `<name>.json` sidecar, then the
/// latest `start` event in the metrics.jsonl beside it.
pub(crate) fn net_config_for(args: &[String], net_path: &Path) -> NetConfig {
    let from_json = |v: &serde_json::Value| {
        Some((
            v["blocks"].as_u64()? as usize,
            v["channels"].as_u64()? as i64,
            v["size"].as_u64().unwrap_or(20) as i64,
        ))
    };
    let sidecar = net_path
        .file_name()
        .and_then(|n| n.to_str())
        .map(|n| net_path.with_file_name(format!("{n}.json")))
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|text| from_json(&serde_json::from_str(&text).ok()?));
    let recorded = sidecar.or_else(|| {
        net_path
            .parent()
            .map(|d| d.join("metrics.jsonl"))
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|text| {
                text.lines().rev().find_map(|l| {
                    let v: serde_json::Value = serde_json::from_str(l).ok()?;
                    if v["event"] != "start" {
                        return None;
                    }
                    from_json(&v)
                })
            })
    });
    let blocks = arg_opt(args, "--blocks")
        .or(recorded.map(|r| r.0))
        .unwrap_or(4);
    let channels = arg_opt(args, "--ch")
        .or(recorded.map(|r| r.1))
        .unwrap_or(64);
    let size = arg_opt(args, "--size")
        .or(recorded.map(|r| r.2))
        .unwrap_or(20);
    NetConfig {
        blocks,
        channels,
        size,
    }
}

/// The most recent `pool_size` snapshot checkpoints (`ckpt-NNNNNN.ot`) in
/// `dir`, oldest-to-newest. The opponent pool draws from these.
fn snapshot_paths(dir: &Path, pool_size: usize) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut ckpts: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("ckpt-") && n.ends_with(".ot"))
        })
        .collect();
    ckpts.sort();
    if ckpts.len() > pool_size {
        ckpts.drain(..ckpts.len() - pool_size);
    }
    ckpts
}

pub(crate) fn append_line(path: &Path, line: &str) {
    use std::io::Write;
    let written = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut f| f.write_all(format!("{line}\n").as_bytes()));
    if let Err(e) = written {
        eprintln!("warning: dropped metrics line ({e})");
    }
}

pub(crate) fn last_lr(path: &Path) -> Option<f64> {
    let text = std::fs::read_to_string(path).ok()?;
    text.lines().rev().find_map(|l| {
        let v: serde_json::Value = serde_json::from_str(l).ok()?;
        if v.get("policy_loss").is_some() {
            v.get("lr")?.as_f64()
        } else {
            None
        }
    })
}

fn last_iter(path: &Path) -> u64 {
    let Ok(text) = std::fs::read_to_string(path) else {
        return 0;
    };
    text.lines()
        .rev()
        .find_map(|l| {
            serde_json::from_str::<serde_json::Value>(l)
                .ok()?
                .get("iter")?
                .as_u64()
        })
        .unwrap_or(0)
}

pub(crate) fn epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

pub(crate) fn device() -> Device {
    if tch::utils::has_mps() {
        Device::Mps
    } else {
        eprintln!("warning: MPS unavailable, training on CPU");
        Device::Cpu
    }
}

fn save_with_retry(trainer: &Trainer, path: &Path) {
    for attempt in 1..=2 {
        match trainer.save(path) {
            Ok(()) => return,
            Err(e) => eprintln!(
                "warning: checkpoint save to {} failed (attempt {attempt}): {e}",
                path.display()
            ),
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
}

#[allow(clippy::too_many_lines)]
fn run(args: &[String]) {
    let hours: f64 = arg(args, "--hours", 5.0);
    let dir: PathBuf = arg(args, "--dir", PathBuf::from("../data/azsnake/run1"));
    let sims: u32 = arg(args, "--sims", 320);
    let leaves: u32 = arg(args, "--leaves", 8);
    let concurrent: usize = arg(args, "--concurrent", 256);
    let samples_per_iter: usize = arg(args, "--samples-per-iter", 8192);
    let temp_plies: u16 = arg(args, "--temp-plies", 12);
    let alpha: f64 = arg(args, "--alpha", 1.0);
    let value_mix: f32 = arg(args, "--value-mix", 0.3);
    let resign_fp_target: f64 = arg(args, "--resign-fp-target", 0.05);
    let resign_q: f64 = arg(args, "--resign-q", 0.95);
    let resign_min_ply: u16 = arg(args, "--resign-ply", 20);
    let resign_off: f64 = arg(args, "--resign-off", 0.1);
    let fast_sims: u32 = arg(args, "--fast-sims", 32);
    let full_sims: u32 = arg(args, "--full-sims", sims.max(256));
    let full_prob: f64 = arg(args, "--full-prob", 0.0);
    let gamma: f32 = arg(args, "--gamma", 0.997);
    let margin_w: f32 = arg(args, "--margin-w", 0.25);
    let pool_frac: f64 = arg(args, "--pool-frac", 0.2);
    let pool_size: usize = arg(args, "--pool-size", 5);
    let forced_k: f32 = arg(args, "--forced-k", 0.0);
    let swa_decay: f64 = arg(args, "--swa-decay", 0.0);
    let use_sgd = arg(args, "--optimizer", String::from("adam")) == "sgd";
    let momentum: f64 = arg(args, "--momentum", 0.9);
    let grad_clip: f64 = arg(args, "--grad-clip", 0.0);
    let warmup_iters: u64 = arg(args, "--warmup-iters", 0);
    let batch: usize = arg(args, "--batch", 512);
    let reuse: f64 = arg(args, "--reuse", 1.8);
    let replay_cap: usize = arg(args, "--replay", 400_000);
    let lr: f64 = arg(args, "--lr", 1e-3);
    let weight_decay: f64 = arg(args, "--wd", 1e-4);
    let eval_every: u64 = arg(args, "--eval-every", 4);
    let eval_pairs: u32 = arg(args, "--eval-pairs", 16);
    let eval_sims: u32 = arg(args, "--eval-sims", 128);
    let snapshot_every: u64 = arg(args, "--snapshot-every", 30);

    let net_cfg = net_config_for(args, &dir.join("latest.ot"));
    let (blocks, channels, size) = (net_cfg.blocks, net_cfg.channels, net_cfg.size);
    let sp_cfg = SelfPlayConfig {
        puct: PuctConfig {
            sims,
            max_leaves: leaves,
            dirichlet_alpha: alpha,
            forced_playouts_k: forced_k,
            ..PuctConfig::default()
        },
        concurrent,
        temp_plies,
        resign_q,
        resign_min_ply,
        resign_off,
        fast_sims,
        full_sims,
        full_prob,
        gamma,
        margin_w,
        pool_frac,
    };

    std::fs::create_dir_all(&dir).expect("create run dir");
    std::fs::write(dir.join("dashboard.html"), DASHBOARD).expect("write dashboard");
    let latest = dir.join("latest.ot");
    let metrics = dir.join("metrics.jsonl");
    let stop = dir.join("STOP");
    if stop.exists() {
        std::fs::remove_file(&stop).expect("clear stale STOP file");
    }

    let dev = device();
    let opt_cfg = OptConfig {
        sgd: use_sgd,
        momentum,
        weight_decay,
        grad_clip,
    };
    let mut trainer = Trainer::new(dev, net_cfg, lr, value_mix, swa_decay, opt_cfg);
    let mut iter = 0u64;
    if latest.exists() {
        trainer.load(&latest).unwrap_or_else(|e| {
            eprintln!("failed to load {}: {e}", latest.display());
            std::process::exit(1);
        });
        iter = last_iter(&metrics);
        println!("resumed {} at iter {iter}", latest.display());
    } else if let Some(seed) = arg_opt::<PathBuf>(args, "--init-from") {
        trainer.init_from(&seed).unwrap_or_else(|e| {
            eprintln!("failed to seed from {}: {e}", seed.display());
            std::process::exit(1);
        });
        println!("seeded weights from {} (transfer)", seed.display());
    }
    let mut current_lr = lr;
    let mut lr_dropped = false;
    if iter > 0
        && let Some(prev) = last_lr(&metrics)
        && prev < lr
    {
        trainer.set_lr(prev);
        current_lr = prev;
        lr_dropped = true;
        println!("restored lr {prev} from the previous leg (base {lr})");
    }
    let mut pool = SelfPlay::new(sp_cfg, mix(0x60A1_5EED, 0));
    let mut replay = Replay::new(replay_cap);
    let mut calib_pool: std::collections::VecDeque<f64> = std::collections::VecDeque::new();
    let mut live_resign_q = resign_q;

    append_line(
        &metrics,
        &serde_json::json!({
            "event": "start", "time": epoch_secs(), "iter": iter,
            "blocks": blocks, "channels": channels, "size": size, "sims": sims,
            "concurrent": concurrent, "samples_per_iter": samples_per_iter,
            "batch_size": batch, "replay_capacity": replay_cap, "lr": lr,
            "eval_every": eval_every, "eval_pairs": eval_pairs,
            "eval_sims": eval_sims, "value_mix": value_mix,
            "fast_sims": fast_sims, "full_sims": full_sims, "full_prob": full_prob,
            "gamma": gamma, "margin_w": margin_w,
            "pool_frac": pool_frac, "pool_size": pool_size,
            "forced_k": forced_k, "swa_decay": swa_decay,
            "optimizer": if use_sgd { "sgd" } else { "adam" },
            "grad_clip": grad_clip, "warmup_iters": warmup_iters,
            "resign_fp_target": resign_fp_target, "alpha": alpha,
            "threads": rayon::current_num_threads(),
        })
        .to_string(),
    );
    println!(
        "run: {hours:.1}h budget, {blocks}x{channels} resnet, {size}x{size} board on {dev:?}, \
         {sims} sims/move, {concurrent} concurrent games, {samples_per_iter} samples/iter, dir {}",
        dir.display()
    );

    let budget_secs = hours * 3600.0;
    let mut work_secs = 0.0f64;
    let start = Instant::now();
    let opponents = [Opponent::Random, Opponent::Greedy, Opponent::Mcts(256)];
    let mut pool_paths: Vec<PathBuf> = Vec::new();
    loop {
        iter += 1;

        if pool_size > 0 && pool_frac > 0.0 {
            let latest_paths = snapshot_paths(&dir, pool_size);
            if latest_paths != pool_paths {
                let nets: Vec<Infer> = latest_paths
                    .iter()
                    .filter_map(|p| Infer::load(p, net_cfg, dev, Kind::Half).ok())
                    .collect();
                println!("opponent pool: {} past checkpoints", nets.len());
                pool.set_pool(nets);
                pool_paths = latest_paths;
            }
        }
        if warmup_iters > 0 {
            if iter <= warmup_iters {
                trainer.set_lr(current_lr * iter as f64 / warmup_iters as f64);
            } else if iter == warmup_iters + 1 {
                trainer.set_lr(current_lr);
            }
        }
        let sp_start = Instant::now();
        let infer = Infer::snapshot(trainer.infer_vs(), net_cfg, Kind::Half);
        let (samples, stats, calib) = pool.collect(&infer, samples_per_iter.max(1));
        let self_play_secs = sp_start.elapsed().as_secs_f32();
        let n_new = samples.len();
        replay.extend(samples);

        calib_pool.extend(calib);
        while calib_pool.len() > 1000 {
            calib_pool.pop_front();
        }
        if calib_pool.len() >= 100 {
            let mut sorted: Vec<f64> = calib_pool.iter().copied().collect();
            sorted.sort_by(f64::total_cmp);
            let q = ((resign_fp_target * sorted.len() as f64) as usize).min(sorted.len() - 1);
            let t = sorted[q];
            live_resign_q = (-t).clamp(0.5, 0.995);
            pool.set_resign_q(live_resign_q);
        }

        let steps = ((n_new as f64 * reuse) / batch as f64).ceil() as usize;
        let train_start = Instant::now();
        let (policy_loss, value_loss) =
            trainer.train(&replay, steps, batch, &mut Rng::new(mix(0xC0FFEE, iter)));
        trainer.update_swa();
        let train_secs = train_start.elapsed().as_secs_f32();

        save_with_retry(&trainer, &latest);
        if let Err(e) = trainer.save_swa(&dir.join("latest_swa.ot")) {
            eprintln!("save_swa failed: {e}");
        }
        if iter.is_multiple_of(snapshot_every) {
            save_with_retry(&trainer, &dir.join(format!("ckpt-{iter:06}.ot")));
        }

        let mut eval_fields: Option<(f32, serde_json::Value)> = None;
        let mut eval_human = String::new();
        let mut eval_work = 0.0f32;
        if iter == 1 || iter.is_multiple_of(eval_every) {
            let infer = Infer::snapshot(trainer.infer_vs(), net_cfg, Kind::Half);
            let t = Instant::now();
            let entries = ladder(&infer, &opponents, eval_pairs, eval_sims, mix(0xE7A1, iter));
            let eval_secs = t.elapsed().as_secs_f32();
            eval_work = eval_secs;
            let table: serde_json::Map<String, serde_json::Value> = entries
                .iter()
                .map(|e| {
                    (
                        e.name.clone(),
                        serde_json::json!({
                            "score": e.score, "w": e.wins, "d": e.draws, "l": e.losses
                        }),
                    )
                })
                .collect();
            eval_fields = Some((eval_secs, serde_json::Value::Object(table)));
            eval_human = format!(
                " | eval [{eval_secs:.0}s] {}",
                entries
                    .iter()
                    .map(|e| format!("{} {:.2}", e.name, e.score))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }

        let elapsed_min = start.elapsed().as_secs_f64() / 60.0;
        let mut line = serde_json::json!({
            "iter": iter, "time": epoch_secs(), "elapsed_min": elapsed_min,
            "policy_loss": policy_loss, "value_loss": value_loss,
            "n_new": n_new, "games": stats.games, "seat0_wins": stats.seat0_wins,
            "avg_plies": stats.avg_plies(), "buffer": replay.len(),
            "self_play_secs": self_play_secs, "train_secs": train_secs,
            "resigned": stats.resigned, "capped": stats.capped,
            "would_resign": stats.would_resign, "resign_fp": stats.resign_fp,
            "resign_q": live_resign_q, "lr": current_lr,
        });
        if let Some((eval_secs, table)) = eval_fields {
            line["eval_secs"] = serde_json::json!(eval_secs);
            line["eval"] = table;
        }
        append_line(&metrics, &line.to_string());
        println!(
            "iter {iter:>4} [{elapsed_min:>6.1}m] loss {:.3} (p {policy_loss:.3} + v {value_loss:.3}) | \
             {} games, {} seat0 wins ({} resign, {} capped), avg {:>3.0} plies, buffer {:>7} | \
             sp {self_play_secs:>5.1}s train {train_secs:>4.1}s{eval_human}",
            policy_loss + value_loss,
            stats.games,
            stats.seat0_wins,
            stats.resigned,
            stats.capped,
            stats.avg_plies(),
            replay.len(),
        );

        work_secs += f64::from(self_play_secs) + f64::from(train_secs) + f64::from(eval_work);
        if !lr_dropped && work_secs > 0.6 * budget_secs {
            current_lr = lr * 0.3;
            trainer.set_lr(current_lr);
            lr_dropped = true;
            println!("lr {lr} -> {current_lr} at 60% of work budget");
        }

        let reason = if stop.exists() {
            Some("STOP file")
        } else if work_secs >= budget_secs {
            Some("work budget reached")
        } else {
            None
        };
        if let Some(reason) = reason {
            append_line(
                &metrics,
                &serde_json::json!({
                    "event": "stop", "time": epoch_secs(), "iter": iter, "reason": reason,
                })
                .to_string(),
            );
            println!(
                "stopping ({reason}) after iter {iter}; checkpoint at {}",
                latest.display()
            );
            break;
        }
    }
}

fn bench(args: &[String]) {
    use game_core::{Game, PolicyValueEncoder};
    use snake::Duel;
    use snake::encode::SnakeEncoder;

    let blocks: usize = arg(args, "--blocks", 4);
    let channels: i64 = arg(args, "--ch", 64);
    let size: i64 = arg(args, "--size", 20);
    let sims: u32 = arg(args, "--sims", 320);
    let leaves: u32 = arg(args, "--leaves", 8);
    let concurrent: usize = arg(args, "--concurrent", 256);
    let samples: usize = arg(args, "--samples", 4096);

    let dev = device();
    let net_cfg = NetConfig {
        blocks,
        channels,
        size,
    };
    let trainer = Trainer::new(
        dev,
        net_cfg,
        1e-3,
        0.3,
        0.0,
        OptConfig {
            sgd: false,
            momentum: 0.9,
            weight_decay: 1e-4,
            grad_clip: 0.0,
        },
    );
    let infer = Infer::snapshot(&trainer.vs, net_cfg, Kind::Half);
    let sp_cfg = SelfPlayConfig {
        puct: PuctConfig {
            sims,
            max_leaves: leaves,
            dirichlet_alpha: 1.0,
            ..PuctConfig::default()
        },
        concurrent,
        ..SelfPlayConfig::default()
    };
    let mut pool = SelfPlay::new(sp_cfg, 0xBE7C);

    println!(
        "bench: {blocks}x{channels} resnet, {size}x{size} board on {dev:?}, {sims} sims, {concurrent} games"
    );

    let game = Duel::new();
    let enc = SnakeEncoder::new();
    let synth = |n: usize| {
        let mut s = game.initial_state();
        let outs = game.chance_outcomes(&s);
        game.apply(&mut s, outs[0].0);
        let actions = game.legal_actions(&s);
        (0..n)
            .map(|_| net::EvalRequest {
                features: enc.encode_state(&game, &s),
                support: actions
                    .iter()
                    .map(|&a| enc.action_index(&game, &s, a) as u16)
                    .collect(),
            })
            .collect::<Vec<_>>()
    };
    for bs in [256usize, 1024, 4096] {
        let reqs = synth(bs);
        infer.forward_batch(&reqs);
        let t = Instant::now();
        let iters = 20;
        for _ in 0..iters {
            infer.forward_batch(&reqs);
        }
        let dt = t.elapsed().as_secs_f64();
        println!(
            "forward_batch {bs:>5}: {:>7.0} evals/s ({:.1} ms/batch)",
            bs as f64 * iters as f64 / dt,
            dt / iters as f64 * 1000.0
        );
    }

    let t0 = Instant::now();
    let (s, stats, _calib) = pool.collect(&infer, samples);
    let dt = t0.elapsed().as_secs_f64();
    let spg = stats.avg_plies().max(1.0) as f64;
    println!(
        "self-play: {} samples in {dt:.1}s = {:.0} samples/s ≈ {:.0} games/h \
         ({} games finished, avg {:.0} plies, {:.0}k sims/s; cpu {:.1}s gpu {:.1}s; \
         ends: {} seat0 wins, {} resign, {} cap)",
        s.len(),
        s.len() as f64 / dt,
        s.len() as f64 / dt / spg * 3600.0,
        stats.games,
        spg,
        s.len() as f64 / dt * f64::from(sims) / 1000.0,
        stats.cpu_secs,
        stats.gpu_secs,
        stats.seat0_wins,
        stats.resigned,
        stats.capped,
    );
    println!(
        "  {} gpu batches, avg width {:.0}, {:.1} ms/batch",
        stats.batches,
        stats.evals as f64 / f64::from(stats.batches.max(1)),
        stats.gpu_secs as f64 * 1000.0 / f64::from(stats.batches.max(1)),
    );

    let mut trainer = trainer;
    let mut replay = Replay::new(1_000_000);
    replay.extend(s);
    let t1 = Instant::now();
    let steps = 30;
    let (pl, vl) = trainer.train(&replay, steps, 512, &mut Rng::new(7));
    let dt = t1.elapsed().as_secs_f64();
    println!(
        "train: {steps} steps of 512 in {dt:.1}s = {:.0} samples/s (losses p {pl:.3} v {vl:.3})",
        steps as f64 * 512.0 / dt
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("run") => run(&args[1..]),
        Some("bench") => bench(&args[1..]),
        Some("rate") => gauge::rate(&args[1..]),
        Some("elo") => gauge::elo_gauge(&args[1..]),
        _ => {
            eprintln!(
                "usage: azsnake run   [--dir ../data/azsnake/run1] [--hours 5] [--size 20] \
                 [--blocks 4] [--ch 64] [--sims 320] [--leaves 8] [--concurrent 256] \
                 [--samples-per-iter 8192] [--temp-plies 12] [--alpha 1.0] [--value-mix 0.3] \
                 [--gamma 0.997] [--margin-w 0.25] [--pool-frac 0.2] [--pool-size 5] \
                 [--resign-fp-target 0.05] [--resign-q 0.95] [--resign-ply 20] [--resign-off 0.1] \
                 [--batch 512] [--reuse 1.8] [--replay 400000] [--lr 1e-3] [--wd 1e-4] \
                 [--eval-every 4] [--eval-pairs 16] [--eval-sims 128] [--snapshot-every 30]"
            );
            eprintln!(
                "       azsnake bench [--size 20] [--blocks 4] [--ch 64] [--sims 320] [--leaves 8] \
                 [--concurrent 256] [--samples 4096]"
            );
            eprintln!(
                "       azsnake rate [--dir ../data/azsnake/run1] [--pairs 8] [--sims 96] \
                 [--watch <minutes>]  (current net vs oldest snapshot)"
            );
            eprintln!(
                "       azsnake elo  [--net ../data/azsnake/run1/latest.ot] [--sims 200] [--pairs 16] \
                 [--watch <minutes>]  (win rate vs random/greedy/mcts baseline)"
            );
            std::process::exit(2);
        }
    }
}
