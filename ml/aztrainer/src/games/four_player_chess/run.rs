//! Four-player chess training loop: batched MPS self-play, four-seat
//! cross-entropy values, replay training, checkpoint league, and seat-rotated
//! field evaluation.

use std::path::{Path, PathBuf};
use std::time::Instant;

use four_player_chess::encode::{PLANE_COUNT, POLICY_LEN};
use game_core::Rng;
use nn_infer::HeadKind;
use tch::Kind;

use super::eval::{self, Opponent};
use super::sample::Sample;
use super::selfplay::{SelfPlay, SelfPlayConfig, mix};
use crate::net::{Infer, NetConfig};
use crate::rundir::{append_line, device, epoch_secs, save_with_retry};
use crate::train::{OptConfig, Replay, Trainer};
use solvers::azero::PuctConfig;

const DASHBOARD: &str = include_str!("../../../../../assets/azero_dashboard.html");

pub(super) fn config(blocks: usize, channels: i64) -> NetConfig {
    NetConfig {
        blocks,
        channels,
        planes: PLANE_COUNT as i64,
        size: 14,
        head: HeadKind::FlatConv,
        policy_len: POLICY_LEN as i64,
        go_aux: false,
        seats: 4,
    }
}

pub(super) fn arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    arg_opt(args, name).unwrap_or(default)
}

pub(super) fn arg_opt<T: std::str::FromStr>(args: &[String], name: &str) -> Option<T> {
    args.windows(2)
        .find(|window| window[0] == name)
        .and_then(|window| window[1].parse().ok())
}

pub(super) fn net_config_for(args: &[String], path: &Path) -> NetConfig {
    let (blocks, channels, _) = crate::rundir::resolve_arch(
        arg_opt(args, "--blocks"),
        arg_opt(args, "--ch"),
        Some(14),
        path,
        (4, 48, 14),
    );
    config(blocks, channels)
}

fn league_checkpoints(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir.join("league")) else {
        return Vec::new();
    };
    let mut paths: Vec<_> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("ot"))
        .collect();
    paths.sort();
    paths
}

fn assert_durable_run_dir(dir: &Path) {
    if dir.components().any(|part| part.as_os_str() == ".Codex") {
        panic!("training output must not live under .Codex; use a durable root runs/ directory");
    }
}

#[allow(clippy::too_many_lines)]
pub fn run(args: &[String]) {
    let hours: f64 = arg(args, "--hours", 8.0);
    let dir: PathBuf = arg(args, "--dir", PathBuf::from("../../runs/four-player-chess"));
    assert_durable_run_dir(&dir);
    let sims: u32 = arg(args, "--sims", 96);
    let leaves: u32 = arg(args, "--leaves", 8);
    let concurrent: usize = arg(args, "--concurrent", 64);
    let samples_per_iter: usize = arg(args, "--samples-per-iter", 2048);
    let temp_plies: u16 = arg(args, "--temp-plies", 48);
    let ply_cap: u16 = arg(args, "--ply-cap", 320);
    let batch: usize = arg(args, "--batch", 128);
    let reuse: f64 = arg(args, "--reuse", 1.5);
    let replay_cap: usize = arg(args, "--replay", 150_000);
    let lr: f64 = arg(args, "--lr", 1e-3);
    let weight_decay: f64 = arg(args, "--wd", 1e-4);
    let use_sgd = arg(args, "--optimizer", String::from("adam")) == "sgd";
    let momentum: f64 = arg(args, "--momentum", 0.9);
    let grad_clip: f64 = arg(args, "--grad-clip", 1.0);
    let swa_decay: f64 = arg(args, "--swa-decay", 0.0);
    let snapshot_every: u64 = arg(args, "--snapshot-every", 5);
    let league_every: u64 = arg(args, "--league-every", 2);
    let eval_every: u64 = arg(args, "--eval-every", 5);
    let eval_games: u32 = arg(args, "--eval-games", 8);
    let eval_sims: u32 = arg(args, "--eval-sims", 48);
    let max_iters: u64 = arg(args, "--max-iters", 0);

    std::fs::create_dir_all(&dir).expect("create run dir");
    std::fs::create_dir_all(dir.join("league")).expect("create league dir");
    std::fs::write(dir.join("dashboard.html"), DASHBOARD).expect("write dashboard");
    let stop = dir.join("STOP");
    if stop.exists() {
        std::fs::remove_file(&stop).expect("clear stale STOP file");
    }
    let latest = dir.join("latest.ot");
    let metrics = dir.join("metrics.jsonl");
    let net_cfg = net_config_for(args, &latest);
    let dev = device();
    let mut trainer = Trainer::new(
        dev,
        net_cfg,
        lr,
        0.0,
        swa_decay,
        OptConfig {
            sgd: use_sgd,
            momentum,
            weight_decay,
            grad_clip,
        },
    );
    let mut iter = 0u64;
    if latest.exists() {
        trainer.load(&latest).expect("resume latest checkpoint");
        iter = crate::rundir::last_iter(&metrics);
        println!("resumed {} at iter {iter}", latest.display());
    }

    let sp_cfg = SelfPlayConfig {
        puct: PuctConfig {
            sims,
            max_leaves: leaves,
            cycle_draws: true,
            ..PuctConfig::default()
        },
        concurrent,
        temp_plies,
        ply_cap,
    };
    let mut pool = SelfPlay::new(sp_cfg, 0x4C48_4553_5346_4641);
    let mut replay: Replay<Sample> = Replay::new(replay_cap);
    append_line(
        &metrics,
        &serde_json::json!({
            "event": "start", "time": epoch_secs(), "iter": iter,
            "blocks": net_cfg.blocks, "channels": net_cfg.channels, "size": 14,
            "planes": net_cfg.planes, "policy": net_cfg.policy_len, "value_seats": 4,
            "sims": sims, "concurrent": concurrent, "samples_per_iter": samples_per_iter,
            "batch_size": batch, "replay_capacity": replay_cap, "lr": lr,
            "ply_cap": ply_cap, "value_target": "terminal-placement", "league_every": league_every,
            "eval_every": eval_every, "eval_games": eval_games, "eval_sims": eval_sims,
            "device": format!("{dev:?}"), "threads": rayon::current_num_threads(),
        })
        .to_string(),
    );
    println!(
        "run: {hours:.2}h, {}x{} four-seat resnet on {dev:?}, {sims} sims, {concurrent} games, dir {}",
        net_cfg.blocks,
        net_cfg.channels,
        dir.display()
    );

    let budget = hours * 3600.0;
    let mut work_secs = 0.0;
    loop {
        iter += 1;
        let checkpoints = league_checkpoints(&dir);
        let past_path = (!checkpoints.is_empty())
            .then(|| checkpoints[(iter as usize - 1) % checkpoints.len()].clone());
        let current = Infer::snapshot(trainer.infer_vs(), net_cfg, Kind::Half);
        let past = past_path.as_ref().map(|path| {
            Infer::load(path, net_cfg, dev, Kind::Half).unwrap_or_else(|error| {
                panic!("load league checkpoint {}: {error}", path.display())
            })
        });
        let started = Instant::now();
        let (samples, stats) = pool.collect(&current, past.as_ref(), samples_per_iter.max(1));
        let self_play_secs = started.elapsed().as_secs_f32();
        let n_new = samples.len();
        replay.extend(samples);

        let steps = ((n_new as f64 * reuse) / batch as f64).ceil() as usize;
        let train_started = Instant::now();
        let (policy_loss, value_loss) =
            trainer.train(&replay, steps, batch, &mut Rng::new(mix(0xC0FFEE, iter)));
        trainer.update_swa();
        let train_secs = train_started.elapsed().as_secs_f32();
        save_with_retry(&trainer, &latest);
        if snapshot_every > 0 && iter.is_multiple_of(snapshot_every) {
            save_with_retry(&trainer, &dir.join(format!("ckpt-{iter:06}.ot")));
        }
        if league_every > 0 && iter.is_multiple_of(league_every) {
            save_with_retry(
                &trainer,
                &dir.join("league").join(format!("ckpt-{iter:06}.ot")),
            );
        }

        let mut eval_json = serde_json::Value::Null;
        let mut eval_text = String::new();
        let mut eval_secs = 0.0f32;
        if eval_every > 0 && iter.is_multiple_of(eval_every) {
            let eval_started = Instant::now();
            let infer = Infer::snapshot(trainer.infer_vs(), net_cfg, Kind::Half);
            let mut entries = Vec::new();
            for opponent in [Opponent::Random, Opponent::Greedy, Opponent::Mobility] {
                entries.push(eval::vs_baseline(
                    &infer,
                    opponent,
                    eval_games.max(4),
                    eval_sims,
                    ply_cap,
                    mix(0xE7A1, iter ^ opponent.name().len() as u64),
                ));
            }
            if let Some(path) = past_path.as_ref() {
                let old = Infer::load(path, net_cfg, dev, Kind::Half).expect("load past eval net");
                entries.push(eval::vs_past(
                    &infer,
                    &old,
                    eval_games.max(4),
                    eval_sims,
                    ply_cap,
                    mix(0x1EA6, iter),
                ));
            }
            eval_secs = eval_started.elapsed().as_secs_f32();
            eval_text = entries
                .iter()
                .map(|entry| {
                    format!(
                        "{} win {:.3}/score {:.3}",
                        entry.name,
                        entry.win_share(),
                        entry.score_share
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            eval_json = serde_json::json!(
                entries
                    .iter()
                    .map(|entry| serde_json::json!({
                        "opponent": entry.name, "games": entry.games,
                        "strict_wins": entry.strict_wins, "win_share": entry.win_share(),
                        "score_share": entry.score_share, "fair": 0.25,
                    }))
                    .collect::<Vec<_>>()
            );
        }

        work_secs += f64::from(self_play_secs + train_secs + eval_secs);
        let line = serde_json::json!({
            "iter": iter, "time": epoch_secs(), "policy_loss": policy_loss,
            "value_loss": value_loss, "n_new": n_new, "buffer": replay.len(),
            "games": stats.games, "league_games": stats.league_games,
            "avg_plies": stats.avg_plies(), "capped": stats.capped,
            "self_play_secs": self_play_secs, "train_secs": train_secs,
            "eval_secs": eval_secs, "eval": eval_json,
            "league_checkpoint": past_path.as_ref().map(|path| path.display().to_string()),
        });
        append_line(&metrics, &line.to_string());
        println!(
            "iter {iter:>4} loss {:.3} (p {policy_loss:.3} + v {value_loss:.3}) | {} games ({} league), avg {:.0} plies, {} samples | sp {:.1}s train {:.1}s{}",
            policy_loss + value_loss,
            stats.games,
            stats.league_games,
            stats.avg_plies(),
            n_new,
            self_play_secs,
            train_secs,
            if eval_text.is_empty() {
                String::new()
            } else {
                format!(" | {eval_text}")
            },
        );

        let reason = if stop.exists() {
            Some("STOP file")
        } else if work_secs >= budget {
            Some("work budget reached")
        } else if max_iters > 0 && iter >= max_iters {
            Some("max iters reached")
        } else {
            None
        };
        if let Some(reason) = reason {
            append_line(
                &metrics,
                &serde_json::json!({
                    "event": "stop", "time": epoch_secs(), "iter": iter, "reason": reason
                })
                .to_string(),
            );
            println!("stopping ({reason}); checkpoint at {}", latest.display());
            break;
        }
    }
}

pub fn evaluate(args: &[String]) {
    let net: PathBuf = arg(
        args,
        "--net",
        PathBuf::from("../../runs/four-player-chess/latest.ot"),
    );
    let games: u32 = arg(args, "--games", 16);
    let sims: u32 = arg(args, "--sims", 96);
    let ply_cap: u16 = arg(args, "--ply-cap", 320);
    let cfg = net_config_for(args, &net);
    let infer = Infer::load(&net, cfg, device(), Kind::Half).expect("load net");
    for opponent in [Opponent::Random, Opponent::Greedy, Opponent::Mobility] {
        let entry = eval::vs_baseline(&infer, opponent, games.max(4), sims, ply_cap, 7);
        println!(
            "{}: strict wins {}/{} = {:.3} (fair 0.250), score share {:.3}",
            entry.name,
            entry.strict_wins,
            entry.games,
            entry.win_share(),
            entry.score_share,
        );
    }
}
