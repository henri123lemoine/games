//! The cheap progress gauge: the current net vs an older snapshot of itself
//! (KataGo-style relative rating), plus a coarse absolute strength readout
//! against the fixed baseline ladder (random / greedy / shallow MCTS). Pente has
//! no external engine anchor, so there is no calibrated Elo.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use tch::Kind;

use super::eval::{self, Opponent, ladder};
use super::run::{device_for, net_config_for, parse_arg as arg};
use super::selfplay::mix;
use crate::net::Infer;
use crate::rundir::{append_line, epoch_secs};

fn metrics_last_iter(path: &Path) -> u64 {
    crate::rundir::last_iter(path)
}

/// The oldest `ckpt-NNNNNN.ot` snapshot in `dir` — the fixed baseline the rate
/// signal measures improvement against.
fn oldest_snapshot(dir: &Path) -> Option<(PathBuf, u64)> {
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| {
            let p = e.ok()?.path();
            let it: u64 = p
                .file_name()?
                .to_str()?
                .strip_prefix("ckpt-")?
                .strip_suffix(".ot")?
                .parse()
                .ok()?;
            Some((p, it))
        })
        .min_by_key(|&(_, it)| it)
}

/// The cheap, continuous progress signal (KataGo-style relative rating): the
/// current net vs an older snapshot of itself at low sims. A win rate above 0.5
/// means the net is still improving. Appends `{"event":"rate",...}`; `--watch N`
/// repeats every N minutes.
pub fn rate(args: &[String]) {
    let dir: PathBuf = arg(args, "--dir", PathBuf::from("../../data/azpente/run1"));
    let pairs: u32 = arg(args, "--pairs", 8);
    let sims: u32 = arg(args, "--sims", 96);
    let watch_min: f64 = arg(args, "--watch", 0.0);
    let latest = dir.join("latest.ot");
    let metrics = dir.join("metrics.jsonl");
    let stop = dir.join("STOP");
    let dev = device_for();
    let cfg = net_config_for(args, &latest);
    loop {
        let latest_iter = metrics_last_iter(&metrics);
        match oldest_snapshot(&dir) {
            Some((ref_path, ref_iter)) if ref_iter < latest_iter => {
                let a = Infer::load(&latest, cfg, dev, Kind::Half);
                let b = Infer::load(&ref_path, cfg, dev, Kind::Half);
                if let (Ok(a), Ok(b)) = (a, b) {
                    let t = Instant::now();
                    let (wins, total) = eval::net_vs_net(
                        &a,
                        &b,
                        pairs,
                        sims,
                        cfg.size as usize,
                        mix(0x4A7E, epoch_secs()),
                    );
                    let wr = f64::from(wins) / f64::from(total.max(1));
                    println!(
                        "rate: iter {latest_iter} vs ref {ref_iter}: {wr:.2} ({wins}/{total}, \
                         {sims} sims, {:.0}s)",
                        t.elapsed().as_secs_f32()
                    );
                    append_line(
                        &metrics,
                        &serde_json::json!({
                            "event": "rate", "time": epoch_secs(), "iter": latest_iter,
                            "ref_iter": ref_iter, "win_rate": wr, "games": total, "sims": sims,
                        })
                        .to_string(),
                    );
                } else {
                    eprintln!("rate: checkpoint load failed; retrying next cycle");
                }
            }
            _ => println!(
                "rate: waiting for a snapshot older than the current net (iter {latest_iter})"
            ),
        }
        if watch_min <= 0.0 {
            break;
        }
        let deadline = Instant::now() + Duration::from_secs_f64(watch_min * 60.0);
        loop {
            if stop.exists() {
                println!("STOP present; rate watcher exiting");
                return;
            }
            if Instant::now() >= deadline {
                break;
            }
            std::thread::sleep(Duration::from_secs(10));
        }
    }
}

/// A coarse absolute strength readout: the checkpoint's win rate against the
/// fixed baseline ladder (random / greedy / shallow MCTS). There is no external
/// engine anchor for Pente, so this is a plain win-rate panel rather than a
/// calibrated Elo. `--watch N` re-runs every N minutes.
pub fn elo_gauge(args: &[String]) {
    let net_path: PathBuf = arg(
        args,
        "--net",
        PathBuf::from("../../data/azpente/run1/latest.ot"),
    );
    let sims: u32 = arg(args, "--sims", 200);
    let pairs: u32 = arg(args, "--pairs", 16);
    let watch_min: f64 = arg(args, "--watch", 0.0);

    let dev = device_for();
    let cfg = net_config_for(args, &net_path);
    let metrics = net_path
        .parent()
        .unwrap_or(Path::new("."))
        .join("metrics.jsonl");
    let opponents = [Opponent::Random, Opponent::Greedy, Opponent::Mcts(256)];
    loop {
        let infer = match Infer::load(&net_path, cfg, dev, Kind::Half) {
            Ok(i) => i,
            Err(e) if watch_min > 0.0 => {
                eprintln!("load failed ({e}); retrying in 30s");
                std::thread::sleep(Duration::from_secs(30));
                continue;
            }
            Err(e) => {
                eprintln!("failed to load {}: {e}", net_path.display());
                std::process::exit(1);
            }
        };

        let t = Instant::now();
        let entries = ladder(
            &infer,
            &opponents,
            pairs,
            sims,
            cfg.size as usize,
            mix(0x510, epoch_secs()),
        );
        let detail = entries
            .iter()
            .map(|e| format!("{}:{:.2}", e.name, e.score))
            .collect::<Vec<_>>()
            .join(" ");
        let games: u32 = entries.iter().map(|e| e.wins + e.draws + e.losses).sum();
        println!(
            "strength vs baseline: [{detail}] ({games} games, {sims} sims, {:.0}s)",
            t.elapsed().as_secs_f32()
        );
        append_line(
            &metrics,
            &serde_json::json!({
                "event": "strength", "time": epoch_secs(), "detail": detail,
                "games": games, "sims": sims,
            })
            .to_string(),
        );
        if watch_min <= 0.0 {
            break;
        }
        let stop = net_path.parent().unwrap_or(Path::new(".")).join("STOP");
        let deadline = Instant::now() + Duration::from_secs_f64(watch_min * 60.0);
        loop {
            if stop.exists() {
                println!("STOP file present; strength watcher exiting");
                return;
            }
            if Instant::now() >= deadline {
                break;
            }
            std::thread::sleep(Duration::from_secs(15));
        }
    }
}
