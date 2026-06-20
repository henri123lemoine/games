//! The run-directory contract shared by every trainer: a `metrics.jsonl` event
//! log, a `dashboard.html` snapshot, a `STOP` file for graceful shutdown, and
//! the resume/architecture bookkeeping. Game-agnostic — the per-game binary
//! supplies its own dashboard HTML and the metric fields it logs.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use tch::Device;

use crate::train::Trainer;

/// The training device: Metal (MPS) when available, else CPU with a warning.
pub fn device() -> Device {
    if tch::utils::has_mps() {
        Device::Mps
    } else {
        eprintln!("warning: MPS unavailable, training on CPU");
        Device::Cpu
    }
}

/// Saves a checkpoint, retrying once after a short pause — a transient
/// filesystem hiccup shouldn't lose an iteration's weights.
pub fn save_with_retry(trainer: &Trainer, path: &Path) {
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

/// Seconds since the Unix epoch, for metric timestamps.
pub fn epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Appends one line to `metrics.jsonl`, durably. A transient I/O error costs one
/// metrics line, never the run.
pub fn append_line(path: &Path, line: &str) {
    let write = || -> std::io::Result<()> {
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        f.write_all(line.as_bytes())?;
        f.write_all(b"\n")?;
        Ok(())
    };
    if let Err(e) = write() {
        eprintln!("warning: dropped metrics line ({e})");
    }
}

/// The last value of `field` over every line of `metrics.jsonl`, as f64.
pub fn last_f64(path: &Path, field: &str) -> Option<f64> {
    std::fs::read_to_string(path).ok().and_then(|t| {
        t.lines().rev().find_map(|l| {
            serde_json::from_str::<serde_json::Value>(l)
                .ok()?
                .get(field)?
                .as_f64()
        })
    })
}

/// The last `iter` recorded in `metrics.jsonl` (0 if none) — the resume point.
pub fn last_iter(path: &Path) -> u64 {
    last_f64(path, "iter").map(|v| v as u64).unwrap_or(0)
}

/// A `<name>.json` sidecar's recorded `(blocks, channels, size)`, if present.
pub fn sidecar_arch(net_path: &Path) -> Option<(usize, i64, i64)> {
    let name = net_path.file_name()?.to_str()?;
    let sidecar = net_path.with_file_name(format!("{name}.json"));
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(sidecar).ok()?).ok()?;
    Some((
        v["blocks"].as_u64()? as usize,
        v["channels"].as_u64()? as i64,
        v["size"].as_u64().unwrap_or(0) as i64,
    ))
}

/// The latest `start` event's `(blocks, channels, size)` in a run's
/// `metrics.jsonl` — the architecture for checkpoints that predate sidecars.
pub fn metrics_arch(metrics: &Path) -> Option<(usize, i64, i64)> {
    let text = std::fs::read_to_string(metrics).ok()?;
    for line in text.lines().rev() {
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if v.get("event").and_then(|e| e.as_str()) == Some("start") {
            return Some((
                v["blocks"].as_u64()? as usize,
                v["channels"].as_u64()? as i64,
                v["size"].as_u64().unwrap_or(0) as i64,
            ));
        }
    }
    None
}

/// Resolves a checkpoint's `(blocks, channels, size)`: explicit flags win, then
/// the checkpoint's own `<name>.json` sidecar, then the run's `metrics.jsonl`
/// `start` event, then the supplied defaults. The same rule every trainer
/// subcommand follows, so `export`/`run`/`eval` agree on a net's architecture.
pub fn resolve_arch(
    flag_blocks: Option<usize>,
    flag_channels: Option<i64>,
    flag_size: Option<i64>,
    net_path: &Path,
    default: (usize, i64, i64),
) -> (usize, i64, i64) {
    let metrics = net_path.parent().map(|d| d.join("metrics.jsonl"));
    let from = sidecar_arch(net_path)
        .or_else(|| metrics.as_deref().and_then(metrics_arch))
        .unwrap_or(default);
    (
        flag_blocks.unwrap_or(from.0),
        flag_channels.unwrap_or(from.1),
        flag_size.unwrap_or(from.2),
    )
}

/// Whether a `STOP` file requests graceful shutdown.
pub fn stop_requested(dir: &Path) -> bool {
    dir.join("STOP").exists()
}

/// The live run directory: paths, the metrics log, and the work-budget / LR
/// schedule the run loop reads each iteration.
pub struct RunDir {
    pub dir: PathBuf,
    pub latest: PathBuf,
    pub metrics: PathBuf,
}

impl RunDir {
    /// Creates `dir`, writes the dashboard HTML, clears any stale `STOP`, and
    /// returns the run paths. `dashboard` is the per-game HTML (an `include_str!`
    /// from the binary).
    pub fn open(dir: &Path, dashboard: &str) -> RunDir {
        std::fs::create_dir_all(dir).expect("create run dir");
        std::fs::write(dir.join("dashboard.html"), dashboard).expect("write dashboard");
        let stop = dir.join("STOP");
        if stop.exists() {
            std::fs::remove_file(&stop).expect("clear stale STOP file");
        }
        RunDir {
            dir: dir.to_path_buf(),
            latest: dir.join("latest.ot"),
            metrics: dir.join("metrics.jsonl"),
        }
    }

    pub fn append(&self, line: &str) {
        append_line(&self.metrics, line);
    }

    pub fn stop_requested(&self) -> bool {
        stop_requested(&self.dir)
    }
}

/// The shared LR schedule: a linear warmup over the first `warmup_iters` (steady
/// SGD early steps), then the base rate, then a single step down to `base·0.3`
/// once `work_secs` passes 60% of the budget. Tracks whether the late drop has
/// fired so resumed legs don't re-shock the run. Mirrors azgo/azt's schedule.
pub struct LrSchedule {
    pub base: f64,
    pub warmup_iters: u64,
    pub budget_secs: f64,
    dropped: bool,
}

impl LrSchedule {
    pub fn new(
        base: f64,
        warmup_iters: u64,
        budget_secs: f64,
        resumed_dropped: bool,
    ) -> LrSchedule {
        LrSchedule {
            base,
            warmup_iters,
            budget_secs,
            dropped: resumed_dropped,
        }
    }

    /// The learning rate for `iter`, given cumulative `work_secs`. Returns the
    /// rate and whether the late step-down just fired (for one-time logging).
    pub fn lr(&mut self, iter: u64, work_secs: f64) -> (f64, bool) {
        if !self.dropped && work_secs > 0.6 * self.budget_secs {
            self.dropped = true;
            return (self.base * 0.3, true);
        }
        let base = if self.dropped {
            self.base * 0.3
        } else {
            self.base
        };
        if self.warmup_iters > 0 && iter <= self.warmup_iters {
            (base * iter as f64 / self.warmup_iters as f64, false)
        } else {
            (base, false)
        }
    }

    pub fn dropped(&self) -> bool {
        self.dropped
    }
}
