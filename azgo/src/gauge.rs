//! The Elo gauge: an anchored opponent panel, the maximum-likelihood fit,
//! Bradley-Terry panel calibration, and the watch loop that appends
//! `{"event":"elo"}` rows for the dashboard.
//!
//! The scale is anchored at **GNU Go level 10 = 1800**, the long-standing
//! CGOS 9×9 convention (komi 7.5, area scoring — the same rules this lab
//! plays). That makes the estimates comparable to published computer-go
//! ratings; "human-equivalent" is approximate the same way CGOS is. There
//! is no stronger local anchor above GNU Go, so estimates beyond ~1800 rely
//! on the MLE extrapolating from win rates against the top rungs and
//! saturate a few hundred points above the anchor.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use tch::Kind;

use crate::eval::{self, Opponent, ladder};
use crate::net::Infer;
use crate::selfplay::mix;
use crate::{append_line, arg, device, epoch_secs, net_config_for};

/// The gauge's anchored opponent panel. Elo values come from
/// `azgo calibrate` (opponent-vs-opponent matches solved as Bradley-Terry,
/// anchored at GNU Go level 10 = 1800); rerun after any rules change.
pub const PANEL: [(Opponent, f64); 7] = [
    (Opponent::Random, -400.0),
    (Opponent::Mcts(128), 189.0),
    (Opponent::Mcts(512), 778.0),
    (Opponent::Mcts(2048), 1141.0),
    (Opponent::GnuGo(1), 1675.0),
    (Opponent::GnuGo(5), 1784.0),
    (Opponent::GnuGo(10), 1800.0),
];

/// Maximum-likelihood Elo from scores against rated opponents: the unique
/// root of the score-vs-expectation excess.
fn mle_elo(anchors: &[(f64, f64, u32)]) -> f64 {
    let mut lo = anchors.iter().map(|a| a.0).fold(f64::MAX, f64::min) - 800.0;
    let mut hi = anchors.iter().map(|a| a.0).fold(f64::MIN, f64::max) + 800.0;
    let excess = |e: f64| -> f64 {
        anchors
            .iter()
            .map(|&(opp, s, n)| {
                let p = 1.0 / (1.0 + 10f64.powf((opp - e) / 400.0));
                f64::from(n) * (s - p)
            })
            .sum()
    };
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if excess(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Plays opponent-vs-opponent matches along the panel chain and solves the
/// anchored Bradley-Terry model, printing Elo values for [`PANEL`].
pub fn calibrate(args: &[String]) {
    let pairs: u32 = arg(args, "--pairs", 12);
    let players: Vec<Opponent> = PANEL.iter().map(|&(o, _)| o).collect();
    let anchor = players.len() - 1;
    let links = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (2, 4),
        (3, 5),
        (4, 6),
    ];
    let mut results = Vec::new();
    for &(i, j) in &links {
        let t = Instant::now();
        let (s, n) = eval::duel(
            players[i],
            players[j],
            pairs,
            mix(0xCA1B, (i * 16 + j) as u64),
        );
        println!(
            "{} vs {}: {s:.3} over {n} games [{:.0}s]",
            players[i].name(),
            players[j].name(),
            t.elapsed().as_secs_f32()
        );
        let s = s.clamp(0.5 / f64::from(n), 1.0 - 0.5 / f64::from(n));
        results.push((i, j, s, n));
    }
    // Gradient ascent on the Bradley-Terry log-likelihood, anchor fixed.
    let mut e = vec![1800.0f64; players.len()];
    for _ in 0..200_000 {
        let mut grad = vec![0.0f64; players.len()];
        for &(i, j, s, n) in &results {
            let p = 1.0 / (1.0 + 10f64.powf((e[j] - e[i]) / 400.0));
            let g = f64::from(n) * (s - p);
            grad[i] += g;
            grad[j] -= g;
        }
        for (k, g) in grad.iter().enumerate() {
            if k != anchor {
                e[k] = (e[k] + 0.05 * g).clamp(-400.0, 3200.0);
            }
        }
    }
    println!("\ncalibrated panel (anchor gnugo-l10 = 1800):");
    for (p, elo) in players.iter().zip(e) {
        println!("  {:<10} {elo:7.0}", p.name());
    }
}

/// Estimates the checkpoint's Elo from paired games against the anchored
/// panel, fit by maximum likelihood — meaningful even far below the GNU Go
/// rungs. Appends an `{"event":"elo",...}` line to the run's metrics for
/// the dashboard. `--watch N` re-gauges the latest checkpoint every N
/// minutes.
pub fn elo_gauge(args: &[String]) {
    let net_path: PathBuf = arg(args, "--net", PathBuf::from("../data/azgo/run1/latest.ot"));
    let sims: u32 = arg(args, "--sims", 400);
    let pairs: u32 = arg(args, "--pairs", 6);
    let watch_min: f64 = arg(args, "--watch", 0.0);

    let dev = device();
    let cfg = net_config_for(args, &net_path);
    let metrics = net_path
        .parent()
        .unwrap_or(Path::new("."))
        .join("metrics.jsonl");
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
        let panel_opps: Vec<Opponent> = PANEL.iter().map(|&(o, _)| o).collect();
        let entries = ladder(&infer, &panel_opps, pairs, sims, mix(0x510, epoch_secs()));
        let mut anchors: Vec<(f64, f64, u32)> = Vec::new();
        let mut detail = Vec::new();
        for (&(_, elo), en) in PANEL.iter().zip(&entries) {
            let n = en.wins + en.draws + en.losses;
            // Clamp clean sweeps off 0/1 (as calibrate does): an unclamped
            // shutout pins the MLE to the bracket edge instead of bounding it.
            let score = en.score.clamp(0.5 / f64::from(n), 1.0 - 0.5 / f64::from(n));
            anchors.push((elo, score, n));
            detail.push(format!("{}:{:.2}", en.name, en.score));
        }

        let est = mle_elo(&anchors);
        let floor = anchors.iter().all(|&(_, s, _)| s <= 0.03);
        let games: u32 = anchors.iter().map(|a| a.2).sum();
        println!(
            "estimated elo: {}{est:.0}  [{}] ({games} games, {sims} sims, {:.0}s)",
            if floor { "<" } else { "" },
            detail.join(" "),
            t.elapsed().as_secs_f32()
        );
        append_line(
            &metrics,
            &serde_json::json!({
                "event": "elo", "time": epoch_secs(), "est": est.round(),
                "floor": floor, "games": games, "sims": sims,
                "detail": detail.join(" "),
            })
            .to_string(),
        );
        if watch_min <= 0.0 {
            break;
        }
        // Honor the run dir's STOP contract like every other process: exit
        // instead of gauging forever after the run ends.
        let stop = net_path.parent().unwrap_or(Path::new(".")).join("STOP");
        let deadline = Instant::now() + Duration::from_secs_f64(watch_min * 60.0);
        loop {
            if stop.exists() {
                println!("STOP file present; elo watcher exiting");
                return;
            }
            if Instant::now() >= deadline {
                break;
            }
            std::thread::sleep(Duration::from_secs(15));
        }
    }
}
