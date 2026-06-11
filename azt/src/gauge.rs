//! The Elo gauge: an anchored opponent panel, the maximum-likelihood fit,
//! Bradley-Terry panel calibration, and the watch loop that appends
//! `{"event":"elo"}` rows for the dashboard.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use tch::Kind;

use crate::eval::{self, Opponent, ladder};
use crate::net::Infer;
use crate::selfplay::mix;
use crate::{append_line, arg, device, epoch_secs, net_config_for};

/// The gauge's anchored opponent panel. Elo values come from
/// `azt calibrate` (opponent-vs-opponent matches solved as Bradley-Terry,
/// anchored at Stockfish's UCI_Elo 1320); rerun after any rules change.
const PANEL: [(Opponent, f64); 8] = [
    (Opponent::Random, 71.0),
    (Opponent::Diluted { pct_random: 85 }, 216.0),
    (Opponent::Diluted { pct_random: 50 }, 813.0),
    (Opponent::Diluted { pct_random: 35 }, 967.0),
    (Opponent::Diluted { pct_random: 20 }, 1079.0),
    (Opponent::AbMaterial(1), 1301.0),
    (Opponent::AbMaterial(2), 1455.0),
    (Opponent::AbMaterial(3), 1663.0),
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
    let pairs: u32 = arg(args, "--pairs", 24);
    let movetime: u32 = arg(args, "--movetime", 50);
    let sf = Opponent::Stockfish {
        elo: 1320,
        movetime_ms: movetime,
    };
    let players = [
        Opponent::Random,
        Opponent::Diluted { pct_random: 85 },
        Opponent::Diluted { pct_random: 50 },
        Opponent::Diluted { pct_random: 35 },
        Opponent::Diluted { pct_random: 20 },
        Opponent::AbMaterial(1),
        Opponent::AbMaterial(2),
        Opponent::AbMaterial(3),
        sf,
    ];
    let links = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (6, 8),
    ];
    let mut results = Vec::new();
    for &(i, j) in &links {
        let t = Instant::now();
        let (s, n) = eval::duel(
            players[i],
            players[j],
            pairs,
            mix(0xCA1B, (i * 8 + j) as u64),
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
    let mut e = [1320.0f64; 9];
    for _ in 0..200_000 {
        let mut grad = [0.0f64; 9];
        for &(i, j, s, n) in &results {
            let p = 1.0 / (1.0 + 10f64.powf((e[j] - e[i]) / 400.0));
            let g = f64::from(n) * (s - p);
            grad[i] += g;
            grad[j] -= g;
        }
        for k in 0..8 {
            e[k] = (e[k] + 0.05 * grad[k]).clamp(-400.0, 3200.0);
        }
    }
    println!("\ncalibrated panel (anchor sf-1320):");
    for (p, elo) in players.iter().zip(e) {
        println!("  {:<10} {elo:7.0}", p.name());
    }
}

/// Estimates the checkpoint's Elo from paired games against the anchored
/// panel (random, the alpha-beta ladder, Stockfish levels), fit by maximum
/// likelihood — meaningful even far below Stockfish's 1320 floor. Appends
/// an `{"event":"elo",...}` line to the run's metrics for the dashboard.
/// `--watch N` re-gauges the latest checkpoint every N minutes.
pub fn elo_gauge(args: &[String]) {
    let net_path: PathBuf = arg(args, "--net", PathBuf::from("../data/azt/run2/latest.ot"));
    let sims: u32 = arg(args, "--sims", 600);
    let pairs: u32 = arg(args, "--pairs", 8);
    let movetime: u32 = arg(args, "--movetime", 50);
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
        // Bracket upward through Stockfish levels while the net keeps up.
        let mut level = 1320u32;
        while anchors.last().is_some_and(|&(_, s, _)| s > 0.62) && level < 2800 {
            level += 200;
            let opp = [Opponent::Stockfish {
                elo: level,
                movetime_ms: movetime,
            }];
            let en = &ladder(&infer, &opp, pairs, sims, mix(0x51F, epoch_secs()))[0];
            let n = en.wins + en.draws + en.losses;
            let score = en.score.clamp(0.5 / f64::from(n), 1.0 - 0.5 / f64::from(n));
            anchors.push((f64::from(level), score, n));
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
