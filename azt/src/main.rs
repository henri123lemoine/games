//! AlphaZero chess on Metal: batched self-play + tch training.
//!
//! ```text
//! cargo run --release -- run --dir ../data/azt/run1 --hours 24
//! cargo run --release -- bench
//! ```
//!
//! `run` mirrors the CPU harness's contract: one JSON line per iteration in
//! `<dir>/metrics.jsonl`, `latest.ot` checkpoints (auto-resume), periodic
//! `ckpt-NNNNNN.ot` snapshots, `dashboard.html` for the live view, and a
//! `STOP` file for graceful shutdown.

mod eval;
mod mcts;
mod net;
mod selfplay;
mod train;
mod uci;

use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{Duration, Instant, SystemTime};

use game_core::Rng;
use tch::{Device, Kind};

use eval::{Opponent, ladder};
use mcts::MctsConfig;
use net::{Infer, NetConfig};
use selfplay::{SelfPlay, SelfPlayConfig, mix};
use train::{Replay, Trainer};

const DASHBOARD: &str = include_str!("../../solvers/examples/azero_dashboard.html");

fn arg<T: FromStr>(args: &[String], name: &str, default: T) -> T {
    args.windows(2)
        .find(|w| w[0] == name)
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(default)
}

fn append_line(path: &Path, line: &str) {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("open metrics file");
    writeln!(f, "{line}").expect("append metrics line");
}

fn last_iter(path: &Path) -> u64 {
    let Ok(text) = std::fs::read_to_string(path) else {
        return 0;
    };
    text.lines()
        .rev()
        .find_map(|l| {
            let i = l.find("\"iter\":")? + 7;
            l[i..]
                .chars()
                .take_while(char::is_ascii_digit)
                .collect::<String>()
                .parse()
                .ok()
        })
        .unwrap_or(0)
}

fn epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn device() -> Device {
    if tch::utils::has_mps() {
        Device::Mps
    } else {
        eprintln!("warning: MPS unavailable, training on CPU");
        Device::Cpu
    }
}

#[allow(clippy::too_many_lines)]
fn run(args: &[String]) {
    let hours: f64 = arg(args, "--hours", 24.0);
    let dir: PathBuf = arg(args, "--dir", PathBuf::from("../data/azt/run1"));
    let blocks: usize = arg(args, "--blocks", 6);
    let channels: i64 = arg(args, "--ch", 64);
    let sims: u32 = arg(args, "--sims", 320);
    let leaves: u32 = arg(args, "--leaves", 8);
    let concurrent: usize = arg(args, "--concurrent", 512);
    let samples_per_iter: usize = arg(args, "--samples-per-iter", 16384);
    let temp_plies: u16 = arg(args, "--temp-plies", 24);
    let value_mix: f32 = arg(args, "--value-mix", 0.3);
    let resign_fp_target: f64 = arg(args, "--resign-fp-target", 0.05);
    let resign_q: f64 = arg(args, "--resign-q", 0.99);
    let resign_min_ply: u16 = arg(args, "--resign-ply", 40);
    let resign_off: f64 = arg(args, "--resign-off", 0.1);
    let batch: usize = arg(args, "--batch", 1024);
    let reuse: f64 = arg(args, "--reuse", 1.7);
    let replay_cap: usize = arg(args, "--replay", 1_500_000);
    let lr: f64 = arg(args, "--lr", 1e-3);
    let weight_decay: f64 = arg(args, "--wd", 1e-4);
    let eval_every: u64 = arg(args, "--eval-every", 4);
    let eval_pairs: u32 = arg(args, "--eval-pairs", 16);
    let eval_sims: u32 = arg(args, "--eval-sims", 256);
    let snapshot_every: u64 = arg(args, "--snapshot-every", 40);

    let net_cfg = NetConfig { blocks, channels };
    let sp_cfg = SelfPlayConfig {
        mcts: MctsConfig {
            sims,
            max_leaves: leaves,
            ..MctsConfig::default()
        },
        concurrent,
        temp_plies,
        resign_q,
        resign_min_ply,
        resign_off,
        ..SelfPlayConfig::default()
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
    let mut trainer = Trainer::new(dev, net_cfg, lr, weight_decay, value_mix);
    let mut iter = 0u64;
    if latest.exists() {
        trainer.load(&latest).unwrap_or_else(|e| {
            eprintln!("failed to load {}: {e}", latest.display());
            std::process::exit(1);
        });
        iter = last_iter(&metrics);
        println!("resumed {} at iter {iter}", latest.display());
    }
    let mut pool = SelfPlay::new(sp_cfg, 0xA12E_5EED);
    let mut replay = Replay::new(replay_cap);
    // Rolling pool of control games' non-loser minimum Qs; the resignation
    // threshold is the fp-target quantile of this distribution (AGZ-style
    // auto-calibration), so no hand-tuned constant survives contact.
    let mut calib_pool: std::collections::VecDeque<f64> = std::collections::VecDeque::new();
    let mut live_resign_q = resign_q;

    append_line(
        &metrics,
        &format!(
            r#"{{"event":"start","time":{},"iter":{iter},"blocks":{blocks},"channels":{channels},"sims":{sims},"concurrent":{concurrent},"samples_per_iter":{samples_per_iter},"batch_size":{batch},"replay_capacity":{replay_cap},"lr":{lr},"eval_every":{eval_every},"eval_pairs":{eval_pairs},"eval_sims":{eval_sims},"value_mix":{value_mix},"resign_fp_target":{resign_fp_target},"threads":{}}}"#,
            epoch_secs(),
            rayon::current_num_threads(),
        ),
    );
    println!(
        "run: {hours:.1}h budget, {blocks}x{channels} resnet on {dev:?}, {sims} sims/move, \
         {concurrent} concurrent games, {samples_per_iter} samples/iter, dir {}",
        dir.display()
    );

    // Budget counts *work* time (self-play + train + eval), not wall clock:
    // closing the laptop lid suspends the process and costs nothing.
    let budget_secs = hours * 3600.0;
    let mut work_secs = 0.0f64;
    let mut lr_dropped = false;
    let start = Instant::now();
    let opponents = [
        Opponent::Random,
        Opponent::AbMaterial(1),
        Opponent::AbMaterial(2),
        Opponent::AbMaterial(3),
    ];
    loop {
        iter += 1;
        let infer = Infer::snapshot(&trainer.vs, net_cfg, Kind::Half);
        let sp_start = Instant::now();
        let (samples, stats, calib) = pool.collect(&infer, samples_per_iter);
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
            let t = sorted[(resign_fp_target * sorted.len() as f64) as usize];
            live_resign_q = (-t).clamp(0.85, 0.995);
            pool.set_resign_q(live_resign_q);
        }

        let steps = ((n_new as f64 * reuse) / batch as f64).ceil() as usize;
        let train_start = Instant::now();
        let (policy_loss, value_loss) =
            trainer.train(&replay, steps, batch, &mut Rng::new(mix(0xC0FFEE, iter)));
        let train_secs = train_start.elapsed().as_secs_f32();

        trainer.save(&latest).expect("write checkpoint");
        if iter.is_multiple_of(snapshot_every) {
            trainer
                .save(&dir.join(format!("ckpt-{iter:06}.ot")))
                .expect("write snapshot");
        }

        let mut eval_json = String::new();
        let mut eval_human = String::new();
        let mut eval_work = 0.0f32;
        if iter == 1 || iter.is_multiple_of(eval_every) {
            let infer = Infer::snapshot(&trainer.vs, net_cfg, Kind::Half);
            let t = Instant::now();
            let entries = ladder(&infer, &opponents, eval_pairs, eval_sims, mix(0xE7A1, iter));
            let eval_secs = t.elapsed().as_secs_f32();
            eval_work = eval_secs;
            let parts: Vec<String> = entries
                .iter()
                .map(|e| {
                    format!(
                        r#""{}":{{"score":{:.4},"w":{},"d":{},"l":{}}}"#,
                        e.name, e.score, e.wins, e.draws, e.losses
                    )
                })
                .collect();
            eval_json = format!(
                r#","eval_secs":{eval_secs:.1},"eval":{{{}}}"#,
                parts.join(",")
            );
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
        append_line(
            &metrics,
            &format!(
                r#"{{"iter":{iter},"time":{},"elapsed_min":{elapsed_min:.2},"policy_loss":{policy_loss:.4},"value_loss":{value_loss:.4},"games":{},"decisive":{},"avg_plies":{:.1},"buffer":{},"self_play_secs":{self_play_secs:.1},"train_secs":{train_secs:.1},"resigned":{},"rep_draws":{},"capped":{},"would_resign":{},"resign_fp":{},"resign_q":{live_resign_q:.3}{eval_json}}}"#,
                epoch_secs(),
                stats.games,
                stats.decisive,
                stats.avg_plies(),
                replay.len(),
                stats.resigned,
                stats.repetition_draws,
                stats.capped,
                stats.would_resign,
                stats.resign_fp,
            ),
        );
        println!(
            "iter {iter:>4} [{elapsed_min:>6.1}m] loss {:.3} (p {policy_loss:.3} + v {value_loss:.3}) | \
             {} games, {} decisive ({} resign), avg {:>3.0} plies, buffer {:>7} | \
             sp {self_play_secs:>5.1}s train {train_secs:>4.1}s{eval_human}",
            policy_loss + value_loss,
            stats.games,
            stats.decisive,
            stats.resigned,
            stats.avg_plies(),
            replay.len(),
        );

        work_secs += f64::from(self_play_secs) + f64::from(train_secs) + f64::from(eval_work);
        if !lr_dropped && work_secs > 0.6 * budget_secs {
            trainer.set_lr(lr * 0.3);
            lr_dropped = true;
            println!("lr {} -> {} at 60% of work budget", lr, lr * 0.3);
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
                &format!(
                    r#"{{"event":"stop","time":{},"iter":{iter},"reason":"{reason}"}}"#,
                    epoch_secs()
                ),
            );
            println!(
                "stopping ({reason}) after iter {iter}; checkpoint at {}",
                latest.display()
            );
            break;
        }
    }
}

/// Times one self-play burst and a training burst at the given sizes.
fn bench(args: &[String]) {
    let blocks: usize = arg(args, "--blocks", 6);
    let channels: i64 = arg(args, "--ch", 64);
    let sims: u32 = arg(args, "--sims", 320);
    let leaves: u32 = arg(args, "--leaves", 8);
    let concurrent: usize = arg(args, "--concurrent", 512);
    let samples: usize = arg(args, "--samples", 8192);

    let dev = device();
    let net_cfg = NetConfig { blocks, channels };
    let trainer = Trainer::new(dev, net_cfg, 1e-3, 1e-4, 0.3);
    let infer = Infer::snapshot(&trainer.vs, net_cfg, Kind::Half);
    let sp_cfg = SelfPlayConfig {
        mcts: MctsConfig {
            sims,
            max_leaves: leaves,
            ..MctsConfig::default()
        },
        concurrent,
        ..SelfPlayConfig::default()
    };
    let mut pool = SelfPlay::new(sp_cfg, 0xBE7C);

    println!("bench: {blocks}x{channels} resnet on {dev:?}, {sims} sims, {concurrent} games");

    let synth = |n: usize| {
        let board = chess::Board::start();
        let moves = chess::legal_moves(&board);
        (0..n)
            .map(|_| net::EvalRequest {
                planes: chess::encode::encode_planes(&board),
                support: moves
                    .iter()
                    .map(|&m| chess::encode::az_move_index(m, board.stm) as u16)
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
         ends: {} decisive, {} resign, {} rep, {} cap)",
        s.len(),
        s.len() as f64 / dt,
        s.len() as f64 / dt / spg * 3600.0,
        stats.games,
        spg,
        s.len() as f64 / dt * f64::from(sims) / 1000.0,
        stats.cpu_secs,
        stats.gpu_secs,
        stats.decisive,
        stats.resigned,
        stats.repetition_draws,
        stats.capped,
    );
    println!(
        "  {} gpu batches, avg width {:.0}, {:.1} ms/batch",
        stats.batches,
        stats.evals as f64 / f64::from(stats.batches.max(1)),
        stats.gpu_secs as f64 * 1000.0 / f64::from(stats.batches.max(1)),
    );

    let mut trainer = trainer;
    let mut replay = Replay::new(2_000_000);
    replay.extend(s);
    let t1 = Instant::now();
    let steps = 30;
    let (pl, vl) = trainer.train(&replay, steps, 1024, &mut Rng::new(7));
    let dt = t1.elapsed().as_secs_f64();
    println!(
        "train: {steps} steps of 1024 in {dt:.1}s = {:.0} samples/s (losses p {pl:.3} v {vl:.3})",
        steps as f64 * 1024.0 / dt
    );
}

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
fn calibrate(args: &[String]) {
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
fn elo_gauge(args: &[String]) {
    let net_path: PathBuf = arg(args, "--net", PathBuf::from("../data/azt/run2/latest.ot"));
    let blocks: usize = arg(args, "--blocks", 8);
    let channels: i64 = arg(args, "--ch", 96);
    let sims: u32 = arg(args, "--sims", 600);
    let pairs: u32 = arg(args, "--pairs", 8);
    let movetime: u32 = arg(args, "--movetime", 50);
    let watch_min: f64 = arg(args, "--watch", 0.0);

    let dev = device();
    let cfg = NetConfig { blocks, channels };
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
            anchors.push((elo, en.score, en.wins + en.draws + en.losses));
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
            anchors.push((f64::from(level), en.score, en.wins + en.draws + en.losses));
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
            &format!(
                r#"{{"event":"elo","time":{},"est":{est:.0},"floor":{floor},"games":{games},"sims":{sims},"detail":"{}"}}"#,
                epoch_secs(),
                detail.join(" ")
            ),
        );
        if watch_min <= 0.0 {
            break;
        }
        std::thread::sleep(Duration::from_secs_f64(watch_min * 60.0));
    }
}

/// Play against a checkpoint from the terminal: moves in coordinate
/// notation (e2e4, e7e8q), `quit` to leave.
fn play(args: &[String]) {
    use chess::{Board, Color, legal_moves};
    use mcts::{Gather, Search};
    use selfplay::argmax;
    use std::collections::HashMap;

    let net_path: PathBuf = arg(args, "--net", PathBuf::from("../data/azt/run2/latest.ot"));
    let blocks: usize = arg(args, "--blocks", 8);
    let channels: i64 = arg(args, "--ch", 96);
    let sims: u32 = arg(args, "--sims", 800);
    let human_is_white = arg(args, "--human", "w".to_string()) != "b";

    let dev = device();
    let cfg = NetConfig { blocks, channels };
    let infer = Infer::load(&net_path, cfg, dev, Kind::Half).unwrap_or_else(|e| {
        eprintln!(
            "failed to load {} as a {blocks}x{channels} net: {e}",
            net_path.display()
        );
        std::process::exit(1);
    });
    let mcts_cfg = MctsConfig {
        sims,
        root_noise: 0.0,
        ..MctsConfig::default()
    };
    println!(
        "playing {} ({blocks}x{channels}, {sims} sims/move); you are {}",
        net_path.display(),
        if human_is_white { "White" } else { "Black" }
    );

    let mut board = Board::start();
    let mut rng = Rng::new(epoch_secs());
    let mut keys: HashMap<u64, u8> = HashMap::new();
    keys.insert(board.key(), 1);
    loop {
        println!("\n{board}\n");
        let moves = legal_moves(&board);
        if moves.is_empty() {
            if board.in_check(board.stm) {
                let mated_human = (board.stm == Color::White) == human_is_white;
                println!(
                    "checkmate — {}",
                    if mated_human {
                        "the engine wins"
                    } else {
                        "you win!"
                    }
                );
            } else {
                println!("stalemate — draw");
            }
            return;
        }
        if board.halfmove >= 100 || board.insufficient_material() || keys.values().any(|&c| c >= 3)
        {
            println!("draw (repetition, fifty-move rule, or bare material)");
            return;
        }

        let human_turn = (board.stm == Color::White) == human_is_white;
        let m = if human_turn {
            let mut line = String::new();
            loop {
                use std::io::Write;
                print!("your move: ");
                std::io::stdout().flush().ok();
                line.clear();
                if std::io::stdin().read_line(&mut line).unwrap_or(0) == 0 {
                    return;
                }
                let text = line.trim();
                if text == "quit" {
                    return;
                }
                match text.parse() {
                    Ok(m) if moves.contains(&m) => break m,
                    _ => {
                        let labels: Vec<String> = moves.iter().map(|m| m.to_string()).collect();
                        println!("illegal; legal moves: {}", labels.join(" "));
                    }
                }
            }
        } else {
            let mut search = Search::new(None);
            let mut results = Vec::new();
            while let Gather::Requests(reqs) = search.advance(
                &board,
                &keys,
                &mcts_cfg,
                &mut rng,
                std::mem::take(&mut results),
            ) {
                results = infer.forward_batch(&reqs);
            }
            let i = argmax(search.root_visits());
            let m = search.root_moves()[i];
            println!("engine plays {m} (q {:+.2})", search.root_q());
            m
        };
        board.apply(m);
        *keys.entry(board.key()).or_insert(0) += 1;
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("run") => run(&args[1..]),
        Some("bench") => bench(&args[1..]),
        Some("play") => play(&args[1..]),
        Some("elo") => elo_gauge(&args[1..]),
        Some("calibrate") => calibrate(&args[1..]),
        _ => {
            eprintln!(
                "usage: azt run   [--dir ../data/azt/run1] [--hours 24] [--blocks 6] [--ch 64] \
                 [--sims 320] [--concurrent 512] [--samples-per-iter 16384] [--batch 1024] \
                 [--reuse 1.7] [--replay 1500000] [--lr 1e-3] [--eval-every 4] [--eval-pairs 16] \
                 [--eval-sims 256] [--snapshot-every 40]"
            );
            eprintln!("       azt bench [--blocks 6] [--ch 64] [--sims 320] [--concurrent 512]");
            eprintln!(
                "       azt play  [--net ../data/azt/run2/latest.ot] [--blocks 8] [--ch 96] \
                 [--sims 800] [--human w|b]"
            );
            eprintln!(
                "       azt elo   [--net ../data/azt/run2/latest.ot] [--blocks 8] [--ch 96] \
                 [--sims 600] [--pairs 8] [--movetime 50] [--watch <minutes>]"
            );
            std::process::exit(2);
        }
    }
}
