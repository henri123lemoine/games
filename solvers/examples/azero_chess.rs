//! AlphaZero self-play training on chess, CPU-only.
//!
//! ```text
//! cargo run --release -p solvers --example azero_chess -- run \
//!     --dir data/azero/run1 --hours 24 --sims 128 --games 96 --hidden 384
//! cargo run --release -p solvers --example azero_chess -- play-eval \
//!     --net data/azero/run1/latest.bin
//! ```
//!
//! `run` is the long-haul harness: it appends one JSON line per iteration to
//! `<dir>/metrics.jsonl`, checkpoints `<dir>/latest.bin` (resuming from it on
//! restart), snapshots `ckpt-NNNNNN.bin`, and serves a live view through
//! `<dir>/dashboard.html` (open it via any static file server). Touch
//! `<dir>/STOP` to finish the current iteration and exit cleanly.
//!
//! Strength is measured on a fixed ladder — uniform random and
//! `AlphaBeta(MaterialEval)` at depths 1–3 — with paired random openings
//! (each side plays both colors of the same opening) and draws scored ½.
//!
//! Honesty: this demonstrates a *correct, learning* AlphaZero loop on laptop
//! hardware. Climbing the material ladder is the goal; strong chess remains
//! a GPU-scale endeavor.

use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{Duration, Instant, SystemTime};

use chess::{Board, Chess, ChessSpec, MaterialEval};
use game_core::{Agent, Game, Rng, Turn};
use rayon::prelude::*;
use solvers::AlphaBeta;
use solvers::azero::{AzeroConfig, Mlp, Puct, PuctAgent, SelfPlayTrainer};

const DASHBOARD: &str = include_str!("../../assets/azero_dashboard.html");

/// Plies of uniform-random opening shared by both games of an eval pair.
const OPENING_PLIES: usize = 4;
/// Eval games longer than this are scored as draws.
const EVAL_PLY_CAP: usize = 300;

fn random_agent() -> impl Agent<Chess> {
    game_core::RandomAgent
}

fn mix(a: u64, b: u64) -> u64 {
    let mut x = a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^ (x >> 27)
}

struct EvalOutcome {
    score: f64,
    wins: u32,
    draws: u32,
    losses: u32,
}

fn play_from(
    game: &Chess,
    mut s: Board,
    white: &dyn Agent<Chess>,
    black: &dyn Agent<Chess>,
    rng: &mut Rng,
) -> f64 {
    for _ in 0..EVAL_PLY_CAP {
        if game.is_terminal(&s) {
            break;
        }
        let Turn::Player(p) = game.turn(&s) else {
            unreachable!("chess has no chance nodes");
        };
        let actions = game.legal_actions(&s);
        let agent: &dyn Agent<Chess> = if p == 0 { white } else { black };
        let i = agent.act(game, &s, p, rng);
        game.apply(&mut s, actions[i]);
    }
    if game.is_terminal(&s) {
        game.returns(&s, 0)
    } else {
        0.0
    }
}

fn random_opening(game: &Chess, rng: &mut Rng) -> Board {
    loop {
        let mut s = game.initial_state();
        for _ in 0..OPENING_PLIES {
            if game.is_terminal(&s) {
                break;
            }
            let actions = game.legal_actions(&s);
            let i = ((rng.unit() * actions.len() as f64) as usize).min(actions.len() - 1);
            game.apply(&mut s, actions[i]);
        }
        if !game.is_terminal(&s) {
            return s;
        }
    }
}

/// `pairs` paired games (net as White, then Black, from the same random
/// opening) against fresh opponents from `make_opponent`. Draws score ½.
fn eval_vs<A: Agent<Chess>>(
    net: &Mlp,
    sims: usize,
    pairs: u32,
    seed: u64,
    make_opponent: impl Fn() -> A + Sync,
) -> EvalOutcome {
    let game = &Chess;
    let returns: Vec<f64> = (0..pairs)
        .into_par_iter()
        .flat_map_iter(|i| {
            let mut rng = Rng::new(mix(seed, u64::from(i) + 1));
            let opening = random_opening(game, &mut rng);
            let bot = PuctAgent(Puct::new(game, &chess::encode::FlatEncoder, net, sims));
            let opp = make_opponent();
            let as_white = play_from(game, opening.clone(), &bot, &opp, &mut rng);
            let as_black = -play_from(game, opening, &opp, &bot, &mut rng);
            [as_white, as_black]
        })
        .collect();
    let wins = returns.iter().filter(|&&r| r > 0.0).count() as u32;
    let losses = returns.iter().filter(|&&r| r < 0.0).count() as u32;
    let games = returns.len() as u32;
    let draws = games - wins - losses;
    EvalOutcome {
        score: (f64::from(wins) + 0.5 * f64::from(draws)) / f64::from(games),
        wins,
        draws,
        losses,
    }
}

fn eval_ladder(net: &Mlp, sims: usize, pairs: u32, seed: u64) -> Vec<(String, EvalOutcome)> {
    let mut ladder = vec![(
        "random".to_string(),
        eval_vs(net, sims, pairs, mix(seed, 1), random_agent),
    )];
    for depth in [1u32, 2, 3] {
        ladder.push((
            format!("ab-mat-d{depth}"),
            eval_vs(
                net,
                sims,
                pairs,
                mix(seed, u64::from(depth) + 1),
                move || AlphaBeta::new(depth, MaterialEval, ChessSpec),
            ),
        ));
    }
    ladder
}

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

/// Last `"iter": N` recorded in the metrics file, for resuming.
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

fn run(args: &[String]) {
    let hours: f64 = arg(args, "--hours", 24.0);
    let dir: PathBuf = arg(args, "--dir", PathBuf::from("data/azero/run1"));
    let sims: usize = arg(args, "--sims", 128);
    let games: usize = arg(args, "--games", 96);
    let hidden: usize = arg(args, "--hidden", 384);
    let eval_every: u64 = arg(args, "--eval-every", 5);
    let eval_pairs: u32 = arg(args, "--eval-pairs", 16);
    let snapshot_every: u64 = arg(args, "--snapshot-every", 100);

    let cfg = AzeroConfig {
        hidden,
        sims,
        games_per_iter: games,
        replay_capacity: 250_000,
        batch_size: 128,
        batches_per_iter: 256,
        ..AzeroConfig::default()
    };
    let (batch_size, batches_per_iter, replay_capacity) =
        (cfg.batch_size, cfg.batches_per_iter, cfg.replay_capacity);

    std::fs::create_dir_all(&dir).expect("create run dir");
    std::fs::write(dir.join("dashboard.html"), DASHBOARD).expect("write dashboard");
    let latest = dir.join("latest.bin");
    let metrics = dir.join("metrics.jsonl");
    let stop = dir.join("STOP");
    if stop.exists() {
        std::fs::remove_file(&stop).expect("clear stale STOP file");
    }

    let game = &Chess;
    let (mut trainer, mut iter) = if latest.exists() {
        let net = Mlp::load(&latest).unwrap_or_else(|e| {
            eprintln!("failed to load {}: {e}", latest.display());
            std::process::exit(1);
        });
        let resumed = last_iter(&metrics);
        println!(
            "resuming {} from iter {resumed} ({} inputs, {}x2 hidden, {} policy)",
            latest.display(),
            net.input_len(),
            net.hidden_len(),
            net.policy_len()
        );
        (
            SelfPlayTrainer::with_net(game, &chess::encode::FlatEncoder, cfg, net),
            resumed,
        )
    } else {
        (
            SelfPlayTrainer::new(game, &chess::encode::FlatEncoder, cfg, 0xA12E),
            0,
        )
    };

    let hidden = trainer.net().hidden_len();
    append_line(
        &metrics,
        &format!(
            r#"{{"event":"start","time":{},"iter":{iter},"hidden":{hidden},"sims":{sims},"games_per_iter":{games},"batch_size":{batch_size},"batches_per_iter":{batches_per_iter},"replay_capacity":{replay_capacity},"eval_every":{eval_every},"eval_pairs":{eval_pairs},"threads":{}}}"#,
            epoch_secs(),
            rayon::current_num_threads(),
        ),
    );
    println!(
        "run: {hours:.1}h budget, {sims} sims/move, {games} games/iter, {hidden}x2 hidden, \
         {} threads, dir {}",
        rayon::current_num_threads(),
        dir.display()
    );

    let budget = Duration::from_secs_f64(hours * 3600.0);
    let start = Instant::now();
    loop {
        iter += 1;
        let stats = trainer.iterate(mix(0xC0FFEE, iter));
        trainer.net().save(&latest).expect("write checkpoint");
        if iter % snapshot_every == 0 {
            let snap = dir.join(format!("ckpt-{iter:06}.bin"));
            trainer.net().save(&snap).expect("write snapshot");
        }

        let mut eval_json = String::new();
        let mut eval_human = String::new();
        if iter == 1 || iter % eval_every == 0 {
            let t = Instant::now();
            let ladder = eval_ladder(trainer.net(), sims, eval_pairs, mix(0xE7A1, iter));
            let eval_secs = t.elapsed().as_secs_f32();
            let entries: Vec<String> = ladder
                .iter()
                .map(|(name, o)| {
                    format!(
                        r#""{name}":{{"score":{:.4},"w":{},"d":{},"l":{}}}"#,
                        o.score, o.wins, o.draws, o.losses
                    )
                })
                .collect();
            eval_json = format!(
                r#","eval_secs":{eval_secs:.1},"eval":{{{}}}"#,
                entries.join(",")
            );
            eval_human = ladder
                .iter()
                .map(|(name, o)| format!("{name} {:.2}", o.score))
                .collect::<Vec<_>>()
                .join(", ");
            eval_human = format!(" | eval [{eval_secs:.0}s] {eval_human}");
        }

        let elapsed_min = start.elapsed().as_secs_f64() / 60.0;
        append_line(
            &metrics,
            &format!(
                r#"{{"iter":{iter},"time":{},"elapsed_min":{elapsed_min:.2},"policy_loss":{:.4},"value_loss":{:.4},"games":{},"decisive":{},"avg_plies":{:.1},"buffer":{},"self_play_secs":{:.1},"train_secs":{:.1}{eval_json}}}"#,
                epoch_secs(),
                stats.policy_loss,
                stats.value_loss,
                stats.games,
                stats.decisive,
                stats.avg_plies,
                stats.samples,
                stats.self_play_secs,
                stats.train_secs,
            ),
        );
        println!(
            "iter {iter:>4} [{elapsed_min:>6.1}m] loss {:.3} (p {:.3} + v {:.3}) | \
             {} games, {} decisive, avg {:>3.0} plies, buffer {:>6} | \
             sp {:>4.1}s train {:>4.1}s{eval_human}",
            stats.total_loss(),
            stats.policy_loss,
            stats.value_loss,
            stats.games,
            stats.decisive,
            stats.avg_plies,
            stats.samples,
            stats.self_play_secs,
            stats.train_secs,
        );

        let reason = if stop.exists() {
            Some("STOP file")
        } else if start.elapsed() >= budget {
            Some("time budget reached")
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

fn train(args: &[String]) {
    let minutes: f64 = arg(args, "--minutes", 10.0);
    let sims: usize = arg(args, "--sims", 96);
    let games: usize = arg(args, "--games", 30);
    let hidden: usize = arg(args, "--hidden", 256);
    let out: PathBuf = arg(args, "--out", PathBuf::from("data/azero/chess.bin"));

    let cfg = AzeroConfig {
        hidden,
        sims,
        games_per_iter: games,
        ..AzeroConfig::default()
    };
    let game = &Chess;
    let mut trainer = if out.exists() {
        let net = Mlp::load(&out).unwrap_or_else(|e| {
            eprintln!("failed to load {}: {e}", out.display());
            std::process::exit(1);
        });
        println!(
            "resuming from {} ({} inputs, {}x2 hidden, {} policy)",
            out.display(),
            net.input_len(),
            net.hidden_len(),
            net.policy_len()
        );
        SelfPlayTrainer::with_net(game, &chess::encode::FlatEncoder, cfg, net)
    } else {
        SelfPlayTrainer::new(game, &chess::encode::FlatEncoder, cfg, 0xA12E)
    };

    println!(
        "training for {minutes:.1} min: {sims} sims/move, {games} games/iteration, checkpoint {}",
        out.display()
    );
    let budget = Duration::from_secs_f64(minutes * 60.0);
    let start = Instant::now();
    let mut iter = 0u64;
    while start.elapsed() < budget {
        iter += 1;
        let stats = trainer.iterate(0xC0FFEE ^ iter);
        trainer.net().save(&out).expect("write checkpoint");
        let o = eval_vs(trainer.net(), sims, 10, mix(0xE7A1, iter), random_agent);
        println!(
            "iter {iter:>3} [{:>5.1}m] loss {:.3} (policy {:.3} + value {:.3}) | \
             {} games, {} decisive, avg {:>3.0} plies, buffer {:>6} | \
             vs random: {:.2} ({}-{}-{})",
            start.elapsed().as_secs_f64() / 60.0,
            stats.total_loss(),
            stats.policy_loss,
            stats.value_loss,
            stats.games,
            stats.decisive,
            stats.avg_plies,
            stats.samples,
            o.score,
            o.wins,
            o.draws,
            o.losses,
        );
    }
    println!("done: {iter} iterations, checkpoint at {}", out.display());
}

fn play_eval(args: &[String]) {
    let path: PathBuf = arg(args, "--net", PathBuf::from("data/azero/chess.bin"));
    let sims: usize = arg(args, "--sims", 96);
    let pairs: u32 = arg(args, "--pairs", 25);
    let net = Mlp::load(&path).unwrap_or_else(|e| {
        eprintln!("failed to load {}: {e}", path.display());
        std::process::exit(1);
    });
    let ladder = eval_ladder(&net, sims, pairs, 0xBEE5);
    println!("{} over {} games per opponent:", path.display(), pairs * 2);
    for (name, o) in ladder {
        println!(
            "  vs {name:<10} score {:.3} ({}-{}-{})",
            o.score, o.wins, o.draws, o.losses
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("run") => run(&args[1..]),
        Some("train") => train(&args[1..]),
        Some("play-eval") => play_eval(&args[1..]),
        _ => {
            eprintln!(
                "usage: azero_chess run       [--dir data/azero/run1] [--hours 24] [--sims 128] \
                 [--games 96] [--hidden 384] [--eval-every 5] [--eval-pairs 16] \
                 [--snapshot-every 100]"
            );
            eprintln!(
                "       azero_chess train     [--minutes 10] [--sims 96] [--games 30] \
                 [--hidden 256] [--out data/azero/chess.bin]"
            );
            eprintln!(
                "       azero_chess play-eval [--net data/azero/chess.bin] [--sims 96] \
                 [--pairs 25]"
            );
            std::process::exit(2);
        }
    }
}
