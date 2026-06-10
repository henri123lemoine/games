//! AlphaZero self-play training on chess, CPU-only.
//!
//! ```text
//! cargo run --release -p solvers --example azero_chess -- train \
//!     --minutes 10 --sims 96 --games 30 --out data/azero/chess.bin
//! cargo run --release -p solvers --example azero_chess -- play-eval \
//!     --net data/azero/chess.bin
//! ```
//!
//! Honesty: this demonstrates a *correct, learning* AlphaZero loop — the bar
//! is beating a uniform-random opponent (draws count as failures), not
//! playing strong chess.

use std::path::PathBuf;
use std::str::FromStr;
use std::time::{Duration, Instant};

use chess::{Board, Chess, Move};
use game_core::{Agent, Game, Rng, play};
use rayon::prelude::*;
use solvers::azero::{AzeroConfig, Mlp, PolicyValueEncoder, Puct, PuctAgent, SelfPlayTrainer};

struct Enc;

impl PolicyValueEncoder<Chess> for Enc {
    fn input_len(&self) -> usize {
        chess::encode::INPUT_LEN
    }

    fn policy_len(&self) -> usize {
        chess::encode::POLICY_LEN
    }

    fn encode_state(&self, _g: &Chess, s: &Board) -> Vec<f32> {
        chess::encode::encode_board(s)
    }

    fn action_index(&self, _g: &Chess, _s: &Board, m: Move) -> usize {
        chess::encode::move_index(m)
    }
}

fn random_agent() -> impl Agent<Chess> {
    |g: &Chess, s: &Board, _p: usize, r: f64| {
        let n = g.legal_actions(s).len();
        ((r * n as f64) as usize).min(n - 1)
    }
}

fn winrate_vs_random(net: &Mlp, sims: usize, games: u32, seed: u64) -> (f64, u32, u32) {
    let game = &Chess;
    let results: Vec<f64> = (0..games)
        .into_par_iter()
        .map(|i| {
            let bot = PuctAgent(Puct::new(game, &Enc, net, sims));
            let rnd = random_agent();
            let mut rng = Rng::new(
                seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(u64::from(i) * 2 + 1),
            );
            if i % 2 == 0 {
                play(game, &bot, &rnd, &mut rng)
            } else {
                -play(game, &rnd, &bot, &mut rng)
            }
        })
        .collect();
    let wins = results.iter().filter(|&&r| r > 0.0).count() as u32;
    let draws = results.iter().filter(|&&r| r == 0.0).count() as u32;
    (f64::from(wins) / f64::from(games), wins, draws)
}

fn arg<T: FromStr>(args: &[String], name: &str, default: T) -> T {
    args.windows(2)
        .find(|w| w[0] == name)
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(default)
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
        SelfPlayTrainer::with_net(game, &Enc, cfg, net)
    } else {
        SelfPlayTrainer::new(game, &Enc, cfg, 0xA12E)
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
        let (wr, wins, draws) = winrate_vs_random(trainer.net(), sims, 20, 0xE7A1 ^ iter);
        println!(
            "iter {iter:>3} [{:>5.1}m] loss {:.3} (policy {:.3} + value {:.3}) | \
             {} games, {} decisive, avg {:>3.0} plies, buffer {:>6} | \
             vs random: {wr:.2} ({wins} wins, {draws} draws / 20)",
            start.elapsed().as_secs_f64() / 60.0,
            stats.total_loss(),
            stats.policy_loss,
            stats.value_loss,
            stats.games,
            stats.decisive,
            stats.avg_plies,
            stats.samples,
        );
    }
    println!("done: {iter} iterations, checkpoint at {}", out.display());
}

fn play_eval(args: &[String]) {
    let path: PathBuf = arg(args, "--net", PathBuf::from("data/azero/chess.bin"));
    let sims: usize = arg(args, "--sims", 96);
    let games: u32 = arg(args, "--games", 50);
    let net = Mlp::load(&path).unwrap_or_else(|e| {
        eprintln!("failed to load {}: {e}", path.display());
        std::process::exit(1);
    });
    let (wr, wins, draws) = winrate_vs_random(&net, sims, games, 0xBEE5);
    println!(
        "{}: win rate vs uniform random over {games} games: {wr:.2} \
         ({wins} wins, {draws} draws, {} losses)",
        path.display(),
        games - wins - draws
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("train") => train(&args[1..]),
        Some("play-eval") => play_eval(&args[1..]),
        _ => {
            eprintln!(
                "usage: azero_chess train     [--minutes 10] [--sims 96] [--games 30] \
                 [--hidden 256] [--out data/azero/chess.bin]"
            );
            eprintln!(
                "       azero_chess play-eval [--net data/azero/chess.bin] [--sims 96] \
                 [--games 50]"
            );
            std::process::exit(2);
        }
    }
}
