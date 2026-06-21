//! Snake (1v1 Duel, 20×20) plugged into the generic trainer: a global-pool
//! dense net over the four absolute headings. The self-play loop resolves food
//! chance nodes, discounts and length-sharpens the terminal value, and mixes an
//! opponent pool of past checkpoints. Eval is a win-rate panel (random / greedy
//! / rollout-MCTS); there is no external engine anchor.

mod eval;
mod gauge;
mod run;
mod sample;
mod selfplay;

use std::path::PathBuf;

use crate::verify::{VerifyGame, verify};
use nn_infer::HeadKind;
use snake::Duel;
use snake::encode::SnakeEncoder;

pub use sample::Sample;

struct SnakeVerify;
impl VerifyGame for SnakeVerify {
    type G = Duel;
    type E = SnakeEncoder;
    fn game(_cfg: &crate::net::NetConfig) -> Duel {
        Duel::new()
    }
    fn encoder(_cfg: &crate::net::NetConfig) -> SnakeEncoder {
        SnakeEncoder::new()
    }
}

fn export(args: &[String]) {
    let net: PathBuf = run::parse_arg(
        args,
        "--net",
        PathBuf::from("../../data/azsnake/run3/latest.ot"),
    );
    let out: PathBuf = run::parse_arg(args, "--out", net.with_file_name("azero-snake.azweb"));
    let cfg = run::net_config_for(args, &net);
    let len = crate::export::export(&net, cfg, &out).expect("export");
    println!(
        "exported {}x{} size-{} net: {} body bytes -> {} (AZNET1)",
        cfg.blocks,
        cfg.channels,
        cfg.size,
        len,
        out.display(),
    );
}

fn verify_export(args: &[String]) {
    let net: PathBuf = run::parse_arg(
        args,
        "--net",
        PathBuf::from("../../data/azsnake/run3/latest.ot"),
    );
    let out: PathBuf = run::parse_arg(args, "--out", net.with_file_name("azero-snake.azweb"));
    let cfg = run::net_config_for(args, &net);
    verify::<SnakeVerify>(&net, cfg, &out, 120).expect("verify");
}

/// The snake binary's command dispatch.
pub fn main(args: &[String]) {
    let _ = HeadKind::GlobalPoolDense;
    match args.first().map(String::as_str) {
        Some("run") => run::run(&args[1..]),
        Some("bench") => run::bench(&args[1..]),
        Some("export") => export(&args[1..]),
        Some("verify-export") => verify_export(&args[1..]),
        Some("rate") => gauge::rate(&args[1..]),
        Some("elo") => gauge::elo_gauge(&args[1..]),
        other => {
            eprintln!(
                "usage: snake <run|bench|export|verify-export|rate|elo> [flags]\n\
                 got: {other:?}"
            );
            std::process::exit(2);
        }
    }
}
