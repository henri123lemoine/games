//! Chess (8×8) plugged into the generic trainer: a flat-conv net over the
//! AlphaZero 64×73 move space. Self-play drives `solvers::azero::Search<Chess>`
//! directly with cycle-draw awareness (threefold repetition). Eval is a
//! Stockfish-anchored Bradley-Terry panel; the binary also serves terminal play
//! and a minimal UCI engine.

mod eval;
mod gauge;
mod play;
mod run;
mod sample;
mod selfplay;
mod uci;

use std::path::PathBuf;

use crate::verify::{VerifyGame, verify};
use chess::Chess;
use chess::encode::PlanesEncoder;

pub use sample::Sample;

struct ChessVerify;
impl VerifyGame for ChessVerify {
    type G = Chess;
    type E = PlanesEncoder;
    fn game(_cfg: &crate::net::NetConfig) -> Chess {
        Chess
    }
    fn encoder(_cfg: &crate::net::NetConfig) -> PlanesEncoder {
        PlanesEncoder
    }
}

fn export(args: &[String]) {
    let net: PathBuf = run::parse_arg(args, "--net", PathBuf::from("../data/azt/run2/latest.ot"));
    let out: PathBuf = run::parse_arg(args, "--out", net.with_file_name("azero-chess.azweb"));
    let cfg = run::net_config_for(args, &net);
    let len = crate::export::export(&net, cfg, &out).expect("export");
    println!(
        "exported {}x{} net: {} body bytes -> {} (AZNET1)",
        cfg.blocks,
        cfg.channels,
        len,
        out.display(),
    );
}

fn verify_export(args: &[String]) {
    let net: PathBuf = run::parse_arg(args, "--net", PathBuf::from("../data/azt/run2/latest.ot"));
    let out: PathBuf = run::parse_arg(args, "--out", net.with_file_name("azero-chess.azweb"));
    let cfg = run::net_config_for(args, &net);
    verify::<ChessVerify>(&net, cfg, &out, 120).expect("verify");
}

/// The chess binary's command dispatch.
pub fn main(args: &[String]) {
    match args.first().map(String::as_str) {
        Some("run") => run::run(&args[1..]),
        Some("bench") => run::bench(&args[1..]),
        Some("play") => play::play(&args[1..]),
        Some("uci") => play::uci_engine(&args[1..]),
        Some("elo") => gauge::elo_gauge(&args[1..]),
        Some("calibrate") => gauge::calibrate(&args[1..]),
        Some("export") => export(&args[1..]),
        Some("verify-export") => verify_export(&args[1..]),
        other => {
            eprintln!(
                "usage: chess <run|bench|play|uci|elo|calibrate|export|verify-export> [flags]\n\
                 got: {other:?}"
            );
            std::process::exit(2);
        }
    }
}
