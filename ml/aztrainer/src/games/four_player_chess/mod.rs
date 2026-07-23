//! Standard Chess.com free-for-all four-player chess plugged into the generic
//! AlphaZero trainer: fixed 14×14 flat policy, four absolute-seat value logits,
//! seat-rotated evaluation, and past-checkpoint league self-play.

mod eval;
mod run;
mod sample;
mod selfplay;

use std::path::PathBuf;

use four_player_chess::FourPlayerChess;
use four_player_chess::encode::FourPlayerChessEncoder;

use crate::verify::{VerifyGame, verify};

struct FourPlayerChessVerify;

impl VerifyGame for FourPlayerChessVerify {
    type G = FourPlayerChess;
    type E = FourPlayerChessEncoder;

    fn game(_cfg: &crate::net::NetConfig) -> FourPlayerChess {
        FourPlayerChess::with_ply_cap(320)
    }

    fn encoder(_cfg: &crate::net::NetConfig) -> FourPlayerChessEncoder {
        FourPlayerChessEncoder
    }
}

fn export(args: &[String]) {
    let net: PathBuf = run::arg(
        args,
        "--net",
        PathBuf::from("../../runs/four-player-chess/latest.ot"),
    );
    let out: PathBuf = run::arg(
        args,
        "--out",
        PathBuf::from("../../runs/four-player-chess/four-player-chess.azweb"),
    );
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).expect("create export directory");
    }
    let cfg = run::net_config_for(args, &net);
    let bytes = crate::export::export(&net, cfg, &out).expect("export");
    println!(
        "exported {}x{} four-seat net: {bytes} body bytes -> {}",
        cfg.blocks,
        cfg.channels,
        out.display()
    );
}

fn verify_export(args: &[String]) {
    let net: PathBuf = run::arg(
        args,
        "--net",
        PathBuf::from("../../runs/four-player-chess/latest.ot"),
    );
    let out: PathBuf = run::arg(
        args,
        "--out",
        PathBuf::from("../../runs/four-player-chess/four-player-chess.azweb"),
    );
    let cfg = run::net_config_for(args, &net);
    verify::<FourPlayerChessVerify>(&net, cfg, &out, 32).expect("verify export");
}

pub fn main(args: &[String]) {
    match args.first().map(String::as_str) {
        Some("run") => run::run(&args[1..]),
        Some("eval") => run::evaluate(&args[1..]),
        Some("export") => export(&args[1..]),
        Some("verify-export") => verify_export(&args[1..]),
        other => {
            eprintln!(
                "usage: four-player-chess <run|eval|export|verify-export> [flags]\ngot: {other:?}"
            );
            std::process::exit(2);
        }
    }
}
