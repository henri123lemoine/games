//! Pente (5×5…19×19) plugged into the generic trainer: a global-pool spatial net
//! over the board, like go but simpler — no komi, no pass, no ownership/score
//! auxiliary heads. Self-play stores bit-packed positions; training augments
//! with the 8-fold dihedral symmetry. The first move is forced to the center;
//! Pente has no pass, so the spatial head's trailing pass slot is allocated for
//! head reuse but is never a legal action and its policy target is always 0. Eval
//! is a win-rate panel (random / greedy / rollout-MCTS); there is no external
//! engine anchor.

mod eval;
mod gauge;
mod run;
mod sample;
mod selfplay;

use std::path::PathBuf;

use crate::verify::{VerifyGame, verify};
use nn_infer::HeadKind;
use pente::Pente;
use pente::encode::PenteEncoder;

pub use sample::Sample;

struct PenteVerify;
impl VerifyGame for PenteVerify {
    type G = Pente;
    type E = PenteEncoder;
    fn game(cfg: &crate::net::NetConfig) -> Pente {
        Pente::new(cfg.size as usize)
    }
    fn encoder(cfg: &crate::net::NetConfig) -> PenteEncoder {
        PenteEncoder::new(cfg.size as usize)
    }
}

fn export(args: &[String]) {
    let net: PathBuf = run::parse_arg(
        args,
        "--net",
        PathBuf::from("../../data/azpente/run1/latest_swa.ot"),
    );
    let out: PathBuf = run::parse_arg(args, "--out", net.with_file_name("azero-pente.azweb"));
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
        PathBuf::from("../../data/azpente/run1/latest_swa.ot"),
    );
    let out: PathBuf = run::parse_arg(args, "--out", net.with_file_name("azero-pente.azweb"));
    let cfg = run::net_config_for(args, &net);
    verify::<PenteVerify>(&net, cfg, &out, 120).expect("verify");
}

/// The pente binary's command dispatch.
pub fn main(args: &[String]) {
    let _ = HeadKind::GlobalPoolSpatial;
    match args.first().map(String::as_str) {
        Some("run") => run::run(&args[1..]),
        Some("bench") => run::bench(&args[1..]),
        Some("export") => export(&args[1..]),
        Some("verify-export") => verify_export(&args[1..]),
        Some("rate") => gauge::rate(&args[1..]),
        Some("elo") => gauge::elo_gauge(&args[1..]),
        other => {
            eprintln!(
                "usage: pente <run|bench|export|verify-export|rate|elo> [flags]\n\
                 got: {other:?}"
            );
            std::process::exit(2);
        }
    }
}
