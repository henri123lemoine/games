//! Go (9×9…19×19) plugged into the generic trainer: a global-pool spatial net
//! with the ownership/score auxiliary heads. Self-play bit-packs positions,
//! randomizes komi, and masks early passes; training augments with the 8-fold
//! dihedral symmetry and mixes board sizes. Eval is a GNU-Go-anchored Elo
//! ladder over GTP.

mod calibrate_pass;
mod eval;
mod gauge;
mod gtp;
mod run;
mod sample;
mod selfplay;

use std::path::PathBuf;

use crate::verify::{VerifyGame, verify};
use go::Go;
use go::encode::GoEncoder;

pub use sample::Sample;

struct GoVerify;
impl VerifyGame for GoVerify {
    type G = Go;
    type E = GoEncoder;
    fn game(cfg: &crate::net::NetConfig) -> Go {
        Go::new(cfg.size as usize)
    }
    fn encoder(cfg: &crate::net::NetConfig) -> GoEncoder {
        GoEncoder::new(cfg.size as usize)
    }
}

fn export(args: &[String]) {
    let net: PathBuf = run::parse_arg(
        args,
        "--net",
        PathBuf::from("../../data/azgo/run_full/latest_swa.ot"),
    );
    let out: PathBuf = run::parse_arg(args, "--out", net.with_file_name("azero-go.azweb"));
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
        PathBuf::from("../../data/azgo/run_full/latest_swa.ot"),
    );
    let out: PathBuf = run::parse_arg(args, "--out", net.with_file_name("azero-go.azweb"));
    let cfg = run::net_config_for(args, &net);
    verify::<GoVerify>(&net, cfg, &out, 120).expect("verify");
}

/// The go binary's command dispatch.
pub fn main(args: &[String]) {
    match args.first().map(String::as_str) {
        Some("run") => run::run(&args[1..]),
        Some("bench") => run::bench(&args[1..]),
        Some("export") => export(&args[1..]),
        Some("verify-export") => verify_export(&args[1..]),
        Some("elo") => gauge::elo_gauge(&args[1..]),
        Some("rate") => gauge::rate(&args[1..]),
        Some("calibrate") => gauge::calibrate(&args[1..]),
        Some("calibrate-pass") => calibrate_pass::run(&args[1..]),
        other => {
            eprintln!(
                "usage: go <run|bench|export|verify-export|elo|rate|calibrate|calibrate-pass> [flags]\n\
                 got: {other:?}"
            );
            std::process::exit(2);
        }
    }
}
