//! The go binary: constructs the global-pool spatial `NetConfig` (with the go
//! ownership/score aux heads) and drives the generic trainer. For now wires
//! `export` / `verify-export` (the parity gate); the full surface follows.

use std::path::PathBuf;

use aztrainer::net::NetConfig;
use aztrainer::verify::VerifyGame;
use aztrainer::{HeadKind, export, verify};
use go::Go;
use go::encode::{GoEncoder, PLANES};

struct GoVerify;
impl VerifyGame for GoVerify {
    type G = Go;
    type E = GoEncoder;
    fn game(cfg: &NetConfig) -> Go {
        Go::new(cfg.size as usize)
    }
    fn encoder(cfg: &NetConfig) -> GoEncoder {
        GoEncoder::new(cfg.size as usize)
    }
}

/// Global-pool spatial go architecture with the ownership/score aux heads.
fn config(blocks: usize, channels: i64, size: i64) -> NetConfig {
    NetConfig {
        blocks,
        channels,
        planes: PLANES as i64,
        size,
        head: HeadKind::GlobalPoolSpatial,
        policy_len: 0,
        go_aux: true,
    }
}

fn arg<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> T {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("export") => {
            let net: PathBuf = arg(
                &args,
                "--net",
                PathBuf::from("../data/azgo/run_full/latest_swa.ot"),
            );
            let legacy: PathBuf = arg(&args, "--out", net.with_file_name("azero-go.azweb"));
            let aznet1: PathBuf = arg(&args, "--aznet1", net.with_file_name("azero-go.aznet1"));
            let blocks = arg(&args, "--blocks", 6);
            let channels = arg(&args, "--ch", 96);
            let size = arg(&args, "--size", 19);
            let len = export::export_dual(&net, config(blocks, channels, size), &legacy, &aznet1)
                .expect("export");
            println!("exported {len}-byte body");
        }
        Some("verify-export") => {
            let net: PathBuf = arg(
                &args,
                "--net",
                PathBuf::from("../data/azgo/run_full/latest_swa.ot"),
            );
            let legacy: PathBuf = arg(&args, "--out", net.with_file_name("azero-go.azweb"));
            let aznet1: PathBuf = arg(&args, "--aznet1", net.with_file_name("azero-go.aznet1"));
            let blocks = arg(&args, "--blocks", 6);
            let channels = arg(&args, "--ch", 96);
            let size = arg(&args, "--size", 19);
            verify::verify::<GoVerify>(&net, config(blocks, channels, size), &legacy, &aznet1, 120)
                .expect("verify");
        }
        other => {
            eprintln!("usage: go <export|verify-export> [--net ...] [--blocks N --ch N --size N]");
            eprintln!("got: {other:?}");
            std::process::exit(2);
        }
    }
}
