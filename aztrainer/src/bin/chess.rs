//! The chess (8×8) binary: constructs the flat-conv `NetConfig` and drives the
//! generic trainer. For now wires `export` / `verify-export` (the parity gate);
//! the full run/play/uci/elo surface follows.

use std::path::PathBuf;

use aztrainer::net::NetConfig;
use aztrainer::verify::VerifyGame;
use aztrainer::{HeadKind, export, verify};
use chess::Chess;
use chess::encode::{AZ_POLICY_LEN, PLANE_COUNT, PlanesEncoder};

struct ChessVerify;
impl VerifyGame for ChessVerify {
    type G = Chess;
    type E = PlanesEncoder;
    fn game(_cfg: &NetConfig) -> Chess {
        Chess
    }
    fn encoder(_cfg: &NetConfig) -> PlanesEncoder {
        PlanesEncoder
    }
}

/// Flat-conv chess architecture. `blocks`/`channels` are read from the
/// checkpoint (flags > sidecar > metrics); the rest is fixed by the chess net.
fn config(blocks: usize, channels: i64) -> NetConfig {
    NetConfig {
        blocks,
        channels,
        planes: PLANE_COUNT as i64,
        size: 8,
        head: HeadKind::FlatConv,
        policy_len: AZ_POLICY_LEN as i64,
        go_aux: false,
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
            let net: PathBuf = arg(&args, "--net", PathBuf::from("../data/azt/run2/latest.ot"));
            let legacy: PathBuf = arg(&args, "--out", net.with_file_name("azero-chess.azweb"));
            let aznet1: PathBuf = arg(&args, "--aznet1", net.with_file_name("azero-chess.aznet1"));
            let blocks = arg(&args, "--blocks", 8);
            let channels = arg(&args, "--ch", 96);
            let len = export::export_dual(&net, config(blocks, channels), &legacy, &aznet1)
                .expect("export");
            println!("exported {len}-byte body");
        }
        Some("verify-export") => {
            let net: PathBuf = arg(&args, "--net", PathBuf::from("../data/azt/run2/latest.ot"));
            let legacy: PathBuf = arg(&args, "--out", net.with_file_name("azero-chess.azweb"));
            let aznet1: PathBuf = arg(&args, "--aznet1", net.with_file_name("azero-chess.aznet1"));
            let blocks = arg(&args, "--blocks", 8);
            let channels = arg(&args, "--ch", 96);
            verify::verify::<ChessVerify>(&net, config(blocks, channels), &legacy, &aznet1, 120)
                .expect("verify");
        }
        other => {
            eprintln!("usage: chess <export|verify-export> [--net ...] [--blocks N --ch N]");
            eprintln!("got: {other:?}");
            std::process::exit(2);
        }
    }
}
