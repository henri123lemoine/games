//! The snake (1v1 Duel) binary: constructs the global-pool dense `NetConfig`
//! (4 headings) and drives the generic trainer. For now wires `export` /
//! `verify-export` (the parity gate); the full surface follows.

use std::path::PathBuf;

use aztrainer::net::NetConfig;
use aztrainer::verify::VerifyGame;
use aztrainer::{HeadKind, export, verify};
use snake::Duel;
use snake::encode::{PLANES, SnakeEncoder};

/// The four absolute headings the snake policy scores.
const ACTIONS: i64 = 4;

struct SnakeVerify;
impl VerifyGame for SnakeVerify {
    type G = Duel;
    type E = SnakeEncoder;
    fn game(_cfg: &NetConfig) -> Duel {
        Duel::new()
    }
    fn encoder(_cfg: &NetConfig) -> SnakeEncoder {
        SnakeEncoder::new()
    }
}

/// Global-pool dense snake architecture (4 headings).
fn config(blocks: usize, channels: i64, size: i64) -> NetConfig {
    NetConfig {
        blocks,
        channels,
        planes: PLANES as i64,
        size,
        head: HeadKind::GlobalPoolDense,
        policy_len: ACTIONS,
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
            let net: PathBuf = arg(
                &args,
                "--net",
                PathBuf::from("../data/azsnake/run3/latest.ot"),
            );
            let legacy: PathBuf = arg(&args, "--out", net.with_file_name("azero-snake.azweb"));
            let aznet1: PathBuf = arg(&args, "--aznet1", net.with_file_name("azero-snake.aznet1"));
            let blocks = arg(&args, "--blocks", 4);
            let channels = arg(&args, "--ch", 64);
            let size = arg(&args, "--size", 20);
            let len = export::export_dual(&net, config(blocks, channels, size), &legacy, &aznet1)
                .expect("export");
            println!("exported {len}-byte body");
        }
        Some("verify-export") => {
            let net: PathBuf = arg(
                &args,
                "--net",
                PathBuf::from("../data/azsnake/run3/latest.ot"),
            );
            let legacy: PathBuf = arg(&args, "--out", net.with_file_name("azero-snake.azweb"));
            let aznet1: PathBuf = arg(&args, "--aznet1", net.with_file_name("azero-snake.aznet1"));
            let blocks = arg(&args, "--blocks", 4);
            let channels = arg(&args, "--ch", 64);
            let size = arg(&args, "--size", 20);
            verify::verify::<SnakeVerify>(
                &net,
                config(blocks, channels, size),
                &legacy,
                &aznet1,
                120,
            )
            .expect("verify");
        }
        other => {
            eprintln!(
                "usage: snake <export|verify-export> [--net ...] [--blocks N --ch N --size N]"
            );
            eprintln!("got: {other:?}");
            std::process::exit(2);
        }
    }
}
