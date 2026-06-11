//! Interactive frontends for a checkpoint: terminal play, and the minimal
//! UCI server that chess GUIs and the local web board speak to.

use std::path::PathBuf;

use azinfer::mcts::MctsConfig;
use game_core::Rng;
use tch::Kind;

use crate::net::Infer;
use crate::{arg, device, epoch_secs, net_config_for};

/// Play against a checkpoint from the terminal: moves in coordinate
/// notation (e2e4, e7e8q), `quit` to leave.
pub fn play(args: &[String]) {
    use crate::selfplay::argmax;
    use azinfer::mcts::{Gather, Search};
    use chess::{Board, Color, legal_moves};
    use std::collections::HashMap;

    let net_path: PathBuf = arg(args, "--net", PathBuf::from("../data/azt/run2/latest.ot"));
    let sims: u32 = arg(args, "--sims", 800);
    let human_is_white = arg(args, "--human", "w".to_string()) != "b";

    let dev = device();
    let cfg = net_config_for(args, &net_path);
    let infer = Infer::load(&net_path, cfg, dev, Kind::Half).unwrap_or_else(|e| {
        eprintln!(
            "failed to load {} as a {}x{} net: {e}",
            net_path.display(),
            cfg.blocks,
            cfg.channels
        );
        std::process::exit(1);
    });
    let mcts_cfg = MctsConfig {
        sims,
        root_noise: 0.0,
        ..MctsConfig::default()
    };
    println!(
        "playing {} ({}x{}, {sims} sims/move); you are {}",
        net_path.display(),
        cfg.blocks,
        cfg.channels,
        if human_is_white { "White" } else { "Black" }
    );

    let mut board = Board::start();
    let mut rng = Rng::new(epoch_secs());
    let mut keys: HashMap<u64, u8> = HashMap::new();
    keys.insert(board.key(), 1);
    loop {
        println!("\n{board}\n");
        let moves = legal_moves(&board);
        if moves.is_empty() {
            if board.in_check(board.stm) {
                let mated_human = (board.stm == Color::White) == human_is_white;
                println!(
                    "checkmate — {}",
                    if mated_human {
                        "the engine wins"
                    } else {
                        "you win!"
                    }
                );
            } else {
                println!("stalemate — draw");
            }
            return;
        }
        if board.halfmove >= 100 || board.insufficient_material() || keys.values().any(|&c| c >= 3)
        {
            println!("draw (repetition, fifty-move rule, or bare material)");
            return;
        }

        let human_turn = (board.stm == Color::White) == human_is_white;
        let m = if human_turn {
            let mut line = String::new();
            loop {
                use std::io::Write;
                print!("your move: ");
                std::io::stdout().flush().ok();
                line.clear();
                if std::io::stdin().read_line(&mut line).unwrap_or(0) == 0 {
                    return;
                }
                let text = line.trim();
                if text == "quit" {
                    return;
                }
                match text.parse() {
                    Ok(m) if moves.contains(&m) => break m,
                    _ => {
                        let labels: Vec<String> = moves.iter().map(|m| m.to_string()).collect();
                        println!("illegal; legal moves: {}", labels.join(" "));
                    }
                }
            }
        } else {
            let mut search = Search::new(None);
            let mut results = Vec::new();
            while let Gather::Requests(reqs) = search.advance(
                &board,
                &keys,
                &mcts_cfg,
                &mut rng,
                std::mem::take(&mut results),
            ) {
                results = infer.forward_batch(&reqs);
            }
            let i = argmax(search.root_visits());
            let m = search.root_moves()[i];
            println!("engine plays {m} (q {:+.2})", search.root_q());
            m
        };
        board.apply(m);
        *keys.entry(board.key()).or_insert(0) += 1;
    }
}

/// Minimal UCI engine over stdin/stdout: enough for chess GUIs and the
/// local web board (position startpos|fen [moves ...], go [movetime N]).
pub fn uci_engine(args: &[String]) {
    use azinfer::argmax;
    use azinfer::mcts::{Gather, MctsConfig, Search};
    use chess::{Board, legal_moves};
    use std::collections::HashMap;
    use std::io::BufRead;

    let net_path: PathBuf = arg(args, "--net", PathBuf::from("../data/azt/run2/latest.ot"));
    let sims: u32 = arg(args, "--sims", 2000);

    let cfg = net_config_for(args, &net_path);
    let infer = Infer::load(&net_path, cfg, device(), Kind::Half).unwrap_or_else(|e| {
        eprintln!("failed to load {}: {e}", net_path.display());
        std::process::exit(1);
    });
    let mut board = Board::start();
    let mut keys: HashMap<u64, u8> = HashMap::new();
    let mut rng = Rng::new(epoch_secs());

    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let line = line.trim();
        let mut words = line.split_whitespace();
        match words.next() {
            Some("uci") => {
                println!("id name azero-azt ({}x{})", cfg.blocks, cfg.channels);
                println!("id author the games room");
                println!("uciok");
            }
            Some("isready") => println!("readyok"),
            Some("ucinewgame") => {
                board = Board::start();
                keys.clear();
            }
            Some("position") => {
                let rest: Vec<&str> = words.collect();
                let (start, move_idx) = if rest.first() == Some(&"startpos") {
                    (Board::start(), 1)
                } else if rest.first() == Some(&"fen") {
                    let fen_end = rest
                        .iter()
                        .position(|&w| w == "moves")
                        .unwrap_or(rest.len());
                    match Board::from_fen(&rest[1..fen_end].join(" ")) {
                        Ok(b) => (b, fen_end),
                        Err(e) => {
                            eprintln!("bad fen: {e}");
                            continue;
                        }
                    }
                } else {
                    continue;
                };
                board = start;
                keys.clear();
                keys.insert(board.key(), 1);
                if rest.get(move_idx) == Some(&"moves") {
                    for m in &rest[move_idx + 1..] {
                        if let Ok(m) = m.parse::<chess::Move>()
                            && legal_moves(&board).contains(&m)
                        {
                            board.apply(m);
                            *keys.entry(board.key()).or_insert(0) += 1;
                        }
                    }
                }
            }
            Some("go") => {
                let mcts_cfg = MctsConfig {
                    sims,
                    root_noise: 0.0,
                    ..MctsConfig::default()
                };
                let mut search = Search::new(None);
                let mut results = Vec::new();
                while let Gather::Requests(reqs) = search.advance(
                    &board,
                    &keys,
                    &mcts_cfg,
                    &mut rng,
                    std::mem::take(&mut results),
                ) {
                    results = infer.forward_batch(&reqs);
                }
                let i = argmax(search.root_visits());
                let q = search.root_q();
                println!("info score cp {} string q {q:+.3}", (q * 600.0) as i64);
                println!("bestmove {}", search.root_moves()[i]);
            }
            Some("quit") => break,
            _ => {}
        }
    }
}
