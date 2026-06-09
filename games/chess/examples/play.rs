//! Terminal human-vs-engine chess.
//!
//! Usage: `cargo run --release -p chess --example play [depth] [black]`
//! Human plays White by default; pass `black` to take Black. Moves are UCI
//! coordinates ("e2e4", promotions "e7e8q").

use chess::{AlphaBetaAgent, Board, Color, Move, legal_moves};
use std::io::{self, BufRead, Write};

fn announce_game_over(board: &Board, moves_empty: bool) -> bool {
    if moves_empty {
        if board.in_check(board.stm) {
            let winner = match board.stm {
                Color::White => "Black",
                Color::Black => "White",
            };
            println!("checkmate — {winner} wins");
        } else {
            println!("stalemate — draw");
        }
        true
    } else if board.halfmove >= 100 {
        println!("draw by the 50-move rule");
        true
    } else if board.insufficient_material() {
        println!("draw by insufficient material");
        true
    } else {
        false
    }
}

fn main() {
    let mut depth = 5u32;
    let mut human_is_white = true;
    for arg in std::env::args().skip(1) {
        if arg == "black" || arg == "--black" {
            human_is_white = false;
        } else if let Ok(d) = arg.parse::<u32>() {
            depth = d.max(1);
        } else {
            eprintln!("usage: play [depth] [black]");
            return;
        }
    }

    let engine = AlphaBetaAgent::new(depth);
    let mut board = Board::start();
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    println!(
        "you are {} — engine depth {depth}; enter moves like e2e4 (promote: e7e8q)",
        if human_is_white { "White" } else { "Black" }
    );

    loop {
        println!("\n{board}\n");
        let moves = legal_moves(&board);
        if announce_game_over(&board, moves.is_empty()) {
            return;
        }
        if board.in_check(board.stm) {
            println!("check!");
        }

        let human_turn = (board.stm == Color::White) == human_is_white;
        let chosen = if human_turn {
            loop {
                print!("your move> ");
                io::stdout().flush().ok();
                let Some(Ok(line)) = lines.next() else {
                    println!("\ninput closed — goodbye");
                    return;
                };
                let line = line.trim();
                if line == "quit" || line == "exit" {
                    println!("goodbye");
                    return;
                }
                match line.parse::<Move>() {
                    Ok(m) if moves.contains(&m) => break m,
                    Ok(m) => println!("'{m}' is not legal here"),
                    Err(e) => println!("{e}"),
                }
            }
        } else {
            let m = engine.best_move(&board).expect("non-terminal position");
            println!("engine plays {m}");
            m
        };
        board.apply(chosen);
    }
}
