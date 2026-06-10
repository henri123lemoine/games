//! The lab's terminal client — one play loop for every registered game.
//!
//!     lab list
//!     lab play <game> [key=value ...]
//!
//! e.g. `lab play chess depth=6 seat=1`, `lab play liars-dice players=3 dice=2`.

mod registry;
mod runner;

use std::collections::HashMap;
use std::io::{self, Write};

use registry::{Opts, entries};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("list") => list(),
        Some("play") if args.len() >= 2 => play(&args[1], &args[2..]),
        _ => {
            eprintln!("usage: lab list | lab play <game> [key=value ...]");
            std::process::exit(2);
        }
    }
}

fn list() {
    println!("games:");
    for e in entries() {
        println!("  {:<12} {}", e.id, e.summary);
        println!("  {:<12}   opts: {}", "", e.opts_help);
    }
}

fn play(game_id: &str, kvs: &[String]) {
    let entry = match entries().into_iter().find(|e| e.id == game_id) {
        Some(e) => e,
        None => {
            eprintln!("unknown game '{game_id}' — try `lab list`");
            std::process::exit(2);
        }
    };
    let mut map = HashMap::new();
    for kv in kvs {
        match kv.split_once('=') {
            Some((k, v)) => {
                map.insert(k.to_string(), v.to_string());
            }
            None => {
                eprintln!("options are key=value, got '{kv}'");
                std::process::exit(2);
            }
        }
    }
    let mut m = match (entry.make)(&Opts(map)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };

    loop {
        for line in m.advance() {
            println!("{line}");
        }
        if m.is_over() {
            println!("\n=== {} ===", m.result_text());
            return;
        }
        println!("\n{}", m.view());
        let labels = m.legal_labels();
        const MENU_CAP: usize = 24;
        for (i, l) in labels.iter().take(MENU_CAP).enumerate() {
            println!("  [{i}] {l}");
        }
        if labels.len() > MENU_CAP {
            println!(
                "  ... {} more (type the move directly)",
                labels.len() - MENU_CAP
            );
        }
        loop {
            print!("> ");
            io::stdout().flush().unwrap();
            let mut line = String::new();
            if io::stdin().read_line(&mut line).unwrap() == 0 {
                println!("\ninput closed — goodbye");
                return;
            }
            if line.trim().is_empty() {
                continue;
            }
            match m.apply_human(&line) {
                Ok(narration) => {
                    for n in narration {
                        println!("{n}");
                    }
                    break;
                }
                Err(e) => println!("{e}"),
            }
        }
    }
}
