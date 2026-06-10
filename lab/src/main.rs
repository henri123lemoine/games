//! The lab's terminal client — one play loop for every registered game, plus
//! bot-vs-bot evaluation.
//!
//!     lab list
//!     lab play <game> [key=value ...]
//!     lab compare <game> a=<bot> b=<bot> [max-games=N elo0= elo1= ...]
//!     lab tourney <game> bots=<bot>,<bot>,... [games=N ...]
//!
//! e.g. `lab play chess depth=6 seat=1`, `lab play liars-dice players=3
//! dice=2`, `lab compare connect4 a=alphabeta:depth=7 b=alphabeta:depth=3`.

mod compare;
mod registry;
mod runner;

use std::collections::HashMap;
use std::io::{self, Write};

use compare::{CompareArgs, TourneyArgs, split_specs};
use registry::{CompareEntry, Opts, compare_entries, entries};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("list") => list(),
        Some("play") if args.len() >= 2 => play(&args[1], &args[2..]),
        Some("compare") if args.len() >= 2 => run_compare(&args[1], &args[2..]),
        Some("tourney") if args.len() >= 2 => run_tourney(&args[1], &args[2..]),
        _ => {
            eprintln!(
                "usage: lab list | lab play <game> [key=value ...] | lab compare <game> a=<bot> \
                 b=<bot> [key=value ...] | lab tourney <game> bots=<bot>,... [key=value ...]"
            );
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
    println!("\ncompare/tourney bots:");
    for c in compare_entries() {
        println!("  {:<12} {}", c.id, c.bots_help);
    }
}

fn parse_kvs(kvs: &[String]) -> HashMap<String, String> {
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
    map
}

fn find_compare_entry(game_id: &str) -> CompareEntry {
    match compare_entries().into_iter().find(|e| e.id == game_id) {
        Some(e) => e,
        None => {
            eprintln!("no compare support for '{game_id}' — try `lab list`");
            std::process::exit(2);
        }
    }
}

fn take<T: std::str::FromStr>(map: &mut HashMap<String, String>, key: &str, default: T) -> T {
    match map.remove(key) {
        Some(v) => match v.parse() {
            Ok(t) => t,
            Err(_) => {
                eprintln!("could not parse {key}={v}");
                std::process::exit(2);
            }
        },
        None => default,
    }
}

fn run_compare(game_id: &str, kvs: &[String]) {
    let entry = find_compare_entry(game_id);
    let mut map = parse_kvs(kvs);
    let (Some(a), Some(b)) = (map.remove("a"), map.remove("b")) else {
        eprintln!(
            "compare needs a=<bot> and b=<bot> (bots: {})",
            entry.bots_help
        );
        std::process::exit(2);
    };
    let args = CompareArgs {
        a,
        b,
        elo0: take(&mut map, "elo0", 0.0),
        elo1: take(&mut map, "elo1", 20.0),
        alpha: take(&mut map, "alpha", 0.05),
        beta: take(&mut map, "beta", 0.05),
        max_games: take(&mut map, "max-games", 1000),
        batch: take(&mut map, "batch", 16),
        delta: take(&mut map, "delta", 0.1),
        seed: take(&mut map, "seed", 0xC0FFEE),
        opts: Opts(map),
    };
    if let Err(e) = (entry.compare)(&args) {
        eprintln!("error: {e}");
        std::process::exit(2);
    }
}

fn run_tourney(game_id: &str, kvs: &[String]) {
    let entry = find_compare_entry(game_id);
    let mut map = parse_kvs(kvs);
    let Some(bots) = map.remove("bots") else {
        eprintln!(
            "tourney needs bots=<bot>,<bot>,... (bots: {})",
            entry.bots_help
        );
        std::process::exit(2);
    };
    let args = TourneyArgs {
        bots: split_specs(&bots),
        games: take(&mut map, "games", 20),
        seed: take(&mut map, "seed", 0xC0FFEE),
        opts: Opts(map),
    };
    if let Err(e) = (entry.tourney)(&args) {
        eprintln!("error: {e}");
        std::process::exit(2);
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
    let map = parse_kvs(kvs);
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
