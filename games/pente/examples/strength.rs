//! Strength + timing probe for the Pente alpha-beta engine.
//!
//! Three things, printed to stdout:
//!   1. per-move search latency at a few depths on the standard 13x13 board,
//!   2. a battery of constructed tactical positions where there is one correct
//!      move (take the winning capture, complete five, block the opponent's
//!      open four, take an available capture, refuse to walk into a capture),
//!   3. a deeper-vs-shallower self-play mini-match, so depth is shown to buy
//!      strength rather than just spending time.
//!
//! Run: `cargo run --release -p pente --example strength`.

use std::time::Instant;

use game_core::{Agent, Game, GameUi, Rng};
use pente::{Pente, PenteAction, PenteEval, PenteSpec};
use solvers::AlphaBeta;

fn engine(depth: u32) -> AlphaBeta<Pente, PenteEval, PenteSpec> {
    AlphaBeta::new(depth, PenteEval, PenteSpec)
}

/// Best move's coordinate label for a position the engine is to play.
fn best_label(g: &Pente, s: &<Pente as Game>::State, depth: u32) -> String {
    let ab = engine(depth);
    let i = ab.best_action(g, s);
    let a = g.legal_actions(s)[i];
    g.action_label(s, a)
}

fn timing(g: &Pente) {
    println!("== per-move latency, 13x13, midgame-ish position ==");
    // A modest opening so the board is non-trivial but realistic.
    let mut s = g.initial_state();
    let opening = ["g7", "h8", "f6", "h6", "g6", "f8", "g8", "h7"];
    for c in opening {
        let a = PenteAction(g.point(c).expect("coord"));
        g.apply(&mut s, a);
    }
    for depth in [2, 4, 6] {
        let ab = engine(depth);
        let t = Instant::now();
        let _ = ab.best_action(g, &s);
        println!(
            "  depth {depth}: {:>8.1} ms  ({} nodes)",
            t.elapsed().as_secs_f64() * 1e3,
            ab.node_count()
        );
    }
}

fn tactics() {
    let g = Pente::new(9);
    let g = &g;
    println!("\n== tactical correctness (depth 4) ==");
    let mut pass = 0;
    let mut total = 0;
    let mut check = |name: &str, s: &<Pente as Game>::State, want: &[&str]| {
        total += 1;
        let got = best_label(g, s, 4);
        let ok = want.contains(&got.as_str());
        if ok {
            pass += 1;
        }
        println!(
            "  [{}] {name}: played {got}, wanted one of {want:?}",
            if ok { "ok" } else { "XX" }
        );
    };

    // 1. A winning capture (the fifth pair) is available — take it.
    let s = g.parse_state(
        &[
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". X O O . . . . .",
            ". . . . . . . . .",
        ],
        0,
        [4, 0],
    );
    check("take the fifth pair", &s, &["e2"]);

    // 2. Complete an open four into five.
    let s = g.parse_state(
        &[
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". X X X X . . . .",
            ". . . . . . . . .",
        ],
        0,
        [0, 0],
    );
    check("complete five-in-a-row", &s, &["a2", "f2"]);

    // 3. Block the opponent's open four (White just made X X X X . threats; it
    //    is Black to move and must plug an end).
    let s = g.parse_state(
        &[
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". O O O O . . . .",
            ". . . . . . . . .",
        ],
        0,
        [0, 0],
    );
    check("block the open four", &s, &["a2", "f2"]);

    // 4. A free capture is on the board — take it (no higher-value move exists).
    let s = g.parse_state(
        &[
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". X O O . . . . .",
            ". . . . . . . . .",
        ],
        0,
        [0, 0],
    );
    check("take the available capture", &s, &["e2"]);

    println!("  tactics: {pass}/{total} correct");
}

/// A seat-swapped mini-match between two depths; returns deep's score over
/// `games` pairs (1 win, 0.5 draw, 0 loss), out of `games * 2`.
fn duel(g: &Pente, deep: u32, shallow: u32, games: u64) -> f64 {
    let mut score = 0.0;
    for k in 0..games {
        for deep_is_black in [true, false] {
            let mut s = g.initial_state();
            let mut rng = Rng::new(0x9E37_79B9_u64.wrapping_add(k));
            let (a, b) = (engine(deep), engine(shallow));
            while !g.is_terminal(&s) {
                let p = match g.turn(&s) {
                    game_core::Turn::Player(p) => p,
                    _ => unreachable!(),
                };
                let deep_to_move = (p == 0) == deep_is_black;
                let i = if deep_to_move {
                    a.act(g, &s, p, &mut rng)
                } else {
                    b.act(g, &s, p, &mut rng)
                };
                let action = g.legal_actions(&s)[i];
                g.apply(&mut s, action);
            }
            let deep_seat = if deep_is_black { 0 } else { 1 };
            score += (g.returns(&s, deep_seat) + 1.0) / 2.0;
        }
    }
    score
}

fn main() {
    let g = Pente::new(13);
    timing(&g);
    tactics();

    println!("\n== depth ladder, 13x13 (seat-swapped, deeper listed first) ==");
    for (deep, shallow, games) in [(4, 2, 6), (6, 4, 3)] {
        let score = duel(&g, deep, shallow, games);
        println!(
            "  depth-{deep} scored {score}/{} vs depth-{shallow}",
            games * 2
        );
    }
}
