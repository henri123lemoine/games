//! Play Liar's Dice against a freshly-solved CFR+ strategy in the terminal.
//!
//!     cargo run --release -p liars-dice --example play [dice] [faces]
//!
//! You are player 0. On your turn, type the number of the action you want.

use std::io::{self, Write};

use cfr_core::{Game, Solver, Turn};
use liars_dice::LiarsDice;

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn sample<T: Copy>(outs: &[(T, f64)], r: f64) -> T {
    let mut acc = 0.0;
    for (a, p) in outs {
        acc += *p;
        if r < acc {
            return *a;
        }
    }
    outs[outs.len() - 1].0
}

fn main() {
    let dice: u8 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let faces: u8 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let seed: u64 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0xD1CE_F00D);

    let game = LiarsDice::two_player(dice, faces);
    eprint!("Solving {dice}x{faces} Liar's Dice... ");
    let mut solver = Solver::new(LiarsDice::two_player(dice, faces), 1);
    solver.solve(3000);
    eprintln!("done ({} infosets).", solver.num_infosets());
    println!(
        "You are Player 0 (the CFR bot is Player 1). Each has {dice} die/dice, faces 1..={faces}.\n"
    );

    let mut s = game.initial_state();
    let mut rng = Rng(seed | 1);
    loop {
        if game.is_terminal(&s) {
            let r0 = game.returns(&s, 0);
            let msg = if r0 > 0.0 {
                "You win!"
            } else if r0 < 0.0 {
                "The bot wins."
            } else {
                "Draw."
            };
            println!("\n=== {msg} ===");
            break;
        }
        match game.turn(&s) {
            Turn::Chance => {
                let a = sample(&game.chance_outcomes(&s), rng.unit());
                game.apply(&mut s, a);
            }
            Turn::Player(0) => {
                let (q, f) = s.current_bid();
                println!("Your hand: {:?}   dice left {:?}", s.hand(0), s.dice_left());
                if q == 0 {
                    println!("Opening bid — choose one:");
                } else {
                    println!("Current bid: {q} x {f}. Choose one:");
                }
                let actions = game.legal_actions(&s);
                for (i, &a) in actions.iter().enumerate() {
                    println!("  [{i}] {}", game.action_label(a));
                }
                print!("> ");
                io::stdout().flush().unwrap();
                let mut line = String::new();
                io::stdin().read_line(&mut line).unwrap();
                let idx: usize = line.trim().parse().unwrap_or(0).min(actions.len() - 1);
                game.apply(&mut s, actions[idx]);
            }
            Turn::Player(_) => {
                let actions = game.legal_actions(&s);
                let i = solver.sample_action(&s, 1, rng.unit());
                println!("Bot: {}", game.action_label(actions[i]));
                game.apply(&mut s, actions[i]);
            }
        }
    }
}
