//! Play N-player Liar's Dice against the tuned probabilistic bots in the terminal.
//!
//!     cargo run --release -p liars-dice --example play [players] [dice] [faces]
//!
//! You are player 0. On your turn, type the number of the action you want.

use std::io::{self, Write};

use cfr_core::{Agent, Game, Turn};
use liars_dice::{Action, LiarsDice, ProbConfig, RolloutAgent};

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

fn arg<T: std::str::FromStr>(i: usize, d: T) -> T {
    std::env::args()
        .nth(i)
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

fn main() {
    let players: u8 = arg(1, 5);
    let dice: u8 = arg(2, 5);
    let faces: u8 = arg(3, 6);
    let seed: u64 = arg(4, 0xD1CED1CE);

    let rollouts: u32 = arg(5, 1000);
    let game = LiarsDice::new(players, dice, faces);
    let bot = RolloutAgent::new(rollouts, ProbConfig::default(), seed ^ 0xBADC0DE);
    let mut rng = Rng(seed | 1);

    println!("Liar's Dice — {players} players, {dice} dice each, faces 1..={faces}.");
    println!("(bots use Monte-Carlo lookahead, {rollouts} rollouts/decision)");
    println!(
        "You are Player 0; Players 1..{} are the bot.\n",
        players - 1
    );

    let mut s = game.initial_state();
    loop {
        if game.is_terminal(&s) {
            let winner = (0..players as usize)
                .find(|&p| game.returns(&s, p) > 0.0)
                .unwrap();
            println!(
                "\n=== {} wins! ===",
                if winner == 0 {
                    "You".into()
                } else {
                    format!("Player {winner}")
                }
            );
            break;
        }
        let p = match game.turn(&s) {
            Turn::Chance => {
                let outs = game.chance_outcomes(&s);
                let r = rng.unit();
                let (mut acc, mut chosen) = (0.0, outs[outs.len() - 1].0);
                for (a, pr) in &outs {
                    acc += *pr;
                    if r < acc {
                        chosen = *a;
                        break;
                    }
                }
                game.apply(&mut s, chosen);
                continue;
            }
            Turn::Player(p) => p,
        };

        let (q, f) = s.current_bid();
        let actions = game.legal_actions(&s);
        let idx = if p == 0 {
            println!("---");
            println!("Your hand: {:?}", s.hand(0));
            println!(
                "Dice left per player: {:?}",
                &s.dice_left()[..players as usize]
            );
            if q == 0 {
                println!("You open the round. Choose:");
            } else {
                println!(
                    "Current bid: {q} x face {f} (by Player {}). Choose:",
                    s.last_bidder()
                );
            }
            for (i, &a) in actions.iter().enumerate() {
                println!("  [{i}] {}", game.action_label(a));
            }
            print!("> ");
            io::stdout().flush().unwrap();
            let mut line = String::new();
            io::stdin().read_line(&mut line).unwrap();
            line.trim().parse().unwrap_or(0usize).min(actions.len() - 1)
        } else {
            let i = bot.act(&game, &s, p, rng.unit());
            println!("Player {p}: {}", game.action_label(actions[i]));
            i
        };

        let a = actions[idx];
        let is_call = matches!(a, Action::CallLiar | Action::CallExact);
        let dice_before: Vec<u8> = s.dice_left()[..players as usize].to_vec();
        let hands: Vec<Vec<u8>> = (0..players as usize).map(|i| s.hand(i)).collect();
        game.apply(&mut s, a);

        if is_call {
            let actual: usize = hands.iter().flatten().filter(|&&d| d == f).count();
            let who = if p == 0 {
                "You".into()
            } else {
                format!("Player {p}")
            };
            println!("  → {who} called on {q}×{f}. Revealed dice: {hands:?}");
            println!("  → actual count of face {f}: {actual} (bid was {q}).");
            let after = &s.dice_left()[..players as usize];
            if let Some(loser) = (0..players as usize).find(|&i| after[i] < dice_before[i]) {
                let l = if loser == 0 {
                    "you".into()
                } else {
                    format!("Player {loser}")
                };
                println!("  → {l} lose a die (now {}).", after[loser]);
            } else {
                println!("  → exact! nobody loses a die.");
            }
        }
    }
}
