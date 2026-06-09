//! Play Twenty-One against the solver in the terminal.
//!
//!     cargo run --release -p twentyone --example play [hearts] [iters_per_subgame | solver.bin]
//!
//! You are player 0. Each round you see your cards and the opponent's face-up
//! card; type `d` to draw or `s` to stand.

use std::io::{self, Write};
use std::time::Instant;

use twentyone::{Action, Env, Solver};

fn main() {
    let hearts: u8 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);
    let trained = std::env::args().nth(2).unwrap_or_default();

    let solver = if trained.ends_with(".bin") {
        println!("Loading solver from {trained}...");
        Solver::load(&trained).expect("load solver")
    } else {
        let iters: u64 = trained.parse().unwrap_or(50_000);
        let mut s = if hearts <= 2 {
            Solver::with_hearts(0xD1CE, hearts)
        } else {
            Solver::abstracted(0xD1CE, hearts)
        };
        eprint!("Training ({iters} iters/subgame)... ");
        let t = Instant::now();
        s.solve(iters);
        eprintln!("done in {:.0}s.", t.elapsed().as_secs_f64());
        s
    };

    let mut env = Env::with_hearts(rand_seed(), hearts);
    println!("\nTwenty-One — first player out of hearts loses. You are Player 0.");
    while !env.is_game_over() {
        env.start_new_round().expect("new round");
        println!(
            "\n=== Round {} ({} damage) — hearts you {} / bot {} ===",
            env.round(),
            env.round(),
            env.hearts(0),
            env.hearts(1)
        );
        let mut bot_announced_stand = false;
        loop {
            let p = env.current_player();
            let action = if p == 0 {
                let o = env.observation(0);
                println!(
                    "your total: {} (up {}, hidden {})   bot shows: {}{}",
                    o.self_total,
                    o.self_face_up,
                    o.self_face_down,
                    o.opp_face_up,
                    if o.opp_stood { "  [bot stood]" } else { "" }
                );
                prompt_action()
            } else {
                let draw_p = solver.play_draw_prob(&env, 1);
                let a = if draw_p > 0.5 {
                    Action::Draw
                } else {
                    Action::Stand
                };
                if a == Action::Draw {
                    println!("bot: draw");
                    bot_announced_stand = false;
                } else if !bot_announced_stand {
                    println!("bot: stand");
                    bot_announced_stand = true;
                }
                a
            };
            let res = env.step(action).expect("step");
            if res.round_over {
                if let Some(out) = res.outcome {
                    println!(
                        "round over: {} ({} damage)",
                        match out.winner {
                            Some(0) => "you win the round",
                            Some(_) => "bot wins the round",
                            None => "push — no damage",
                        },
                        out.damage
                    );
                }
                break;
            }
        }
    }
    println!(
        "\n=== {} ===",
        if env.utility(0) > 0.0 {
            "You win!"
        } else {
            "The bot wins."
        }
    );
}

fn prompt_action() -> Action {
    loop {
        print!("[d]raw or [s]tand> ");
        io::stdout().flush().unwrap();
        let mut line = String::new();
        if io::stdin().read_line(&mut line).unwrap() == 0 {
            std::process::exit(0);
        }
        match line.trim() {
            "d" | "draw" => return Action::Draw,
            "s" | "stand" => return Action::Stand,
            _ => println!("type d or s"),
        }
    }
}

fn rand_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos() as u64
        | 1
}
