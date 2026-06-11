//! Train ES-MCCFR on a config, then run a round-robin win-rate gauntlet between
//! the trained policy, the probabilistic agent, and random.
//!
//!     cargo run --release -p liars-dice --example gauntlet [players] [dice] [faces] [iters]

use game_core::{Agent, RandomAgent, winrate_vs_field};
use liars_dice::{LdState, LiarsDice, ProbabilisticAgent};
use solvers::Mccfr;

fn arg<T: std::str::FromStr>(i: usize, d: T) -> T {
    std::env::args()
        .nth(i)
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

fn main() {
    let players: u8 = arg(1, 2);
    let dice: u8 = arg(2, 2);
    let faces: u8 = arg(3, 6);
    let iters: u64 = arg(4, 300_000);
    let games = 8000u32;

    let game = LiarsDice::new(players, dice, faces);
    eprint!("Training MCCFR on {players}p{dice}d{faces}f for {iters} iters... ");
    let t = std::time::Instant::now();
    let mut mccfr = Mccfr::new(LiarsDice::new(players, dice, faces), 0xC0FFEE);
    mccfr.run(iters);
    eprintln!(
        "done in {:.1}s ({} infosets).",
        t.elapsed().as_secs_f64(),
        mccfr.num_infosets()
    );

    let prob = ProbabilisticAgent::default_agent();
    let rand = RandomAgent;
    let mc = |_g: &LiarsDice, s: &LdState, p: usize, rng: &mut game_core::Rng| {
        mccfr.sample_action(s, p, rng)
    };

    let named: [(&str, &dyn Agent<LiarsDice>); 3] =
        [("mccfr", &mc), ("prob", &prob), ("random", &rand)];

    println!(
        "\nwin-rate of row vs a field of column (fair = {:.3}):",
        1.0 / players as f64
    );
    print!("{:>8}", "");
    for (n, _) in &named {
        print!("{:>9}", n);
    }
    println!();
    for (hn, hero) in &named {
        print!("{hn:>8}");
        for (bn, base) in &named {
            if hn == bn {
                print!("{:>9}", "-");
            } else {
                let wr = winrate_vs_field(&game, *hero, *base, games, 0x1234);
                print!("{wr:>9.3}");
            }
        }
        println!();
    }
}
