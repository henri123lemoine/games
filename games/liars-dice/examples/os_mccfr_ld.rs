//! Can outcome-sampling MCCFR learn Liar's Dice through the existing
//! abstracted information-set key (own hand + bid + last-K history)?
//!
//! Trains [`solvers::os_mccfr::OsMccfr`] to several iteration checkpoints per
//! configuration and evaluates the learned *average strategy* as an arena
//! agent against (a) uniform-random and (b) the tuned probabilistic belief
//! player, ~1000 seat-swapped games each.
//!
//!     cargo run --release -p liars-dice --example os_mccfr_ld

use std::time::Instant;

use game_core::{RandomAgent, win_rate};
use liars_dice::{LiarsDice, ProbabilisticAgent};
use solvers::os_mccfr::OsMccfr;

const EVAL_GAMES: u32 = 1000;

fn main() {
    let configs: [(u8, u8, u8, &[u64]); 3] = [
        (2, 1, 3, &[1_000, 10_000, 100_000, 1_000_000, 10_000_000]),
        (2, 2, 4, &[1_000, 10_000, 100_000, 1_000_000, 10_000_000]),
        (2, 2, 6, &[10_000, 100_000, 1_000_000, 10_000_000]),
    ];
    println!(
        "{:>8} {:>9} {:>10} {:>10} {:>10} {:>9}",
        "config", "iters", "vs-random", "vs-belief", "infosets", "train-t"
    );
    println!("{}", "-".repeat(62));
    for (p, d, f, checkpoints) in configs {
        let mut solver = OsMccfr::new(LiarsDice::new(p, d, f), 0xD1CE);
        let mut done = 0u64;
        for &iters in checkpoints {
            let t = Instant::now();
            solver.run(iters - done);
            done = iters;
            let train_t = t.elapsed();
            let vs_random = win_rate(solver.game(), &solver, &RandomAgent, EVAL_GAMES, 0xA11CE);
            let belief = ProbabilisticAgent::default_agent();
            let vs_belief = win_rate(solver.game(), &solver, &belief, EVAL_GAMES, 0xB0B);
            println!(
                "{:>8} {:>9} {:>10.3} {:>10.3} {:>10} {:>8.1?}",
                format!("{p}p{d}d{f}f"),
                iters,
                vs_random,
                vs_belief,
                solver.num_infosets(),
                train_t
            );
        }
        println!();
    }
}
