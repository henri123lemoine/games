//! Train the decomposed CFR+ solver and report quality.
//!
//!     cargo run --release -p twentyone --example solve [hearts] [iters_per_subgame] [out.bin]
//!
//! Small variants (1-2 hearts) use lossless information sets and also report
//! exact best-response exploitability; larger ones use the band abstraction
//! (the bake-off champion — see BAKEOFF.md).

use std::time::Instant;

use twentyone::Solver;

fn arg<T: std::str::FromStr>(i: usize, d: T) -> T {
    std::env::args()
        .nth(i)
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

fn main() {
    let hearts: u8 = arg(1, 1);
    let iters: u64 = arg(2, 100_000);
    let out = std::env::args().nth(3);

    let mut solver = if hearts <= 2 {
        Solver::with_hearts(0xD1CE, hearts)
    } else {
        Solver::abstracted(0xD1CE, hearts)
    };
    println!("Training {hearts}-heart Twenty-One, {iters} iters/subgame...");
    let t = Instant::now();
    solver.solve(iters);
    println!(
        "done in {:.1}s — {} infosets",
        t.elapsed().as_secs_f64(),
        solver.num_infosets()
    );

    if hearts <= 2 {
        let t = Instant::now();
        let (br0, br1, nashconv) = solver.exploitability(0, 0);
        println!(
            "exact best response: br0 {br0:.4}  br1 {br1:.4}  exploitability {:.4}  ({:.1}s)",
            nashconv / 2.0,
            t.elapsed().as_secs_f64()
        );
    }
    if let Some(path) = out {
        solver.save_play(&path).expect("save solver");
        println!("saved play artifact to {path}");
    }
}
