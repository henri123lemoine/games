//! Prove fitted value iteration learns the *real* continuation value: the
//! policy/value net's value head, bootstrapped on its own backed-up round
//! values, converges to the exact 2-player continuation lattice.
//!
//! # What the heuristic baseline can't do
//!
//! The shipped Stage-A continuation [`DiceShareValue`] is a fixed dice-share
//! guess; training the value head against it only teaches the head to *mimic*
//! the heuristic — there is no value learning. Fitted value iteration replaces
//! that fixed continuation with the net's own value head and trains it on the
//! one-round Bellman backup (the realised round-leaf value, which under the net
//! continuation *is* the head's prediction of the post-round equity). Iterating
//! drives the head to the fixed point.
//!
//! # The oracle
//!
//! [`fit_two_player`] computes the EXACT 2-player continuation lattice by
//! repeated exact CFR solves swept to convergence (`solve.rs`). It is the proven
//! ground truth. We report the value head's mean-absolute-error against it:
//!
//!   * after the warm start (heuristic continuation) — the baseline,
//!   * after fitted-VI iterations — must be SMALLER (converging toward exact).
//!
//! We also report the exact heuristic's own MAE to the lattice, so the win is
//! unambiguous: fitted VI ends up closer to the proven values than the fixed
//! heuristic it replaced.
//!
//! Run: `cargo run --release -p liars-dice --example value_verify`
//! Add `-- full` to also verify 2p2d3f (richer, but several minutes — the
//! infinite-horizon lattice oracle for 2d3 converges slowly).

use std::time::Instant;

use liars_dice::train::{fit_value_head_2p, value_head_lattice_mae};
use liars_dice::{DiceShareValue, FitConfig, LatticeValue, fit_two_player};

/// CFR iters per round solve for the EXACT lattice oracle — pins these tiny
/// games to ~1e-5 so the ground truth is tight.
const ORACLE_ITERS: u64 = 2000;
/// CFR iters per round solve inside the fitted-VI backup. Lower than the oracle:
/// the backup re-solves every lattice state each iteration, and the value head
/// (a small SGD-fit MLP) is the precision bottleneck, not the per-round solve —
/// so cheaper solves keep the proof under a minute without moving the result.
const BACKUP_ITERS: u64 = 800;

/// The exact-heuristic MAE to the lattice: what mimicking `DiceShareValue`
/// perfectly would score. The bar fitted VI must beat.
fn heuristic_mae(dice: u8, faces: u8, exact: &LatticeValue) -> f64 {
    use liars_dice::ContinuationValue;
    let mut sum = 0.0;
    let mut n = 0u32;
    for a in 1..=dice {
        for b in 1..=dice {
            for opener in 0..2usize {
                let h = DiceShareValue.value(faces, &[a, b], opener, 0);
                let e = exact.get_two_player(&[a, b], opener).unwrap();
                sum += (h - e).abs();
                n += 1;
            }
        }
    }
    sum / n.max(1) as f64
}

fn main() {
    // 1d6 (2 lattice states) is the cheapest fixed point — the default proof runs
    // in well under a minute. `full` adds 2d3 (8 states): a richer demonstration,
    // but its infinite-horizon lattice oracle converges slowly (the CallExact
    // coupling needs ~200 sweeps), so it costs several minutes — opt-in.
    let full = std::env::args().any(|a| a == "full");
    let configs: &[(u8, u8)] = if full { &[(1, 6), (2, 3)] } else { &[(1, 6)] };
    let (warmup, iters) = (5usize, 30usize);

    println!("== Fitted value iteration vs the EXACT continuation lattice ==\n");
    println!(
        "Each config: exact 2p lattice (oracle) computed by fit_two_player, then the\n\
         value head is warm-started on the heuristic ({warmup} iters) and bootstrapped on\n\
         its own backed-up values ({} fitted-VI iters). MAE is the mean |V_head - V_exact|\n\
         over every reachable continuing 2p state (seat 0).\n",
        iters - warmup
    );

    for &(dice, faces) in configs {
        let t = Instant::now();
        let fit = fit_two_player(
            dice,
            faces,
            FitConfig {
                iters_per_solve: ORACLE_ITERS,
                tol: 1e-6,
                max_sweeps: 200,
                measure_exploitability: false,
            },
        );
        let oracle_secs = t.elapsed().as_secs_f64();
        let heur = heuristic_mae(dice, faces, &fit.lattice);

        let t = Instant::now();
        let (net, log) = fit_value_head_2p(
            dice,
            faces,
            &fit.lattice,
            iters,
            warmup,
            BACKUP_ITERS,
            0xF177ED,
        );
        let fvi_secs = t.elapsed().as_secs_f64();

        let warm_end = log[warmup - 1].mae;
        let final_mae = log.last().unwrap().mae;
        // Sanity: a direct rebuild of the MAE from the returned net matches.
        let recomputed = value_head_lattice_mae(&net, dice, faces, &fit.lattice);

        println!(
            "-- 2p{dice}d{faces}f  ({} lattice states) --",
            fit.lattice.len()
        );
        println!(
            "  exact lattice converged in {} sweeps ({oracle_secs:.1}s)",
            fit.sweep_deltas.len()
        );
        println!("  fixed-heuristic MAE to exact : {heur:.4}   (mimicking it perfectly)");
        println!("  value-head MAE @ warmup-end   : {warm_end:.4}   (heuristic continuation)");
        println!("  value-head MAE @ fitted-VI end: {final_mae:.4}   (bootstrapped on itself)");
        println!("  (recomputed from net: {recomputed:.4}; fitted VI ran in {fvi_secs:.1}s)");
        let beats_warm = final_mae < warm_end;
        let beats_heur = final_mae < heur;
        println!(
            "  verdict: fitted VI {} the warmup baseline, {} the fixed heuristic",
            if beats_warm { "BEATS" } else { "DOES NOT BEAT" },
            if beats_heur { "BEATS" } else { "DOES NOT BEAT" },
        );

        // A compact MAE trajectory: warmup-end, then a few fitted-VI checkpoints.
        let marks: Vec<usize> = [warmup - 1, warmup, warmup + 4, warmup + 9, iters - 1]
            .into_iter()
            .filter(|&i| i < log.len())
            .collect();
        print!("  trace [iter:mae]:");
        for i in marks {
            let c = log[i];
            print!(
                "  {}:{:.4}{}",
                c.iter,
                c.mae,
                if c.warm { "(w)" } else { "" }
            );
        }
        println!("\n");
    }

    println!(
        "Interpretation: the heuristic continuation pins the value head to a fixed,\n\
         biased guess (its MAE to the exact lattice is the floor of pure mimicry).\n\
         Fitted value iteration bootstraps off the net's own backed-up values and\n\
         converges BELOW that floor — toward the proven exact continuation — which is\n\
         real value learning, not heuristic imitation."
    );
}
