//! K-iteration leaf-value cache sweep for the ReBeL vector-CFR solver.
//!
//! QUALITY (gating): depth-2 PerfectOracleLeaf exploitability on
//! StandardLiarsDice (1x4f, 1x5f) at `leaf_refresh_every=K`. K=1 is the exact
//! reference (~0.026 on 1x4f).
//!
//! THROUGHPUT: 5p5d6f data-gen samples/s at hidden=256, num_iters=256, parallel
//! — the production path (NetLeaf continuation, principled open cap).
//!
//!     cargo run --release -p liars-dice --features parallel --example rebel_kcache_sweep

use std::time::Instant;

use game_core::Rng;
use rayon::prelude::*;

use liars_dice::rebel::deploy_train::sample_fixed_round;
use liars_dice::rebel::{
    Belief, CfrParams, DeployCont, LiarsDiceAdapter, NetContinuation, PbsNet, PerfectOracleLeaf,
    RebelGame, SelfPlayParams, Solver, StandardLiarsDice, TerminalLeaf, Tree, exploitability,
    generate_episode,
};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Stitch the depth-2 (oracle-leaf) top policy onto the full-game equilibrium and
/// measure full-game exploitability — mirrors the in-tree gate test exactly,
/// parameterized by `leaf_refresh_every`.
fn depth2_oracle_exploitability(d: u8, f: u8, k: usize) -> f64 {
    let game = StandardLiarsDice::new(d, f);
    let initial = Belief::uniform_prior(&game.root());

    let exact = TerminalLeaf::new(&game);
    let full_params = CfrParams {
        num_iters: 1024,
        max_depth: u32::MAX,
        ..CfrParams::default()
    };
    let mut full_solver = Solver::new(&game, full_params, &exact, initial.clone());
    full_solver.multistep();
    let mut stitched = full_solver.average_strategy().to_vec();

    let oracle = PerfectOracleLeaf::new(&game, 256);
    let depth2_params = CfrParams {
        num_iters: 512,
        max_depth: 2,
        leaf_refresh_every: k,
        ..CfrParams::default()
    };
    let mut depth2_solver = Solver::new(&game, depth2_params, &oracle, initial);
    depth2_solver.multistep();
    let top = depth2_solver.average_strategy();

    let depth2_tree = Tree::build(&game, 2);
    for idx in 0..depth2_tree.len() {
        if !depth2_tree.nodes[idx].is_leaf {
            stitched[idx].clone_from(&top[idx]);
        }
    }
    exploitability(&game, &stitched)
}

fn throughput_5p5d6f(
    k: usize,
    hidden: usize,
    num_iters: usize,
    episodes: usize,
    base_seed: u64,
) -> (usize, f64) {
    let net = PbsNet::new(hidden, 2, 12_345);
    let sp = SelfPlayParams {
        cfr: CfrParams {
            num_iters,
            max_depth: 2,
            leaf_refresh_every: k,
            ..CfrParams::default()
        },
        explore_eps: 0.25,
    };
    let t0 = Instant::now();
    let total: usize = (0..episodes)
        .into_par_iter()
        .map(|e| {
            let seed = base_seed ^ (e as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut rng = Rng::new(seed);
            let round = sample_fixed_round(&mut rng, 5, 5, 6, 0.0);
            let cont = DeployCont::Net(NetContinuation::new(&net));
            let adapter = LiarsDiceAdapter::new(
                round.players,
                round.faces,
                round.dice_left,
                round.opener,
                round.first_round,
                &cont,
            )
            .with_principled_open_cap();
            generate_episode(&adapter, sp, &net, &mut rng).len()
        })
        .sum();
    (total, t0.elapsed().as_secs_f64())
}

fn main() {
    let threads = rayon::current_num_threads();
    let hidden = env_usize("HIDDEN", 256);
    let num_iters = env_usize("NUM_ITERS", 256);
    let episodes = env_usize("EPISODES", threads * 2);
    let ks = [1usize, 2, 4, 8];

    println!("threads={threads} hidden={hidden} num_iters={num_iters} episodes={episodes}");

    if env_usize("QUALITY", 1) != 0 {
        println!("\n=== QUALITY: depth-2 perfect-oracle exploitability (K=1 exact ref) ===");
        println!("{:>3} | {:>10} | {:>10}", "K", "1x4f", "1x5f");
        for &k in &ks {
            let e4 = depth2_oracle_exploitability(1, 4, k);
            let e5 = depth2_oracle_exploitability(1, 5, k);
            println!("{k:>3} | {e4:>10.6} | {e5:>10.6}");
        }
    }

    println!("\n=== THROUGHPUT: 5p5d6f data-gen samples/s ===");
    println!(
        "{:>3} | {:>12} | {:>8} | {:>8}",
        "K", "samples/s", "speedup", "samples"
    );
    let mut base_sps = 0.0f64;
    for &k in &ks {
        let (n, secs) = throughput_5p5d6f(k, hidden, num_iters, episodes, 0xABCD_1234);
        let sps = n as f64 / secs;
        if k == 1 {
            base_sps = sps;
        }
        println!("{k:>3} | {sps:>12.1} | {:>7.2}x | {n:>8}", sps / base_sps);
    }
}
