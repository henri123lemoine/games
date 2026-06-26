//! Lossless + speedup gates for the ReBeL opening action abstraction.
//!
//! The deploy adapter's round-OPEN node enumerates `Bid::Raise{q,f}` for every
//! quantity `q∈1..=total`; at 5p5d6f that is 150 openings and the depth-2 solve
//! over it is the data-gen bottleneck. [`principled_open_cap`] prunes the high
//! quantities (dominated junk — claiming far more of a face than the dice could
//! plausibly show). This harness proves the prune is LOSSLESS and measures the
//! speedup:
//!
//!  A. EXACT value preservation — on a small config solved against the EXACT 2p
//!     continuation lattice, sweep the opening cap and show the equilibrium value
//!     and exploitability are unchanged down to the smallest lossless cap (and
//!     change only once a *non*-dominated opening is cut). No net error.
//!  B. ~0-MASS sanity — solve the FULL (uncapped) opening to equilibrium and show
//!     the openings the principled cap prunes carry ~0 average-strategy
//!     probability (so the cap removes only dominated actions).
//!  C. STRENGTH — train two nets (capped vs full data-gen, same seed/budget) on a
//!     pruning config and compare `RebelAgent` field-win-share vs Rollout.
//!  D. SPEEDUP — 5p5d6f deploy data-gen throughput, capped vs uncapped.
//!
//!     cargo run --release -p liars-dice --example rebel_open_cap
//!
//! Env overrides: EX_DICE EX_FACES FIT_ITERS FIT_SWEEPS EVAL_ITERS (exact gate);
//! PLAYERS DICE FACES STEPS GEN_PER NUM_ITERS HIDDEN WARMUP GAMES ROLLOUTS
//! AGENT_ITERS (strength); MASS_ITERS (sanity); GEN_EPISODES PERF_HIDDEN
//! PERF_ITERS (speedup).

use std::time::Instant;

use game_core::{Rng, winrate_vs_field};
use liars_dice::rebel::deploy_train::sample_fixed_round;
use liars_dice::rebel::{
    Belief, Bid, CfrParams, DeployCont, DeployTrainConfig, DeployTrainer, LiarsDiceAdapter,
    NetContinuation, NetLeaf, PbsNet, RebelAgent, RebelGame, SelfPlayParams, Solver, TerminalLeaf,
    exploitability, generate_episode, principled_open_cap,
};
use liars_dice::{
    BidConditioned, ContinuationValue, DiceShareValue, FitConfig, LatticeValue, LiarsDice,
    ProbabilisticAgent, fit_two_player,
};
use rayon::prelude::*;
use solvers::Rollout;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn dice_vec(counts: &[u8]) -> [u8; 8] {
    let mut d = [0u8; 8];
    d[..counts.len()].copy_from_slice(counts);
    d
}

/// Solve every continuing 2p state against the EXACT `lattice`, with the opening
/// capped at `cap` (`None` = full range). Returns the max per-round exploitability
/// (vs the same exact game) and the seat-0 root value per state — the lossless
/// fingerprint: if a cap removes a strategically-relevant opening, the opener's
/// equilibrium value drops.
fn exact_cap_solve(
    lattice: &LatticeValue,
    dice: u8,
    faces: u8,
    cap: Option<u8>,
    iters: usize,
) -> (f64, Vec<f64>) {
    let params = CfrParams {
        num_iters: iters,
        max_depth: u32::MAX,
        ..CfrParams::default()
    };
    let mut max_expl = 0.0f64;
    let mut values = Vec::new();
    for a in 1..=dice {
        for b in 1..=dice {
            for opener in 0..2usize {
                let d = dice_vec(&[a, b]);
                let ad =
                    LiarsDiceAdapter::new(2, faces, d, opener, false, lattice).with_open_cap(cap);
                let initial = Belief::uniform_prior(&ad.root());
                let terminal = TerminalLeaf::new(&ad);
                let mut solver = Solver::new(&ad, params, &terminal, initial.clone());
                solver.multistep();
                let v0: f64 = solver
                    .root_values_mean(0)
                    .iter()
                    .zip(&initial.per_seat[0])
                    .map(|(v, p)| v * p)
                    .sum();
                let avg = solver.average_strategy().to_vec();
                max_expl = max_expl.max(exploitability(&ad, &avg));
                values.push(v0);
            }
        }
    }
    (max_expl, values)
}

fn max_abs_delta(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Opener average-strategy mass per opening quantity, the principled cap, and the
/// mass that cap would prune. `avg0` is the acting (opener) seat's per-hand
/// strategy at the root, `legal` the root openings, `prior` the opener's hand
/// prior.
fn mass_per_quantity(
    avg0: &[Vec<f64>],
    legal: &[Bid],
    prior: &[f64],
    total: u8,
    faces: u8,
) -> (u8, Vec<f64>, f64) {
    let cap = principled_open_cap(total, faces);
    let mut mass_per_q = vec![0.0f64; total as usize + 1];
    for (hand, row) in avg0.iter().enumerate() {
        let pw = prior[hand];
        for (ai, bid) in legal.iter().enumerate() {
            if let Bid::Raise { qty, .. } = bid {
                mass_per_q[*qty as usize] += pw * row[ai];
            }
        }
    }
    let pruned: f64 = mass_per_q
        .iter()
        .enumerate()
        .filter(|&(q, _)| q as u8 > cap)
        .map(|(_, m)| m)
        .sum();
    (cap, mass_per_q, pruned)
}

/// FULL-depth uncapped equilibrium opener mass (exact terminal payoffs via
/// `cont`); only tractable for tiny rounds (the per-path round tree is large).
fn opening_mass_exact<C: ContinuationValue>(
    cont: &C,
    players: usize,
    faces: u8,
    dice: [u8; 8],
    opener: usize,
    iters: usize,
) -> (u8, Vec<f64>, f64) {
    let adapter = LiarsDiceAdapter::new(players, faces, dice, opener, false, cont);
    let initial = Belief::uniform_prior(&adapter.root());
    let leaf = TerminalLeaf::new(&adapter);
    let params = CfrParams {
        num_iters: iters,
        max_depth: u32::MAX,
        ..CfrParams::default()
    };
    let mut solver = Solver::new(&adapter, params, &leaf, initial.clone());
    solver.multistep();
    let total: u8 = dice[..players].iter().sum();
    mass_per_quantity(
        &solver.average_strategy()[0],
        &solver.tree().nodes[0].legal,
        &initial.per_seat[opener],
        total,
        faces,
    )
}

/// Depth-2 uncapped opener mass at deploy scale. Round-end calls are valued by
/// the [`DiceShareValue`] continuation (which prices the die a beaten bidder
/// loses), the depth-2 raise leaves by a fresh net. A huge opening is decided
/// right here — the responder calls it (its count convolution puts ~0 mass on the
/// claim) and the opener loses a die, so the opener avoids it; the dominance is
/// structural (driven by the exact call resolution + die-loss price), so an
/// untrained net for the raise leaves suffices.
fn opening_mass_depth2(
    players: usize,
    faces: u8,
    dice: [u8; 8],
    opener: usize,
    iters: usize,
) -> (u8, Vec<f64>, f64) {
    let cont = DiceShareValue;
    let adapter = LiarsDiceAdapter::new(players, faces, dice, opener, false, &cont);
    let net = PbsNet::new(64, 2, 0);
    let leaf = NetLeaf::new(&net, &adapter);
    let initial = Belief::uniform_prior(&adapter.root());
    let params = CfrParams {
        num_iters: iters,
        max_depth: 2,
        ..CfrParams::default()
    };
    let mut solver = Solver::new(&adapter, params, &leaf, initial.clone());
    solver.multistep();
    let total: u8 = dice[..players].iter().sum();
    mass_per_quantity(
        &solver.average_strategy()[0],
        &solver.tree().nodes[0].legal,
        &initial.per_seat[opener],
        total,
        faces,
    )
}

fn report_mass(label: &str, total: u8, faces: u8, cap: u8, mass: &[f64], pruned: f64) {
    print!("  {label}: total={total} faces={faces} cap={cap}  mass[q]=");
    for (q, m) in mass.iter().enumerate().skip(1) {
        if *m > 5e-4 {
            print!("{q}:{m:.3} ");
        }
    }
    let smallest_lossless = mass
        .iter()
        .enumerate()
        .rev()
        .find(|&(_, m)| *m > 1e-3)
        .map_or(1, |(q, _)| q.max(1));
    println!(
        "\n    -> pruned mass (q>{cap}) = {pruned:.5}   smallest lossless cap = {smallest_lossless}"
    );
}

#[allow(clippy::too_many_arguments)]
fn train_cfg(
    players: usize,
    dice: u8,
    faces: u8,
    open_cap: bool,
    steps: usize,
    gen_per: usize,
    num_iters: usize,
    hidden: usize,
    warmup: usize,
    burn_in: usize,
) -> DeployTrainConfig {
    DeployTrainConfig {
        steps,
        warmup_steps: warmup,
        num_iters,
        max_depth: 2,
        batch: 256,
        lr: 1e-3,
        gen_per_step: gen_per,
        train_gen_ratio: 16,
        burn_in,
        eval_every: steps.max(1),
        eval_iters: 32,
        // Np configs have no exact-lattice gate; the fit is skipped, so this is moot.
        eval_fit_iters: 1,
        hidden,
        n_layers: 2,
        buffer_cap: 500_000,
        seed: 0,
        log: false,
        fixed_config: Some((players, dice, faces)),
        principled_open_cap: open_cap,
        ..DeployTrainConfig::default()
    }
}

fn measure_gen(use_cap: bool, episodes: usize, hidden: usize, num_iters: usize) -> (usize, f64) {
    let net = PbsNet::new(hidden, 2, 0);
    let sp = SelfPlayParams {
        cfr: CfrParams {
            num_iters,
            max_depth: 2,
            ..CfrParams::default()
        },
        explore_eps: 0.25,
    };
    let t = Instant::now();
    let total: usize = (0..episodes)
        .into_par_iter()
        .map(|e| {
            let mut r = Rng::new(0xD1CE ^ e as u64);
            let round = sample_fixed_round(&mut r, 5, 5, 6, 0.5);
            let cont = DeployCont::Net(NetContinuation::new(&net));
            let adapter = LiarsDiceAdapter::new(
                round.players,
                round.faces,
                round.dice_left,
                round.opener,
                round.first_round,
                &cont,
            );
            let adapter = if use_cap {
                adapter.with_principled_open_cap()
            } else {
                adapter
            };
            generate_episode(&adapter, sp, &net, &mut r).len()
        })
        .sum();
    (total, t.elapsed().as_secs_f64())
}

fn main() {
    let wall = Instant::now();

    // ---- A. EXACT value preservation across opening caps (no net error) ----
    let ex_dice = env_usize("EX_DICE", 2) as u8;
    let ex_faces = env_usize("EX_FACES", 4) as u8;
    let fit_iters = env_usize("FIT_ITERS", 4000) as u64;
    let fit_sweeps = env_usize("FIT_SWEEPS", 60);
    let eval_iters = env_usize("EVAL_ITERS", 4096);
    println!("=== ReBeL opening action abstraction gates ===");
    println!(
        "--- A. exact value preservation vs opening cap ({ex_dice}d{ex_faces}f, exact lattice) ---"
    );
    eprintln!("[stage] fitting exact lattice {ex_dice}d{ex_faces}f...");
    let lattice = fit_two_player(
        ex_dice,
        ex_faces,
        FitConfig {
            iters_per_solve: fit_iters,
            tol: 1e-3,
            max_sweeps: fit_sweeps,
            measure_exploitability: false,
        },
    )
    .lattice;
    eprintln!(
        "[stage] lattice fit at {:.1}s",
        wall.elapsed().as_secs_f64()
    );

    let (e_full, v_full) = exact_cap_solve(&lattice, ex_dice, ex_faces, None, eval_iters);
    let max_total = 2 * ex_dice;
    println!("  full (no cap): max_exploitability={e_full:.5}");
    for cap in (1..=max_total).rev() {
        let (e_c, v_c) = exact_cap_solve(&lattice, ex_dice, ex_faces, Some(cap), eval_iters);
        let vd = max_abs_delta(&v_full, &v_c);
        let tag = if vd < 5e-3 { "LOSSLESS" } else { "LOSSY   " };
        println!("  cap={cap:<2} {tag}  max|Δ value vs full|={vd:.5}  max_exploitability={e_c:.5}");
    }

    // ---- B. ~0-MASS sanity: pruned openings are dominated in the open solve ----
    println!("\n--- B. ~0-mass sanity (uncapped equilibrium, opener mass per quantity) ---");
    let mass_iters = env_usize("MASS_ITERS", 4096);
    // Small config: the exact FULL-depth equilibrium (tractable here).
    let d = dice_vec(&[ex_dice, ex_dice]);
    let (cap, mass, pruned) = opening_mass_exact(&lattice, 2, ex_faces, d, 0, mass_iters);
    report_mass(
        "2p exact full-depth",
        2 * ex_dice,
        ex_faces,
        cap,
        &mass,
        pruned,
    );
    // Deploy scale: the per-path round tree is too large for full depth, so a
    // depth-2 solve (where the responder's call already decides a big opening).
    let (cap, mass, pruned) = opening_mass_depth2(3, 4, dice_vec(&[4, 4, 4]), 0, mass_iters);
    report_mass("3p4d4f depth-2", 12, 4, cap, &mass, pruned);
    let (cap, mass, pruned) = opening_mass_depth2(5, 6, dice_vec(&[5, 5, 5, 5, 5]), 0, mass_iters);
    report_mass("5p5d6f depth-2", 25, 6, cap, &mass, pruned);
    eprintln!(
        "[stage] exact+mass done at {:.1}s",
        wall.elapsed().as_secs_f64()
    );

    // ---- C. STRENGTH: trained capped vs full data-gen, win-share vs Rollout ----
    let players = env_usize("PLAYERS", 3);
    let dice = env_usize("DICE", 4) as u8;
    let faces = env_usize("FACES", 4) as u8;
    let steps = env_usize("STEPS", 120);
    let gen_per = env_usize("GEN_PER", 48);
    let num_iters = env_usize("NUM_ITERS", 128);
    let hidden = env_usize("HIDDEN", 64);
    let warmup = env_usize("WARMUP", 20);
    let burn_in = env_usize("BURN_IN", 384);
    let games = env_usize("GAMES", 900) as u32;
    let rollouts = env_usize("ROLLOUTS", 150) as u32;
    let agent_iters = env_usize("AGENT_ITERS", 256);
    println!(
        "\n--- C. strength: train full (cap off) vs capped (cap on) on {players}p{dice}d{faces}f ---"
    );
    let mut trainer_full = DeployTrainer::new(train_cfg(
        players, dice, faces, false, steps, gen_per, num_iters, hidden, warmup, burn_in,
    ));
    trainer_full.run();
    eprintln!(
        "[stage] full trained at {:.1}s",
        wall.elapsed().as_secs_f64()
    );
    let mut trainer_capped = DeployTrainer::new(train_cfg(
        players, dice, faces, true, steps, gen_per, num_iters, hidden, warmup, burn_in,
    ));
    trainer_capped.run();
    eprintln!(
        "[stage] capped trained at {:.1}s",
        wall.elapsed().as_secs_f64()
    );

    let game = LiarsDice::new(players as u8, dice, faces).with_max_rounds(24);
    let baseline = Rollout::new(
        rollouts,
        ProbabilisticAgent::default_agent(),
        BidConditioned::default(),
    );
    let fair = 1.0 / players as f64;
    let tmp = std::env::temp_dir();
    let full_path = tmp.join("ld_opencap_full.bin");
    let capped_path = tmp.join("ld_opencap_capped.bin");
    trainer_full.net().save(&full_path).unwrap();
    trainer_capped.net().save(&capped_path).unwrap();
    let agent_full = RebelAgent::with_config(PbsNet::load(&full_path).unwrap(), agent_iters, 2)
        .with_opening_abstraction(false);
    let agent_capped = RebelAgent::with_config(PbsNet::load(&capped_path).unwrap(), agent_iters, 2)
        .with_opening_abstraction(true);
    let t1 = Instant::now();
    let share_full = winrate_vs_field(&game, &agent_full, &baseline, games, 0x2024);
    let share_capped = winrate_vs_field(&game, &agent_capped, &baseline, games, 0x2024);
    let stderr = (fair * (1.0 - fair) / games as f64).sqrt();
    println!(
        "  field_win_share vs Rollout ({games} games, fair {fair:.3}, ~1σ≈{stderr:.4}):\n\
         \x20   full (uncapped agent) = {share_full:.4} (Δ {:+.4})   capped (capped agent) = \
         {share_capped:.4} (Δ {:+.4})   |full-capped|={:.4}  [{:.1}s]",
        share_full - fair,
        share_capped - fair,
        (share_full - share_capped).abs(),
        t1.elapsed().as_secs_f64()
    );

    // ---- D. SPEEDUP: 5p5d6f deploy data-gen, capped vs uncapped ----
    println!("\n--- D. speedup: 5p5d6f deploy data-gen throughput ---");
    let gen_episodes = env_usize("GEN_EPISODES", 64);
    let perf_hidden = env_usize("PERF_HIDDEN", 256);
    let perf_iters = env_usize("PERF_ITERS", 256);
    let _ = measure_gen(false, 6, perf_hidden, perf_iters);
    let (s_off, t_off) = measure_gen(false, gen_episodes, perf_hidden, perf_iters);
    let (s_on, t_on) = measure_gen(true, gen_episodes, perf_hidden, perf_iters);
    let sps_off = s_off as f64 / t_off;
    let sps_on = s_on as f64 / t_on;
    println!(
        "  hidden={perf_hidden} num_iters={perf_iters} threads={}  {gen_episodes} episodes each",
        rayon::current_num_threads()
    );
    println!("  uncapped: {s_off} samples in {t_off:.2}s  =>  {sps_off:.0} samples/s");
    println!("  capped:   {s_on} samples in {t_on:.2}s  =>  {sps_on:.0} samples/s");
    println!("  speedup = {:.2}x", sps_on / sps_off);
    eprintln!("[stage] total wall {:.1}s", wall.elapsed().as_secs_f64());
}
