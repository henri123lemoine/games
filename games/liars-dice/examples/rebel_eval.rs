//! Strength + perf gates for the test-time ReBeL agent.
//!
//! 1. Train a SMALL value net via `DeployTrainer` on a fixed small config, load
//!    it into a `RebelAgent`, and report its `winrate_vs_field` against the
//!    Rollout baseline (hero rotated through every seat; fair = 1/players).
//! 2. Report the average per-move ReBeL solve wall-time at 5p5d6f.
//! 3. Report the deploy data-gen throughput at 5p5d6f (samples/sec).
//!
//!     cargo run --release -p liars-dice --example rebel_eval
//!
//! Env overrides: PLAYERS DICE FACES (train config), STEPS GEN_PER NUM_ITERS
//! HIDDEN (train budget), GAMES ROLLOUTS EVAL_ITERS DEPTH (eval/deploy depth),
//! PERF_ITERS PERF_HIDDEN GEN_EPISODES (perf).

use std::time::Instant;

use game_core::{Agent, Game, Rng, Turn, winrate_vs_field};
use liars_dice::rebel::deploy_train::sample_fixed_round;
use liars_dice::rebel::{
    CfrParams, DeployCont, DeployTrainConfig, DeployTrainer, LiarsDiceAdapter, NetContinuation,
    PbsNet, RebelAgent, SelfPlayParams, generate_episode,
};
use liars_dice::{BidConditioned, LiarsDice, ProbabilisticAgent};
use rayon::prelude::*;
use solvers::Rollout;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let players = env_usize("PLAYERS", 2);
    let dice = env_usize("DICE", 2) as u8;
    let faces = env_usize("FACES", 3) as u8;
    let steps = env_usize("STEPS", 160);
    let gen_per = env_usize("GEN_PER", 48);
    let num_iters = env_usize("NUM_ITERS", 128);
    let hidden = env_usize("HIDDEN", 64);
    let games = env_usize("GAMES", 1200) as u32;
    let rollouts = env_usize("ROLLOUTS", 200) as u32;
    let eval_iters = env_usize("EVAL_ITERS", 256);
    // Deploy/solve depth for the agent gates (2, 3a) and data-gen (3b). The net is
    // trained at depth 2 but can be RE-SOLVED deeper at deploy; default 2 keeps the
    // gates backward-compatible.
    let depth = env_usize("DEPTH", 2) as u32;

    // ---- Gate 2: train a small net (reuse a cached one), then field-win-share ----
    let outdir = std::env::temp_dir().join(format!("ld_rebel_{players}p{dice}d{faces}f"));
    std::fs::create_dir_all(&outdir).unwrap();
    let net_path = outdir.join("net.bin");

    if net_path.exists() {
        println!("=== reusing cached net at {} ===", net_path.display());
    } else {
        let cfg = DeployTrainConfig {
            steps,
            warmup_steps: 20,
            num_iters,
            max_depth: 2,
            batch: 256,
            lr: 1e-3,
            gen_per_step: gen_per,
            train_gen_ratio: 16,
            burn_in: 1024,
            eval_every: steps.max(1),
            eval_iters: 128,
            eval_fit_iters: 400,
            hidden,
            n_layers: 2,
            buffer_cap: 500_000,
            seed: 0,
            log: false,
            fixed_config: Some((players, dice, faces)),
            ..DeployTrainConfig::default()
        };
        println!(
            "=== training small net on {players}p{dice}d{faces}f: hidden={hidden} steps={steps} \
             gen_per={gen_per} num_iters={num_iters} ==="
        );
        let t0 = Instant::now();
        let mut trainer = DeployTrainer::new(cfg);
        let report = trainer.run();
        trainer.net().save(&net_path).unwrap();
        println!(
            "trained in {:.1}s  samples={} train_steps={} best_gate_expl={:.4}",
            t0.elapsed().as_secs_f64(),
            report.samples_generated,
            report.train_steps,
            report.best_exploitability
        );
    }

    let agent = RebelAgent::with_config(PbsNet::load(&net_path).unwrap(), eval_iters, depth);
    // Cap round count so a rare exact-call stall can't make one game run minutes
    // of re-solves; the natural game is far shorter than this.
    let game = LiarsDice::new(players as u8, dice, faces)
        .with_max_rounds(env_usize("MAX_ROUNDS", 24) as u16);
    let baseline = Rollout::new(
        rollouts,
        ProbabilisticAgent::default_agent(),
        BidConditioned::default(),
    );
    let fair = 1.0 / players as f64;
    println!(
        "=== gate 2: RebelAgent({eval_iters} iters) vs Rollout({rollouts}) on \
         {players}p{dice}d{faces}f, {games} games (fair {fair:.3}) ==="
    );
    let t1 = Instant::now();
    let share = winrate_vs_field(&game, &agent, &baseline, games, 0x2024);
    println!(
        "field_win_share = {share:.4}  (fair {fair:.3}, Δ {:+.4})  [{:.1}s]",
        share - fair,
        t1.elapsed().as_secs_f64()
    );

    // ---- Gate 3a: per-move solve wall-time at 5p5d6f ----
    let perf_hidden = env_usize("PERF_HIDDEN", 256);
    let perf_iters = env_usize("PERF_ITERS", 1024);
    let perf_net = PbsNet::new(perf_hidden, 2, 0);
    let perf_agent = RebelAgent::with_config(perf_net, perf_iters, depth);
    let perf_moves = env_usize("PERF_MOVES", 3);
    let big = LiarsDice::new(5, 5, 6);
    let mut rng = Rng::new(99);
    let mut times = Vec::new();
    for _ in 0..2 {
        if times.len() >= perf_moves {
            break;
        }
        let mut s = big.initial_state();
        let mut moves = 0;
        while !big.is_terminal(&s) && times.len() < perf_moves && moves < 8 {
            match big.turn(&s) {
                Turn::Chance => {
                    let a = big.sample_chance_action(&s, &mut rng);
                    big.apply(&mut s, a);
                }
                Turn::Player(p) => {
                    let acts = big.legal_actions(&s);
                    if acts.len() > 1 {
                        let t = Instant::now();
                        let i = perf_agent.act(&big, &s, p, &mut rng);
                        times.push(t.elapsed().as_secs_f64());
                        moves += 1;
                        big.apply(&mut s, acts[i]);
                    } else {
                        big.apply(&mut s, acts[0]);
                    }
                }
            }
        }
    }
    let avg_ms = 1000.0 * times.iter().sum::<f64>() / times.len() as f64;
    let max_ms = 1000.0 * times.iter().cloned().fold(0.0, f64::max);
    println!(
        "=== gate 3a: 5p5d6f per-move solve (hidden={perf_hidden}, {perf_iters} iters, depth {depth}) ==="
    );
    println!(
        "avg {avg_ms:.0} ms/move  max {max_ms:.0} ms/move  over {} moves",
        times.len()
    );

    // ---- Gate 3b: deploy data-gen throughput at 5p5d6f ----
    let gen_episodes = env_usize("GEN_EPISODES", 24);
    let gen_net = PbsNet::new(perf_hidden, 2, 0);
    let sp = SelfPlayParams {
        cfr: CfrParams {
            num_iters,
            max_depth: depth,
            ..CfrParams::default()
        },
        explore_eps: 0.25,
    };
    let t2 = Instant::now();
    let total_samples: usize = (0..gen_episodes)
        .into_par_iter()
        .map(|e| {
            let mut r = Rng::new(0xD1CE ^ e as u64);
            let round = sample_fixed_round(&mut r, 5, 5, 6, 0.5);
            let cont = DeployCont::Net(NetContinuation::new(&gen_net));
            let adapter = LiarsDiceAdapter::new(
                round.players,
                round.faces,
                round.dice_left,
                round.opener,
                round.first_round,
                &cont,
            )
            .with_principled_open_cap();
            generate_episode(&adapter, sp, &gen_net, &mut r).len()
        })
        .sum();
    let gen_secs = t2.elapsed().as_secs_f64();
    println!(
        "=== gate 3b: 5p5d6f deploy data-gen \
         (hidden={perf_hidden}, num_iters={num_iters}, depth={depth}, principled_open_cap) ==="
    );
    println!(
        "{gen_episodes} episodes in {gen_secs:.2}s  =>  {:.1} episodes/s  {:.0} samples/s  \
         (parallel, {} threads)",
        gen_episodes as f64 / gen_secs,
        total_samples as f64 / gen_secs,
        rayon::current_num_threads(),
    );
}
