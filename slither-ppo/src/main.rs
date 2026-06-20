//! PPO self-play league trainer for the slither encircle bot.
//!
//! One iteration: collect a `T × N` self-play rollout (learner seat 0 in every
//! arena, opponents from the PFSP pool), run the PPO update, anneal the encircle
//! shaping as real kills appear, advance the curriculum, and periodically snapshot
//! the learner into the pool, checkpoint, and run the eval panel.
//!
//! All tch work runs on the trainer thread / one device. The env rollout is the
//! CPU-parallel part; the net forwards (learner + neural opponents) are batched.
//!
//! Usage:
//!   cargo run --release -- [iters=N] [arenas=N] [steps=N] [device=cpu|mps|cuda]
//!                          [out=DIR] [eval-every=N] [snapshot-every=N]

mod curriculum;
mod eval;
mod net;
mod obs_batch;
mod opponent;
mod ppo;
mod rollout;

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use tch::nn::OptimizerConfig;
use tch::{Device, nn};

use slither_rl::world::WorldConfig;

use curriculum::Stage;
use eval::Opp;
use net::Policy;
use opponent::{Neural, Pool};
use rollout::{Collector, LearnerActions, Transition};

struct Args {
    iters: usize,
    arenas: usize,
    steps: usize,
    device: Device,
    out: PathBuf,
    eval_every: usize,
    snapshot_every: usize,
    lr: f64,
    seed: u64,
}

fn parse_args() -> Args {
    let mut a = Args {
        iters: 200,
        arenas: 256,
        steps: 64,
        device: default_device(),
        out: PathBuf::from("runs/dev"),
        eval_every: 10,
        snapshot_every: 25,
        lr: 2.5e-4,
        seed: 1,
    };
    for arg in std::env::args().skip(1) {
        let Some((k, v)) = arg.split_once('=') else {
            continue;
        };
        match k {
            "iters" => a.iters = v.parse().unwrap(),
            "arenas" => a.arenas = v.parse().unwrap(),
            "steps" => a.steps = v.parse().unwrap(),
            "device" => {
                a.device = match v {
                    "cpu" => Device::Cpu,
                    "mps" => Device::Mps,
                    "cuda" => Device::Cuda(0),
                    _ => default_device(),
                }
            }
            "out" => a.out = PathBuf::from(v),
            "eval-every" => a.eval_every = v.parse().unwrap(),
            "snapshot-every" => a.snapshot_every = v.parse().unwrap(),
            "lr" => a.lr = v.parse().unwrap(),
            "seed" => a.seed = v.parse().unwrap(),
            _ => eprintln!("ignoring unknown arg {k}"),
        }
    }
    a
}

fn default_device() -> Device {
    if tch::utils::has_mps() {
        Device::Mps
    } else if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    }
}

/// Anneal the encircle shaping prior: hold it at full strength while the learner
/// is still finding the behavior, then decay — slowly by iteration, faster once it
/// is reliably killing on its own — so the *final* policy is learned, not scripted
/// (the blueprint's requirement). The kill term only *accelerates* an already-slow
/// schedule and is clamped, so a noisy early kill spike can't collapse the prior
/// before the curriculum has had time to work.
///
/// `kills_per_episode` is the measured self-play kill rate from the previous
/// rollout. A rate near/above 1 means the learner kills about once per life — at
/// that point the prior has done its job and can come off.
fn shaping_weight(iter: usize, kills_per_episode: f32) -> f32 {
    const HOLD: usize = 20; // full strength for the first HOLD iterations
    const SPAN: f32 = 130.0; // iterations from HOLD to zero on the slow schedule
    if iter < HOLD {
        return 1.0;
    }
    let by_iter = (1.0 - (iter - HOLD) as f32 / SPAN).clamp(0.0, 1.0);
    // Kill-driven acceleration: 0 below 0.25 kills/ep, ramping to a 0.6× pull-down
    // by 1.0 kills/ep. Never fully zeros on its own — iteration finishes the job.
    let kill_accel = 1.0 - 0.6 * ((kills_per_episode - 0.25) / 0.75).clamp(0.0, 1.0);
    (by_iter * kill_accel).clamp(0.0, 1.0)
}

fn stage_for(iter: usize) -> Stage {
    match iter {
        0..=40 => Stage::OversizedVsPrey,
        41..=90 => Stage::Mixed,
        _ => Stage::EvenSelfPlay,
    }
}

fn main() {
    let args = parse_args();
    tch::manual_seed(args.seed as i64);
    std::fs::create_dir_all(&args.out).expect("create out dir");
    let metrics_path = args.out.join("metrics.jsonl");
    let mut metrics = std::fs::File::create(&metrics_path).expect("metrics file");

    println!(
        "slither-ppo  device={:?}  arenas={}  steps={}  iters={}  out={}",
        args.device,
        args.arenas,
        args.steps,
        args.iters,
        args.out.display()
    );
    println!(
        "  obs grid {:?} + {} scalars   action {} turn buckets x boost",
        net::CHANNELS as usize,
        net::SCALARS,
        net::TURN_BUCKETS
    );

    let vs = nn::VarStore::new(args.device);
    let policy = Policy::new(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, args.lr).expect("optimizer");

    let cfg = WorldConfig {
        worms: 6,
        pellet_target: 600,
        ..WorldConfig::default()
    };

    let mut pool = Pool::seeded(args.seed ^ 0x5151, true, true);

    let ppo_cfg = ppo::PpoConfig {
        gamma: 0.995,
        lambda: 0.95,
        clip: 0.2,
        value_coef: 0.5,
        entropy_coef: 0.01,
        max_grad_norm: 0.5,
        epochs: 4,
        minibatches: 8,
        steps: args.steps,
    };

    let mut stage = stage_for(0);
    let mut collector = Collector::new(args.arenas, cfg, stage, &mut pool, args.seed ^ 0xC0DE);
    let mut buf: Vec<Transition> = Vec::with_capacity(args.steps * args.arenas);
    let mut kills_per_episode_last = 0.0f32;

    for iter in 0..args.iters {
        let t0 = Instant::now();
        let new_stage = stage_for(iter);
        if new_stage != stage {
            stage = new_stage;
            collector.set_stage(stage);
            println!("  [iter {iter}] curriculum -> {stage:?}");
        }

        // Shaping uses the previous iteration's measured kill rate; iter 0 starts
        // at full strength (no kills observed yet).
        let shaping = shaping_weight(iter, kills_per_episode_last);

        buf.clear();
        collector.outcomes.clear();
        collector.learner_kills = 0;
        let mut rollout_reward = 0.0f64;
        let mut rollout_deaths = 0u32;

        for _ in 0..args.steps {
            let inputs = collector.step_inputs(&pool);

            // Learner forward (one batched GPU call).
            let (lg, ls) = obs_batch::pack(&inputs.learner_obs, args.device);
            let (turn, boost, logp, value) = policy.act(&lg, &ls);
            let learner = LearnerActions {
                turn: tensor_i64(&turn),
                boost: tensor_i64(&boost),
                log_prob: tensor_f32(&logp),
                value: tensor_f32(&value),
            };

            // Neural opponents: one forward per snapshot bucket in play.
            let mut neural_actions: Vec<(Vec<i64>, Vec<i64>)> = Vec::new();
            for b in &inputs.neural {
                let (g, s) = obs_batch::pack(&b.obs, args.device);
                let np = &pool.neural[b.neural_idx].policy;
                let (t, bo, _lp, _v) = np.act(&g, &s);
                neural_actions.push((tensor_i64(&t), tensor_i64(&bo)));
            }

            let before = buf.len();
            collector.step(
                &mut pool,
                &inputs,
                &learner,
                &neural_actions,
                &mut buf,
                shaping,
            );
            for tr in &buf[before..] {
                rollout_reward += tr.reward as f64;
                if tr.done {
                    rollout_deaths += 1;
                }
            }
        }

        // Bootstrap values for GAE tail.
        let boot_obs = collector.bootstrap_obs();
        let (bg, bs) = obs_batch::pack(&boot_obs, args.device);
        let boot = tensor_f32(&policy.value(&bg, &bs));

        let stats = ppo::update(&policy, &mut opt, args.device, &buf, &boot, &ppo_cfg);

        // Update PFSP win-rates from the rollout's finished episodes.
        for &(pi, won) in &collector.outcomes {
            pool.entries[pi].update(won);
        }

        // Self-play kill rate per finished episode this rollout — the honest
        // signal that drives the shaping anneal next iteration.
        let episodes = rollout_deaths.max(1);
        kills_per_episode_last = collector.learner_kills as f32 / episodes as f32;

        let dt = t0.elapsed().as_secs_f64();
        let reward_per_step = rollout_reward / (args.steps as f64 * args.arenas as f64);

        // Periodic eval (also yields the kill rate that drives shaping).
        let mut ev_rand = None;
        let mut ev_heur = None;
        if iter % args.eval_every == 0 || iter == args.iters - 1 {
            let r = eval::evaluate(
                &policy,
                args.device,
                128,
                400,
                Opp::Random,
                args.seed ^ 0x11,
            );
            let h = eval::evaluate(
                &policy,
                args.device,
                128,
                400,
                Opp::Heuristic,
                args.seed ^ 0x22,
            );
            println!(
                "  [iter {iter:>4}] r/step {reward_per_step:+.4}  ent {:.3}  kl {:.4}  clip {:.2}  ev {:+.2}  vloss {:.3}  shaping {shaping:.2}  | vs RAND win {:.2} k {:.2}  | vs HEUR win {:.2} k {:.2}  opp-k {:.2}  {dt:.1}s",
                stats.entropy,
                stats.approx_kl,
                stats.clip_frac,
                stats.explained_variance,
                stats.value_loss,
                r.winrate,
                r.learner_kills_per_game,
                h.winrate,
                h.learner_kills_per_game,
                h.opp_kills_per_game,
            );
            ev_rand = Some(r);
            ev_heur = Some(h);
        } else {
            println!(
                "  [iter {iter:>4}] r/step {reward_per_step:+.4}  ent {:.3}  kl {:.4}  clip {:.2}  ev {:+.2}  vloss {:.3}  shaping {shaping:.2}  episodes {episodes}  {dt:.1}s",
                stats.entropy,
                stats.approx_kl,
                stats.clip_frac,
                stats.explained_variance,
                stats.value_loss,
            );
        }

        write_metrics(
            &mut metrics,
            iter,
            reward_per_step,
            &stats,
            shaping,
            stage,
            ev_rand.as_ref(),
            ev_heur.as_ref(),
        );

        // Snapshot the learner into the pool, and checkpoint.
        if iter > 0 && iter % args.snapshot_every == 0 {
            snapshot(&vs, &policy, &mut pool, iter as u64, args.device);
            println!(
                "  [iter {iter}] snapshot -> pool ({} entries)",
                pool.entries.len()
            );
        }
        if iter % args.eval_every == 0 || iter == args.iters - 1 {
            let ckpt = args.out.join(format!("ckpt_{iter:05}.ot"));
            net::save(&vs, &ckpt).expect("save checkpoint");
            write_sidecar(&args.out, iter, &args);
        }
    }

    let final_ckpt = args.out.join("ckpt_final.ot");
    net::save(&vs, &final_ckpt).expect("save final");
    write_sidecar(&args.out, args.iters, &args);
    println!("done. final checkpoint {}", final_ckpt.display());
}

fn snapshot(train_vs: &nn::VarStore, _policy: &Policy, pool: &mut Pool, iter: u64, device: Device) {
    let mut vs = nn::VarStore::new(device);
    let policy = Policy::new(&vs.root());
    vs.copy(train_vs).expect("copy weights to snapshot");
    vs.freeze();
    pool.add_snapshot(Neural { vs, policy, iter });
}

fn tensor_i64(t: &tch::Tensor) -> Vec<i64> {
    Vec::<i64>::try_from(t).unwrap()
}
fn tensor_f32(t: &tch::Tensor) -> Vec<f32> {
    Vec::<f32>::try_from(t).unwrap()
}

#[allow(clippy::too_many_arguments)]
fn write_metrics(
    f: &mut std::fs::File,
    iter: usize,
    reward_per_step: f64,
    s: &ppo::UpdateStats,
    shaping: f32,
    stage: Stage,
    rand: Option<&eval::EvalResult>,
    heur: Option<&eval::EvalResult>,
) {
    let mut o = serde_json::Map::new();
    o.insert("iter".into(), iter.into());
    o.insert("reward_per_step".into(), reward_per_step.into());
    o.insert("policy_loss".into(), (s.policy_loss as f64).into());
    o.insert("value_loss".into(), (s.value_loss as f64).into());
    o.insert("entropy".into(), (s.entropy as f64).into());
    o.insert("approx_kl".into(), (s.approx_kl as f64).into());
    o.insert("clip_frac".into(), (s.clip_frac as f64).into());
    o.insert(
        "explained_variance".into(),
        (s.explained_variance as f64).into(),
    );
    o.insert("shaping".into(), (shaping as f64).into());
    o.insert("stage".into(), format!("{stage:?}").into());
    if let Some(r) = rand {
        o.insert("rand_winrate".into(), (r.winrate as f64).into());
        o.insert(
            "rand_kills".into(),
            (r.learner_kills_per_game as f64).into(),
        );
    }
    if let Some(h) = heur {
        o.insert("heur_winrate".into(), (h.winrate as f64).into());
        o.insert(
            "heur_kills".into(),
            (h.learner_kills_per_game as f64).into(),
        );
        o.insert(
            "heur_opp_kills".into(),
            (h.opp_kills_per_game as f64).into(),
        );
        o.insert("heur_lifespan".into(), (h.mean_lifespan as f64).into());
        o.insert("heur_final_len".into(), (h.mean_final_len as f64).into());
    }
    let line = serde_json::to_string(&serde_json::Value::Object(o)).unwrap();
    writeln!(f, "{line}").ok();
    f.flush().ok();
}

fn write_sidecar(out: &std::path::Path, iter: usize, args: &Args) {
    let sidecar = serde_json::json!({
        "iter": iter,
        "arenas": args.arenas,
        "steps": args.steps,
        "lr": args.lr,
        "channels": net::CHANNELS,
        "grid": net::GRID,
        "scalars": net::SCALARS,
        "turn_buckets": net::TURN_BUCKETS,
        "arch": "3conv_cnn[32,64,64]_trunk256",
    });
    let path = out.join("model.json");
    std::fs::write(path, serde_json::to_string_pretty(&sidecar).unwrap()).ok();
}
