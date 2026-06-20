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
//!                          [out=DIR] [eval-every=N] [eval-games=N] [snapshot-every=N]
//!                          [lr=F] [seed=N] [init=CKPT.ot]
//!
//! `init` warm-starts from a checkpoint and skips the early curriculum (a
//! fine-tune leg). The best net by the combined kill-aware score vs the heuristic
//! is kept automatically as `<out>/best.ot`.

mod curriculum;
mod eval;
mod export;
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
    eval_games: usize,
    snapshot_every: usize,
    lr: f64,
    seed: u64,
    /// Warm-start the learner from this checkpoint instead of random init. Lets a
    /// fine-tune leg build on a strong net (e.g. retune on changed dynamics).
    init: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut a = Args {
        iters: 200,
        arenas: 256,
        steps: 64,
        device: default_device(),
        out: PathBuf::from("runs/dev"),
        eval_every: 10,
        eval_games: 128,
        snapshot_every: 25,
        lr: 2.5e-4,
        seed: 1,
        init: None,
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
            "eval-games" => a.eval_games = v.parse().unwrap(),
            "snapshot-every" => a.snapshot_every = v.parse().unwrap(),
            "lr" => a.lr = v.parse().unwrap(),
            "seed" => a.seed = v.parse().unwrap(),
            "init" => a.init = Some(PathBuf::from(v)),
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

/// Curriculum stage for an iteration. A fine-tune (`warm`) skips the early
/// oversized/mixed ramp — the warm-started net already plays even-size, and the
/// prey stages would only un-teach it — and trains straight in even self-play
/// (with the close-encounter practice geometry).
fn stage_for(iter: usize, warm: bool) -> Stage {
    if warm {
        return Stage::EvenSelfPlay;
    }
    match iter {
        0..=40 => Stage::OversizedVsPrey,
        41..=90 => Stage::Mixed,
        _ => Stage::EvenSelfPlay,
    }
}

/// Linear LR decay from the configured peak down to a small floor over the run —
/// the cleanrl default. Late training stabilizes (smaller steps) so the policy
/// settles at its plateau instead of drifting off it into the regression.
fn lr_for(iter: usize, iters: usize, base_lr: f64) -> f64 {
    let frac = iter as f64 / iters.max(1) as f64;
    let floor = 0.1; // never below 10% of base, so it keeps learning to the end
    base_lr * (1.0 - (1.0 - floor) * frac)
}

/// Entropy floor controller: nudge the entropy coefficient up when the policy
/// has collapsed below `ENTROPY_FLOOR` and back down toward `ENTROPY_COEF_BASE`
/// when it has room to explore. The long run collapsed to ent≈0.65 and drifted
/// into reckless aggression; holding a floor keeps the policy from over-sharpening
/// into the brittle self-play mode that forgot how to beat the heuristic.
const ENTROPY_FLOOR: f32 = 0.9;
const ENTROPY_COEF_BASE: f64 = 0.01;
const ENTROPY_COEF_MAX: f64 = 0.05;

fn adapt_entropy_coef(coef: f64, measured_entropy: f32) -> f64 {
    let next = if measured_entropy < ENTROPY_FLOOR {
        coef * 1.5
    } else {
        coef * 0.9
    };
    next.clamp(ENTROPY_COEF_BASE, ENTROPY_COEF_MAX)
}

/// Keep-best combined score = winrate + KILL_WIN_WEIGHT * kill_winrate. Weight
/// chosen so a 0.10 gain in kill-win rate (a real encircling improvement) can
/// outweigh ~0.05 of overall winrate, pulling the kept net toward decisive
/// predation — but WINRATE_SLACK caps how much overall winrate it may give up, so
/// the kept net can't regress the stable plateau for a flashier-killer one.
const KILL_WIN_WEIGHT: f32 = 0.5;
const WINRATE_SLACK: f32 = 0.04;

fn main() {
    // Subcommand dispatch: `export` / `verify-export` for the browser net, else
    // the default is a training run (`[k=v]` knobs).
    let raw: Vec<String> = std::env::args().skip(1).collect();
    match raw.first().map(String::as_str) {
        Some("export") => return export::export(&raw[1..]),
        Some("verify-export") => return export::verify_export(&raw[1..]),
        Some("compare") => return compare(&raw[1..]),
        _ => {}
    }
    train();
}

/// `compare net=A.ot [net2=B.ot] [games=N] [steps=N] [seed=S]` — load one or two
/// checkpoints and run the eval panel (greedy, vs the heuristic, even footing) on
/// the same seeds, printing winrate + kill/death rates side by side. The honest
/// apples-to-apples for "is the new net actually better" — both nets see the same
/// arenas, so the difference isn't a lucky draw.
fn compare(args: &[String]) {
    let device = default_device();
    let get = |k: &str| args.iter().find_map(|a| a.strip_prefix(&format!("{k}=")));
    let games: usize = get("games").and_then(|v| v.parse().ok()).unwrap_or(512);
    let steps: usize = get("steps").and_then(|v| v.parse().ok()).unwrap_or(400);
    let seed: u64 = get("seed").and_then(|v| v.parse().ok()).unwrap_or(777);

    let eval_one = |path: &str| -> eval::EvalResult {
        let mut vs = nn::VarStore::new(device);
        let policy = Policy::new(&vs.root());
        vs.load(path).unwrap_or_else(|e| panic!("load {path}: {e}"));
        eval::evaluate(&policy, device, games, steps, Opp::Heuristic, seed)
    };

    let print = |label: &str, r: &eval::EvalResult| {
        println!(
            "{label:>40}  win {:.3}  kill-win {:.3}  learner-kills/g {:.3}  deaths-to-opp/g {:.3}  lifespan {:.0}  final-len {:.1}",
            r.winrate,
            r.kill_winrate,
            r.learner_kills_per_game,
            r.opp_kills_per_game,
            r.mean_lifespan,
            r.mean_final_len
        );
    };

    println!("compare vs HEURISTIC  device={device:?}  games={games}  steps={steps}  seed={seed}");
    if let Some(a) = get("net") {
        print(a, &eval_one(a));
    }
    if let Some(b) = get("net2") {
        print(b, &eval_one(b));
    }
}

fn train() {
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

    let mut vs = nn::VarStore::new(args.device);
    let policy = Policy::new(&vs.root());
    if let Some(init) = &args.init {
        vs.load(init)
            .unwrap_or_else(|e| panic!("warm-start load {}: {e}", init.display()));
        println!("  warm-started from {}", init.display());
    }
    let mut opt = nn::Adam::default().build(&vs, args.lr).expect("optimizer");

    let cfg = WorldConfig {
        worms: 6,
        ..WorldConfig::default()
    };

    let mut pool = Pool::seeded(args.seed ^ 0x5151, true, true);

    let mut ppo_cfg = ppo::PpoConfig {
        gamma: 0.995,
        lambda: 0.95,
        clip: 0.2,
        value_coef: 0.5,
        entropy_coef: ENTROPY_COEF_BASE,
        max_grad_norm: 0.5,
        epochs: 4,
        minibatches: 8,
        steps: args.steps,
    };

    let warm = args.init.is_some();
    let mut stage = stage_for(0, warm);
    let mut collector = Collector::new(args.arenas, cfg, stage, &mut pool, args.seed ^ 0xC0DE);
    let mut buf: Vec<Transition> = Vec::with_capacity(args.steps * args.arenas);
    let mut kills_per_episode_last = 0.0f32;

    // Keep-best by a COMBINED score vs the heuristic: overall winrate plus a
    // weighted kill-win rate, so the kept net is a decisive *encircler*, not just
    // a survivor — but a winrate floor (a flashy killer that tanks overall winrate
    // can't win the gate) keeps it from regressing the stable plateau.
    let best_path = args.out.join("best.ot");
    let mut best_score = f32::NEG_INFINITY;
    let mut best_winrate = 0.0f32;
    let mut best_iter = 0usize;

    for iter in 0..args.iters {
        let t0 = Instant::now();
        let new_stage = stage_for(iter, warm);
        if new_stage != stage {
            stage = new_stage;
            collector.set_stage(stage);
            println!("  [iter {iter}] curriculum -> {stage:?}");
        }

        opt.set_lr(lr_for(iter, args.iters, args.lr));

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

        // Adapt the entropy coefficient toward the floor for the *next* update.
        ppo_cfg.entropy_coef = adapt_entropy_coef(ppo_cfg.entropy_coef, stats.entropy);

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
                args.eval_games,
                400,
                Opp::Random,
                args.seed ^ 0x11,
            );
            let h = eval::evaluate(
                &policy,
                args.device,
                args.eval_games,
                400,
                Opp::Heuristic,
                args.seed ^ 0x22,
            );
            println!(
                "  [iter {iter:>4}] r/step {reward_per_step:+.4}  ent {:.3}  kl {:.4}  clip {:.2}  ev {:+.2}  vloss {:.3}  shaping {shaping:.2}  | vs RAND win {:.2} k {:.2}  | vs HEUR win {:.2} kill-win {:.2} k {:.2}  opp-k {:.2}  {dt:.1}s",
                stats.entropy,
                stats.approx_kl,
                stats.clip_frac,
                stats.explained_variance,
                stats.value_loss,
                r.winrate,
                r.learner_kills_per_game,
                h.winrate,
                h.kill_winrate,
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

        // Eval-gated keep-best on the combined score (winrate + weighted
        // kill-win), with a winrate floor so a flashier killer can't ship at the
        // cost of overall winrate. The deployable net is then always the peak by
        // this metric, automatically — and a later regression can't overwrite it.
        if let Some(h) = ev_heur.as_ref() {
            let score = h.winrate + KILL_WIN_WEIGHT * h.kill_winrate;
            let floor_ok = h.winrate >= best_winrate - WINRATE_SLACK;
            if score > best_score && floor_ok {
                best_score = score;
                best_winrate = best_winrate.max(h.winrate);
                best_iter = iter;
                net::save(&vs, &best_path).expect("save best");
                write_best_sidecar(&args.out, iter, h, &args);
                println!(
                    "  [iter {iter}] new best  win {:.3}  kill-win {:.3}  score {score:.3} -> best.ot",
                    h.winrate, h.kill_winrate
                );
            }
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
    println!(
        "done. final checkpoint {}\nbest combined score {:.3} (winrate floor {:.3}) at iter {} -> {}",
        final_ckpt.display(),
        best_score,
        best_winrate,
        best_iter,
        best_path.display()
    );
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
        o.insert("heur_kill_winrate".into(), (h.kill_winrate as f64).into());
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

/// Sidecar for `best.ot`: the same arch dims plus the eval that earned it, so a
/// later export knows which iteration / winrate / kill-rate the deployed net
/// came from.
fn write_best_sidecar(out: &std::path::Path, iter: usize, h: &eval::EvalResult, args: &Args) {
    let sidecar = serde_json::json!({
        "iter": iter,
        "heur_winrate": h.winrate,
        "heur_kill_winrate": h.kill_winrate,
        "heur_kills_per_game": h.learner_kills_per_game,
        "arenas": args.arenas,
        "steps": args.steps,
        "lr": args.lr,
        "channels": net::CHANNELS,
        "grid": net::GRID,
        "scalars": net::SCALARS,
        "turn_buckets": net::TURN_BUCKETS,
        "arch": "3conv_cnn[32,64,64]_trunk256",
    });
    let path = out.join("best.json");
    std::fs::write(path, serde_json::to_string_pretty(&sidecar).unwrap()).ok();
}
