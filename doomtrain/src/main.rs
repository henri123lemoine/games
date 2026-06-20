mod dfp;
mod env;
mod export;
mod ffi;
mod net;
mod ppo;
mod ppo_net;

use std::path::PathBuf;

use tch::nn::OptimizerConfig;
use tch::{nn, Device};

use env::DoomEnv;
use net::DfpNet;

fn arg(name: &str, default: &str) -> String {
    let key = format!("--{name}=");
    std::env::args()
        .find(|a| a.starts_with(&key))
        .map(|a| a[key.len()..].to_string())
        .unwrap_or_else(|| default.to_string())
}

fn epsilon_at(iter: usize, iters: usize, start: f64, end: f64) -> f64 {
    if iters <= 1 {
        return end;
    }
    let frac = iter as f64 / (iters - 1) as f64;
    start + (end - start) * frac
}

fn main() {
    let cmd = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "smoke".to_string());

    let iwad = arg("iwad", "../web/app/public/doom/doom1.wad");
    let arena = arg("arena", "../doomrl/assets/flatarena.wad");
    let steps: usize = arg("steps", "256").parse().unwrap();
    let lr: f64 = arg("lr", "1e-3").parse().unwrap();

    let device = if std::env::var("DOOMTRAIN_MPS").is_ok() {
        Device::Mps
    } else {
        Device::Cpu
    };

    // PPO is its own VarStore/net; dispatch before the DFP net is built.
    if cmd == "ppo" {
        train_ppo(&iwad, &arena, device, steps, lr);
        return;
    }
    if cmd == "ppo-eval" {
        ppo_eval_cmd(&iwad, &arena, device, steps);
        return;
    }
    if cmd == "ppo-export" {
        ppo_export_cmd(device);
        return;
    }

    let mut vs = nn::VarStore::new(device);
    let net = DfpNet::new(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, lr).unwrap();

    match cmd.as_str() {
        "smoke" => smoke(&env_new(&iwad, &arena), &net, &mut opt, device, steps),
        "train" => train(&iwad, &arena, &net, &mut opt, &mut vs, device, steps),
        "eval" => {
            let ckpt = arg("net", "");
            if !ckpt.is_empty() {
                export::load_checkpoint(&mut vs, &PathBuf::from(&ckpt));
            }
            let episodes: usize = arg("episodes", "10").parse().unwrap();
            let env = env_new(&iwad, &arena);
            run_eval(&env, &net, device, episodes, steps);
        }
        "export" => {
            let ckpt = arg("net", "");
            if !ckpt.is_empty() {
                export::load_checkpoint(&mut vs, &PathBuf::from(&ckpt));
            }
            let out = PathBuf::from(arg("out", "doomdfp.bin"));
            export::export(&vs, &out);
            let ok = export::verify_roundtrip(&vs, &out);
            println!(
                "export: wrote {} ({} tensors) round_trip={}",
                out.display(),
                vs.variables().len(),
                if ok { "OK" } else { "FAILED" }
            );
            assert!(ok, "export round-trip mismatch");
        }
        other => {
            eprintln!(
                "unknown command: {other} \
                 (try: ppo | ppo-eval | ppo-export | smoke | train | eval | export)"
            );
            std::process::exit(2);
        }
    }
}

/// Curriculum stage: spawn distance + opponent skill at a given progress 0..1.
fn curriculum(progress: f64) -> (f32, f32) {
    // Hold CLOSE for the first 60% so PPO refines the BC-cloned fragging at
    // point-blank before navigation re-enters; gentle ramp after. Skill ramps to
    // 0.7 (still beatable).
    let p = progress as f32;
    let spawn = if p < 0.6 {
        256.0
    } else {
        256.0 + (1100.0 - 256.0) * ((p - 0.6) / 0.4)
    };
    let skill = 0.7 * p;
    (spawn, skill)
}

#[allow(clippy::too_many_arguments)]
fn train_ppo(iwad: &str, arena: &str, device: Device, steps: usize, lr: f64) {
    use ppo_net::PpoNet;

    let iters: usize = arg("iters", "300").parse().unwrap();
    let epochs: usize = arg("epochs", "4").parse().unwrap();
    let eval_every: usize = arg("eval-every", "10").parse().unwrap();
    let eval_episodes: usize = arg("eval-episodes", "8").parse().unwrap();
    let self_play_at: f64 = arg("self-play-at", "0.7").parse().unwrap();
    let save: String = arg("save", "doomppo.ot");
    let best_path: String = arg("best", "doomppo_best.ot");

    let vs = nn::VarStore::new(device);
    let net = PpoNet::new(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, lr).unwrap();
    let n_params: i64 = vs
        .trainable_variables()
        .iter()
        .map(|t| t.numel() as i64)
        .sum();
    println!(
        "doomtrain ppo: iters={iters} steps={steps} epochs={epochs} actions={} params={n_params}",
        env::NUM_ACTIONS
    );

    let env = env_new(iwad, arena);

    // Behaviour-cloning warmstart: imitate the scripted hunter so the policy
    // starts with competent aim. Undirected RL exploration cannot land the 7+
    // accurate shots a kill needs, so without this the policy never frags and
    // never reinforces the +frag reward — the cold-start both DFP and raw PPO hit.
    let bc_iters: usize = arg("bc-iters", "150").parse().unwrap();
    if bc_iters > 0 {
        let bc_loss = ppo::bc_pretrain(
            &env,
            &net,
            &mut opt,
            device,
            bc_iters,
            steps.min(512),
            256.0,
        );
        let (nf, _nd, bf, share) = ppo_eval_avg(&env, &net, device, 6, steps);
        println!(
            "bc warmstart: {bc_iters} iters, final ce={bc_loss:.4} -> eval net[frags={nf}] \
             bot[frags={bf}] net_frag_share={share:.3}"
        );
    }

    // frozen self-play snapshot, used once progress passes self_play_at.
    let mut opp_vs = nn::VarStore::new(device);
    let opp_net = PpoNet::new(&opp_vs.root());
    opp_vs.copy(&vs).unwrap();
    opp_vs.freeze();

    let mut best = -1.0f64;

    for it in 0..iters {
        let progress = it as f64 / (iters.max(1) - 1).max(1) as f64;
        let (spawn, skill) = curriculum(progress);

        let mut bot = env::BeatableBot::for_skill(skill, 0x9E37 ^ it as u64);
        let mut foe = if progress >= self_play_at {
            ppo::Foe::Snapshot(&opp_net)
        } else {
            ppo::Foe::Bot(&mut bot)
        };

        let roll = ppo::collect(&env, &net, &mut foe, device, steps, spawn);
        let (pi, vl, ent) = ppo::update(&net, &mut opt, device, &roll, epochs);

        if progress >= self_play_at && (it + 1) % 20 == 0 {
            opp_vs.copy(&vs).unwrap();
        }

        let avg_r: f32 = roll.steps.iter().map(|s| s.reward).sum::<f32>() / roll.steps.len() as f32;
        println!(
            "iter {it}: prog={progress:.2} spawn={spawn:.0} skill={skill:.2} \
             frags={} deaths={} avg_r={avg_r:.4} pi={pi:.4} v={vl:.4} ent={ent:.3}",
            roll.frags, roll.deaths
        );

        if eval_every > 0 && (it + 1) % eval_every == 0 {
            let (nf, nd, bf, share) = ppo_eval_avg(&env, &net, device, eval_episodes, steps);
            let mut tag = "";
            if share > best {
                best = share;
                vs.save(&best_path).unwrap();
                tag = " <- new best";
            }
            println!(
                "  eval @iter{it} ({eval_episodes} eps, beatable bot): net[frags={nf} deaths={nd}] \
                 bot[frags={bf}] net_frag_share={share:.3} best={best:.3}{tag}"
            );
        }
    }

    vs.save(&save).unwrap();
    println!("saved ppo checkpoint: {save} (best net_frag_share={best:.3} -> {best_path})");
}

/// Eval the PPO net over N episodes vs a fixed mid-skill beatable bot.
fn ppo_eval_avg(
    env: &DoomEnv,
    net: &ppo_net::PpoNet,
    device: Device,
    episodes: usize,
    steps: usize,
) -> (i64, i64, i64, f64) {
    let (mut nf, mut nd, mut bf) = (0i64, 0i64, 0i64);
    for e in 0..episodes {
        let mut bot = env::BeatableBot::for_skill(0.5, 0xEEE ^ e as u64);
        let (a, b, c) = ppo::eval(env, net, &mut bot, device, steps, 384.0);
        nf += a;
        nd += b;
        bf += c;
    }
    let share = nf as f64 / (nf + bf).max(1) as f64;
    (nf, nd, bf, share)
}

fn ppo_eval_cmd(iwad: &str, arena: &str, device: Device, steps: usize) {
    let ckpt = arg("net", "");
    let mut vs = nn::VarStore::new(device);
    let net = ppo_net::PpoNet::new(&vs.root());
    if !ckpt.is_empty() {
        export::load_checkpoint(&mut vs, &PathBuf::from(&ckpt));
    }
    let episodes: usize = arg("episodes", "10").parse().unwrap();
    let env = env_new(iwad, arena);
    let (nf, nd, bf, share) = ppo_eval_avg(&env, &net, device, episodes, steps);
    println!(
        "PPO EVAL ({episodes} eps vs beatable bot): net[frags={nf} deaths={nd}] bot[frags={bf}] \
         net_frag_share={share:.3}"
    );
}

fn ppo_export_cmd(device: Device) {
    let ckpt = arg("net", "");
    let mut vs = nn::VarStore::new(device);
    let _net = ppo_net::PpoNet::new(&vs.root());
    if !ckpt.is_empty() {
        export::load_checkpoint(&mut vs, &PathBuf::from(&ckpt));
    }
    let out = PathBuf::from(arg("out", "doomppo.bin"));
    export::export(&vs, &out);
    let ok = export::verify_roundtrip(&vs, &out);
    println!(
        "ppo-export: wrote {} ({} tensors) round_trip={}",
        out.display(),
        vs.variables().len(),
        if ok { "OK" } else { "FAILED" }
    );
    assert!(ok, "export round-trip mismatch");
}

fn env_new(iwad: &str, arena: &str) -> DoomEnv {
    DoomEnv::new(iwad, Some(arena))
}

fn n_params(vs: &nn::VarStore) -> i64 {
    vs.trainable_variables()
        .iter()
        .map(|t| t.numel() as i64)
        .sum()
}

fn smoke(env: &DoomEnv, net: &DfpNet, opt: &mut nn::Optimizer, device: Device, steps: usize) {
    println!("doomtrain smoke (BPTT): num_players={}", env.num_players());

    let episodes: usize = arg("episodes", "3").parse().unwrap();
    let epsilon: f64 = arg("epsilon", "0.3").parse().unwrap();
    let mut replay = dfp::ReplayBuffer::new(256);

    for ep in 0..episodes {
        let (r0, r1, frags) = dfp::collect_episode(env, net, device, steps, epsilon);
        assert_eq!(r0.transitions.len(), steps, "rollout length mismatch");

        for c in dfp::chunk_rollout(&r0) {
            replay.push(c);
        }
        for c in dfp::chunk_rollout(&r1) {
            replay.push(c);
        }
        assert!(!replay.is_empty(), "no BPTT chunks produced");

        let mut last_loss = 0.0;
        for _ in 0..8 {
            let batch = replay.sample(8.min(replay.len()));
            last_loss = dfp::train_chunk(net, opt, device, &batch);
        }

        let obs_std = {
            let flat: Vec<f32> = r0.transitions.iter().flat_map(|t| t.obs).collect();
            tch::Tensor::from_slice(&flat).std(false).double_value(&[])
        };
        let mut act_hist = [0usize; env::NUM_ACTIONS];
        for tr in r0.transitions.iter().chain(r1.transitions.iter()) {
            act_hist[tr.action] += 1;
        }
        let distinct = act_hist.iter().filter(|&&c| c > 0).count();

        println!(
            "ep{ep}: frags={frags} chunks_in_replay={} bptt_loss={last_loss:.5} \
             obs_std={obs_std:.3} distinct_actions={distinct}/{}",
            replay.len(),
            env::NUM_ACTIONS
        );
        assert!(last_loss.is_finite(), "loss went non-finite");
        assert!(obs_std > 0.0, "observations constant — env not varying");
    }
    println!("smoke OK: BPTT over rollout chunks, replay buffer, train steps ran");
}

#[allow(clippy::too_many_arguments)]
fn train(
    iwad: &str,
    arena: &str,
    net: &DfpNet,
    opt: &mut nn::Optimizer,
    vs: &mut nn::VarStore,
    device: Device,
    steps: usize,
) {
    let iters: usize = arg("iters", "20").parse().unwrap();
    let updates: usize = arg("updates", "16").parse().unwrap();
    let batch: usize = arg("batch", "16").parse().unwrap();
    let eps_start: f64 = arg("eps-start", "0.9").parse().unwrap();
    let eps_end: f64 = arg("eps-end", "0.1").parse().unwrap();
    let eval_every: usize = arg("eval-every", "5").parse().unwrap();
    let eval_episodes: usize = arg("eval-episodes", "6").parse().unwrap();
    let self_play: bool = std::env::args().any(|a| a == "--self-play");
    let vs_hunter: bool = std::env::args().any(|a| a == "--vs-hunter");
    let refresh_every: usize = arg("refresh-every", "10").parse().unwrap();
    let save: String = arg("save", "doomdfp.ot");
    let best_path: String = arg("best", "doomdfp_best.ot");

    let mode = if vs_hunter {
        "vs-hunter"
    } else if self_play {
        "self-play"
    } else {
        "self-mirror"
    };
    println!(
        "doomtrain train: iters={iters} steps={steps} updates={updates} batch={batch} \
         mode={mode} eval_episodes={eval_episodes} params={}",
        n_params(vs)
    );
    let env = env_new(iwad, arena);
    let mut replay = dfp::ReplayBuffer::new(4096);

    // Frozen self-play opponent snapshot (seat 1), refreshed from the live net.
    let mut opp_vs = nn::VarStore::new(device);
    let opp_net = DfpNet::new(&opp_vs.root());
    opp_vs.copy(vs).expect("init opponent snapshot");
    opp_vs.freeze();

    let mut best_share = -1.0f64;

    for it in 0..iters {
        let epsilon = epsilon_at(it, iters, eps_start, eps_end);
        let opponent = if vs_hunter {
            dfp::Opponent::Hunter
        } else if self_play {
            dfp::Opponent::Snapshot(&opp_net)
        } else {
            dfp::Opponent::SelfMirror
        };
        let (r0, r1, frags) = dfp::collect_episode_vs(&env, net, opponent, device, steps, epsilon);
        for c in dfp::chunk_rollout(&r0) {
            replay.push(c);
        }
        // In vs-hunter mode seat 1 is the scripted hunter, not the net — don't
        // train on its rollout. Otherwise both seats are net-driven (valid data).
        if !vs_hunter {
            for c in dfp::chunk_rollout(&r1) {
                replay.push(c);
            }
        }

        let mut loss = 0.0;
        for _ in 0..updates {
            let b = replay.sample(batch.min(replay.len()));
            loss = dfp::train_chunk(net, opt, device, &b);
        }

        if self_play && refresh_every > 0 && (it + 1) % refresh_every == 0 {
            opp_vs.copy(vs).expect("refresh opponent snapshot");
            println!("  refreshed self-play opponent @iter{it}");
        }

        println!(
            "iter {it}: eps={epsilon:.3} frags={frags} replay={} loss={loss:.5}",
            replay.len()
        );

        if eval_every > 0 && (it + 1) % eval_every == 0 {
            let (nf, nd, hf, hd, share) = eval_avg(&env, net, device, eval_episodes, steps);
            let mut tag = "";
            if share > best_share {
                best_share = share;
                vs.save(&best_path).expect("save best checkpoint");
                tag = " <- new best, saved";
            }
            println!(
                "  eval @iter{it} ({eval_episodes} eps): net[frags={nf} deaths={nd}] \
                 hunter[frags={hf} deaths={hd}] net_frag_share={share:.3} best={best_share:.3}{tag}"
            );
        }
    }

    vs.save(&save).expect("save checkpoint");
    println!("saved final checkpoint: {save} (best net_frag_share={best_share:.3} -> {best_path})");
}

/// Average eval over N episodes vs the scripted hunter; returns totals and the
/// net's frag share.
fn eval_avg(
    env: &DoomEnv,
    net: &DfpNet,
    device: Device,
    episodes: usize,
    steps: usize,
) -> (i64, i64, i64, i64, f64) {
    let (mut nf, mut nd, mut hf, mut hd) = (0i64, 0i64, 0i64, 0i64);
    for _ in 0..episodes {
        let (a, b, c, d) = dfp::eval_episode(env, net, device, steps);
        nf += a;
        nd += b;
        hf += c;
        hd += d;
    }
    let share = nf as f64 / (nf + hf).max(1) as f64;
    (nf, nd, hf, hd, share)
}

fn run_eval(env: &DoomEnv, net: &DfpNet, device: Device, episodes: usize, steps: usize) {
    let mut net_frags = 0i64;
    let mut net_deaths = 0i64;
    let mut hunter_frags = 0i64;
    let mut hunter_deaths = 0i64;
    for ep in 0..episodes {
        let (nf, nd, hf, hd) = dfp::eval_episode(env, net, device, steps);
        net_frags += nf;
        net_deaths += nd;
        hunter_frags += hf;
        hunter_deaths += hd;
        println!("ep{ep}: net[frags={nf} deaths={nd}] hunter[frags={hf} deaths={hd}]");
    }
    println!(
        "EVAL over {episodes} eps: net frags={net_frags} deaths={net_deaths} | \
         hunter frags={hunter_frags} deaths={hunter_deaths} | net_frag_share={:.2}",
        net_frags as f64 / (net_frags + hunter_frags).max(1) as f64
    );
}
