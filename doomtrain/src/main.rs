mod dfp;
mod env;
mod ffi;
mod net;

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

fn main() {
    let cmd = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "smoke".to_string());

    let iwad = arg("iwad", "../web/app/public/doom/doom1.wad");
    let arena = arg("arena", "../doomrl/assets/flatarena.wad");
    let episodes: usize = arg("episodes", "3").parse().unwrap();
    let steps: usize = arg("steps", "256").parse().unwrap();
    let lr: f64 = arg("lr", "1e-3").parse().unwrap();
    let epsilon: f64 = arg("epsilon", "0.25").parse().unwrap();

    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let net = DfpNet::new(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, lr).unwrap();

    match cmd.as_str() {
        "smoke" => {
            println!(
                "doomtrain smoke: iwad={iwad} arena={arena} episodes={episodes} steps={steps}"
            );
            let env = DoomEnv::new(&iwad, Some(&arena));
            println!("engine up: num_players={}", env.num_players());

            let n_params: i64 = vs
                .trainable_variables()
                .iter()
                .map(|t| t.numel() as i64)
                .sum();
            println!("net params: {n_params}");

            for ep in 0..episodes {
                let (r0, r1, frags) = dfp::collect_episode(&env, &net, device, steps, epsilon);
                assert_eq!(r0.transitions.len(), steps, "rollout length mismatch");

                let (o0, m0, a0, t0) = dfp::build_targets(&r0, device);
                let (o1, m1, a1, t1) = dfp::build_targets(&r1, device);

                let obs = tch::Tensor::cat(&[o0, o1], 0);
                let meas = tch::Tensor::cat(&[m0, m1], 0);
                let act = tch::Tensor::cat(&[a0, a1], 0);
                let tgt = tch::Tensor::cat(&[t0, t1], 0);

                // shape sanity
                let n = obs.size()[0];
                assert_eq!(obs.size(), vec![n, env::OBS_DIM as i64]);
                assert_eq!(tgt.size(), vec![n, net::PRED_PER_ACTION]);

                let mut last_loss = 0.0;
                for _ in 0..8 {
                    last_loss = dfp::train_step(&net, &mut opt, device, &obs, &meas, &act, &tgt);
                }

                let tgt_mean = tgt.mean(tch::Kind::Float).double_value(&[]);
                let tgt_absmax = tgt.abs().max().double_value(&[]);

                // obs/action sanity: spread of obs and the action histogram
                let obs_std = obs.std(false).double_value(&[]);
                let mut act_hist = [0usize; env::NUM_ACTIONS];
                for tr in r0.transitions.iter().chain(r1.transitions.iter()) {
                    act_hist[tr.action] += 1;
                }
                let distinct_actions = act_hist.iter().filter(|&&c| c > 0).count();

                println!(
                    "ep{ep}: frags(both seats, end)={frags} steps={steps} batch_n={n} \
                     loss={last_loss:.5} tgt_mean={tgt_mean:.4} tgt_absmax={tgt_absmax:.3} \
                     obs_std={obs_std:.3} distinct_actions={distinct_actions}/{}",
                    env::NUM_ACTIONS
                );

                assert!(last_loss.is_finite(), "loss went non-finite");
                assert!(obs_std > 0.0, "observations are constant — env not varying");
            }
            println!("smoke OK: compiled, env stepped, DFP targets built, train steps ran");
        }
        other => {
            eprintln!("unknown command: {other} (try: smoke)");
            std::process::exit(2);
        }
    }
}
