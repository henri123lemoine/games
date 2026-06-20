use tch::{Kind, Tensor};

use crate::env::{
    decode_action, observation, shaped_reward, BeatableBot, DoomEnv, NUM_ACTIONS, OBS_DIM,
};
use crate::ppo_net::PpoNet;

pub const GAMMA: f64 = 0.99;
pub const LAMBDA: f64 = 0.95;
pub const CLIP: f64 = 0.2;
pub const ENT_COEF: f64 = 0.01;
pub const VF_COEF: f64 = 0.5;
pub const WINDOW: usize = 32;

/// One learner step of experience (seat 0). value/logp recorded at action time.
pub struct Step {
    pub obs: [f32; OBS_DIM],
    pub action: i64,
    pub logp: f32,
    pub value: f32,
    pub reward: f32,
    pub done: f32, // 1.0 if the seat died this step (episode boundary for GAE)
}

pub struct Rollout {
    pub steps: Vec<Step>,
    pub adv: Vec<f32>,
    pub ret: Vec<f32>,
    pub frags: i64,
    pub deaths: i64,
}

/// Seat 1 driver during collection.
pub enum Foe<'a> {
    Bot(&'a mut BeatableBot),
    Snapshot(&'a PpoNet),
}

fn sample_action(logits: &Tensor) -> (i64, f32) {
    // logits: [1, A]
    let probs = logits.softmax(-1, Kind::Float);
    let idx = probs.multinomial(1, true).int64_value(&[0, 0]);
    let logp = logits.log_softmax(-1, Kind::Float).double_value(&[0, idx]) as f32;
    (idx, logp)
}

/// Collect one episode for the learner (seat 0) vs `foe` (seat 1), with the
/// shaped reward. Spawns the two players `spawn_dist` apart each episode
/// (curriculum). Returns the rollout with GAE advantages + returns filled.
pub fn collect(
    env: &DoomEnv,
    net: &PpoNet,
    foe: &mut Foe,
    device: tch::Device,
    steps: usize,
    spawn_dist: f32,
) -> Rollout {
    env.reset();
    if spawn_dist > 0.0 {
        env.spawn_near(spawn_dist);
    }

    let mut steps_v: Vec<Step> = Vec::with_capacity(steps);
    let mut state = PpoNet::zero_state(1, device);
    let mut foe_state = PpoNet::zero_state(1, device);

    let mut prev0 = env.player_state(0);
    let mut prev1 = env.player_state(1);

    for _ in 0..steps {
        let st0 = env.player_state(0);
        let obs = observation(&st0);
        let obs_t = Tensor::from_slice(&obs).unsqueeze(0).to_device(device);

        let (action, logp, value, new_state) = tch::no_grad(|| {
            let (logits, value, ns) = net.step(&obs_t, &state);
            let (a, lp) = sample_action(&logits);
            (a, lp, value.double_value(&[0]) as f32, ns)
        });
        state = new_state;

        // seat 1
        let st1 = env.player_state(1);
        let a1 = match foe {
            Foe::Bot(bot) => bot.act(&st1),
            Foe::Snapshot(opp) => {
                let o1 = observation(&st1);
                let o1t = Tensor::from_slice(&o1).unsqueeze(0).to_device(device);
                let (idx, ns) = tch::no_grad(|| {
                    let (logits, _v, ns) = opp.step(&o1t, &foe_state);
                    let i = logits.argmax(-1, false).int64_value(&[0]);
                    (i, ns)
                });
                foe_state = ns;
                decode_action(idx as usize)
            }
        };

        env.step(decode_action(action as usize), a1);

        let cur0 = env.player_state(0);
        let cur1 = env.player_state(1);
        let reward = shaped_reward(&prev0, &cur0, &prev1, &cur1);
        let done = (prev0.alive != 0 && cur0.alive == 0) as i32 as f32;

        steps_v.push(Step {
            obs,
            action,
            logp,
            value,
            reward,
            done,
        });

        prev0 = cur0;
        prev1 = cur1;
    }

    // bootstrap value of the final state
    let last = env.player_state(0);
    let last_obs = observation(&last);
    let last_v = tch::no_grad(|| {
        let t = Tensor::from_slice(&last_obs).unsqueeze(0).to_device(device);
        let (_l, v, _s) = net.step(&t, &state);
        v.double_value(&[0]) as f32
    });

    let (adv, ret) = gae(&steps_v, last_v);
    let frags = last.frags as i64;
    let deaths = last.deaths as i64;
    Rollout {
        steps: steps_v,
        adv,
        ret,
        frags,
        deaths,
    }
}

/// Generalized Advantage Estimation. `done` marks a death (value resets to 0
/// past it, since the seat respawns fresh — a soft episode boundary).
fn gae(steps: &[Step], last_v: f32) -> (Vec<f32>, Vec<f32>) {
    let n = steps.len();
    let mut adv = vec![0f32; n];
    let mut last_gae = 0f32;
    let mut next_v = last_v;
    for t in (0..n).rev() {
        let nonterminal = 1.0 - steps[t].done;
        let delta = steps[t].reward + (GAMMA as f32) * next_v * nonterminal - steps[t].value;
        last_gae = delta + (GAMMA as f32) * (LAMBDA as f32) * nonterminal * last_gae;
        adv[t] = last_gae;
        next_v = steps[t].value;
    }
    let ret: Vec<f32> = (0..n).map(|t| adv[t] + steps[t].value).collect();
    (adv, ret)
}

/// PPO update over the rollout: split into BPTT windows of `WINDOW`, run K
/// epochs of minibatch (= window) updates with the clipped surrogate + value
/// loss + entropy bonus, BPTT through each window. Returns (pi_loss, v_loss,
/// entropy) from the last minibatch.
pub fn update(
    net: &PpoNet,
    opt: &mut tch::nn::Optimizer,
    device: tch::Device,
    roll: &Rollout,
    epochs: usize,
) -> (f64, f64, f64) {
    let n = roll.steps.len();
    let n_win = n / WINDOW;
    if n_win == 0 {
        return (0.0, 0.0, 0.0);
    }

    // normalize advantages over the whole rollout
    let adv_t = Tensor::from_slice(&roll.adv[..n_win * WINDOW]);
    let mean = adv_t.mean(Kind::Float).double_value(&[]) as f32;
    let std = adv_t.std(true).double_value(&[]).max(1e-6) as f32;

    let mut last = (0.0, 0.0, 0.0);
    let w = WINDOW as i64;

    for _ in 0..epochs {
        for win in 0..n_win {
            let base = win * WINDOW;
            let mut obs_v = Vec::with_capacity(WINDOW * OBS_DIM);
            let mut act_v = Vec::with_capacity(WINDOW);
            let mut oldlp_v = Vec::with_capacity(WINDOW);
            let mut adv_v = Vec::with_capacity(WINDOW);
            let mut ret_v = Vec::with_capacity(WINDOW);
            for i in 0..WINDOW {
                let s = &roll.steps[base + i];
                obs_v.extend_from_slice(&s.obs);
                act_v.push(s.action);
                oldlp_v.push(s.logp);
                adv_v.push((roll.adv[base + i] - mean) / std);
                ret_v.push(roll.ret[base + i]);
            }

            let obs = Tensor::from_slice(&obs_v)
                .view([1, w, OBS_DIM as i64])
                .to_device(device);
            let act = Tensor::from_slice(&act_v).view([w]).to_device(device);
            let oldlp = Tensor::from_slice(&oldlp_v).view([w]).to_device(device);
            let adv = Tensor::from_slice(&adv_v).view([w]).to_device(device);
            let ret = Tensor::from_slice(&ret_v).view([w]).to_device(device);

            let state = PpoNet::zero_state(1, device);
            let (logits, value) = net.forward_seq(&obs, &state); // [1,T,A], [1,T]
            let logits = logits.view([w, NUM_ACTIONS as i64]);
            let value = value.view([w]);

            let logp_all = logits.log_softmax(-1, Kind::Float);
            let logp = logp_all
                .gather(1, &act.unsqueeze(-1), false)
                .squeeze_dim(-1);
            let ratio = (&logp - &oldlp).exp();
            let surr1 = &ratio * &adv;
            let surr2 = ratio.clamp(1.0 - CLIP, 1.0 + CLIP) * &adv;
            let pi_loss = -surr1.minimum(&surr2).mean(Kind::Float);

            let v_loss = (&value - &ret).square().mean(Kind::Float);

            let probs = logits.softmax(-1, Kind::Float);
            let entropy = -(&probs * &logp_all)
                .sum_dim_intlist(-1, false, Kind::Float)
                .mean(Kind::Float);

            let loss = &pi_loss + VF_COEF * &v_loss - ENT_COEF * &entropy;
            opt.backward_step(&loss);

            last = (
                pi_loss.double_value(&[]),
                v_loss.double_value(&[]),
                entropy.double_value(&[]),
            );
        }
    }
    last
}

/// Greedy eval of the learner (seat 0) vs a BeatableBot (seat 1) for one episode
/// spawned `spawn_dist` apart. Returns (net_frags, net_deaths, bot_frags).
pub fn eval(
    env: &DoomEnv,
    net: &PpoNet,
    bot: &mut BeatableBot,
    device: tch::Device,
    steps: usize,
    spawn_dist: f32,
) -> (i64, i64, i64) {
    env.reset();
    if spawn_dist > 0.0 {
        env.spawn_near(spawn_dist);
    }
    let mut state = PpoNet::zero_state(1, device);
    for _ in 0..steps {
        let st0 = env.player_state(0);
        let obs = observation(&st0);
        let obs_t = Tensor::from_slice(&obs).unsqueeze(0).to_device(device);
        let (a0, ns) = tch::no_grad(|| {
            let (logits, _v, ns) = net.step(&obs_t, &state);
            (logits.argmax(-1, false).int64_value(&[0]), ns)
        });
        state = ns;
        let st1 = env.player_state(1);
        let a1 = bot.act(&st1);
        env.step(decode_action(a0 as usize), a1);
    }
    let s0 = env.player_state(0);
    let s1 = env.player_state(1);
    (s0.frags as i64, s0.deaths as i64, s1.frags as i64)
}
