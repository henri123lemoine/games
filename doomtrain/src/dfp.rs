use rand::Rng;
use tch::Tensor;

use crate::env::{decode_action, measurements, observation, DoomEnv, MEAS_DIM, NUM_ACTIONS};
use crate::net::{DfpNet, NUM_OFFSETS, OFFSETS, PRED_PER_ACTION};

pub struct Transition {
    pub obs: [f32; crate::env::OBS_DIM],
    pub meas: [f32; MEAS_DIM],
    pub action: usize,
}

/// One seat's rollout: a sequence of transitions plus the measurement stream so
/// future-offset targets can be sliced.
pub struct Rollout {
    pub transitions: Vec<Transition>,
    pub meas_stream: Vec<[f32; MEAS_DIM]>,
}

pub fn goal_vector() -> [f32; MEAS_DIM] {
    // weight: health a little, ammo a little, frags a lot — the DFP "goal".
    [0.5, 0.5, 1.0]
}

/// Tile the per-measurement goal across all offsets (later offsets weighted up).
pub fn goal_full() -> Vec<f32> {
    let g = goal_vector();
    let mut v = Vec::with_capacity(MEAS_DIM * NUM_OFFSETS);
    for oi in 0..NUM_OFFSETS {
        let w = (oi as f32 + 1.0) / NUM_OFFSETS as f32;
        for &gm in g.iter() {
            v.push(gm * w);
        }
    }
    v
}

/// Greedy action: argmax over actions of goal · predicted future deltas.
pub fn greedy_action(pred: &Tensor, goal_full: &Tensor) -> usize {
    // pred: [1, NUM_ACTIONS, PRED_PER_ACTION], goal_full: [PRED_PER_ACTION]
    let scores = pred
        .squeeze_dim(0)
        .matmul(&goal_full.unsqueeze(1))
        .squeeze();
    scores.argmax(0, false).int64_value(&[]) as usize
}

#[allow(clippy::too_many_arguments)]
pub fn collect_episode(
    env: &DoomEnv,
    net: &DfpNet,
    device: tch::Device,
    steps: usize,
    epsilon: f64,
) -> (Rollout, Rollout, i64) {
    env.reset();
    let goal = Tensor::from_slice(&goal_full()).to_device(device);

    let mut roll = [
        Rollout {
            transitions: Vec::new(),
            meas_stream: Vec::new(),
        },
        Rollout {
            transitions: Vec::new(),
            meas_stream: Vec::new(),
        },
    ];
    let mut state = [DfpNet::zero_state(1, device), DfpNet::zero_state(1, device)];
    let mut rng = rand::thread_rng();
    let mut total_frags = 0i64;

    for _ in 0..steps {
        let mut actions = [0usize; 2];
        let mut obs_now = [[0f32; crate::env::OBS_DIM]; 2];
        let mut meas_now = [[0f32; MEAS_DIM]; 2];

        for seat in 0..2 {
            let st = env.player_state(seat as i32);
            let obs = observation(&st);
            let meas = measurements(&st);
            obs_now[seat] = obs;
            meas_now[seat] = meas;

            let obs_t = Tensor::from_slice(&obs).unsqueeze(0).to_device(device);
            let meas_t = Tensor::from_slice(&meas).unsqueeze(0).to_device(device);
            let goal_t = goal.unsqueeze(0);

            let (pred, new_state) =
                tch::no_grad(|| net.step(&obs_t, &meas_t, &goal_t, &state[seat]));
            state[seat] = new_state;

            let a = if rng.gen::<f64>() < epsilon {
                rng.gen_range(0..NUM_ACTIONS)
            } else {
                greedy_action(&pred, &goal)
            };
            actions[seat] = a;
        }

        for seat in 0..2 {
            roll[seat].meas_stream.push(meas_now[seat]);
            roll[seat].transitions.push(Transition {
                obs: obs_now[seat],
                meas: meas_now[seat],
                action: actions[seat],
            });
        }

        env.step(decode_action(actions[0]), decode_action(actions[1]));
    }

    // append a final measurement so the last offsets have a target tail
    for (seat, r) in roll.iter_mut().enumerate() {
        let st = env.player_state(seat as i32);
        r.meas_stream.push(measurements(&st));
        total_frags += st.frags as i64;
    }

    let [r0, r1] = roll;
    (r0, r1, total_frags)
}

/// Build DFP training tensors from a rollout. Target for offset o at step t is
/// (meas[t+o] - meas[t]); clamped to the available horizon.
pub fn build_targets(roll: &Rollout, device: tch::Device) -> (Tensor, Tensor, Tensor, Tensor) {
    let n = roll.transitions.len();
    let stream_len = roll.meas_stream.len();

    let mut obs_v = Vec::with_capacity(n * crate::env::OBS_DIM);
    let mut meas_v = Vec::with_capacity(n * MEAS_DIM);
    let mut act_v = Vec::with_capacity(n);
    let mut tgt_v = Vec::with_capacity(n * PRED_PER_ACTION as usize);

    for t in 0..n {
        obs_v.extend_from_slice(&roll.transitions[t].obs);
        meas_v.extend_from_slice(&roll.transitions[t].meas);
        act_v.push(roll.transitions[t].action as i64);

        let cur = roll.meas_stream[t];
        for &off in OFFSETS.iter() {
            let idx = (t + off).min(stream_len - 1);
            let future = roll.meas_stream[idx];
            for m in 0..MEAS_DIM {
                tgt_v.push(future[m] - cur[m]);
            }
        }
    }

    let obs = Tensor::from_slice(&obs_v)
        .view([n as i64, crate::env::OBS_DIM as i64])
        .to_device(device);
    let meas = Tensor::from_slice(&meas_v)
        .view([n as i64, MEAS_DIM as i64])
        .to_device(device);
    let act = Tensor::from_slice(&act_v).to_device(device);
    let tgt = Tensor::from_slice(&tgt_v)
        .view([n as i64, PRED_PER_ACTION])
        .to_device(device);
    (obs, meas, act, tgt)
}

/// One DFP training pass over a batch of steps (no BPTT across the GRU here —
/// the GRU is run fresh-stateless per step, which is the simplest correct
/// smoke; full BPTT is a later refinement). Returns the scalar loss.
pub fn train_step(
    net: &DfpNet,
    opt: &mut tch::nn::Optimizer,
    device: tch::Device,
    obs: &Tensor,
    meas: &Tensor,
    act: &Tensor,
    tgt: &Tensor,
) -> f64 {
    let n = obs.size()[0];
    let goal = Tensor::from_slice(&goal_full())
        .unsqueeze(0)
        .expand([n, PRED_PER_ACTION], false)
        .to_device(device);
    let state = DfpNet::zero_state(n, device);

    let (pred, _) = net.step(obs, meas, &goal, &state); // [N, A, P]
    let act_idx = act.view([n, 1, 1]).expand([n, 1, PRED_PER_ACTION], false);
    let taken = pred.gather(1, &act_idx, false).squeeze_dim(1); // [N, P]

    let loss = taken.mse_loss(tgt, tch::Reduction::Mean);
    opt.backward_step(&loss);
    loss.double_value(&[])
}
