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
    // weight: health a little, ammo slightly negative (spending ammo to fight is
    // fine), frags a lot, opp_damage a lot — opp_damage is the dense signal that
    // lets the policy learn to shoot before it ever lands a (sparse) frag.
    [0.5, -0.1, 1.0, 1.0]
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

/// Who drives seat 1 during collection.
pub enum Opponent<'a> {
    /// Seat 1 uses the same live net (self-mirror).
    SelfMirror,
    /// Seat 1 uses a frozen snapshot (self-play).
    Snapshot(&'a DfpNet),
    /// Seat 1 uses the scripted hunter — forces firefights so seat 0's rollout
    /// actually contains opp_damage / frag signal (the cold-start cure).
    Hunter,
}

#[allow(clippy::too_many_arguments)]
pub fn collect_episode(
    env: &DoomEnv,
    net: &DfpNet,
    device: tch::Device,
    steps: usize,
    epsilon: f64,
) -> (Rollout, Rollout, i64) {
    collect_episode_vs(env, net, Opponent::SelfMirror, device, steps, epsilon)
}

/// Collect an episode. Seat 0 always acts with the live `net` (epsilon-greedy on
/// the DFP goal). Seat 1 is driven per `opponent`. Both seats' rollouts are valid
/// DFP regression data (only seat 0's is used for training in vs-hunter mode,
/// since the hunter isn't the net — the caller chooses).
#[allow(clippy::too_many_arguments)]
pub fn collect_episode_vs(
    env: &DoomEnv,
    net: &DfpNet,
    opponent: Opponent,
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
        let mut hunter_act = None;

        for seat in 0..2 {
            let st = env.player_state(seat as i32);
            let obs = observation(&st);
            let meas = measurements(&st);
            obs_now[seat] = obs;
            meas_now[seat] = meas;

            // Seat 1 may be the scripted hunter — its real (continuous) action
            // drives the env; the stored index is the nearest discrete one.
            if seat == 1 {
                if let Opponent::Hunter = opponent {
                    let ha = crate::env::scripted_hunter(&st);
                    actions[1] = crate::env::encode_action(&ha);
                    hunter_act = Some(ha);
                    continue;
                }
            }

            let obs_t = Tensor::from_slice(&obs).unsqueeze(0).to_device(device);
            let meas_t = Tensor::from_slice(&meas).unsqueeze(0).to_device(device);
            let goal_t = goal.unsqueeze(0);

            let actor = match (seat, &opponent) {
                (1, Opponent::Snapshot(o)) => *o,
                _ => net,
            };
            let (pred, new_state) =
                tch::no_grad(|| actor.step(&obs_t, &meas_t, &goal_t, &state[seat]));
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

        let a1 = hunter_act.unwrap_or_else(|| decode_action(actions[1]));
        env.step(decode_action(actions[0]), a1);
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

/// Greedy DFP policy (seat 0, GRU state carried) vs the scripted hunter (seat 1)
/// for one episode. Returns (net_frags, net_deaths, hunter_frags, hunter_deaths).
pub fn eval_episode(
    env: &DoomEnv,
    net: &DfpNet,
    device: tch::Device,
    steps: usize,
) -> (i64, i64, i64, i64) {
    env.reset();
    let goal = Tensor::from_slice(&goal_full()).to_device(device);
    let mut state = DfpNet::zero_state(1, device);

    for _ in 0..steps {
        let st0 = env.player_state(0);
        let obs = observation(&st0);
        let meas = measurements(&st0);
        let obs_t = Tensor::from_slice(&obs).unsqueeze(0).to_device(device);
        let meas_t = Tensor::from_slice(&meas).unsqueeze(0).to_device(device);
        let goal_t = goal.unsqueeze(0);

        let (pred, new_state) = tch::no_grad(|| net.step(&obs_t, &meas_t, &goal_t, &state));
        state = new_state;
        let a0 = greedy_action(&pred, &goal);

        let st1 = env.player_state(1);
        let a1 = crate::env::scripted_hunter(&st1);

        env.step(decode_action(a0), a1);
    }

    let s0 = env.player_state(0);
    let s1 = env.player_state(1);
    (
        s0.frags as i64,
        s0.deaths as i64,
        s1.frags as i64,
        s1.deaths as i64,
    )
}

/// A fixed-length window of one seat's experience, ready for BPTT. Stored flat
/// (length `WINDOW`) so a minibatch of chunks stacks into [B, WINDOW, *].
pub struct Chunk {
    pub obs: Vec<f32>,  // WINDOW * OBS_DIM
    pub meas: Vec<f32>, // WINDOW * MEAS_DIM
    pub act: Vec<i64>,  // WINDOW
    pub tgt: Vec<f32>,  // WINDOW * PRED_PER_ACTION
}

pub const WINDOW: usize = 32;

/// Slice a rollout into non-overlapping BPTT windows of length `WINDOW`.
/// Each window starts the GRU from zero state (standard truncated BPTT — the
/// episode is short enough that this is a fine approximation and keeps chunks
/// independent for off-policy replay).
pub fn chunk_rollout(roll: &Rollout) -> Vec<Chunk> {
    let n = roll.transitions.len();
    let stream_len = roll.meas_stream.len();
    let mut chunks = Vec::new();
    let mut start = 0;
    while start + WINDOW <= n {
        let mut c = Chunk {
            obs: Vec::with_capacity(WINDOW * crate::env::OBS_DIM),
            meas: Vec::with_capacity(WINDOW * MEAS_DIM),
            act: Vec::with_capacity(WINDOW),
            tgt: Vec::with_capacity(WINDOW * PRED_PER_ACTION as usize),
        };
        for t in start..start + WINDOW {
            c.obs.extend_from_slice(&roll.transitions[t].obs);
            c.meas.extend_from_slice(&roll.transitions[t].meas);
            c.act.push(roll.transitions[t].action as i64);
            let cur = roll.meas_stream[t];
            for &off in OFFSETS.iter() {
                let idx = (t + off).min(stream_len - 1);
                let future = roll.meas_stream[idx];
                for m in 0..MEAS_DIM {
                    c.tgt.push(future[m] - cur[m]);
                }
            }
        }
        chunks.push(c);
        start += WINDOW;
    }
    chunks
}

/// Capped FIFO replay buffer of BPTT chunks for off-policy DFP updates.
pub struct ReplayBuffer {
    chunks: Vec<Chunk>,
    capacity: usize,
    next: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> ReplayBuffer {
        ReplayBuffer {
            chunks: Vec::with_capacity(capacity),
            capacity,
            next: 0,
        }
    }

    pub fn push(&mut self, c: Chunk) {
        if self.chunks.len() < self.capacity {
            self.chunks.push(c);
        } else {
            self.chunks[self.next] = c;
            self.next = (self.next + 1) % self.capacity;
        }
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    pub fn sample(&self, batch: usize) -> Vec<&Chunk> {
        let mut rng = rand::thread_rng();
        (0..batch)
            .map(|_| &self.chunks[rng.gen_range(0..self.chunks.len())])
            .collect()
    }
}

/// One BPTT update over a minibatch of chunks. Stacks the chunks into
/// [B, WINDOW, *], runs the GRU over the whole window (gradients through all
/// WINDOW steps), and regresses the taken-action future predictions onto the
/// DFP targets. Returns the scalar loss.
pub fn train_chunk(
    net: &DfpNet,
    opt: &mut tch::nn::Optimizer,
    device: tch::Device,
    batch: &[&Chunk],
) -> f64 {
    let b = batch.len() as i64;
    let w = WINDOW as i64;

    let mut obs_v = Vec::new();
    let mut meas_v = Vec::new();
    let mut act_v = Vec::new();
    let mut tgt_v = Vec::new();
    for c in batch {
        obs_v.extend_from_slice(&c.obs);
        meas_v.extend_from_slice(&c.meas);
        act_v.extend_from_slice(&c.act);
        tgt_v.extend_from_slice(&c.tgt);
    }

    let obs = Tensor::from_slice(&obs_v)
        .view([b, w, crate::env::OBS_DIM as i64])
        .to_device(device);
    let meas = Tensor::from_slice(&meas_v)
        .view([b, w, MEAS_DIM as i64])
        .to_device(device);
    let act = Tensor::from_slice(&act_v).view([b, w]).to_device(device);
    let tgt = Tensor::from_slice(&tgt_v)
        .view([b, w, PRED_PER_ACTION])
        .to_device(device);

    let goal = Tensor::from_slice(&goal_full())
        .view([1, 1, PRED_PER_ACTION])
        .expand([b, w, PRED_PER_ACTION], false)
        .to_device(device);
    let state = DfpNet::zero_state(b, device);

    let (pred, _) = net.forward_seq(&obs, &meas, &goal, &state); // [B, W, A, P]
    let act_idx = act
        .view([b, w, 1, 1])
        .expand([b, w, 1, PRED_PER_ACTION], false);
    let taken = pred.gather(2, &act_idx, false).squeeze_dim(2); // [B, W, P]

    let loss = taken.mse_loss(&tgt, tch::Reduction::Mean);
    opt.backward_step(&loss);
    loss.double_value(&[])
}
