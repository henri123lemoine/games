use tch::nn::{self, RNN};
use tch::{Kind, Tensor};

use crate::env::{MEAS_DIM, NUM_ACTIONS, OBS_DIM};

pub const OFFSETS: [usize; 6] = [1, 2, 4, 8, 16, 32];
pub const NUM_OFFSETS: usize = OFFSETS.len();
pub const PRED_PER_ACTION: i64 = (NUM_OFFSETS * MEAS_DIM) as i64;
pub const GRU_HIDDEN: i64 = 128;

pub struct DfpNet {
    obs_fc: nn::Linear,
    meas_fc: nn::Linear,
    goal_fc: nn::Linear,
    gru: nn::GRU,
    expectation: nn::Sequential,
    advantage: nn::Sequential,
}

fn mlp(p: &nn::Path, sizes: &[i64]) -> nn::Sequential {
    let mut seq = nn::seq();
    for i in 0..sizes.len() - 1 {
        seq = seq.add(nn::linear(
            p / format!("l{i}"),
            sizes[i],
            sizes[i + 1],
            Default::default(),
        ));
        if i < sizes.len() - 2 {
            seq = seq.add_fn(|x| x.relu());
        }
    }
    seq
}

impl DfpNet {
    pub fn new(vs: &nn::Path) -> DfpNet {
        let obs_fc = nn::linear(vs / "obs_fc", OBS_DIM as i64, 128, Default::default());
        let meas_fc = nn::linear(vs / "meas_fc", MEAS_DIM as i64, 64, Default::default());
        let goal_fc = nn::linear(
            vs / "goal_fc",
            (MEAS_DIM * NUM_OFFSETS) as i64,
            64,
            Default::default(),
        );
        let gru = nn::gru(vs / "gru", 128, GRU_HIDDEN, Default::default());

        let joint = GRU_HIDDEN + 64 + 64;
        let expectation = mlp(&(vs / "exp"), &[joint, 256, PRED_PER_ACTION]);
        let advantage = mlp(
            &(vs / "adv"),
            &[joint, 256, PRED_PER_ACTION * NUM_ACTIONS as i64],
        );

        DfpNet {
            obs_fc,
            meas_fc,
            goal_fc,
            gru,
            expectation,
            advantage,
        }
    }

    /// Single-step forward with explicit GRU state.
    /// obs: [B, OBS_DIM], meas: [B, MEAS_DIM], goal: [B, MEAS_DIM*NUM_OFFSETS],
    /// state: [B, GRU_HIDDEN]. Returns (pred [B, NUM_ACTIONS, PRED_PER_ACTION], new_state).
    pub fn step(
        &self,
        obs: &Tensor,
        meas: &Tensor,
        goal: &Tensor,
        state: &Tensor,
    ) -> (Tensor, Tensor) {
        let o = obs.apply(&self.obs_fc).relu();
        let gru_in = o.unsqueeze(1); // [B, 1, 128]
        let h0 = nn::GRUState(state.unsqueeze(0)); // [1, B, H]
        let (out, new_h) = self.gru.seq_init(&gru_in, &h0);
        let h = out.squeeze_dim(1); // [B, H]
        let new_state = new_h.0.squeeze_dim(0); // [B, H]

        let m = meas.apply(&self.meas_fc).relu();
        let g = goal.apply(&self.goal_fc).relu();
        let joint = Tensor::cat(&[h, m, g], 1);

        let exp = joint.apply(&self.expectation); // [B, PRED_PER_ACTION]
        let adv = joint
            .apply(&self.advantage)
            .view([-1, NUM_ACTIONS as i64, PRED_PER_ACTION]);
        let adv = &adv - adv.mean_dim(1, true, Kind::Float);
        let pred = exp.unsqueeze(1) + adv; // [B, NUM_ACTIONS, PRED_PER_ACTION]
        (pred, new_state)
    }

    pub fn zero_state(b: i64, device: tch::Device) -> Tensor {
        Tensor::zeros([b, GRU_HIDDEN], (Kind::Float, device))
    }
}
