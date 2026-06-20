use tch::nn::{self, RNN};
use tch::{Kind, Tensor};

use crate::env::{NUM_ACTIONS, OBS_DIM};

pub const GRU_HIDDEN: i64 = 128;

/// Recurrent actor-critic for PPO: an obs encoder → GRU trunk (partial-
/// observability memory) → policy-logit head + scalar value head. Reuses the
/// DFP net's trunk shape so the BPTT-over-window machinery carries over.
pub struct PpoNet {
    obs_fc: nn::Linear,
    gru: nn::GRU,
    pi_hidden: nn::Linear,
    pi: nn::Linear,
    v_hidden: nn::Linear,
    v: nn::Linear,
}

impl PpoNet {
    pub fn new(vs: &nn::Path) -> PpoNet {
        let obs_fc = nn::linear(vs / "obs_fc", OBS_DIM as i64, 128, Default::default());
        let gru = nn::gru(vs / "gru", 128, GRU_HIDDEN, Default::default());
        let pi_hidden = nn::linear(vs / "pi_hidden", GRU_HIDDEN, 256, Default::default());
        let pi = nn::linear(vs / "pi", 256, NUM_ACTIONS as i64, Default::default());
        let v_hidden = nn::linear(vs / "v_hidden", GRU_HIDDEN, 256, Default::default());
        let v = nn::linear(vs / "v", 256, 1, Default::default());
        PpoNet {
            obs_fc,
            gru,
            pi_hidden,
            pi,
            v_hidden,
            v,
        }
    }

    fn heads(&self, h: &Tensor) -> (Tensor, Tensor) {
        let logits = h.apply(&self.pi_hidden).relu().apply(&self.pi);
        let value = h
            .apply(&self.v_hidden)
            .relu()
            .apply(&self.v)
            .squeeze_dim(-1);
        (logits, value)
    }

    /// Single-step forward (acting). obs: [B, OBS_DIM], state: [B, H].
    /// Returns (logits [B, A], value [B], new_state [B, H]).
    pub fn step(&self, obs: &Tensor, state: &Tensor) -> (Tensor, Tensor, Tensor) {
        let o = obs.apply(&self.obs_fc).relu().unsqueeze(1); // [B,1,128]
        let h0 = nn::GRUState(state.unsqueeze(0));
        let (out, new_h) = self.gru.seq_init(&o, &h0);
        let h = out.squeeze_dim(1); // [B,H]
        let new_state = new_h.0.squeeze_dim(0);
        let (logits, value) = self.heads(&h);
        (logits, value, new_state)
    }

    /// Sequence forward for BPTT. obs: [B, T, OBS_DIM], state: [B, H].
    /// Returns (logits [B, T, A], value [B, T]).
    pub fn forward_seq(&self, obs: &Tensor, state: &Tensor) -> (Tensor, Tensor) {
        let (b, t) = (obs.size()[0], obs.size()[1]);
        let o = obs.apply(&self.obs_fc).relu(); // [B,T,128]
        let h0 = nn::GRUState(state.unsqueeze(0));
        let (gru_out, _) = self.gru.seq_init(&o, &h0); // [B,T,H]
        let h = gru_out.reshape([b * t, GRU_HIDDEN]);
        let (logits, value) = self.heads(&h);
        (logits.view([b, t, NUM_ACTIONS as i64]), value.view([b, t]))
    }

    pub fn zero_state(b: i64, device: tch::Device) -> Tensor {
        Tensor::zeros([b, GRU_HIDDEN], (Kind::Float, device))
    }
}
