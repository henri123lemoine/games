//! Replay buffer and the fp32 training step: policy cross-entropy against
//! the search's visit distribution (full softmax over all 82 logits) plus
//! value MSE, AdamW on MPS.
//!
//! Every minibatch draw applies a uniformly random dihedral symmetry to both
//! the planes and the policy target — go's 8-fold board symmetry is an 8×
//! data multiplier the chess harness never had.

use std::collections::VecDeque;

use game_core::Rng;
use go::encode::{d8, d8_policy};
use tch::nn::{self, OptimizerConfig};
use tch::{Device, Kind, Tensor};

use crate::net::{CELLS, Net, NetConfig, PLANE_COUNT, POLICY, SIZE};

/// Planes 0..=6 of the encoding are position-dependent bitsets (81 bits
/// each); plane 7 is the constant stm-is-white fill and plane 8 the constant
/// ones fill, reconstructed at expansion.
pub struct Sample {
    pub planes: [u128; 7],
    pub stm_white: bool,
    /// Sparse visit distribution over policy indices.
    pub policy: Vec<(u16, f32)>,
    /// Game outcome from the perspective of the player to move.
    pub z: f32,
    /// The search's root value at this position (player to move) — mixed
    /// into the value target to de-noise the raw outcome.
    pub q: f32,
}

/// Packs the encoder's f32 features into per-plane bitsets.
pub fn compact(features: &[f32]) -> ([u128; 7], bool) {
    debug_assert_eq!(features.len(), PLANE_COUNT * CELLS);
    let mut planes = [0u128; 7];
    for (p, plane) in planes.iter_mut().enumerate() {
        for cell in 0..CELLS {
            if features[p * CELLS + cell] != 0.0 {
                *plane |= 1 << cell;
            }
        }
    }
    (planes, features[7 * CELLS] != 0.0)
}

/// Expands packed planes under symmetry `t` (0..8) into the net's input
/// layout.
pub fn expand(planes: &[u128; 7], stm_white: bool, t: u8, out: &mut [f32]) {
    debug_assert_eq!(out.len(), PLANE_COUNT * CELLS);
    out.fill(0.0);
    for (p, &bits) in planes.iter().enumerate() {
        let mut b = bits;
        while b != 0 {
            let cell = b.trailing_zeros() as usize;
            out[p * CELLS + d8(cell, t, SIZE as usize)] = 1.0;
            b &= b - 1;
        }
    }
    if stm_white {
        out[7 * CELLS..8 * CELLS].fill(1.0);
    }
    out[8 * CELLS..9 * CELLS].fill(1.0);
}

pub struct Replay {
    buf: VecDeque<Sample>,
    cap: usize,
}

impl Replay {
    pub fn new(cap: usize) -> Replay {
        Replay {
            buf: VecDeque::new(),
            cap,
        }
    }

    pub fn extend(&mut self, samples: Vec<Sample>) {
        for s in samples {
            if self.buf.len() == self.cap {
                self.buf.pop_front();
            }
            self.buf.push_back(s);
        }
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    fn get(&self, rng: &mut Rng) -> &Sample {
        let i = (rng.unit() * self.buf.len() as f64) as usize;
        &self.buf[i.min(self.buf.len() - 1)]
    }
}

pub struct Trainer {
    pub vs: nn::VarStore,
    net: Net,
    opt: nn::Optimizer,
    cfg: NetConfig,
    /// Weight of the search's root value in the value target:
    /// `target = (1-mix)·z + mix·q`. De-noises the raw game outcome and
    /// softens the self-labeling loop resignation introduces.
    value_mix: f32,
}

impl Trainer {
    pub fn new(
        device: Device,
        cfg: NetConfig,
        lr: f64,
        weight_decay: f64,
        value_mix: f32,
    ) -> Trainer {
        let vs = nn::VarStore::new(device);
        let net = Net::new(&vs.root(), cfg);
        let opt = nn::Adam {
            wd: weight_decay,
            ..Default::default()
        }
        .build(&vs, lr)
        .expect("build optimizer");
        Trainer {
            vs,
            net,
            opt,
            cfg,
            value_mix,
        }
    }

    /// `steps` minibatch updates; returns mean (policy loss, value loss).
    pub fn train(
        &mut self,
        replay: &Replay,
        steps: usize,
        batch: usize,
        rng: &mut Rng,
    ) -> (f32, f32) {
        if replay.is_empty() || steps == 0 {
            return (0.0, 0.0);
        }
        let device = self.vs.device();
        let plane_len = PLANE_COUNT * CELLS;
        let mut planes = vec![0.0f32; batch * plane_len];
        let mut targets = vec![0.0f32; batch * POLICY as usize];
        let mut zs = vec![0.0f32; batch];
        let (mut pl_sum, mut vl_sum) = (0.0f64, 0.0f64);

        for _ in 0..steps {
            targets.fill(0.0);
            for i in 0..batch {
                let s = replay.get(rng);
                let t = rng.below(8) as u8;
                expand(
                    &s.planes,
                    s.stm_white,
                    t,
                    &mut planes[i * plane_len..(i + 1) * plane_len],
                );
                for &(idx, p) in &s.policy {
                    let ti = d8_policy(idx, t, SIZE as usize);
                    targets[i * POLICY as usize + usize::from(ti)] = p;
                }
                zs[i] = (1.0 - self.value_mix) * s.z + self.value_mix * s.q;
            }
            let x = Tensor::from_slice(&planes)
                .reshape([batch as i64, PLANE_COUNT as i64, SIZE, SIZE])
                .to_device(device);
            let tp = Tensor::from_slice(&targets)
                .reshape([batch as i64, POLICY])
                .to_device(device);
            let tz = Tensor::from_slice(&zs).to_device(device);

            let (logits, v) = self.net.forward(&x, true);
            let logp = logits.log_softmax(-1, Kind::Float);
            let pl = -(tp * logp)
                .sum_dim_intlist(-1, false, Kind::Float)
                .mean(Kind::Float);
            let vl = (v - tz).square().mean(Kind::Float);
            let loss = &pl + &vl;
            self.opt.backward_step(&loss);
            pl_sum += pl.double_value(&[]);
            vl_sum += vl.double_value(&[]);
        }
        (
            (pl_sum / steps as f64) as f32,
            (vl_sum / steps as f64) as f32,
        )
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.opt.set_lr(lr);
    }

    /// Saves via temp file + rename so concurrent readers (the elo gauge)
    /// never see a torn checkpoint. A `<name>.json` sidecar records the
    /// architecture, so the checkpoint stays loadable away from its run's
    /// metrics.jsonl.
    pub fn save(&self, path: &std::path::Path) -> Result<(), tch::TchError> {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("checkpoint");
        let tmp = path.with_file_name(format!("{name}.{}.tmp", std::process::id()));
        self.vs.save(&tmp)?;
        std::fs::rename(&tmp, path)?;
        std::fs::write(
            path.with_file_name(format!("{name}.json")),
            format!(
                "{{\"blocks\":{},\"channels\":{}}}\n",
                self.cfg.blocks, self.cfg.channels
            ),
        )?;
        Ok(())
    }

    pub fn load(&mut self, path: &std::path::Path) -> Result<(), tch::TchError> {
        self.vs.load(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_core::{Game, PolicyValueEncoder};
    use go::encode::GoEncoder;
    use go::{Go, GoAction};

    #[test]
    fn compact_expand_roundtrip_under_identity() {
        let g = Go::new(9);
        let mut s = g.initial_state();
        for coord in ["e5", "c3", "g7", "d4"] {
            g.apply(&mut s, GoAction::Place(g.point(coord).unwrap()));
        }
        let x = GoEncoder.encode_state(&g, &s);
        let (planes, stm_white) = compact(&x);
        let mut back = vec![0.0f32; x.len()];
        expand(&planes, stm_white, 0, &mut back);
        assert_eq!(x, back);
    }

    #[test]
    fn expand_under_symmetry_matches_encoding_of_transformed_board() {
        let g = Go::new(9);
        let coords = ["e5", "c3", "g7", "d4", "f6"];
        let mut s = g.initial_state();
        for c in &coords {
            g.apply(&mut s, GoAction::Place(g.point(c).unwrap()));
        }
        let (planes, stm_white) = compact(&GoEncoder.encode_state(&g, &s));
        for t in 0..8u8 {
            let mut ts = g.initial_state();
            for c in &coords {
                let p = g.point(c).unwrap() as usize;
                g.apply(&mut ts, GoAction::Place(d8(p, t, 9) as u16));
            }
            let want = GoEncoder.encode_state(&g, &ts);
            let mut got = vec![0.0f32; want.len()];
            expand(&planes, stm_white, t, &mut got);
            assert_eq!(got, want, "symmetry {t}");
        }
    }
}
