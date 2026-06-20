//! Replay buffer and the fp32 training step: policy cross-entropy against
//! the search's visit distribution (softmax over the four headings) plus
//! value MSE, AdamW on MPS.

use std::collections::VecDeque;

use game_core::Rng;
use tch::nn::{self, OptimizerConfig};
use tch::{Device, Kind, Tensor};

use crate::net::{Net, NetConfig, PLANE_COUNT};

/// One self-play training example. Planes are stored unpacked — the board is
/// small (20×20 × ~9 planes) and the replay buffer is bounded, so the
/// memory win from bit-packing is not worth the complexity here.
pub struct Sample {
    pub planes: Vec<f32>,
    /// Sparse visit distribution over policy indices (the four headings).
    pub policy: Vec<(u16, f32)>,
    /// Game outcome from the perspective of the player to move.
    pub z: f32,
    /// The search's root value at this position (player to move) — mixed
    /// into the value target to de-noise the raw outcome.
    pub q: f32,
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
    /// Stochastic Weight Averaging: an exponential moving average of the
    /// weights (`Some` iff `swa_decay > 0`). Training always updates `vs`;
    /// the averaged copy, which generalizes better, is what eval reads.
    swa_vs: Option<nn::VarStore>,
    swa_decay: f64,
    swa_inited: bool,
    /// Global gradient-norm clip applied before each step; 0 disables.
    grad_clip: f64,
}

/// Optimizer choice: AdamW by default, or SGD+Nesterov-momentum, optionally
/// with global gradient-norm clipping.
pub struct OptConfig {
    pub sgd: bool,
    pub momentum: f64,
    pub weight_decay: f64,
    pub grad_clip: f64,
}

impl Trainer {
    pub fn new(
        device: Device,
        cfg: NetConfig,
        lr: f64,
        value_mix: f32,
        swa_decay: f64,
        opt_cfg: OptConfig,
    ) -> Trainer {
        let vs = nn::VarStore::new(device);
        let net = Net::new(&vs.root(), cfg);
        let opt = if opt_cfg.sgd {
            nn::Sgd {
                momentum: opt_cfg.momentum,
                wd: opt_cfg.weight_decay,
                nesterov: true,
                ..Default::default()
            }
            .build(&vs, lr)
            .expect("build optimizer")
        } else {
            nn::Adam {
                wd: opt_cfg.weight_decay,
                ..Default::default()
            }
            .build(&vs, lr)
            .expect("build optimizer")
        };
        let grad_clip = opt_cfg.grad_clip;
        let swa_vs = (swa_decay > 0.0).then(|| {
            let mut s = nn::VarStore::new(device);
            let _ = Net::new(&s.root(), cfg);
            s.freeze();
            s
        });
        Trainer {
            vs,
            net,
            opt,
            cfg,
            value_mix,
            swa_vs,
            swa_decay,
            swa_inited: false,
            grad_clip,
        }
    }

    /// Fold the current weights into the SWA exponential moving average; the
    /// first call seeds it. No-op when SWA is disabled.
    pub fn update_swa(&mut self) {
        if self.swa_vs.is_none() {
            return;
        }
        let (decay, inited) = (self.swa_decay, self.swa_inited);
        let cur = self.vs.variables();
        let mut avg = self.swa_vs.as_ref().unwrap().variables();
        tch::no_grad(|| {
            for (name, swa_t) in &mut avg {
                let cur_t = &cur[name];
                if inited {
                    let updated = &*swa_t * decay + cur_t * (1.0 - decay);
                    swa_t.copy_(&updated);
                } else {
                    swa_t.copy_(cur_t);
                }
            }
        });
        self.swa_inited = true;
    }

    /// The VarStore eval should read: the SWA average once seeded, otherwise
    /// the raw trained weights.
    pub fn infer_vs(&self) -> &nn::VarStore {
        match &self.swa_vs {
            Some(s) if self.swa_inited => s,
            _ => &self.vs,
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
        let size = self.cfg.size as usize;
        let cells = size * size;
        let policy = self.cfg.policy() as usize;
        let plane_len = PLANE_COUNT * cells;
        let (mut pl_sum, mut vl_sum) = (0.0f64, 0.0f64);

        for _ in 0..steps {
            let mut planes = vec![0.0f32; batch * plane_len];
            let mut targets = vec![0.0f32; batch * policy];
            let mut zs = vec![0.0f32; batch];
            for i in 0..batch {
                let s = replay.get(rng);
                planes[i * plane_len..(i + 1) * plane_len].copy_from_slice(&s.planes);
                for &(idx, p) in &s.policy {
                    targets[i * policy + usize::from(idx)] = p;
                }
                zs[i] = (1.0 - self.value_mix) * s.z + self.value_mix * s.q;
            }
            let sz = size as i64;
            let x = Tensor::from_slice(&planes)
                .reshape([batch as i64, PLANE_COUNT as i64, sz, sz])
                .to_device(device);
            let tp = Tensor::from_slice(&targets)
                .reshape([batch as i64, policy as i64])
                .to_device(device);
            let tz = Tensor::from_slice(&zs).to_device(device);

            let (logits, v) = self.net.forward(&x, true);
            let logp = logits.log_softmax(-1, Kind::Float);
            let pl = -(tp * logp)
                .sum_dim_intlist(-1, false, Kind::Float)
                .mean(Kind::Float);
            let vl = (v - tz).square().mean(Kind::Float);
            let total = &pl + &vl;

            self.opt.zero_grad();
            total.backward();
            if self.grad_clip > 0.0 {
                self.opt.clip_grad_norm(self.grad_clip);
            }
            self.opt.step();

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

    /// Saves via temp file + rename so concurrent readers (the rate gauge)
    /// never see a torn checkpoint. A `<name>.json` sidecar records the
    /// architecture so the checkpoint stays loadable away from its run's
    /// metrics.jsonl.
    pub fn save(&self, path: &std::path::Path) -> Result<(), tch::TchError> {
        self.save_vs(&self.vs, path)
    }

    /// Saves the SWA average (for eval). No-op when SWA is off or not yet
    /// seeded, so callers can invoke it unconditionally.
    pub fn save_swa(&self, path: &std::path::Path) -> Result<(), tch::TchError> {
        match &self.swa_vs {
            Some(s) if self.swa_inited => self.save_vs(s, path),
            _ => Ok(()),
        }
    }

    fn save_vs(&self, vs: &nn::VarStore, path: &std::path::Path) -> Result<(), tch::TchError> {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("checkpoint");
        let tmp = path.with_file_name(format!("{name}.{}.tmp", std::process::id()));
        vs.save(&tmp)?;
        std::fs::rename(&tmp, path)?;
        std::fs::write(
            path.with_file_name(format!("{name}.json")),
            format!(
                "{{\"blocks\":{},\"channels\":{},\"size\":{}}}\n",
                self.cfg.blocks, self.cfg.channels, self.cfg.size
            ),
        )?;
        Ok(())
    }

    pub fn load(&mut self, path: &std::path::Path) -> Result<(), tch::TchError> {
        self.vs.load(path)
    }

    /// Seeds the trainable weights from another checkpoint (transfer learning).
    pub fn init_from(&mut self, path: &std::path::Path) -> Result<(), tch::TchError> {
        self.vs.load(path)
    }
}
