//! Replay buffer and the fp32 training step: policy cross-entropy against
//! the search's visit distribution (full softmax over all 4672 logits) plus
//! value MSE, AdamW on MPS.

use std::collections::VecDeque;

use chess::encode::PLANE_COUNT;
use game_core::Rng;
use tch::nn::{self, OptimizerConfig};
use tch::{Device, Kind, Tensor};

use crate::net::{Net, NetConfig, POLICY};
use crate::selfplay::{Sample, expand_planes};

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
        let plane_len = PLANE_COUNT * 64;
        let mut planes = vec![0.0f32; batch * plane_len];
        let mut targets = vec![0.0f32; batch * POLICY as usize];
        let mut zs = vec![0.0f32; batch];
        let (mut pl_sum, mut vl_sum) = (0.0f64, 0.0f64);

        for _ in 0..steps {
            targets.fill(0.0);
            for i in 0..batch {
                let s = replay.get(rng);
                expand_planes(
                    &s.planes,
                    s.halfmove,
                    &mut planes[i * plane_len..(i + 1) * plane_len],
                );
                for &(idx, p) in &s.policy {
                    targets[i * POLICY as usize + usize::from(idx)] = p;
                }
                zs[i] = (1.0 - self.value_mix) * s.z + self.value_mix * s.q;
            }
            let x = Tensor::from_slice(&planes)
                .reshape([batch as i64, PLANE_COUNT as i64, 8, 8])
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

    /// Saves via temp file + rename so concurrent readers (the elo gauge,
    /// `azt play`) never see a torn checkpoint. A `<name>.json` sidecar
    /// records the architecture, so the checkpoint stays loadable away from
    /// its run's metrics.jsonl.
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
