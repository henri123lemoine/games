//! The replay buffer and the fp32 training step, unified across the neural
//! games. The optimizer is the most-evolved of the originals (azgo): AdamW or
//! SGD+Nesterov-momentum, optional global gradient-norm clipping, and
//! Stochastic Weight Averaging. The loss is policy cross-entropy against the
//! search's visit distribution plus value MSE, with the go auxiliary heads
//! (ownership MSE, score Huber) added when the batch carries those targets.
//!
//! Per-game knowledge enters through [`TrainSample`]: each game packs its own
//! example and expands a minibatch into the input planes, the dense policy
//! target, the value target, and (go) the auxiliary targets. The Trainer owns
//! the optimizer, the SWA average, and the loss — the algorithm, once.

use std::collections::VecDeque;

use game_core::Rng;
use tch::nn::{self, OptimizerConfig};
use tch::{Device, Kind, Tensor};

use crate::net::{Net, NetConfig};

/// Weight of the auxiliary ownership MSE in the total loss (go). KataGo-ish:
/// large enough to shape the trunk, small enough not to swamp policy/value.
const OWNERSHIP_WEIGHT: f64 = 1.0;
/// Weight of the auxiliary score-margin (Huber) loss in the total loss (go).
const SCORE_WEIGHT: f64 = 0.3;

/// A minibatch expanded to flat tensors, ready for the forward pass. All vectors
/// are row-major over the `n` examples; `planes` is `n · planes · size²`.
pub struct Batch {
    pub n: usize,
    pub size: usize,
    pub planes: Vec<f32>,
    /// Dense policy target, `n · policy_len`.
    pub policy: Vec<f32>,
    /// Value target (already z/q-mixed by the caller), length `n` for scalar
    /// heads or `n * value_seats` for multiplayer heads.
    pub value: Vec<f32>,
    /// Per-point ownership target (mover view), `n · size²`; empty without aux.
    pub ownership: Vec<f32>,
    /// Score-margin target (mover view, points), length `n`; empty without aux.
    pub score: Vec<f32>,
}

/// Per-game training examples: the replay buffer's element and how a minibatch
/// of them expands into network tensors. The `value_mix` (z/q blend) is applied
/// here so each game controls its own value target.
pub trait TrainSample: Send + Sync {
    /// Expand `n` randomly drawn examples into one (or, for mixed-size games,
    /// several) fixed-size [`Batch`]es. `value_mix` weights the search root
    /// value into the value target: `(1-mix)·z + mix·q`. Augmentation (e.g.
    /// go's dihedral symmetry) is applied here, per draw.
    fn expand(
        buf: &VecDeque<Self>,
        n: usize,
        cfg: &NetConfig,
        value_mix: f32,
        rng: &mut Rng,
    ) -> Vec<Batch>
    where
        Self: Sized;
}

pub struct Replay<S> {
    buf: VecDeque<S>,
    cap: usize,
}

impl<S> Replay<S> {
    pub fn new(cap: usize) -> Replay<S> {
        Replay {
            buf: VecDeque::new(),
            cap,
        }
    }

    pub fn extend(&mut self, samples: impl IntoIterator<Item = S>) {
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

    pub fn buf(&self) -> &VecDeque<S> {
        &self.buf
    }
}

/// Optimizer choice: AdamW by default, or KataGo-style SGD+Nesterov-momentum
/// (`sgd`, which wants a larger lr, ~0.02), optionally with global
/// gradient-norm clipping.
pub struct OptConfig {
    pub sgd: bool,
    pub momentum: f64,
    pub weight_decay: f64,
    /// Clip the global gradient norm to this before each step; 0 disables.
    pub grad_clip: f64,
}

pub struct Trainer {
    pub vs: nn::VarStore,
    net: Net,
    opt: nn::Optimizer,
    cfg: NetConfig,
    /// `target = (1-mix)·z + mix·q`. De-noises the raw game outcome and softens
    /// the self-labeling loop resignation introduces.
    value_mix: f32,
    /// Stochastic Weight Averaging: an exponential moving average of the weights
    /// (`Some` iff `swa_decay > 0`). Training always updates `vs`; the averaged
    /// copy, which generalizes better, is what eval and export read. Includes
    /// BatchNorm running stats so the average stays usable without a
    /// stat-recomputation pass.
    swa_vs: Option<nn::VarStore>,
    swa_decay: f64,
    swa_inited: bool,
    /// Global gradient-norm clip applied before each step; 0 disables.
    grad_clip: f64,
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
            // Allocate variables matching `vs` (names + shapes); the Net handle
            // is dropped but the VarStore owns the tensors. Freeze it — it is
            // never optimized, only averaged into in place.
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

    /// The VarStore eval/export should read: the SWA average once seeded,
    /// otherwise the raw trained weights.
    pub fn infer_vs(&self) -> &nn::VarStore {
        match &self.swa_vs {
            Some(s) if self.swa_inited => s,
            _ => &self.vs,
        }
    }

    pub fn value_mix(&self) -> f32 {
        self.value_mix
    }

    /// `steps` minibatch updates; returns mean (policy loss, value loss). Each
    /// step draws `batch` examples, expands them (per-game, possibly split by
    /// board size), runs one forward/backward, and steps the optimizer.
    pub fn train<S: TrainSample>(
        &mut self,
        replay: &Replay<S>,
        steps: usize,
        batch: usize,
        rng: &mut Rng,
    ) -> (f32, f32) {
        if replay.is_empty() || steps == 0 {
            return (0.0, 0.0);
        }
        let device = self.vs.device();
        let planes_in = self.cfg.planes;
        let (mut pl_sum, mut vl_sum) = (0.0f64, 0.0f64);

        for _ in 0..steps {
            let groups = S::expand(replay.buf(), batch, &self.cfg, self.value_mix, rng);
            let mut total: Option<Tensor> = None;
            let (mut pl_acc, mut vl_acc) = (0.0f64, 0.0f64);
            for b in &groups {
                let n = b.n as i64;
                let sz = b.size as i64;
                let policy = (b.policy.len() / b.n) as i64;
                let x = Tensor::from_slice(&b.planes)
                    .reshape([n, planes_in, sz, sz])
                    .to_device(device);
                let tp = Tensor::from_slice(&b.policy)
                    .reshape([n, policy])
                    .to_device(device);
                let tz = Tensor::from_slice(&b.value).to_device(device);

                let (logits, v, own, score) = self.net.forward_train(&x, true);
                let logp = logits.log_softmax(-1, Kind::Float);
                let pl = -(tp * logp)
                    .sum_dim_intlist(-1, false, Kind::Float)
                    .mean(Kind::Float);
                // Scalar head: MSE against the (-1,1) outcome. Multi-seat head:
                // cross-entropy against the win-share distribution.
                let vl = if self.cfg.seats > 1 {
                    let tz = tz.reshape([n, self.cfg.seats]);
                    let logv = v.log_softmax(-1, Kind::Float);
                    -(tz * logv)
                        .sum_dim_intlist(-1, false, Kind::Float)
                        .mean(Kind::Float)
                } else {
                    (v - tz).square().mean(Kind::Float)
                };
                let mut group_loss = &pl + &vl;

                // Go auxiliary losses, when the batch carries those targets and
                // the net has the heads.
                if !b.ownership.is_empty()
                    && let Some(own) = own
                {
                    let to = Tensor::from_slice(&b.ownership)
                        .reshape([n, (b.size * b.size) as i64])
                        .to_device(device);
                    let ol = (own - to).square().mean(Kind::Float);
                    group_loss += OWNERSHIP_WEIGHT * ol;
                }
                if !b.score.is_empty()
                    && let Some(score) = score
                {
                    let ts = Tensor::from_slice(&b.score).to_device(device);
                    let scale = b.size as f64;
                    let sl =
                        (score / scale).smooth_l1_loss(&(ts / scale), tch::Reduction::Mean, 1.0);
                    group_loss += SCORE_WEIGHT * sl;
                }

                // Weight each size-group by its batch share so the summed loss is
                // a proper batch-mean across mixed sizes.
                let w = b.n as f64 / batch as f64;
                let group_loss = group_loss * w;
                total = Some(match total {
                    Some(t) => t + group_loss,
                    None => group_loss,
                });
                pl_acc += pl.double_value(&[]) * w;
                vl_acc += vl.double_value(&[]) * w;
            }
            if let Some(total) = total {
                self.opt.zero_grad();
                total.backward();
                if self.grad_clip > 0.0 {
                    self.opt.clip_grad_norm(self.grad_clip);
                }
                self.opt.step();
            }
            pl_sum += pl_acc;
            vl_sum += vl_acc;
        }
        (
            (pl_sum / steps as f64) as f32,
            (vl_sum / steps as f64) as f32,
        )
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.opt.set_lr(lr);
    }

    /// Saves via temp file + rename so concurrent readers (an eval gauge) never
    /// see a torn checkpoint. A `<name>.json` sidecar records the architecture so
    /// the checkpoint stays loadable away from its run's metrics.jsonl.
    pub fn save(&self, path: &std::path::Path) -> Result<(), tch::TchError> {
        self.save_vs(&self.vs, path)
    }

    /// Saves the SWA average (for eval/export). No-op when SWA is off or not yet
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
    /// Missing auxiliary heads are tolerated, so a net trained without the go aux
    /// (or at another board size — every weight is size-independent) loads.
    pub fn init_from(&mut self, path: &std::path::Path) -> Result<(), tch::TchError> {
        crate::net::load_inference_weights(&mut self.vs, path)
    }
}
