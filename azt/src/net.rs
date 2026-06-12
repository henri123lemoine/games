//! The policy/value resnet in tch: a conv stem, residual tower, AlphaZero
//! 73-plane convolutional policy head (logits laid out as
//! `oriented from-square · 73 + movement plane`, matching
//! [`chess::encode::az_move_index`]), and a scalar tanh value head.
//!
//! Training runs in fp32 on the `Trainer`'s VarStore; self-play uses
//! [`Infer`], a frozen (optionally fp16) copy refreshed between iterations.
//! All tch calls must stay on one thread — MPS streams are not Sync.

use chess::encode::{AZ_POLICY_LEN, PLANE_COUNT};
use tch::nn;
use tch::{Device, Kind, Tensor};

pub const PLANES: i64 = PLANE_COUNT as i64;
pub const POLICY: i64 = AZ_POLICY_LEN as i64;

#[derive(Clone, Copy)]
pub struct NetConfig {
    pub blocks: usize,
    pub channels: i64,
}

struct Block {
    c1: nn::Conv2D,
    b1: nn::BatchNorm,
    c2: nn::Conv2D,
    b2: nn::BatchNorm,
}

impl Block {
    fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let y = x.apply(&self.c1).apply_t(&self.b1, train).relu();
        let y = y.apply(&self.c2).apply_t(&self.b2, train);
        (x + y).relu()
    }
}

pub struct Net {
    stem_c: nn::Conv2D,
    stem_b: nn::BatchNorm,
    tower: Vec<Block>,
    p1: nn::Conv2D,
    pb: nn::BatchNorm,
    p2: nn::Conv2D,
    v1: nn::Conv2D,
    vb: nn::BatchNorm,
    vf1: nn::Linear,
    vf2: nn::Linear,
}

fn conv(p: nn::Path, cin: i64, cout: i64, k: i64) -> nn::Conv2D {
    let cfg = nn::ConvConfig {
        padding: (k - 1) / 2,
        bias: false,
        ..Default::default()
    };
    nn::conv2d(p, cin, cout, k, cfg)
}

impl Net {
    pub fn new(root: &nn::Path, cfg: NetConfig) -> Net {
        let c = cfg.channels;
        let tower = (0..cfg.blocks)
            .map(|i| {
                let p = root / format!("block{i}");
                Block {
                    c1: conv(&p / "c1", c, c, 3),
                    b1: nn::batch_norm2d(&p / "b1", c, Default::default()),
                    c2: conv(&p / "c2", c, c, 3),
                    b2: nn::batch_norm2d(&p / "b2", c, Default::default()),
                }
            })
            .collect();
        Net {
            stem_c: conv(root / "stem_c", PLANES, c, 3),
            stem_b: nn::batch_norm2d(root / "stem_b", c, Default::default()),
            tower,
            p1: conv(root / "p1", c, c, 1),
            pb: nn::batch_norm2d(root / "pb", c, Default::default()),
            p2: conv(root / "p2", c, 73, 1),
            v1: conv(root / "v1", c, 8, 1),
            vb: nn::batch_norm2d(root / "vb", 8, Default::default()),
            vf1: nn::linear(root / "vf1", 8 * 64, 256, Default::default()),
            vf2: nn::linear(root / "vf2", 256, 1, Default::default()),
        }
    }

    /// `x`: `[B, 18, 8, 8]` → (policy logits `[B, 4672]`, value `[B]`).
    pub fn forward(&self, x: &Tensor, train: bool) -> (Tensor, Tensor) {
        let mut t = x.apply(&self.stem_c).apply_t(&self.stem_b, train).relu();
        for b in &self.tower {
            t = b.forward(&t, train);
        }
        let p = t
            .apply(&self.p1)
            .apply_t(&self.pb, train)
            .relu()
            .apply(&self.p2);
        // [B, 73, 8, 8] → square-major [B, 8·8, 73] → [B, 4672]
        let p = p.permute([0, 2, 3, 1]).reshape([-1, POLICY]);
        let v = t
            .apply(&self.v1)
            .apply_t(&self.vb, train)
            .relu()
            .flatten(1, -1);
        let v = v
            .apply(&self.vf1)
            .relu()
            .apply(&self.vf2)
            .tanh()
            .squeeze_dim(-1);
        (p, v)
    }
}

pub use azinfer::{EvalRequest, EvalResult};

/// A frozen inference copy of the net, optionally fp16.
pub struct Infer {
    _vs: nn::VarStore,
    net: Net,
    device: Device,
    kind: Kind,
}

impl Infer {
    pub fn snapshot(train_vs: &nn::VarStore, cfg: NetConfig, kind: Kind) -> Infer {
        let device = train_vs.device();
        let mut vs = nn::VarStore::new(device);
        let net = Net::new(&vs.root(), cfg);
        vs.copy(train_vs)
            .expect("copy weights into inference store");
        if kind == Kind::Half {
            vs.half();
        }
        vs.freeze();
        Infer {
            _vs: vs,
            net,
            device,
            kind,
        }
    }

    /// Loads a checkpoint saved by `Trainer::save`; `cfg` must match the
    /// checkpoint's architecture.
    pub fn load(
        path: &std::path::Path,
        cfg: NetConfig,
        device: Device,
        kind: Kind,
    ) -> Result<Infer, tch::TchError> {
        let mut vs = nn::VarStore::new(device);
        let net = Net::new(&vs.root(), cfg);
        vs.load(path)?;
        if kind == Kind::Half {
            vs.half();
        }
        vs.freeze();
        Ok(Infer {
            _vs: vs,
            net,
            device,
            kind,
        })
    }

    /// Evaluates a batch of requests in one GPU round trip. Only the legal
    /// (`support`) logits come back from the GPU — pulling all 4672 per row
    /// would dominate the cycle at wide batches.
    pub fn forward_batch(&self, reqs: &[EvalRequest]) -> Vec<EvalResult> {
        if reqs.is_empty() {
            return Vec::new();
        }
        // Pad the batch to bucket sizes: libtorch's MPS backend caches a
        // compiled graph per tensor shape, and self-play batch widths vary
        // every cycle — unbucketed, the cache grows without bound until the
        // OS kills the process.
        let bucket = reqs.len().next_multiple_of(256);
        let b = bucket as i64;
        let mut planes = vec![0.0f32; bucket * PLANE_COUNT * 64];
        let mut gather: Vec<i64> = Vec::with_capacity(reqs.len() * 40);
        for (i, r) in reqs.iter().enumerate() {
            debug_assert_eq!(r.features.len(), PLANE_COUNT * 64);
            planes[i * PLANE_COUNT * 64..(i + 1) * PLANE_COUNT * 64].copy_from_slice(&r.features);
            let base = i as i64 * POLICY;
            gather.extend(r.support.iter().map(|&s| base + i64::from(s)));
        }
        // Same shape-bucketing for the index tensor; padding rows point at
        // row 0 and their outputs are ignored.
        gather.resize(gather.len().next_multiple_of(4096), 0);
        let (legal_logits, values) = tch::no_grad(|| {
            let x = Tensor::from_slice(&planes)
                .reshape([b, PLANES, 8, 8])
                .to_device(self.device)
                .to_kind(self.kind);
            let idx = Tensor::from_slice(&gather).to_device(self.device);
            let (p, v) = self.net.forward(&x, false);
            (
                p.reshape([-1])
                    .index_select(0, &idx)
                    .to_kind(Kind::Float)
                    .to_device(Device::Cpu),
                v.to_kind(Kind::Float).to_device(Device::Cpu),
            )
        });
        let legal: Vec<f32> = legal_logits.try_into().expect("legal logits to vec");
        let values: Vec<f32> = values.reshape([b]).try_into().expect("values to vec");

        let mut offset = 0;
        reqs.iter()
            .enumerate()
            .map(|(i, r)| {
                let mut priors = legal[offset..offset + r.support.len()].to_vec();
                offset += r.support.len();
                let max = priors.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for q in &mut priors {
                    *q = (*q - max).exp();
                    sum += *q;
                }
                for q in &mut priors {
                    *q /= sum;
                }
                EvalResult {
                    priors,
                    value: values[i],
                }
            })
            .collect()
    }
}
