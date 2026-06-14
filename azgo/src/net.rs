//! The go policy/value resnet in tch: a conv stem, residual tower, an
//! AlphaGo Zero-style policy head (1×1 conv to two planes, then a linear to
//! the 82 policy logits — 81 placements + pass, matching
//! [`go::encode::GoEncoder::action_index`]), and a scalar tanh value head.
//!
//! Training runs in fp32 on the `Trainer`'s VarStore; self-play uses
//! [`Infer`], a frozen (optionally fp16) copy refreshed between iterations.
//! All tch calls must stay on one thread — MPS streams are not Sync.

use tch::nn;
use tch::{Device, Kind, Tensor};

/// Load weights for inference/export, tolerating missing training-only heads.
///
/// The ownership (`o1.*`) and score (`sf.*`) heads are training-only
/// auxiliaries — never exported, never read during inference — so checkpoints
/// saved before either existed lack them (the 9×9 `run1` net predates `o1`).
/// Strict `load` is tried first; only an auxiliary-head shortfall is tolerated,
/// so a genuine architecture mismatch still fails loud.
pub(crate) fn load_inference_weights(
    vs: &mut nn::VarStore,
    path: &std::path::Path,
) -> Result<(), tch::TchError> {
    match vs.load(path) {
        Ok(()) => Ok(()),
        Err(e) => {
            let missing = vs.load_partial(path)?;
            if missing
                .iter()
                .all(|n| n.starts_with("o1") || n.starts_with("sf"))
            {
                Ok(())
            } else {
                Err(e)
            }
        }
    }
}

pub const PLANES: i64 = go::encode::PLANES as i64;
pub const PLANE_COUNT: usize = go::encode::PLANES;

#[derive(Clone, Copy)]
pub struct NetConfig {
    pub blocks: usize,
    pub channels: i64,
    pub size: i64,
}

impl NetConfig {
    /// Policy width: one logit per board point plus the pass.
    pub fn policy(&self) -> i64 {
        self.size * self.size + 1
    }
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
    // Policy head — board-size-agnostic: a 1×1 conv gives per-point placement
    // logits, a global-pooled vector both biases that conv (whole-board
    // context) and produces the single pass logit.
    p1: nn::Conv2D,
    pb: nn::BatchNorm,
    pgb: nn::Linear,
    pfc: nn::Conv2D,
    ppass: nn::Linear,
    // Value head — board-size-agnostic: a 1×1 conv then global pooling into an
    // MLP, so it no longer flattens a size-locked spatial vector.
    v1: nn::Conv2D,
    vb: nn::BatchNorm,
    vf1: nn::Linear,
    vf2: nn::Linear,
    /// Score head: predicts the final area-score margin (points, mover's view)
    /// off the same pooled value features. A KataGo-style auxiliary — a far
    /// denser gradient than win/loss — that enriches the trunk; not exported.
    sf: nn::Linear,
    /// Auxiliary ownership head (1×1 conv → per-point tanh). Trained against
    /// the final board's [`go::Go::ownership`]; not exported (inference needs
    /// only policy + value). Its gradient shapes the shared trunk.
    o1: nn::Conv2D,
}

fn conv(p: nn::Path, cin: i64, cout: i64, k: i64) -> nn::Conv2D {
    let cfg = nn::ConvConfig {
        padding: (k - 1) / 2,
        bias: false,
        ..Default::default()
    };
    nn::conv2d(p, cin, cout, k, cfg)
}

/// Global pooling (KataGo): collapse `[B, C, H, W]` to `[B, 3C]` —
/// per-channel mean, max, and a board-size-scaled mean. The scaled channel
/// hands the net an explicit board-size signal, and pooling over the whole
/// board is what makes the value/policy heads independent of board size, so a
/// single net can serve 9×9…19×19. `19.0` only centers the size scale; it is
/// not a board-size assumption.
fn global_pool(t: &Tensor) -> Tensor {
    let (_, _, h, w) = t.size4().expect("gpool expects [B,C,H,W]");
    let dims = [2i64, 3];
    let mean = t.mean_dim(dims.as_slice(), false, t.kind());
    let max = t.amax(dims.as_slice(), false);
    let scaled = &mean * (((h * w) as f64).sqrt() / 19.0);
    Tensor::cat(&[mean, scaled, max], 1)
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
            pgb: nn::linear(root / "pgb", 3 * c, c, Default::default()),
            pfc: conv(root / "pfc", c, 1, 1),
            ppass: nn::linear(root / "ppass", 3 * c, 1, Default::default()),
            v1: conv(root / "v1", c, c, 1),
            vb: nn::batch_norm2d(root / "vb", c, Default::default()),
            vf1: nn::linear(root / "vf1", 3 * c, 128, Default::default()),
            vf2: nn::linear(root / "vf2", 128, 1, Default::default()),
            sf: nn::linear(root / "sf", 128, 1, Default::default()),
            o1: conv(root / "o1", c, 1, 1),
        }
    }

    /// `x`: `[B, 9, size, size]` → (policy logits `[B, size²+1]`, value `[B]`,
    /// ownership `[B, size²]` in `(-1, 1)`, score margin `[B]` in points from
    /// the mover's view). All heads are board-size-agnostic.
    pub fn forward(&self, x: &Tensor, train: bool) -> (Tensor, Tensor, Tensor, Tensor) {
        let mut t = x.apply(&self.stem_c).apply_t(&self.stem_b, train).relu();
        for b in &self.tower {
            t = b.forward(&t, train);
        }

        // Policy: per-point conv features, biased by their global-pool summary;
        // placement logits from a 1×1 conv, the pass logit from the pool.
        let pol = t.apply(&self.p1).apply_t(&self.pb, train).relu();
        let pol_g = global_pool(&pol);
        let pol = (&pol + pol_g.apply(&self.pgb).unsqueeze(-1).unsqueeze(-1)).relu();
        let placement = pol.apply(&self.pfc).flatten(1, -1);
        let pass = pol_g.apply(&self.ppass);
        let p = Tensor::cat(&[placement, pass], 1);

        // Value + score share the pooled MLP features: value is win/loss
        // (tanh), score is the raw margin in points.
        let v = t.apply(&self.v1).apply_t(&self.vb, train).relu();
        let vh = global_pool(&v).apply(&self.vf1).relu();
        let value = vh.apply(&self.vf2).tanh().squeeze_dim(-1);
        let score = vh.apply(&self.sf).squeeze_dim(-1);

        let o = t.apply(&self.o1).flatten(1, -1).tanh();
        (p, value, o, score)
    }
}

pub use solvers::azero::{EvalRequest, EvalResult};

/// In-place softmax: logits → distribution.
pub fn softmax(logits: &mut [f32]) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for q in logits.iter_mut() {
        *q = (*q - max).exp();
        sum += *q;
    }
    for q in logits.iter_mut() {
        *q /= sum;
    }
}

/// A frozen inference copy of the net, optionally fp16.
pub struct Infer {
    _vs: nn::VarStore,
    net: Net,
    device: Device,
    kind: Kind,
    size: i64,
    policy: i64,
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
            size: cfg.size,
            policy: cfg.policy(),
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
        load_inference_weights(&mut vs, path)?;
        if kind == Kind::Half {
            vs.half();
        }
        vs.freeze();
        Ok(Infer {
            _vs: vs,
            net,
            device,
            kind,
            size: cfg.size,
            policy: cfg.policy(),
        })
    }

    /// Evaluates a batch of requests in one GPU round trip. Only the legal
    /// (`support`) logits come back from the GPU.
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
        let cells = (self.size * self.size) as usize;
        let plane_len = PLANE_COUNT * cells;
        let mut planes = vec![0.0f32; bucket * plane_len];
        let mut gather: Vec<i64> = Vec::with_capacity(reqs.len() * 48);
        for (i, r) in reqs.iter().enumerate() {
            debug_assert_eq!(r.features.len(), plane_len);
            planes[i * plane_len..(i + 1) * plane_len].copy_from_slice(&r.features);
            let base = i as i64 * self.policy;
            gather.extend(r.support.iter().map(|&s| base + i64::from(s)));
        }
        // Same shape-bucketing for the index tensor; padding rows point at
        // row 0 and their outputs are ignored.
        gather.resize(gather.len().next_multiple_of(4096), 0);
        let (legal_logits, values) = tch::no_grad(|| {
            let x = Tensor::from_slice(&planes)
                .reshape([b, PLANES, self.size, self.size])
                .to_device(self.device)
                .to_kind(self.kind);
            let idx = Tensor::from_slice(&gather).to_device(self.device);
            // Inference needs only policy + value; ownership and score are
            // training-only auxiliaries.
            let (p, v, _own, _score) = self.net.forward(&x, false);
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
                softmax(&mut priors);
                EvalResult {
                    priors,
                    value: values[i],
                }
            })
            .collect()
    }
}
