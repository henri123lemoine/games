//! The policy/value resnet in tch, config-driven by [`NetConfig`]: a conv stem,
//! a residual tower, and a policy/value head pair selected by [`HeadKind`]. One
//! net definition replacing `azt`/`azgo`/`azsnake`'s three `net.rs` modules — the
//! training mirror of [`nn_infer::Net`], the tch-free forward the browser and the
//! export check evaluate.
//!
//! The tensor names match the original per-game nets (`stem_c`, `block{i}.c1`,
//! `p1`, `pgb`, …) so every committed checkpoint loads unchanged. [`HeadKind`]
//! pairs a policy shape with a value shape — the same closed enumeration the
//! `AZNET1` container carries — so a checkpoint's architecture fully determines
//! the heads.
//!
//! Training runs in fp32 on the trainer's VarStore; self-play uses [`Infer`], a
//! frozen (optionally fp16) copy refreshed between iterations. All tch calls
//! must stay on one thread — MPS streams are not Sync.

use nn_infer::HeadKind;
use tch::nn;
use tch::{Device, Kind, Tensor};

pub use solvers::azero::{EvalRequest, EvalResult};

/// Chess value head: `v1` reduces the trunk to this many channels before the
/// dense MLP (matches `azt`'s net and `nn_infer`'s `CHESS_VALUE_CHANNELS`).
const CHESS_VALUE_CHANNELS: i64 = 8;
const CHESS_VALUE_HIDDEN: i64 = 256;
/// Go/snake global-pool value head's hidden width.
const POOL_VALUE_HIDDEN: i64 = 128;

/// Everything the resnet needs to lay itself out. The training mirror of
/// [`nn_infer::Arch`]; `head` selects both the policy and the value head.
#[derive(Clone, Copy)]
pub struct NetConfig {
    pub blocks: usize,
    pub channels: i64,
    /// Input feature channels (planes).
    pub planes: i64,
    /// Board side; 8 for chess (fixed), the go/snake board size otherwise.
    pub size: i64,
    pub head: HeadKind,
    /// Flat policy width for [`HeadKind::FlatConv`] / [`HeadKind::GlobalPoolDense`];
    /// `size²+1` is computed for [`HeadKind::GlobalPoolSpatial`].
    pub policy_len: i64,
    /// The go auxiliary heads (ownership conv + score linear). Off elsewhere.
    pub go_aux: bool,
}

impl NetConfig {
    /// Policy head width at this net's board size.
    pub fn policy(&self) -> i64 {
        match self.head {
            HeadKind::GlobalPoolSpatial => self.size * self.size + 1,
            HeadKind::FlatConv | HeadKind::GlobalPoolDense => self.policy_len,
        }
    }

    /// Move planes per square for the chess flat-conv policy (`policy_len`
    /// laid out as `square · move_planes`).
    pub fn move_planes(&self) -> i64 {
        self.policy_len / (self.size * self.size)
    }
}

fn conv(p: nn::Path, cin: i64, cout: i64, k: i64) -> nn::Conv2D {
    let cfg = nn::ConvConfig {
        padding: (k - 1) / 2,
        bias: false,
        ..Default::default()
    };
    nn::conv2d(p, cin, cout, k, cfg)
}

/// Global pooling (KataGo): collapse `[B, C, H, W]` to `[B, 3C]` — per-channel
/// mean, a board-size-scaled mean, and max. Pooling over the whole board makes
/// the heads board-size-agnostic; `19.0` only centers the size scale. Matches
/// every trainer's `global_pool` and [`nn_infer`]'s `POOL_SIZE_REF`.
fn global_pool(t: &Tensor) -> Tensor {
    let (_, _, h, w) = t.size4().expect("gpool expects [B,C,H,W]");
    let dims = [2i64, 3];
    let mean = t.mean_dim(dims.as_slice(), false, t.kind());
    let max = t.amax(dims.as_slice(), false);
    let scaled = &mean * (((h * w) as f64).sqrt() / 19.0);
    Tensor::cat(&[mean, scaled, max], 1)
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

/// Chess flat-conv policy: `p1` (1×1, relu) → `p2` (1×1 → move planes), laid out
/// square-major to a fixed move space.
struct FlatPolicy {
    p1: nn::Conv2D,
    pb: nn::BatchNorm,
    p2: nn::Conv2D,
    policy_len: i64,
}

/// Go global-pool spatial policy: `p1` biased per channel by a global-pool
/// linear (`pgb`), a placement conv (`pfc`), and a pooled pass logit (`ppass`).
struct SpatialPolicy {
    p1: nn::Conv2D,
    pb: nn::BatchNorm,
    pgb: nn::Linear,
    pfc: nn::Conv2D,
    ppass: nn::Linear,
}

/// Snake global-pool dense policy: `p1` → global pool → MLP (`pf1`, `pf2`).
struct DensePolicy {
    p1: nn::Conv2D,
    pb: nn::BatchNorm,
    pf1: nn::Linear,
    pf2: nn::Linear,
}

enum Policy {
    Flat(FlatPolicy),
    Spatial(SpatialPolicy),
    Dense(DensePolicy),
}

/// Chess value head: `v1` (1×1 → `vc`) flattened over the full board to a dense
/// MLP. Board-fixed.
struct FlatValue {
    v1: nn::Conv2D,
    vb: nn::BatchNorm,
    vf1: nn::Linear,
    vf2: nn::Linear,
}

/// Go/snake value head: `v1` (1×1 → C) → global pool → MLP. Board-size-agnostic.
struct PoolValue {
    v1: nn::Conv2D,
    vb: nn::BatchNorm,
    vf1: nn::Linear,
    vf2: nn::Linear,
}

enum Value {
    Flat(FlatValue),
    Pool(PoolValue),
}

/// The go auxiliary heads: a per-point ownership conv (`o1`, exported in
/// `AZNET1` when present) and a score-margin linear (`sf`, training-only, never
/// exported). Both enrich the shared trunk with denser gradients.
struct GoAux {
    o1: nn::Conv2D,
    sf: nn::Linear,
}

pub struct Net {
    stem_c: nn::Conv2D,
    stem_b: nn::BatchNorm,
    tower: Vec<Block>,
    policy: Policy,
    value: Value,
    aux: Option<GoAux>,
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

        let policy = match cfg.head {
            HeadKind::FlatConv => Policy::Flat(FlatPolicy {
                p1: conv(root / "p1", c, c, 1),
                pb: nn::batch_norm2d(root / "pb", c, Default::default()),
                p2: conv(root / "p2", c, cfg.move_planes(), 1),
                policy_len: cfg.policy_len,
            }),
            HeadKind::GlobalPoolSpatial => Policy::Spatial(SpatialPolicy {
                p1: conv(root / "p1", c, c, 1),
                pb: nn::batch_norm2d(root / "pb", c, Default::default()),
                pgb: nn::linear(root / "pgb", 3 * c, c, Default::default()),
                pfc: conv(root / "pfc", c, 1, 1),
                ppass: nn::linear(root / "ppass", 3 * c, 1, Default::default()),
            }),
            HeadKind::GlobalPoolDense => Policy::Dense(DensePolicy {
                p1: conv(root / "p1", c, c, 1),
                pb: nn::batch_norm2d(root / "pb", c, Default::default()),
                pf1: nn::linear(root / "pf1", 3 * c, c, Default::default()),
                pf2: nn::linear(root / "pf2", c, cfg.policy_len, Default::default()),
            }),
        };

        let value = match cfg.head {
            HeadKind::FlatConv => Value::Flat(FlatValue {
                v1: conv(root / "v1", c, CHESS_VALUE_CHANNELS, 1),
                vb: nn::batch_norm2d(root / "vb", CHESS_VALUE_CHANNELS, Default::default()),
                vf1: nn::linear(
                    root / "vf1",
                    CHESS_VALUE_CHANNELS * cfg.size * cfg.size,
                    CHESS_VALUE_HIDDEN,
                    Default::default(),
                ),
                vf2: nn::linear(root / "vf2", CHESS_VALUE_HIDDEN, 1, Default::default()),
            }),
            HeadKind::GlobalPoolSpatial | HeadKind::GlobalPoolDense => Value::Pool(PoolValue {
                v1: conv(root / "v1", c, c, 1),
                vb: nn::batch_norm2d(root / "vb", c, Default::default()),
                vf1: nn::linear(root / "vf1", 3 * c, POOL_VALUE_HIDDEN, Default::default()),
                vf2: nn::linear(root / "vf2", POOL_VALUE_HIDDEN, 1, Default::default()),
            }),
        };

        let aux = cfg.go_aux.then(|| GoAux {
            o1: conv(root / "o1", c, 1, 1),
            sf: nn::linear(root / "sf", POOL_VALUE_HIDDEN, 1, Default::default()),
        });

        Net {
            stem_c: conv(root / "stem_c", cfg.planes, c, 3),
            stem_b: nn::batch_norm2d(root / "stem_b", c, Default::default()),
            tower,
            policy,
            value,
            aux,
        }
    }

    fn trunk(&self, x: &Tensor, train: bool) -> Tensor {
        let mut t = x.apply(&self.stem_c).apply_t(&self.stem_b, train).relu();
        for b in &self.tower {
            t = b.forward(&t, train);
        }
        t
    }

    fn policy_forward(&self, t: &Tensor, train: bool) -> Tensor {
        match &self.policy {
            Policy::Flat(p) => {
                let h = t.apply(&p.p1).apply_t(&p.pb, train).relu().apply(&p.p2);
                // [B, move_planes, H, W] → square-major [B, H·W, move_planes] → flat.
                h.permute([0, 2, 3, 1]).reshape([-1, p.policy_len])
            }
            Policy::Spatial(p) => {
                let pol = t.apply(&p.p1).apply_t(&p.pb, train).relu();
                let pol_g = global_pool(&pol);
                let pol = (&pol + pol_g.apply(&p.pgb).unsqueeze(-1).unsqueeze(-1)).relu();
                let placement = pol.apply(&p.pfc).flatten(1, -1);
                let pass = pol_g.apply(&p.ppass);
                Tensor::cat(&[placement, pass], 1)
            }
            Policy::Dense(p) => {
                let pol = t.apply(&p.p1).apply_t(&p.pb, train).relu();
                global_pool(&pol).apply(&p.pf1).relu().apply(&p.pf2)
            }
        }
    }

    /// Value (tanh) and, for the go aux config, the score margin (raw, mover's
    /// view). The score head shares the pooled value features.
    fn value_forward(&self, t: &Tensor, train: bool) -> (Tensor, Option<Tensor>) {
        match &self.value {
            Value::Flat(v) => {
                let h = t
                    .apply(&v.v1)
                    .apply_t(&v.vb, train)
                    .relu()
                    .flatten(1, -1)
                    .apply(&v.vf1)
                    .relu();
                (h.apply(&v.vf2).tanh().squeeze_dim(-1), None)
            }
            Value::Pool(v) => {
                let conv = t.apply(&v.v1).apply_t(&v.vb, train).relu();
                let vh = global_pool(&conv).apply(&v.vf1).relu();
                let value = vh.apply(&v.vf2).tanh().squeeze_dim(-1);
                let score = self.aux.as_ref().map(|a| vh.apply(&a.sf).squeeze_dim(-1));
                (value, score)
            }
        }
    }

    /// Per-point ownership in `(-1, 1)` (mover's view); `None` without the aux.
    fn ownership_forward(&self, t: &Tensor) -> Option<Tensor> {
        self.aux
            .as_ref()
            .map(|a| t.apply(&a.o1).flatten(1, -1).tanh())
    }

    /// Policy logits + scalar value. The aux heads (ownership, score) are read
    /// only during training; inference and search need only policy + value.
    pub fn forward(&self, x: &Tensor, train: bool) -> (Tensor, Tensor) {
        let t = self.trunk(x, train);
        let p = self.policy_forward(&t, train);
        let (v, _score) = self.value_forward(&t, train);
        (p, v)
    }

    /// The full training forward: policy, value, and the go aux heads (ownership,
    /// score) when present.
    pub fn forward_train(
        &self,
        x: &Tensor,
        train: bool,
    ) -> (Tensor, Tensor, Option<Tensor>, Option<Tensor>) {
        let t = self.trunk(x, train);
        let p = self.policy_forward(&t, train);
        let (v, score) = self.value_forward(&t, train);
        let own = self.ownership_forward(&t);
        (p, v, own, score)
    }
}

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

/// Load weights for inference/export, tolerating missing training-only heads.
///
/// The ownership (`o1.*`) and score (`sf.*`) heads are training-only auxiliaries
/// — never exported, never read during inference — so checkpoints saved before
/// either existed lack them. Strict `load` is tried first; only an
/// auxiliary-head shortfall is tolerated, so a genuine architecture mismatch
/// still fails loud.
pub fn load_inference_weights(
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

/// A frozen inference copy of the net, optionally fp16. Answers the search's
/// [`EvalRequest`]s in one GPU round trip per batch.
pub struct Infer {
    _vs: nn::VarStore,
    net: Net,
    device: Device,
    kind: Kind,
    planes: i64,
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
            planes: cfg.planes,
            size: cfg.size,
            policy: cfg.policy(),
        }
    }

    /// Loads a checkpoint saved by [`crate::train::Trainer::save`]; `cfg` must
    /// match the checkpoint's architecture.
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
            planes: cfg.planes,
            size: cfg.size,
            policy: cfg.policy(),
        })
    }

    /// The ownership head's mover-view per-point output for one position — used
    /// by `verify_export` to check the exported `o1` head against tch. Panics
    /// without the aux head.
    pub fn ownership(&self, features: &[f32]) -> Vec<f32> {
        tch::no_grad(|| {
            let x = Tensor::from_slice(features)
                .reshape([1, self.planes, self.size, self.size])
                .to_device(self.device)
                .to_kind(self.kind);
            let own = self
                .net
                .ownership_forward(&self.net.trunk(&x, false))
                .expect("net has an ownership head");
            own.reshape([-1])
                .to_kind(Kind::Float)
                .to_device(Device::Cpu)
        })
        .try_into()
        .expect("ownership to vec")
    }

    /// Evaluates a batch of requests in one GPU round trip. Only the legal
    /// (`support`) logits come back from the GPU.
    pub fn forward_batch(&self, reqs: &[EvalRequest]) -> Vec<EvalResult> {
        if reqs.is_empty() {
            return Vec::new();
        }
        // Pad the batch to bucket sizes: libtorch's MPS backend caches a
        // compiled graph per tensor shape, and self-play batch widths vary every
        // cycle — unbucketed, the cache grows without bound until the OS kills
        // the process.
        let bucket = reqs.len().next_multiple_of(256);
        let b = bucket as i64;
        let cells = (self.size * self.size) as usize;
        let plane_len = self.planes as usize * cells;
        let mut planes = vec![0.0f32; bucket * plane_len];
        let mut gather: Vec<i64> = Vec::with_capacity(reqs.len() * 48);
        for (i, r) in reqs.iter().enumerate() {
            debug_assert_eq!(r.features.len(), plane_len);
            planes[i * plane_len..(i + 1) * plane_len].copy_from_slice(&r.features);
            let base = i as i64 * self.policy;
            gather.extend(r.support.iter().map(|&s| base + i64::from(s)));
        }
        // Same shape-bucketing for the index tensor; padding rows point at row 0
        // and their outputs are ignored.
        gather.resize(gather.len().next_multiple_of(4096), 0);
        let (legal_logits, values) = tch::no_grad(|| {
            let x = Tensor::from_slice(&planes)
                .reshape([b, self.planes, self.size, self.size])
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
                softmax(&mut priors);
                EvalResult {
                    priors,
                    value: values[i],
                }
            })
            .collect()
    }
}
