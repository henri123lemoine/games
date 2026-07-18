//! Torch-free generic conv-resnet forward + the unified versioned `AZNET1`
//! weight format. One reference fp32 forward — a 3×3 conv stem, a residual
//! tower of BN-folded `(conv, conv)` pairs, then a config-selected policy/value
//! head — for every exported AlphaZero net (chess, go, snake). Plain loops,
//! built for correctness and wasm portability; the browser's WebGPU path and the
//! trainer's `verify-export` check validate against it.
//!
//! The head topology is data, not a game identity: [`HeadKind::FlatConv`]
//! (chess), [`HeadKind::GlobalPoolSpatial`] (go, `size²+1` policy + optional
//! ownership), and [`HeadKind::GlobalPoolDense`] (snake, fixed action set). The
//! global-pool heads are board-size-agnostic — the same conv weights run at any
//! `size`. slither's strided non-resnet CNN is a different architecture and is
//! not parsed here; it shares only the low-level conv/linear primitives.
//!
//! [`stratego`] is the second architecture family: the `ATRX1` transformer-pair
//! format and forward for the stratego move/setup nets, sharing the dense
//! [`Linear`] primitive with the conv-resnet path.

pub mod format;
pub mod math;
pub mod stratego;

pub use format::{Arch, Conv, HeadFlags, HeadKind, Linear, Reader};
pub use stratego::{MoveOutput, StrategoNet};

use math::{POOL_SIZE_REF, conv_fwd, conv_fwd_vec, global_pool, linear_fwd};

/// In-place softmax: logits → distribution. The same reduction every `*infer`
/// crate used to restrict priors to the legal support.
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

/// The flat policy head (chess): `p1` (1×1 C→C, relu) → `p2` (1×1 C→planes, no
/// relu), then the channel-major `[planes, area]` block is transposed to
/// square-major `logits[sq·planes + plane]`.
struct FlatPolicy {
    p1: Conv,
    p2: Conv,
    move_planes: usize,
}

/// The go policy head: `p1` (1×1 C→C, relu) biased per channel by a global-pool
/// linear (`pgb`, 3C→C), a bias-less placement conv (`pfc`, C→1), and a pooled
/// pass logit (`ppass`, 3C→1). One logit per board point plus pass.
struct SpatialPolicy {
    p1: Conv,
    pgb: Linear,
    pfc: Conv,
    ppass: Linear,
}

/// The snake policy head: `p1` (1×1 C→C, relu) → global pool → MLP
/// (`pf1` 3C→C relu, `pf2` C→actions).
struct DensePolicy {
    p1: Conv,
    pf1: Linear,
    pf2: Linear,
}

enum Policy {
    Flat(FlatPolicy),
    Spatial(SpatialPolicy),
    Dense(DensePolicy),
}

/// The chess value head: `v1` (1×1 C→`vc`, relu) flattened over the full board
/// to a dense MLP (`vf1` `vc·area`→256 relu, `vf2` 256→1), then tanh.
struct FlatValue {
    v1: Conv,
    vf1: Linear,
    vf2: Linear,
}

/// The go/snake value head: `v1` (1×1 C→C, relu) → global pool → MLP
/// (`vf1` 3C→128 relu, `vf2` 128→1), then tanh.
struct PoolValue {
    v1: Conv,
    vf1: Linear,
    vf2: Linear,
}

enum Value {
    Flat(FlatValue),
    Pool(PoolValue),
}

/// A parsed AZNET1 net: the architecture header, the BN-folded trunk, the
/// config-selected heads, and (go GO3) an optional ownership conv.
pub struct Net {
    arch: Arch,
    stem: Conv,
    tower: Vec<(Conv, Conv)>,
    policy: Policy,
    value: Value,
    ownership: Option<Conv>,
}

/// One forward pass: policy logits (head-dependent width), the value in
/// `(-1, 1)`, and the per-point ownership when the net carries that head.
pub struct Output {
    pub policy: Vec<f32>,
    pub value: f32,
    pub ownership: Option<Vec<f32>>,
}

impl Net {
    pub fn arch(&self) -> &Arch {
        &self.arch
    }

    /// Parses an `AZNET1` export: header, then the BN-folded weight stream in
    /// the fixed layer order, rejecting any trailing bytes.
    pub fn parse(data: &[u8]) -> Result<Net, String> {
        let (arch, body) = Arch::parse(data)?;
        let c = arch.channels;
        let mut r = Reader::new(data, body);

        let stem = r.conv(arch.planes, c, 3)?;
        let mut tower = Vec::with_capacity(arch.blocks);
        for _ in 0..arch.blocks {
            tower.push((r.conv(c, c, 3)?, r.conv(c, c, 3)?));
        }

        let policy = match arch.head {
            HeadKind::FlatConv => {
                // `policy_len` is square-major `area·move_planes`; the stored
                // conv is C→move_planes over the board.
                let move_planes = arch.policy_len / (arch.size * arch.size);
                Policy::Flat(FlatPolicy {
                    p1: r.conv(c, c, 1)?,
                    p2: r.conv(c, move_planes, 1)?,
                    move_planes,
                })
            }
            HeadKind::GlobalPoolSpatial => Policy::Spatial(SpatialPolicy {
                p1: r.conv(c, c, 1)?,
                pgb: r.linear(3 * c, c)?,
                pfc: r.conv_nobias(c, 1, 1)?,
                ppass: r.linear(3 * c, 1)?,
            }),
            HeadKind::GlobalPoolDense => Policy::Dense(DensePolicy {
                p1: r.conv(c, c, 1)?,
                pf1: r.linear(3 * c, c)?,
                pf2: r.linear(c, arch.policy_len)?,
            }),
        };

        let value = match arch.head {
            HeadKind::FlatConv => {
                let vc = CHESS_VALUE_CHANNELS;
                Value::Flat(FlatValue {
                    v1: r.conv(c, vc, 1)?,
                    vf1: r.linear(vc * arch.size * arch.size, CHESS_VALUE_HIDDEN)?,
                    vf2: r.linear(CHESS_VALUE_HIDDEN, 1)?,
                })
            }
            HeadKind::GlobalPoolSpatial | HeadKind::GlobalPoolDense => Value::Pool(PoolValue {
                v1: r.conv(c, c, 1)?,
                vf1: r.linear(3 * c, POOL_VALUE_HIDDEN)?,
                vf2: r.linear(POOL_VALUE_HIDDEN, 1)?,
            }),
        };

        let ownership = arch
            .flags
            .ownership()
            .then(|| r.conv_nobias(c, 1, 1))
            .transpose()?;

        r.finish()?;
        Ok(Net {
            arch,
            stem,
            tower,
            policy,
            value,
            ownership,
        })
    }

    /// Forward at the export's stored board size. For the global-pool heads this
    /// equals [`forward_at`](Self::forward_at) at `arch.size`; the flat head is
    /// board-fixed and ignores any other size.
    pub fn forward(&self, planes: &[f32], scalars: &[f32]) -> Output {
        self.forward_at(planes, scalars, self.arch.size)
    }

    /// Forward at an arbitrary board `size` (`planes·size²` flat features). Only
    /// the global-pool heads are size-agnostic; the flat (chess) head is fixed
    /// at `arch.size` and `size` is ignored there.
    pub fn forward_at(&self, planes: &[f32], scalars: &[f32], size: usize) -> Output {
        debug_assert_eq!(scalars.len(), self.arch.scalars, "scalar side-input width");
        let size = match self.arch.head {
            HeadKind::FlatConv => self.arch.size,
            _ => size,
        };
        let area = size * size;
        debug_assert_eq!(planes.len(), self.arch.planes * area, "feature width");

        let trunk = self.trunk(planes, size);
        let policy = self.policy_forward(&trunk, size, area);
        let value = self.value_forward(&trunk, size, area);
        let ownership = self.ownership.as_ref().map(|o1| {
            let mut o = conv_fwd_vec(o1, &trunk, size, false);
            for v in &mut o {
                *v = v.tanh();
            }
            o
        });
        Output {
            policy,
            value,
            ownership,
        }
    }

    /// The PUCT leaf-eval bridge: forward over `planes` (size inferred from the
    /// feature length for the size-agnostic heads), restrict the policy to the
    /// legal `support` indices, and softmax that subset — returning the
    /// priors-over-support and the value an `azero::Search` leaf wants. Mirrors
    /// the old `Model::eval` per request, so a search driven by this is identical
    /// to one driven by the per-game forwards. `scalars` is empty for the AZ games.
    pub fn forward_support(
        &self,
        planes: &[f32],
        scalars: &[f32],
        support: &[u16],
    ) -> (Vec<f32>, f32) {
        let size = self.infer_size(planes.len());
        let out = self.forward_at(planes, scalars, size);
        let mut priors: Vec<f32> = support
            .iter()
            .map(|&s| out.policy[usize::from(s)])
            .collect();
        softmax(&mut priors);
        (priors, out.value)
    }

    /// Board size implied by a flat feature length (`planes·size²`). For the
    /// board-fixed flat head this is always `arch.size`.
    fn infer_size(&self, features_len: usize) -> usize {
        match self.arch.head {
            HeadKind::FlatConv => self.arch.size,
            _ => (features_len / self.arch.planes).isqrt(),
        }
    }

    /// Residual-tower output `[channels, area]` shared by every head.
    fn trunk(&self, planes: &[f32], size: usize) -> Vec<f32> {
        let mut t = conv_fwd_vec(&self.stem, planes, size, true);
        let mut y = Vec::new();
        for (c1, c2) in &self.tower {
            conv_fwd(c1, &t, size, true, &mut y);
            let mut z = Vec::new();
            conv_fwd(c2, &y, size, false, &mut z);
            for (zv, tv) in z.iter_mut().zip(&t) {
                *zv = (*zv + *tv).max(0.0);
            }
            t = z;
        }
        t
    }

    fn policy_forward(&self, trunk: &[f32], size: usize, area: usize) -> Vec<f32> {
        let c = self.arch.channels;
        match &self.policy {
            Policy::Flat(p) => {
                let h = conv_fwd_vec(&p.p1, trunk, size, true);
                let planes = conv_fwd_vec(&p.p2, &h, size, false); // [move_planes, area]
                // Channel-major [move_planes, area] → square-major.
                let mut logits = vec![0.0f32; area * p.move_planes];
                for plane in 0..p.move_planes {
                    for sq in 0..area {
                        logits[sq * p.move_planes + plane] = planes[plane * area + sq];
                    }
                }
                logits
            }
            Policy::Spatial(p) => {
                let pol = conv_fwd_vec(&p.p1, trunk, size, true);
                let pol_g = global_pool(&pol, c, area);
                let bias = linear_fwd(&p.pgb, &pol_g, false);
                let mut biased = pol;
                for ch in 0..c {
                    let b = bias[ch];
                    for v in &mut biased[ch * area..(ch + 1) * area] {
                        *v = (*v + b).max(0.0);
                    }
                }
                let mut logits = conv_fwd_vec(&p.pfc, &biased, size, false); // [1, area]
                let pass = linear_fwd(&p.ppass, &pol_g, false);
                logits.push(pass[0]);
                logits
            }
            Policy::Dense(p) => {
                let pol = conv_fwd_vec(&p.p1, trunk, size, true);
                let pol_g = global_pool(&pol, c, area);
                let h = linear_fwd(&p.pf1, &pol_g, true);
                linear_fwd(&p.pf2, &h, false)
            }
        }
    }

    fn value_forward(&self, trunk: &[f32], size: usize, area: usize) -> f32 {
        let c = self.arch.channels;
        let out = match &self.value {
            Value::Flat(v) => {
                let conv = conv_fwd_vec(&v.v1, trunk, size, true); // [vc, area], flattened
                let h = linear_fwd(&v.vf1, &conv, true);
                linear_fwd(&v.vf2, &h, false)
            }
            Value::Pool(v) => {
                let conv = conv_fwd_vec(&v.v1, trunk, size, true);
                let v_g = global_pool(&conv, c, area);
                let h = linear_fwd(&v.vf1, &v_g, true);
                linear_fwd(&v.vf2, &h, false)
            }
        };
        out[0].tanh()
    }
}

/// Chess value head: `v1` reduces the trunk to this many channels before the
/// dense MLP. Fixed by the chess net's architecture (aztrainer's `net.rs`).
const CHESS_VALUE_CHANNELS: usize = 8;
const CHESS_VALUE_HIDDEN: usize = 256;
/// Go/snake global-pool value head's hidden width.
const POOL_VALUE_HIDDEN: usize = 128;

/// Confirms `POOL_SIZE_REF` is the value every trainer's `global_pool` uses; a
/// drift here silently rescales the size-scaled pooling channel.
const _: () = assert!(POOL_SIZE_REF == 19.0);
