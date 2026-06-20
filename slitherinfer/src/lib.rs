//! Torch-free slither inference, shared by the trainer (`slither-ppo`) and the
//! browser. [`Model`] parses the `SLNET1` export and runs a reference fp32
//! forward — the ground truth the tch export is validated against
//! (`slither-ppo verify-export`) and the net the browser bot plays through.
//!
//! The net is a strided 3-conv CNN over the egocentric semantic grid
//! (`CHANNELS×GRID×GRID`), relu'd, flattened channel-major, concatenated with
//! the scalar vector, through a trunk linear (relu) into three independent
//! heads: `TURN_BUCKETS` turn logits, one boost logit, and a scalar value.
//! Strides `(1, 2, 2)` and same-padding are fixed by the architecture, so the
//! file carries only weights; this module hard-codes the topology to match
//! [`slither_rl`]'s shapes.
//!
//! Plain fp32 loops — built for correctness and wasm portability, not speed.

pub mod obs;

use slither_rl::env::SHAPES;

/// `(c_out, c_in, k, k)` and `(stride, same-padding)` of the three convs, in
/// order. Padding is `(k-1)/2`; with `k=3` that is 1 throughout.
const CONVS: [ConvSpec; 3] = [
    ConvSpec {
        c_in: CHANNELS,
        c_out: 32,
        stride: 1,
    },
    ConvSpec {
        c_in: 32,
        c_out: 64,
        stride: 2,
    },
    ConvSpec {
        c_in: 64,
        c_out: 64,
        stride: 2,
    },
];
const KERNEL: usize = 3;
const HIDDEN: usize = 256;

pub const CHANNELS: usize = SHAPES.grid.0;
pub const GRID: usize = SHAPES.grid.1;
pub const SCALARS: usize = SHAPES.scalars;
pub const TURN_BUCKETS: usize = SHAPES.turn_buckets;

const MAGIC: &[u8; 6] = b"SLNET1";

#[derive(Clone, Copy)]
struct ConvSpec {
    c_in: usize,
    c_out: usize,
    stride: usize,
}

struct Conv {
    /// `[c_out, c_in, k, k]` flattened.
    w: Vec<f32>,
    b: Vec<f32>,
    c_in: usize,
    c_out: usize,
    stride: usize,
}

struct Linear {
    /// `[n_out, n_in]` flattened.
    w: Vec<f32>,
    b: Vec<f32>,
    n_in: usize,
    n_out: usize,
}

pub struct Model {
    convs: Vec<Conv>,
    trunk: Linear,
    turn: Linear,
    boost: Linear,
    value: Linear,
    /// Flattened conv output width (`c_out · h · w` of the last conv), the
    /// trunk's spatial input before the scalars are concatenated.
    conv_flat: usize,
}

/// One forward pass's head outputs: the turn-bucket logits, the single boost
/// logit (pre-sigmoid), and the scalar value (pre-`tanh`; the value head is
/// linear, matching the trainer).
pub struct Forward {
    pub turn_logits: Vec<f32>,
    pub boost_logit: f32,
    pub value: f32,
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn floats(&mut self, n: usize) -> Result<Vec<f32>, String> {
        let bytes = n
            .checked_mul(4)
            .filter(|b| self.data.len() - self.pos >= *b)
            .ok_or_else(|| format!("truncated export at offset {}", self.pos))?;
        let out = self.data[self.pos..self.pos + bytes]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        self.pos += bytes;
        Ok(out)
    }

    fn conv(&mut self, spec: ConvSpec) -> Result<Conv, String> {
        Ok(Conv {
            w: self.floats(spec.c_out * spec.c_in * KERNEL * KERNEL)?,
            b: self.floats(spec.c_out)?,
            c_in: spec.c_in,
            c_out: spec.c_out,
            stride: spec.stride,
        })
    }

    fn linear(&mut self, n_in: usize, n_out: usize) -> Result<Linear, String> {
        Ok(Linear {
            w: self.floats(n_out * n_in)?,
            b: self.floats(n_out)?,
            n_in,
            n_out,
        })
    }
}

/// Spatial output extent of a same-padded conv: `floor((h + 2p - k)/s) + 1`
/// with `p = (k-1)/2`.
const fn out_hw(h: usize, stride: usize) -> usize {
    let pad = (KERNEL - 1) / 2;
    (h + 2 * pad - KERNEL) / stride + 1
}

impl Model {
    pub fn parse(data: &[u8]) -> Result<Model, String> {
        if data.get(..6) != Some(MAGIC.as_slice()) {
            return Err("not a SLNET1 export".into());
        }
        if data.len() < 18 {
            return Err("truncated header".into());
        }
        let u32_at = |i: usize| u32::from_le_bytes(data[i..i + 4].try_into().unwrap()) as usize;
        let (c, g, s) = (u32_at(6), u32_at(10), u32_at(14));
        if c != CHANNELS || g != GRID || s != SCALARS {
            return Err(format!(
                "export dims {c}x{g}x{g}+{s} != built-in {CHANNELS}x{GRID}x{GRID}+{SCALARS}"
            ));
        }

        let mut r = Reader { data, pos: 18 };
        let mut convs = Vec::with_capacity(CONVS.len());
        let mut hw = GRID;
        for spec in CONVS {
            convs.push(r.conv(spec)?);
            hw = out_hw(hw, spec.stride);
        }
        let conv_flat = CONVS[CONVS.len() - 1].c_out * hw * hw;
        let trunk = r.linear(conv_flat + SCALARS, HIDDEN)?;
        let turn = r.linear(HIDDEN, TURN_BUCKETS)?;
        let boost = r.linear(HIDDEN, 1)?;
        let value = r.linear(HIDDEN, 1)?;
        if r.pos != data.len() {
            return Err(format!("{} trailing bytes in export", data.len() - r.pos));
        }
        Ok(Model {
            convs,
            trunk,
            turn,
            boost,
            value,
            conv_flat,
        })
    }

    /// Forward over one observation: `grid` is `CHANNELS·GRID·GRID` flat
    /// (channel-major, row-major within a channel — exactly as
    /// [`slither_rl::obs::Obs::grid`] stores it), `scalars` is `SCALARS` long.
    pub fn forward(&self, grid: &[f32], scalars: &[f32]) -> Forward {
        assert_eq!(grid.len(), CHANNELS * GRID * GRID, "grid length");
        assert_eq!(scalars.len(), SCALARS, "scalars length");

        // Conv stack, relu after each, channel-major storage throughout.
        let mut x = grid.to_vec();
        let mut hw = GRID;
        for conv in &self.convs {
            let next_hw = out_hw(hw, conv.stride);
            x = conv_fwd(conv, &x, hw, next_hw);
            hw = next_hw;
        }
        debug_assert_eq!(x.len(), self.conv_flat);

        // Flatten (already flat, channel-major) + concat scalars → trunk → relu.
        let mut feat = x;
        feat.extend_from_slice(scalars);
        let hidden = linear_fwd(&self.trunk, &feat, true);

        let turn_logits = linear_fwd(&self.turn, &hidden, false);
        let boost_logit = linear_fwd(&self.boost, &hidden, false)[0];
        let value = linear_fwd(&self.value, &hidden, false)[0];
        Forward {
            turn_logits,
            boost_logit,
            value,
        }
    }
}

/// Same-padded strided convolution, channel-major `[c, h·w]` in and out, relu
/// applied. `in_hw`/`out_hw` are the square spatial extents.
fn conv_fwd(conv: &Conv, x: &[f32], in_hw: usize, out_hw: usize) -> Vec<f32> {
    let pad = (KERNEL - 1) / 2;
    let in_area = in_hw * in_hw;
    let out_area = out_hw * out_hw;
    let mut out = vec![0.0f32; conv.c_out * out_area];
    let stride = conv.stride as isize;
    let pad = pad as isize;
    let in_hw_i = in_hw as isize;
    for co in 0..conv.c_out {
        for oy in 0..out_hw {
            for ox in 0..out_hw {
                let mut acc = conv.b[co];
                for ci in 0..conv.c_in {
                    let wbase = (co * conv.c_in + ci) * KERNEL * KERNEL;
                    let xbase = ci * in_area;
                    for ky in 0..KERNEL {
                        let sy = oy as isize * stride - pad + ky as isize;
                        if !(0..in_hw_i).contains(&sy) {
                            continue;
                        }
                        for kx in 0..KERNEL {
                            let sx = ox as isize * stride - pad + kx as isize;
                            if !(0..in_hw_i).contains(&sx) {
                                continue;
                            }
                            let w = conv.w[wbase + ky * KERNEL + kx];
                            acc += w * x[xbase + (sy as usize) * in_hw + sx as usize];
                        }
                    }
                }
                out[co * out_area + oy * out_hw + ox] = acc.max(0.0);
            }
        }
    }
    out
}

fn linear_fwd(l: &Linear, x: &[f32], relu: bool) -> Vec<f32> {
    debug_assert_eq!(x.len(), l.n_in);
    (0..l.n_out)
        .map(|o| {
            let acc = l.b[o]
                + l.w[o * l.n_in..(o + 1) * l.n_in]
                    .iter()
                    .zip(x)
                    .map(|(w, v)| w * v)
                    .sum::<f32>();
            if relu { acc.max(0.0) } else { acc }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `SLNET1` buffer of the right length filled with `fill`.
    fn buf(fill: f32) -> Vec<u8> {
        let mut hw = GRID;
        let mut floats = 0usize;
        for spec in CONVS {
            floats += spec.c_out * spec.c_in * KERNEL * KERNEL + spec.c_out;
            hw = out_hw(hw, spec.stride);
        }
        let conv_flat = CONVS[CONVS.len() - 1].c_out * hw * hw;
        floats += HIDDEN * (conv_flat + SCALARS) + HIDDEN; // trunk
        floats += TURN_BUCKETS * HIDDEN + TURN_BUCKETS; // turn
        floats += HIDDEN + 1; // boost
        floats += HIDDEN + 1; // value
        let mut b = Vec::new();
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&(CHANNELS as u32).to_le_bytes());
        b.extend_from_slice(&(GRID as u32).to_le_bytes());
        b.extend_from_slice(&(SCALARS as u32).to_le_bytes());
        for _ in 0..floats {
            b.extend_from_slice(&fill.to_le_bytes());
        }
        b
    }

    #[test]
    fn parse_consumes_exactly_and_zero_net_is_zero() {
        let model = Model::parse(&buf(0.0)).expect("parse");
        let out = model.forward(&vec![0.5; CHANNELS * GRID * GRID], &[0.1; SCALARS]);
        assert_eq!(out.turn_logits.len(), TURN_BUCKETS);
        assert!(
            out.turn_logits.iter().all(|&l| l == 0.0),
            "zero net → zero turn logits"
        );
        assert_eq!(out.boost_logit, 0.0);
        assert_eq!(out.value, 0.0);
    }

    #[test]
    fn out_hw_matches_torch_conv_arithmetic() {
        // 32 →(s1) 32 →(s2) 16 →(s2) 8, the trainer's CONV_OUT_HW = GRID/4.
        assert_eq!(out_hw(32, 1), 32);
        assert_eq!(out_hw(32, 2), 16);
        assert_eq!(out_hw(16, 2), 8);
        let model = Model::parse(&buf(0.0)).expect("parse");
        assert_eq!(model.conv_flat, 64 * (GRID / 4) * (GRID / 4));
    }

    #[test]
    fn parse_rejects_bad_magic_and_dims() {
        assert!(Model::parse(b"NOPE01\0\0\0\0").is_err());
        let mut data = buf(0.0);
        data.truncate(data.len() - 4);
        assert!(Model::parse(&data).is_err(), "truncated body rejected");
        let mut extra = buf(0.0);
        extra.extend_from_slice(&[0u8; 4]);
        assert!(Model::parse(&extra).is_err(), "trailing bytes rejected");
    }

    /// A single non-trivial conv cell computed by hand against the reference
    /// loop: one input channel, identity-ish weights, verifies indexing + relu.
    #[test]
    fn conv_cell_matches_hand_computation() {
        // A 1-channel 3x3 input, one output channel, stride 1, weights all 1 →
        // each output cell is the same-padded 3x3 sum of its neighborhood.
        let conv = Conv {
            w: vec![1.0; 9],
            b: vec![0.0],
            c_in: 1,
            c_out: 1,
            stride: 1,
        };
        #[rustfmt::skip]
        let x = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let out = conv_fwd(&conv, &x, 3, 3);
        // Center cell (1,1) sees all nine; corner (0,0) sees the top-left 2x2.
        assert_eq!(out[3 + 1], 45.0);
        assert_eq!(out[0], 1.0 + 2.0 + 4.0 + 5.0);
    }
}
