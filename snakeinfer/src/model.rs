//! Parser and reference forward pass for the `AZSNK1` export: BN-folded convs
//! (so every conv carries a bias), the residual tower, and the global-pooling
//! policy/value heads. Plain fp32 loops — built for correctness and wasm
//! portability, not speed; the tch export check (`azsnake verify-export`) must
//! agree with this to 1e-3.
//!
//! The policy head is one logit per absolute heading (`Up/Right/Down/Left`),
//! matching [`snake::encode::SnakeEncoder::action_index`]. Global pooling
//! collapses the `[C,H,W]` trunk to `[3C]` (mean, board-size-scaled mean, max),
//! matching the trainer's `global_pool`. Both heads are board-size-agnostic.

use snake::encode::PLANES;

use crate::{EvalRequest, EvalResult};

/// The four absolute headings the policy head scores (Up/Right/Down/Left).
const ACTIONS: usize = 4;
/// `19.0` centers the global-pool size-scale; matches the trainer.
const POOL_SIZE_REF: f32 = 19.0;

const MAGIC: &[u8; 6] = b"AZSNK1";

pub struct Conv {
    /// `[c_out, c_in, k, k]` flattened, k ∈ {1, 3}.
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub c_in: usize,
    pub c_out: usize,
    pub k: usize,
}

pub struct Linear {
    /// `[out, in]` flattened.
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub n_in: usize,
    pub n_out: usize,
}

pub struct Model {
    pub blocks: usize,
    pub channels: usize,
    pub size: usize,
    pub stem: Conv,
    /// Per block: (c1, c2).
    pub tower: Vec<(Conv, Conv)>,
    // Policy head.
    pub p1: Conv,
    pub pf1: Linear,
    pub pf2: Linear,
    // Value head.
    pub v1: Conv,
    pub vf1: Linear,
    pub vf2: Linear,
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

    fn conv(&mut self, c_in: usize, c_out: usize, k: usize) -> Result<Conv, String> {
        Ok(Conv {
            w: self.floats(c_out * c_in * k * k)?,
            b: self.floats(c_out)?,
            c_in,
            c_out,
            k,
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

impl Model {
    pub fn parse(data: &[u8]) -> Result<Model, String> {
        if data.get(..6) != Some(MAGIC.as_slice()) {
            return Err("not an AZSNK1 export".into());
        }
        if data.len() < 18 {
            return Err("truncated header".into());
        }
        let u32_at = |i: usize| u32::from_le_bytes(data[i..i + 4].try_into().unwrap()) as usize;
        let (blocks, c, size) = (u32_at(6), u32_at(10), u32_at(14));
        if blocks == 0 || blocks > 64 || c == 0 || c > 1024 || !(2..=64).contains(&size) {
            return Err(format!("implausible architecture {blocks}x{c} size {size}"));
        }
        let mut r = Reader { data, pos: 18 };
        let stem = r.conv(PLANES, c, 3)?;
        let mut tower = Vec::with_capacity(blocks);
        for _ in 0..blocks {
            tower.push((r.conv(c, c, 3)?, r.conv(c, c, 3)?));
        }
        // Policy head: p1 conv (+BN folded) → global pool → MLP (3C→C, C→4).
        let p1 = r.conv(c, c, 1)?;
        let pf1 = r.linear(3 * c, c)?;
        let pf2 = r.linear(c, ACTIONS)?;
        // Value head: v1 conv (+BN folded) → global pool → MLP (3C→128, 128→1).
        let v1 = r.conv(c, c, 1)?;
        let vf1 = r.linear(3 * c, 128)?;
        let vf2 = r.linear(128, 1)?;
        if r.pos != data.len() {
            return Err(format!("{} trailing bytes in export", data.len() - r.pos));
        }
        Ok(Model {
            blocks,
            channels: c,
            size,
            stem,
            tower,
            p1,
            pf1,
            pf2,
            v1,
            vf1,
            vf2,
        })
    }

    /// Policy logits (the four absolute headings) and value for one position at
    /// the export's stored board size.
    pub fn forward(&self, planes: &[f32]) -> (Vec<f32>, f32) {
        self.forward_at(planes, self.size)
    }

    /// Forward at an arbitrary board `size` (`PLANES·size²` flat features). The
    /// global-pooling net is board-size-agnostic — the convolution weights are
    /// shared across sizes and only the spatial extent changes.
    pub fn forward_at(&self, planes: &[f32], size: usize) -> (Vec<f32>, f32) {
        self.forward_with(planes, size, &mut Scratch::default())
    }

    /// Like [`forward_at`](Self::forward_at) but reuses caller-owned scratch
    /// buffers, so a search that evaluates thousands of leaves allocates the
    /// conv working set once instead of per leaf.
    fn forward_with(&self, planes: &[f32], size: usize, sc: &mut Scratch) -> (Vec<f32>, f32) {
        let area = size * size;
        self.trunk(planes, size, sc);
        // Policy: conv → global pool → MLP to the four heading logits.
        conv_fwd(&self.p1, &sc.trunk, size, true, &mut sc.head);
        let pol_g = global_pool(&sc.head, self.channels, area);
        let h = linear_fwd(&self.pf1, &pol_g, true);
        let logits = linear_fwd(&self.pf2, &h, false);
        // Value: conv → global pool → MLP → tanh.
        conv_fwd(&self.v1, &sc.trunk, size, true, &mut sc.head);
        let v_g = global_pool(&sc.head, self.channels, area);
        let vh = linear_fwd(&self.vf1, &v_g, true);
        let out = linear_fwd(&self.vf2, &vh, false);
        (logits, out[0].tanh())
    }

    /// Computes the residual-tower output `[channels, size²]` shared by both
    /// heads, leaving it in `sc.trunk`.
    fn trunk(&self, planes: &[f32], size: usize, sc: &mut Scratch) {
        debug_assert_eq!(planes.len(), PLANES * size * size);
        let Scratch {
            trunk,
            skip,
            tmp,
            head: _,
        } = sc;
        conv_fwd(&self.stem, planes, size, true, trunk);
        for (c1, c2) in &self.tower {
            std::mem::swap(trunk, skip);
            conv_fwd(c1, skip, size, true, tmp);
            conv_fwd(c2, tmp, size, false, trunk);
            for (yv, tv) in trunk.iter_mut().zip(skip.iter()) {
                *yv = (*yv + *tv).max(0.0);
            }
        }
    }

    /// Evaluates requests one by one (reference path; no batching). Each
    /// request's board size is read from its feature length. Conv scratch is
    /// allocated once and reused across the batch.
    pub fn eval(&self, reqs: &[EvalRequest]) -> Vec<EvalResult> {
        let mut sc = Scratch::default();
        reqs.iter()
            .map(|r| {
                let size = isqrt_planes(r.features.len());
                let (logits, value) = self.forward_with(&r.features, size, &mut sc);
                let mut priors: Vec<f32> =
                    r.support.iter().map(|&s| logits[usize::from(s)]).collect();
                crate::softmax(&mut priors);
                EvalResult { priors, value }
            })
            .collect()
    }
}

/// Conv working buffers reused across a forward (and across a whole batch in
/// [`Model::eval`]). `conv_fwd` resizes them as needed, so the default-empty
/// state is a valid start.
#[derive(Default)]
struct Scratch {
    trunk: Vec<f32>,
    skip: Vec<f32>,
    tmp: Vec<f32>,
    head: Vec<f32>,
}

/// `size`×`size` same-padding convolution, channel-major `[c, area]` layout,
/// writing into `out` (resized to `c_out·area`).
///
/// Formulated as a sum of shifted, weight-scaled input planes: each output
/// plane starts at its bias, then every (input channel, kernel tap) contributes
/// a contiguous row-wise `out += w · shift(x)` over the in-bounds region. The
/// inner loop is a branch-free unit-stride axpy — autovectorized, and lowered
/// to `f32x4` under `+simd128` on wasm. The per-output-cell summation order
/// (bias, then input channel, then kernel row, then kernel column) is identical
/// to the textbook nested-loop reference, so results are bit-for-bit unchanged.
fn conv_fwd(conv: &Conv, x: &[f32], size: usize, relu: bool, out: &mut Vec<f32>) {
    let area = size * size;
    out.clear();
    out.resize(conv.c_out * area, 0.0);
    let k = conv.k;
    let half = (k / 2) as isize;
    let s = size as isize;
    for co in 0..conv.c_out {
        let out_plane = &mut out[co * area..(co + 1) * area];
        out_plane.fill(conv.b[co]);
        for ci in 0..conv.c_in {
            let wbase = (co * conv.c_in + ci) * k * k;
            let in_plane = &x[ci * area..(ci + 1) * area];
            for ky in 0..k {
                let dy = ky as isize - half;
                // Output rows whose source row `y + dy` is in bounds.
                let y0 = (-dy).max(0);
                let y1 = (s - dy).min(s);
                for kx in 0..k {
                    let w = conv.w[wbase + ky * k + kx];
                    if w == 0.0 {
                        continue;
                    }
                    let dx = kx as isize - half;
                    // Output columns whose source column `x + dx` is in bounds.
                    let x0 = (-dx).max(0);
                    let x1 = (s - dx).min(s);
                    if x0 >= x1 {
                        continue;
                    }
                    let span = (x1 - x0) as usize;
                    for y in y0..y1 {
                        let o = (y * s + x0) as usize;
                        let i = ((y + dy) * s + (x0 + dx)) as usize;
                        let dst = &mut out_plane[o..o + span];
                        let src = &in_plane[i..i + span];
                        axpy(w, src, dst);
                    }
                }
            }
        }
        if relu {
            for v in out_plane.iter_mut() {
                *v = v.max(0.0);
            }
        }
    }
}

/// `dst[i] += w · src[i]`. A fused multiply-add over equal-length unit-stride
/// slices; the compiler vectorizes the slice-zip into `f32x4` fma lanes.
#[inline(always)]
fn axpy(w: f32, src: &[f32], dst: &mut [f32]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d += w * s;
    }
}

/// Global pooling: channel-major `[c, area]` → `[3c]` = per-channel mean, then
/// board-size-scaled mean, then max. Mirrors the trainer's `global_pool`.
fn global_pool(x: &[f32], c: usize, area: usize) -> Vec<f32> {
    let scale = (area as f32).sqrt() / POOL_SIZE_REF;
    let mut out = vec![0.0f32; 3 * c];
    for ch in 0..c {
        let plane = &x[ch * area..(ch + 1) * area];
        let mut sum = 0.0f32;
        let mut mx = f32::NEG_INFINITY;
        for &v in plane {
            sum += v;
            mx = mx.max(v);
        }
        let mean = sum / area as f32;
        out[ch] = mean;
        out[c + ch] = mean * scale;
        out[2 * c + ch] = mx;
    }
    out
}

/// Board size from a flat feature length (`PLANES·size²`).
fn isqrt_planes(features_len: usize) -> usize {
    (features_len / PLANES).isqrt()
}

fn linear_fwd(l: &Linear, x: &[f32], relu: bool) -> Vec<f32> {
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

    /// Builds an `AZSNK1` buffer of the right length for the given dims.
    fn buf(blocks: usize, c: usize, size: usize, fill: f32) -> Vec<u8> {
        let floats = c * PLANES * 9 + c              // stem
            + blocks * 2 * (c * c * 9 + c)           // tower
            + (c * c + c)                            // p1 (1×1, folded)
            + (c * 3 * c + c)                        // pf1 3c→c
            + (ACTIONS * c + ACTIONS)                // pf2 c→4
            + (c * c + c)                            // v1 (1×1, folded)
            + (128 * 3 * c + 128)                    // vf1 3c→128
            + (128 + 1); // vf2 128→1
        let mut b = Vec::new();
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&(blocks as u32).to_le_bytes());
        b.extend_from_slice(&(c as u32).to_le_bytes());
        b.extend_from_slice(&(size as u32).to_le_bytes());
        for _ in 0..floats {
            b.extend_from_slice(&fill.to_le_bytes());
        }
        b
    }

    #[test]
    fn parse_consumes_exactly_and_zero_net_is_uniform() {
        for (blocks, c, size) in [(1usize, 4usize, 5usize), (4, 8, 20), (2, 16, 11)] {
            let data = buf(blocks, c, size, 0.0);
            let model = Model::parse(&data).expect("parse");
            assert_eq!(
                (model.blocks, model.channels, model.size),
                (blocks, c, size)
            );
            let (logits, value) = model.forward(&vec![0.5; PLANES * size * size]);
            assert_eq!(logits.len(), ACTIONS, "policy = four headings");
            assert!(logits.iter().all(|&l| l == 0.0), "zero net → zero logits");
            assert_eq!(value, 0.0, "zero net → tanh(0)");
        }
    }

    #[test]
    fn forward_is_size_agnostic_for_the_global_pool_heads() {
        let model = Model::parse(&buf(2, 8, 20, 0.03)).expect("parse");
        let planes = vec![0.25f32; PLANES * 20 * 20];
        assert_eq!(model.forward(&planes), model.forward_at(&planes, 20));
        let (logits10, value10) = model.forward_at(&vec![0.1f32; PLANES * 100], 10);
        assert_eq!(logits10.len(), ACTIONS);
        assert!(logits10.iter().all(|x| x.is_finite()) && value10.is_finite());
    }

    #[test]
    fn eval_softmaxes_the_support() {
        let model = Model::parse(&buf(1, 6, 20, 0.02)).expect("parse");
        let req = EvalRequest {
            features: vec![0.3; PLANES * 400],
            support: vec![0, 1, 2, 3],
        };
        let res = &model.eval(std::slice::from_ref(&req))[0];
        let sum: f32 = res.priors.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "priors sum to 1: {sum}");
        assert!(res.value.abs() <= 1.0 && res.value.is_finite());
    }

    #[test]
    fn parse_rejects_truncated_and_wrong_magic() {
        assert!(Model::parse(b"NOPE01\0\0\0\0\0\0").is_err());
        let mut data = buf(1, 4, 5, 0.0);
        data.truncate(data.len() - 8);
        assert!(Model::parse(&data).is_err(), "truncated body rejected");
        let mut extra = buf(1, 4, 5, 0.0);
        extra.extend_from_slice(&[0u8; 4]);
        assert!(Model::parse(&extra).is_err(), "trailing bytes rejected");
    }

    /// A single non-trivial conv cell against a hand computation: one input
    /// channel, all-ones 3×3 weights, stride-1 same-padding → each output cell
    /// is the same-padded 3×3 sum of its neighborhood.
    #[test]
    fn conv_cell_matches_hand_computation() {
        let conv = Conv {
            w: vec![1.0; 9],
            b: vec![0.0],
            c_in: 1,
            c_out: 1,
            k: 3,
        };
        #[rustfmt::skip]
        let x = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let mut out = Vec::new();
        conv_fwd(&conv, &x, 3, false, &mut out);
        assert_eq!(out[3 + 1], 45.0, "center sees all nine");
        assert_eq!(out[0], 1.0 + 2.0 + 4.0 + 5.0, "corner sees the 2x2");
    }
}
