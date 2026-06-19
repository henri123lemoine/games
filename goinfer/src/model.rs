//! Parser and reference forward pass for the `AZWEBGO2` export: BN-folded
//! convs (so every conv carries a bias), the residual tower, and go's
//! global-pooling policy/value heads. Plain fp32 loops — built for correctness
//! and wasm portability, not speed; the browser's WebGPU path and the tch
//! export check (`azgo verify-export`) must agree with this to 1e-3.
//!
//! The policy head is one logit per board point plus the pass (`size²+1`),
//! matching [`go::encode::GoEncoder::action_index`]. Global pooling collapses
//! the `[C,H,W]` trunk to `[3C]` (mean, board-size-scaled mean, max), matching
//! the trainer's `global_pool`.

use go::encode::PLANES;

use crate::{EvalRequest, EvalResult};

/// `19.0` centers the global-pool size-scale; matches the trainer.
const POOL_SIZE_REF: f32 = 19.0;

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
    pub pgb: Linear,
    pub pfc: Conv,
    pub ppass: Linear,
    // Value head.
    pub v1: Conv,
    pub vf1: Linear,
    pub vf2: Linear,
    /// `AZWEBGO3` only: the `o1` ownership head (1×1 `C`→1, no bias), applied to
    /// the trunk then `tanh`, mover's-view per point. `None` for `AZWEBGO2`.
    pub ownership: Option<Conv>,
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

    /// A conv with no stored bias (the bias-less placement conv); bias = 0.
    fn conv_nobias(&mut self, c_in: usize, c_out: usize, k: usize) -> Result<Conv, String> {
        Ok(Conv {
            w: self.floats(c_out * c_in * k * k)?,
            b: vec![0.0; c_out],
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
        let with_ownership = match data.get(..8) {
            Some(b"AZWEBGO3") => true,
            Some(b"AZWEBGO2") => false,
            _ => return Err("not an AZWEBGO2/AZWEBGO3 export".into()),
        };
        if data.len() < 20 {
            return Err("truncated header".into());
        }
        let u32_at = |i: usize| u32::from_le_bytes(data[i..i + 4].try_into().unwrap()) as usize;
        let (blocks, c, size) = (u32_at(8), u32_at(12), u32_at(16));
        if blocks == 0 || blocks > 64 || c == 0 || c > 1024 || !(2..=25).contains(&size) {
            return Err(format!("implausible architecture {blocks}x{c} size {size}"));
        }
        let mut r = Reader { data, pos: 20 };
        let stem = r.conv(PLANES, c, 3)?;
        let mut tower = Vec::new();
        for _ in 0..blocks {
            tower.push((r.conv(c, c, 3)?, r.conv(c, c, 3)?));
        }
        // Policy head: p1 conv (+BN), pool-bias linear (3C→C), placement conv
        // (C→1, no bias), pass linear (3C→1).
        let p1 = r.conv(c, c, 1)?;
        let pgb = r.linear(3 * c, c)?;
        let pfc = r.conv_nobias(c, 1, 1)?;
        let ppass = r.linear(3 * c, 1)?;
        // Value head: v1 conv (+BN) → global pool → MLP.
        let v1 = r.conv(c, c, 1)?;
        let vf1 = r.linear(3 * c, 128)?;
        let vf2 = r.linear(128, 1)?;
        let ownership = with_ownership.then(|| r.conv_nobias(c, 1, 1)).transpose()?;
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
            pgb,
            pfc,
            ppass,
            v1,
            vf1,
            vf2,
            ownership,
        })
    }

    /// Policy logits (`size²+1`, placements then pass) and value for one
    /// position at the export's stored board size.
    pub fn forward(&self, planes: &[f32]) -> (Vec<f32>, f32) {
        self.forward_at(planes, self.size)
    }

    /// Forward at an arbitrary board `size` (`PLANES·size²` flat features). The
    /// global-pooling net is board-size-agnostic — the convolution weights are
    /// shared across sizes and only the spatial extent changes — so this serves
    /// any `size ≤ self.size`, matching the WebGPU path's per-call sizing.
    pub fn forward_at(&self, planes: &[f32], size: usize) -> (Vec<f32>, f32) {
        let area = size * size;
        let t = self.trunk(planes, size);
        // Policy: per-point conv features biased by their global-pool summary;
        // placement logits from a bias-less 1×1 conv, pass from the pool.
        let pol = conv_fwd(&self.p1, &t, size, true);
        let pol_g = global_pool(&pol, self.channels, area);
        let bias = linear_fwd(&self.pgb, &pol_g, false);
        let mut pol_biased = pol;
        for ch in 0..self.channels {
            let b = bias[ch];
            for v in &mut pol_biased[ch * area..(ch + 1) * area] {
                *v = (*v + b).max(0.0);
            }
        }
        let placement = conv_fwd(&self.pfc, &pol_biased, size, false); // [1, area]
        let pass = linear_fwd(&self.ppass, &pol_g, false);
        let mut logits = placement;
        logits.push(pass[0]);
        // Value: conv → global pool → MLP.
        let v = conv_fwd(&self.v1, &t, size, true);
        let v_g = global_pool(&v, self.channels, area);
        let h = linear_fwd(&self.vf1, &v_g, true);
        let out = linear_fwd(&self.vf2, &h, false);
        (logits, out[0].tanh())
    }

    /// The residual-tower output `[channels, size²]` shared by every head.
    fn trunk(&self, planes: &[f32], size: usize) -> Vec<f32> {
        debug_assert_eq!(planes.len(), PLANES * size * size);
        let mut t = conv_fwd(&self.stem, planes, size, true);
        for (c1, c2) in &self.tower {
            let y = conv_fwd(c1, &t, size, true);
            let mut y = conv_fwd(c2, &y, size, false);
            for (yv, tv) in y.iter_mut().zip(&t) {
                *yv = (*yv + *tv).max(0.0);
            }
            t = y;
        }
        t
    }

    /// Per-point ownership in `(-1, 1)` from the mover's view (`+1` ≈ the side
    /// to move ends up owning the point), or `None` when the export carries no
    /// ownership head (`AZWEBGO2`).
    pub fn ownership_at(&self, planes: &[f32], size: usize) -> Option<Vec<f32>> {
        let o1 = self.ownership.as_ref()?;
        let t = self.trunk(planes, size);
        let mut o = conv_fwd(o1, &t, size, false);
        for v in &mut o {
            *v = v.tanh();
        }
        Some(o)
    }

    /// Evaluates requests one by one (reference path; no batching). Each
    /// request's board size is read from its feature length, so a batch may
    /// mix sizes — and play at 9×9 uses the same 19×19-trained weights.
    pub fn eval(&self, reqs: &[EvalRequest]) -> Vec<EvalResult> {
        reqs.iter()
            .map(|r| {
                let size = isqrt_planes(r.features.len());
                let (logits, value) = self.forward_at(&r.features, size);
                let mut priors: Vec<f32> =
                    r.support.iter().map(|&s| logits[usize::from(s)]).collect();
                crate::softmax(&mut priors);
                EvalResult { priors, value }
            })
            .collect()
    }
}

/// `size`×`size` same-padding convolution, channel-major `[c, area]` layout.
fn conv_fwd(conv: &Conv, x: &[f32], size: usize, relu: bool) -> Vec<f32> {
    let s = size as isize;
    let area = size * size;
    let mut out = vec![0.0f32; conv.c_out * area];
    let k = conv.k as isize;
    let half = k / 2;
    for co in 0..conv.c_out {
        for y in 0..s {
            for xx in 0..s {
                let mut acc = conv.b[co];
                for ci in 0..conv.c_in {
                    let wbase = ((co * conv.c_in + ci) * conv.k * conv.k) as isize;
                    for dy in -half..=half {
                        let sy = y + dy;
                        if !(0..s).contains(&sy) {
                            continue;
                        }
                        for dx in -half..=half {
                            let sx = xx + dx;
                            if !(0..s).contains(&sx) {
                                continue;
                            }
                            let w = conv.w[(wbase + (dy + half) * k + (dx + half)) as usize];
                            acc += w * x[ci * area + (sy * s + sx) as usize];
                        }
                    }
                }
                let v = if relu { acc.max(0.0) } else { acc };
                out[co * area + (y * s + xx) as usize] = v;
            }
        }
    }
    out
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

    /// Builds an `AZWEBGO2` buffer of the right length for the given dims.
    fn buf(blocks: usize, c: usize, size: usize, fill: f32) -> Vec<u8> {
        let floats = c * PLANES * 9 + c                  // stem
            + blocks * 2 * (c * c * 9 + c)               // tower
            + (c * c + c)                                // p1 (1×1, folded)
            + (c * 3 * c + c)                            // pgb 3c→c
            + c                                          // pfc c→1 (no bias)
            + (3 * c + 1)                                // ppass 3c→1
            + (c * c + c)                                // v1 (1×1, folded)
            + (128 * 3 * c + 128)                        // vf1 3c→128
            + (128 + 1); // vf2 128→1
        let mut b = Vec::new();
        b.extend_from_slice(b"AZWEBGO2");
        b.extend_from_slice(&(blocks as u32).to_le_bytes());
        b.extend_from_slice(&(c as u32).to_le_bytes());
        b.extend_from_slice(&(size as u32).to_le_bytes());
        for _ in 0..floats {
            b.extend_from_slice(&fill.to_le_bytes());
        }
        b
    }

    /// An `AZWEBGO2` buffer with the `o1` ownership head (`c` floats) appended
    /// and the magic bumped — a synthetic `AZWEBGO3`.
    fn buf3(blocks: usize, c: usize, size: usize, fill: f32, o1: f32) -> Vec<u8> {
        let mut b = buf(blocks, c, size, fill);
        b[..8].copy_from_slice(b"AZWEBGO3");
        for _ in 0..c {
            b.extend_from_slice(&o1.to_le_bytes());
        }
        b
    }

    #[test]
    fn azwebgo2_has_no_ownership_head() {
        let model = Model::parse(&buf(2, 8, 9, 0.1)).expect("parse");
        assert!(model.ownership.is_none());
        assert!(model.ownership_at(&vec![0.5; PLANES * 81], 9).is_none());
    }

    #[test]
    fn azwebgo3_ownership_head_round_trips() {
        let model = Model::parse(&buf3(2, 8, 9, 0.0, 0.0)).expect("parse");
        let own = model
            .ownership_at(&vec![0.5; PLANES * 81], 9)
            .expect("ownership");
        assert_eq!(own.len(), 81);
        assert!(
            own.iter().all(|&o| o == 0.0),
            "zero net → tanh(0) ownership"
        );
    }

    #[test]
    fn ownership_is_tanh_of_o1_over_the_trunk() {
        let model = Model::parse(&buf3(2, 6, 9, 0.05, 0.2)).expect("parse");
        let planes: Vec<f32> = (0..PLANES * 81).map(|i| (i % 5) as f32 * 0.1).collect();
        let t = model.trunk(&planes, 9);
        let o1 = model.ownership.as_ref().unwrap();
        let expected: Vec<f32> = conv_fwd(o1, &t, 9, false)
            .iter()
            .map(|x| x.tanh())
            .collect();
        let got = model.ownership_at(&planes, 9).unwrap();
        assert_eq!(got, expected);
        assert!(got.iter().all(|o| o.abs() <= 1.0 && o.is_finite()));
    }

    #[test]
    fn parse_consumes_exactly_and_zero_net_is_uniform() {
        for (blocks, c, size) in [(1usize, 4usize, 5usize), (3, 8, 9), (2, 16, 19)] {
            let data = buf(blocks, c, size, 0.0);
            let model = Model::parse(&data).expect("parse");
            assert_eq!(
                (model.blocks, model.channels, model.size),
                (blocks, c, size)
            );
            let (logits, value) = model.forward(&vec![0.5; PLANES * size * size]);
            assert_eq!(logits.len(), size * size + 1);
            assert!(logits.iter().all(|&l| l == 0.0), "zero net → zero logits");
            assert_eq!(value, 0.0, "zero net → tanh(0)");
        }
    }

    #[test]
    fn parse_rejects_truncated_and_wrong_magic() {
        assert!(Model::parse(b"AZWEB001\0\0\0\0").is_err());
        let mut data = buf(1, 4, 5, 0.0);
        data.truncate(data.len() - 8);
        assert!(Model::parse(&data).is_err(), "truncated body rejected");
        let mut extra = buf(1, 4, 5, 0.0);
        extra.extend_from_slice(&[0u8; 4]);
        assert!(Model::parse(&extra).is_err(), "trailing bytes rejected");
    }
}
