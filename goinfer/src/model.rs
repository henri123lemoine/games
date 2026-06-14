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
        if data.len() < 20 || &data[..8] != b"AZWEBGO2" {
            return Err("not an AZWEBGO2 export".into());
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
        })
    }

    /// Policy logits (`size²+1`, placements then pass) and value for one
    /// position (`PLANES·size²` flat features).
    pub fn forward(&self, planes: &[f32]) -> (Vec<f32>, f32) {
        let area = self.size * self.size;
        debug_assert_eq!(planes.len(), PLANES * area);
        let mut t = conv_fwd(&self.stem, planes, self.size, true);
        for (c1, c2) in &self.tower {
            let y = conv_fwd(c1, &t, self.size, true);
            let mut y = conv_fwd(c2, &y, self.size, false);
            for (yv, tv) in y.iter_mut().zip(&t) {
                *yv = (*yv + *tv).max(0.0);
            }
            t = y;
        }
        // Policy: per-point conv features biased by their global-pool summary;
        // placement logits from a bias-less 1×1 conv, pass from the pool.
        let pol = conv_fwd(&self.p1, &t, self.size, true);
        let pol_g = global_pool(&pol, self.channels, area);
        let bias = linear_fwd(&self.pgb, &pol_g, false);
        let mut pol_biased = pol;
        for ch in 0..self.channels {
            let b = bias[ch];
            for v in &mut pol_biased[ch * area..(ch + 1) * area] {
                *v = (*v + b).max(0.0);
            }
        }
        let placement = conv_fwd(&self.pfc, &pol_biased, self.size, false); // [1, area]
        let pass = linear_fwd(&self.ppass, &pol_g, false);
        let mut logits = placement;
        logits.push(pass[0]);
        // Value: conv → global pool → MLP.
        let v = conv_fwd(&self.v1, &t, self.size, true);
        let v_g = global_pool(&v, self.channels, area);
        let h = linear_fwd(&self.vf1, &v_g, true);
        let out = linear_fwd(&self.vf2, &h, false);
        (logits, out[0].tanh())
    }

    /// Evaluates requests one by one (reference path; no batching).
    pub fn eval(&self, reqs: &[EvalRequest]) -> Vec<EvalResult> {
        reqs.iter()
            .map(|r| {
                let (logits, value) = self.forward(&r.features);
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
