//! Parser and reference forward pass for the `AZWEBGO1` export: BN-folded
//! convs (so every conv carries a bias), the residual tower, and go's
//! conv→linear policy/value heads. Plain fp32 loops — built for correctness
//! and wasm portability, not speed; the browser's WebGPU path and the tch
//! export check (`azgo verify-export`) must agree with this to 1e-3.
//!
//! The policy head is one logit per board point plus the pass (`size²+1`),
//! matching [`go::encode::GoEncoder::action_index`], so no channel-major
//! rearrange is needed (unlike chess's 73-plane head).

use go::encode::PLANES;

use crate::{EvalRequest, EvalResult};

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
    pub p1: Conv,
    pub pf: Linear,
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
}

impl Model {
    pub fn parse(data: &[u8]) -> Result<Model, String> {
        if data.len() < 20 || &data[..8] != b"AZWEBGO1" {
            return Err("not an AZWEBGO1 export".into());
        }
        let u32_at = |i: usize| u32::from_le_bytes(data[i..i + 4].try_into().unwrap()) as usize;
        let (blocks, c, size) = (u32_at(8), u32_at(12), u32_at(16));
        if blocks == 0 || blocks > 64 || c == 0 || c > 1024 || !(2..=25).contains(&size) {
            return Err(format!("implausible architecture {blocks}x{c} size {size}"));
        }
        let area = size * size;
        let policy = area + 1;
        let mut r = Reader { data, pos: 20 };
        let stem = r.conv(PLANES, c, 3)?;
        let mut tower = Vec::new();
        for _ in 0..blocks {
            tower.push((r.conv(c, c, 3)?, r.conv(c, c, 3)?));
        }
        let p1 = r.conv(c, 2, 1)?;
        let pf = Linear {
            w: r.floats(policy * 2 * area)?,
            b: r.floats(policy)?,
            n_in: 2 * area,
            n_out: policy,
        };
        let v1 = r.conv(c, 2, 1)?;
        let vf1 = Linear {
            w: r.floats(128 * 2 * area)?,
            b: r.floats(128)?,
            n_in: 2 * area,
            n_out: 128,
        };
        let vf2 = Linear {
            w: r.floats(128)?,
            b: r.floats(1)?,
            n_in: 128,
            n_out: 1,
        };
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
            pf,
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
        let p = conv_fwd(&self.p1, &t, self.size, true);
        let logits = linear_fwd(&self.pf, &p, false);
        let v = conv_fwd(&self.v1, &t, self.size, true);
        let h = linear_fwd(&self.vf1, &v, true);
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

    /// Builds an `AZWEBGO1` buffer of the right length for the given dims.
    fn buf(blocks: usize, c: usize, size: usize, fill: f32) -> Vec<u8> {
        let area = size * size;
        let floats = c * PLANES * 9
            + c
            + blocks * 2 * (c * c * 9 + c)
            + (2 * c + 2)
            + ((area + 1) * 2 * area + (area + 1))
            + (2 * c + 2)
            + (128 * 2 * area + 128)
            + (128 + 1);
        let mut b = Vec::new();
        b.extend_from_slice(b"AZWEBGO1");
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
