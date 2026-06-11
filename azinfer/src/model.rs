//! Parser and reference forward pass for the `AZWEB001` export: BN-folded
//! convs (so every conv has a bias), the residual tower, the 73-plane
//! policy head (logits ordered `square·73 + plane`, matching
//! [`chess::encode::az_move_index`]) and the tanh value head. Plain fp32
//! loops — built for correctness and wasm portability, not speed; the
//! browser's WebGPU path must agree with this to ~1e-4.

use chess::encode::{AZ_POLICY_LEN, PLANE_COUNT};

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
    pub stem: Conv,
    /// Per block: (c1, c2).
    pub tower: Vec<(Conv, Conv)>,
    pub p1: Conv,
    pub p2: Conv,
    pub v1: Conv,
    pub vf1: Linear,
    pub vf2: Linear,
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn floats(&mut self, n: usize) -> Result<Vec<f32>, String> {
        let bytes = n * 4;
        if self.pos + bytes > self.data.len() {
            return Err(format!("truncated export at offset {}", self.pos));
        }
        let out = self.data[self.pos..self.pos + bytes]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        self.pos += bytes;
        Ok(out)
    }
}

impl Model {
    pub fn parse(data: &[u8]) -> Result<Model, String> {
        if data.len() < 16 || &data[..8] != b"AZWEB001" {
            return Err("not an AZWEB001 export".into());
        }
        let u32_at =
            |i: usize| u32::from_le_bytes(data[i..i + 4].try_into().unwrap()) as usize;
        let (blocks, c) = (u32_at(8), u32_at(12));
        let mut r = Reader { data, pos: 16 };
        let mut conv = |c_in: usize, c_out: usize, k: usize| -> Result<Conv, String> {
            Ok(Conv {
                w: r.floats(c_out * c_in * k * k)?,
                b: r.floats(c_out)?,
                c_in,
                c_out,
                k,
            })
        };
        let stem = conv(PLANE_COUNT, c, 3)?;
        let mut tower = Vec::new();
        for _ in 0..blocks {
            tower.push((conv(c, c, 3)?, conv(c, c, 3)?));
        }
        let p1 = conv(c, c, 1)?;
        let p2 = conv(c, 73, 1)?;
        let v1 = conv(c, 8, 1)?;
        let vf1 = Linear {
            w: r.floats(256 * 8 * 64)?,
            b: r.floats(256)?,
            n_in: 8 * 64,
            n_out: 256,
        };
        let vf2 = Linear {
            w: r.floats(256)?,
            b: r.floats(1)?,
            n_in: 256,
            n_out: 1,
        };
        if r.pos != data.len() {
            return Err(format!("{} trailing bytes in export", data.len() - r.pos));
        }
        Ok(Model {
            blocks,
            channels: c,
            stem,
            tower,
            p1,
            p2,
            v1,
            vf1,
            vf2,
        })
    }

    /// Full logits (`square·73 + plane`) and value for one position.
    pub fn forward(&self, planes: &[f32]) -> (Vec<f32>, f32) {
        debug_assert_eq!(planes.len(), PLANE_COUNT * 64);
        let mut t = conv_fwd(&self.stem, planes, true);
        for (c1, c2) in &self.tower {
            let y = conv_fwd(c1, &t, true);
            let mut y = conv_fwd(c2, &y, false);
            for (yv, tv) in y.iter_mut().zip(&t) {
                *yv = (*yv + *tv).max(0.0);
            }
            t = y;
        }
        let p = conv_fwd(&self.p1, &t, true);
        let p = conv_fwd(&self.p2, &p, false);
        // [73, 64] channel-major → square-major logits.
        let mut logits = vec![0.0f32; AZ_POLICY_LEN];
        for plane in 0..73 {
            for sq in 0..64 {
                logits[sq * 73 + plane] = p[plane * 64 + sq];
            }
        }
        let v = conv_fwd(&self.v1, &t, true);
        let h = linear_fwd(&self.vf1, &v, true);
        let out = linear_fwd(&self.vf2, &h, false);
        (logits, out[0].tanh())
    }

    /// Evaluates requests one by one (reference path; no batching).
    pub fn eval(&self, reqs: &[EvalRequest]) -> Vec<EvalResult> {
        reqs.iter()
            .map(|r| {
                let (logits, value) = self.forward(&r.planes);
                let mut priors: Vec<f32> = r
                    .support
                    .iter()
                    .map(|&s| logits[usize::from(s)])
                    .collect();
                let max = priors.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for q in &mut priors {
                    *q = (*q - max).exp();
                    sum += *q;
                }
                for q in &mut priors {
                    *q /= sum;
                }
                EvalResult { priors, value }
            })
            .collect()
    }
}

/// 8×8 same-padding convolution, channel-major `[c, 64]` layout.
fn conv_fwd(conv: &Conv, x: &[f32], relu: bool) -> Vec<f32> {
    let mut out = vec![0.0f32; conv.c_out * 64];
    let k = conv.k as isize;
    let half = k / 2;
    for co in 0..conv.c_out {
        for y in 0..8isize {
            for xx in 0..8isize {
                let mut acc = conv.b[co];
                for ci in 0..conv.c_in {
                    let wbase = ((co * conv.c_in + ci) * conv.k * conv.k) as isize;
                    for dy in -half..=half {
                        let sy = y + dy;
                        if !(0..8).contains(&sy) {
                            continue;
                        }
                        for dx in -half..=half {
                            let sx = xx + dx;
                            if !(0..8).contains(&sx) {
                                continue;
                            }
                            let w = conv.w
                                [(wbase + (dy + half) * k + (dx + half)) as usize];
                            acc += w * x[ci * 64 + (sy * 8 + sx) as usize];
                        }
                    }
                }
                let v = if relu { acc.max(0.0) } else { acc };
                out[co * 64 + (y * 8 + xx) as usize] = v;
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
