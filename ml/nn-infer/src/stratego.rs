//! The `ATRX1` container and forward pass for the stratego transformer pair —
//! the move net and the setup (arrangement) net exported from an
//! `ml/stratego-trainer` checkpoint (`export_web.py`).
//!
//! Both nets are pre-LayerNorm Transformers with ReLU FFN (`4d` hidden),
//! separate q/k/v/out projections, learned absolute positional embeddings, and
//! `1/sqrt(head_dim)` scaled-dot-product attention. The move net consumes the
//! `(92, 643)` token matrix from `games/stratego`'s `encode_tokens`, prepends a
//! **zero** value-slot row (index 0 — no learned content embedding, only the
//! positional row distinguishes it), and emits a key-query policy grid
//! `grid[src·92 + dst] = q_src · k_dst / sqrt(d)` plus a 3-way categorical
//! value. The setup net is decoder-only causal over the placed-so-far one-hots
//! (a stored start token prepended) and emits next-placement type logits.
//!
//! Scattering the 92x92 grid into the 1800-slot env action space is game
//! semantics and deliberately lives with the action encoding in
//! `games/stratego`, not here.
//!
//! **Body layout.** fp16 little-endian weights (converted to fp32 at parse) in
//! fixed order, move net then setup net. Per trunk: embedder linear, positional
//! encoding, then per layer `ln1, q, k, v, out, ln2, ff1, ff2`, then the final
//! layernorm. The move net appends its policy `q_proj`/`k_proj` and the value
//! head; the setup net stores its start token after the embedder and appends
//! the policy/value/entropy heads. Every linear stores `[out, in]` weights then
//! the bias, matching MLX `Linear` (`y = x·Wᵀ + b`).

use crate::format::Linear;
use crate::math::linear_fwd;

/// `ATRX1\0\0\0`: the stratego transformer-pair magic, padded to 8 bytes.
pub const MAGIC: &[u8; 8] = b"ATRX1\0\0\0";

/// The only format version.
pub const VERSION: u32 = 1;

/// Number of `u32` header fields after the magic.
const HEADER_FIELDS: usize = 12;
/// Total header byte count: the 8-byte magic plus the twelve `u32` fields (56).
pub const HEADER_LEN: usize = MAGIC.len() + HEADER_FIELDS * 4;

/// One trunk's shape, read from the header.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TrunkDims {
    pub depth: usize,
    pub dim: usize,
    pub heads: usize,
    pub in_dim: usize,
    /// Content tokens the trunk attends over. The move net's positional
    /// encoding spans `tokens + 1` rows (the prepended value slot); the setup
    /// net's spans exactly `tokens`.
    pub tokens: usize,
}

impl TrunkDims {
    fn validate(&self, name: &str) -> Result<(), String> {
        if self.depth == 0 || self.depth > 64 {
            return Err(format!("implausible {name} depth {}", self.depth));
        }
        if self.dim == 0
            || self.dim > 4096
            || self.heads == 0
            || !self.dim.is_multiple_of(self.heads)
        {
            return Err(format!(
                "implausible {name} dim/heads {}/{}",
                self.dim, self.heads
            ));
        }
        if self.in_dim == 0 || self.in_dim > 4096 {
            return Err(format!("implausible {name} in_dim {}", self.in_dim));
        }
        if self.tokens == 0 || self.tokens > 1024 {
            return Err(format!("implausible {name} tokens {}", self.tokens));
        }
        Ok(())
    }
}

/// The header: both trunks' shapes, read entirely from the file.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TxArch {
    pub mv: TrunkDims,
    pub setup: TrunkDims,
}

impl TxArch {
    /// Parses and bounds-checks the header, returning it alongside the byte
    /// offset where the weight stream begins.
    pub fn parse(data: &[u8]) -> Result<(TxArch, usize), String> {
        if data.len() < HEADER_LEN {
            return Err("truncated ATRX1 header".into());
        }
        if &data[..MAGIC.len()] != MAGIC.as_slice() {
            return Err("not an ATRX1 export".into());
        }
        let u32_at = |i: usize| {
            let off = MAGIC.len() + i * 4;
            u32::from_le_bytes(data[off..off + 4].try_into().unwrap()) as usize
        };
        if u32_at(0) as u32 != VERSION {
            return Err(format!("unsupported ATRX1 version {}", u32_at(0)));
        }
        let dims = |base: usize| TrunkDims {
            depth: u32_at(base),
            dim: u32_at(base + 1),
            heads: u32_at(base + 2),
            in_dim: u32_at(base + 3),
            tokens: u32_at(base + 4),
        };
        let arch = TxArch {
            mv: dims(1),
            setup: dims(6),
        };
        arch.mv.validate("move")?;
        arch.setup.validate("setup")?;
        if u32_at(11) != 0 {
            return Err(format!("nonzero reserved header word {}", u32_at(11)));
        }
        Ok((arch, HEADER_LEN))
    }

    /// Serializes the 56-byte header (magic + twelve `u32`s).
    pub fn header_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(HEADER_LEN);
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&VERSION.to_le_bytes());
        for t in [&self.mv, &self.setup] {
            for word in [t.depth, t.dim, t.heads, t.in_dim, t.tokens] {
                b.extend_from_slice(&(word as u32).to_le_bytes());
            }
        }
        b.extend_from_slice(&0u32.to_le_bytes());
        b
    }
}

/// IEEE 754 binary16 → binary32, handling subnormals, infinities, and NaN.
#[inline]
fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match (exp, frac) {
        (0, 0) => sign,
        (0, f) => {
            // Subnormal: renormalize the fraction.
            let shift = f.leading_zeros() - 21;
            let f = (f << (shift + 1)) & 0x3ff;
            sign | ((113 - shift) << 23) | (f << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, f) => sign | 0x7f80_0000 | (f << 13),
        (e, f) => sign | ((e + 112) << 23) | (f << 13),
    };
    f32::from_bits(bits)
}

/// Sequential reader over the fp16 weight stream; converts to fp32 as it reads.
struct F16Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> F16Reader<'a> {
    fn floats(&mut self, n: usize) -> Result<Vec<f32>, String> {
        let bytes = n
            .checked_mul(2)
            .filter(|b| self.data.len() - self.pos >= *b)
            .ok_or_else(|| format!("truncated export at offset {}", self.pos))?;
        let out = self.data[self.pos..self.pos + bytes]
            .chunks_exact(2)
            .map(|c| half_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
            .collect();
        self.pos += bytes;
        Ok(out)
    }

    fn linear(&mut self, n_in: usize, n_out: usize) -> Result<Linear, String> {
        Ok(Linear {
            w: self.floats(n_out * n_in)?,
            b: self.floats(n_out)?,
            n_in,
            n_out,
        })
    }

    fn norm(&mut self, dim: usize) -> Result<Norm, String> {
        Ok(Norm {
            w: self.floats(dim)?,
            b: self.floats(dim)?,
        })
    }

    fn finish(&self) -> Result<(), String> {
        if self.pos != self.data.len() {
            return Err(format!(
                "{} trailing bytes in export",
                self.data.len() - self.pos
            ));
        }
        Ok(())
    }
}

/// LayerNorm scale and shift (`eps = 1e-5`, matching MLX).
struct Norm {
    w: Vec<f32>,
    b: Vec<f32>,
}

const NORM_EPS: f32 = 1e-5;

impl Norm {
    /// Normalizes each `dim`-wide row of `x` in place.
    fn apply_rows(&self, x: &mut [f32]) {
        let dim = self.w.len();
        for row in x.chunks_exact_mut(dim) {
            let mean = row.iter().sum::<f32>() / dim as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
            let inv = 1.0 / (var + NORM_EPS).sqrt();
            for (v, (w, b)) in row.iter_mut().zip(self.w.iter().zip(&self.b)) {
                *v = (*v - mean) * inv * w + b;
            }
        }
    }
}

struct TxLayer {
    ln1: Norm,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    ln2: Norm,
    ff1: Linear,
    ff2: Linear,
}

/// One transformer trunk: embedder, positional rows, layers, final norm.
struct Trunk {
    dims: TrunkDims,
    embed: Linear,
    /// `pos_rows · dim` learned positional encoding (`tokens + 1` rows for the
    /// move net's value slot, `tokens` for the setup net).
    pos: Vec<f32>,
    layers: Vec<TxLayer>,
    norm_out: Norm,
}

impl Trunk {
    fn read(r: &mut F16Reader, dims: TrunkDims, pos_rows: usize) -> Result<Trunk, String> {
        let d = dims.dim;
        let embed = r.linear(dims.in_dim, d)?;
        let pos = r.floats(pos_rows * d)?;
        let mut layers = Vec::with_capacity(dims.depth);
        for _ in 0..dims.depth {
            layers.push(TxLayer {
                ln1: r.norm(d)?,
                q: r.linear(d, d)?,
                k: r.linear(d, d)?,
                v: r.linear(d, d)?,
                out: r.linear(d, d)?,
                ln2: r.norm(d)?,
                ff1: r.linear(d, 4 * d)?,
                ff2: r.linear(4 * d, d)?,
            });
        }
        let norm_out = r.norm(d)?;
        Ok(Trunk {
            dims,
            embed,
            pos,
            layers,
            norm_out,
        })
    }

    /// Runs the trunk over `seq` (`rows · dim`, positional encoding already
    /// added), in place. `causal` masks attention to positions `<= i`.
    fn forward(&self, seq: &mut [f32], causal: bool) {
        let d = self.dims.dim;
        let rows = seq.len() / d;
        let heads = self.dims.heads;
        let hd = d / heads;
        let scale = 1.0 / (hd as f32).sqrt();

        let mut normed = vec![0.0f32; seq.len()];
        let mut q = vec![0.0f32; seq.len()];
        let mut k = vec![0.0f32; seq.len()];
        let mut v = vec![0.0f32; seq.len()];
        let mut attn = vec![0.0f32; seq.len()];
        let mut scores = vec![0.0f32; rows];

        for layer in &self.layers {
            normed.copy_from_slice(seq);
            layer.ln1.apply_rows(&mut normed);
            for (i, row) in normed.chunks_exact(d).enumerate() {
                q[i * d..(i + 1) * d].copy_from_slice(&linear_fwd(&layer.q, row, false));
                k[i * d..(i + 1) * d].copy_from_slice(&linear_fwd(&layer.k, row, false));
                v[i * d..(i + 1) * d].copy_from_slice(&linear_fwd(&layer.v, row, false));
            }
            attn.fill(0.0);
            for h in 0..heads {
                let off = h * hd;
                for i in 0..rows {
                    let visible = if causal { i + 1 } else { rows };
                    let qi = &q[i * d + off..i * d + off + hd];
                    let mut max = f32::NEG_INFINITY;
                    for (j, s) in scores[..visible].iter_mut().enumerate() {
                        let kj = &k[j * d + off..j * d + off + hd];
                        *s = qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale;
                        max = max.max(*s);
                    }
                    let mut total = 0.0;
                    for s in scores[..visible].iter_mut() {
                        *s = (*s - max).exp();
                        total += *s;
                    }
                    let out_row = &mut attn[i * d + off..i * d + off + hd];
                    for (j, s) in scores[..visible].iter().enumerate() {
                        let w = s / total;
                        for (o, vv) in out_row.iter_mut().zip(&v[j * d + off..j * d + off + hd]) {
                            *o += w * vv;
                        }
                    }
                }
            }
            for (i, row) in attn.chunks_exact(d).enumerate() {
                let projected = linear_fwd(&layer.out, row, false);
                for (s, p) in seq[i * d..(i + 1) * d].iter_mut().zip(&projected) {
                    *s += p;
                }
            }

            normed.copy_from_slice(seq);
            layer.ln2.apply_rows(&mut normed);
            for (i, row) in normed.chunks_exact(d).enumerate() {
                let hidden = linear_fwd(&layer.ff1, row, true);
                let projected = linear_fwd(&layer.ff2, &hidden, false);
                for (s, p) in seq[i * d..(i + 1) * d].iter_mut().zip(&projected) {
                    *s += p;
                }
            }
        }
        self.norm_out.apply_rows(seq);
    }
}

/// The move net's output: the `92·92` src-dst policy grid and the categorical
/// value, acting-player POV.
pub struct MoveOutput {
    /// `grid[src · tokens + dst]`, `src`/`dst` in reduced (lake-free) cell
    /// order. Scatter into the 1800-slot action space game-side.
    pub grid: Vec<f32>,
    /// `[P(loss), P(tie), P(win)]`.
    pub value_probs: [f32; 3],
    /// Expectation of `value_probs` over `(-1, 0, 1)`.
    pub value: f32,
}

/// The parsed transformer pair.
pub struct StrategoNet {
    arch: TxArch,
    mv: Trunk,
    policy_q: Linear,
    policy_k: Linear,
    value_head: Linear,
    setup: Trunk,
    start_token: Vec<f32>,
    setup_policy: Linear,
}

impl StrategoNet {
    /// Parses a full `ATRX1` export; rejects trailing bytes.
    pub fn parse(data: &[u8]) -> Result<StrategoNet, String> {
        let (arch, body) = TxArch::parse(data)?;
        let mut r = F16Reader { data, pos: body };
        let mv = Trunk::read(&mut r, arch.mv, arch.mv.tokens + 1)?;
        let policy_q = r.linear(arch.mv.dim, arch.mv.dim)?;
        let policy_k = r.linear(arch.mv.dim, arch.mv.dim)?;
        let value_head = r.linear(arch.mv.dim, 3)?;
        let start_token = r.floats(arch.setup.in_dim)?;
        let setup = Trunk::read(&mut r, arch.setup, arch.setup.tokens)?;
        let setup_policy = r.linear(arch.setup.dim, arch.setup.in_dim)?;
        // The value and entropy heads are training-time only; parse (so the
        // no-trailing-bytes check holds) and drop.
        let _ = r.linear(arch.setup.dim, 3)?;
        let _ = r.linear(arch.setup.dim, 1)?;
        r.finish()?;
        Ok(StrategoNet {
            arch,
            mv,
            policy_q,
            policy_k,
            value_head,
            setup,
            start_token,
            setup_policy,
        })
    }

    pub fn arch(&self) -> &TxArch {
        &self.arch
    }

    /// Scores one move decision from the `(tokens, in_dim)` row-major token
    /// matrix (from `encode_tokens`).
    pub fn move_forward(&self, tokens: &[f32]) -> MoveOutput {
        let TrunkDims {
            dim: d,
            tokens: n,
            in_dim,
            ..
        } = self.arch.mv;
        assert_eq!(tokens.len(), n * in_dim, "move token matrix shape");

        // Row 0 is the zero value slot; content rows follow. Positional rows
        // cover all n + 1.
        let rows = n + 1;
        let mut seq = vec![0.0f32; rows * d];
        for (i, tok) in tokens.chunks_exact(in_dim).enumerate() {
            seq[(i + 1) * d..(i + 2) * d].copy_from_slice(&linear_fwd(&self.mv.embed, tok, false));
        }
        for (s, p) in seq.iter_mut().zip(&self.mv.pos) {
            *s += p;
        }
        self.mv.forward(&mut seq, false);

        let value_logits = linear_fwd(&self.value_head, &seq[..d], false);
        let value_probs = softmax3(&value_logits);
        let value = value_probs[2] - value_probs[0];

        let cells = &seq[d..];
        let mut q = vec![0.0f32; n * d];
        let mut k = vec![0.0f32; n * d];
        for (i, row) in cells.chunks_exact(d).enumerate() {
            q[i * d..(i + 1) * d].copy_from_slice(&linear_fwd(&self.policy_q, row, false));
            k[i * d..(i + 1) * d].copy_from_slice(&linear_fwd(&self.policy_k, row, false));
        }
        let scale = 1.0 / (d as f32).sqrt();
        let mut grid = vec![0.0f32; n * n];
        for src in 0..n {
            let qs = &q[src * d..(src + 1) * d];
            for dst in 0..n {
                let kd = &k[dst * d..(dst + 1) * d];
                grid[src * n + dst] = qs.iter().zip(kd).map(|(a, b)| a * b).sum::<f32>() * scale;
            }
        }
        MoveOutput {
            grid,
            value_probs,
            value,
        }
    }

    /// Scores one deployment decision from the `(placed, in_dim)` row-major
    /// one-hot prefix (`deploy_obs` truncated to the placed rows). Returns the
    /// next-placement type logits, unmasked — legality is the caller's.
    pub fn setup_forward(&self, placed: &[f32]) -> Vec<f32> {
        let TrunkDims {
            dim: d,
            tokens,
            in_dim,
            ..
        } = self.arch.setup;
        let n_placed = placed.len() / in_dim;
        assert!(
            placed.len() == n_placed * in_dim && n_placed < tokens,
            "deploy prefix shape"
        );

        let rows = n_placed + 1;
        let mut seq = vec![0.0f32; rows * d];
        seq[..d].copy_from_slice(&linear_fwd(&self.setup.embed, &self.start_token, false));
        for (i, tok) in placed.chunks_exact(in_dim).enumerate() {
            seq[(i + 1) * d..(i + 2) * d].copy_from_slice(&linear_fwd(
                &self.setup.embed,
                tok,
                false,
            ));
        }
        for (s, p) in seq.iter_mut().zip(&self.setup.pos) {
            *s += p;
        }
        self.setup.forward(&mut seq, true);
        linear_fwd(&self.setup_policy, &seq[(rows - 1) * d..], false)
    }
}

fn softmax3(logits: &[f32]) -> [f32; 3] {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let total: f32 = exp.iter().sum();
    [exp[0] / total, exp[1] / total, exp[2] / total]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_to_half(v: f32) -> u16 {
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32;
        let frac = bits & 0x7f_ffff;
        if exp == 0xff {
            return sign | 0x7c00 | ((frac >> 13) as u16) | u16::from(frac != 0);
        }
        let e = exp - 127 + 15;
        if e >= 0x1f {
            return sign | 0x7c00;
        }
        if e <= 0 {
            if e < -10 {
                return sign;
            }
            let f = (frac | 0x80_0000) >> (1 - e + 13);
            return sign | f as u16;
        }
        sign | ((e as u16) << 10) | ((frac >> 13) as u16)
    }

    fn arch() -> TxArch {
        TxArch {
            mv: TrunkDims {
                depth: 2,
                dim: 8,
                heads: 2,
                in_dim: 5,
                tokens: 4,
            },
            setup: TrunkDims {
                depth: 1,
                dim: 6,
                heads: 2,
                in_dim: 3,
                tokens: 4,
            },
        }
    }

    /// Deterministic small weights: a low-discrepancy walk in `[-0.5, 0.5)`.
    fn tiny_export(a: &TxArch) -> Vec<u8> {
        let mut bytes = a.header_bytes();
        let mut x = 0.0f64;
        let mut push = |bytes: &mut Vec<u8>, n: usize| {
            for _ in 0..n {
                x = (x + 0.754_877_666_246_693) % 1.0;
                bytes.extend_from_slice(&f32_to_half((x - 0.5) as f32).to_le_bytes());
            }
        };
        let trunk = |bytes: &mut Vec<u8>,
                     push: &mut dyn FnMut(&mut Vec<u8>, usize),
                     t: &TrunkDims,
                     pos_rows: usize| {
            let d = t.dim;
            push(bytes, d * t.in_dim + d);
            push(bytes, pos_rows * d);
            for _ in 0..t.depth {
                push(bytes, 2 * d); // ln1
                for _ in 0..4 {
                    push(bytes, d * d + d); // q, k, v, out
                }
                push(bytes, 2 * d); // ln2
                push(bytes, 4 * d * d + 4 * d); // ff1
                push(bytes, 4 * d * d + d); // ff2
            }
            push(bytes, 2 * d); // norm_out
        };
        trunk(&mut bytes, &mut push, &a.mv, a.mv.tokens + 1);
        push(&mut bytes, a.mv.dim * a.mv.dim + a.mv.dim); // policy q
        push(&mut bytes, a.mv.dim * a.mv.dim + a.mv.dim); // policy k
        push(&mut bytes, 3 * a.mv.dim + 3); // value head
        push(&mut bytes, a.setup.in_dim); // start token
        trunk(&mut bytes, &mut push, &a.setup, a.setup.tokens);
        push(&mut bytes, a.setup.in_dim * a.setup.dim + a.setup.in_dim); // policy
        push(&mut bytes, 3 * a.setup.dim + 3); // value
        push(&mut bytes, a.setup.dim + 1); // entropy
        bytes
    }

    #[test]
    fn header_round_trips_through_parse() {
        let a = arch();
        let bytes = a.header_bytes();
        let (parsed, body) = TxArch::parse(&bytes).expect("parse");
        assert_eq!(parsed, a);
        assert_eq!(body, bytes.len());
    }

    #[test]
    fn rejects_wrong_magic_version_and_reserved() {
        let a = arch();
        let mut bad = a.header_bytes();
        bad[0] = b'X';
        assert!(TxArch::parse(&bad).is_err(), "wrong magic rejected");
        let mut bad = a.header_bytes();
        bad[MAGIC.len()] = 2;
        assert!(TxArch::parse(&bad).is_err(), "unknown version rejected");
        let mut bad = a.header_bytes();
        let off = HEADER_LEN - 4;
        bad[off..].copy_from_slice(&7u32.to_le_bytes());
        assert!(TxArch::parse(&bad).is_err(), "nonzero reserved rejected");
    }

    #[test]
    fn rejects_indivisible_heads() {
        let mut a = arch();
        a.mv.heads = 3;
        assert!(TxArch::parse(&a.header_bytes()).is_err());
    }

    #[test]
    fn parse_consumes_exactly_and_rejects_trailing_bytes() {
        let a = arch();
        let bytes = tiny_export(&a);
        StrategoNet::parse(&bytes).expect("full export parses");
        assert!(
            StrategoNet::parse(&bytes[..bytes.len() - 2]).is_err(),
            "truncated body rejected"
        );
        let mut extra = bytes.clone();
        extra.extend_from_slice(&[0, 0]);
        assert!(
            StrategoNet::parse(&extra).is_err(),
            "trailing bytes rejected"
        );
    }

    #[test]
    fn half_conversion_round_trips_key_values() {
        for v in [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            0.5,
            65504.0,
            6.1035156e-5,
            3.0517578e-5,
        ] {
            assert_eq!(half_to_f32(f32_to_half(v)), v, "exact half value {v}");
        }
        assert!(half_to_f32(0x7c00).is_infinite());
        assert!(half_to_f32(0x7e00).is_nan());
        // Subnormal: smallest positive half.
        assert_eq!(half_to_f32(0x0001), 5.9604645e-8);
    }

    #[test]
    fn move_forward_shapes_and_value_normalization() {
        let a = arch();
        let net = StrategoNet::parse(&tiny_export(&a)).unwrap();
        let tokens = vec![0.25f32; a.mv.tokens * a.mv.in_dim];
        let out = net.move_forward(&tokens);
        assert_eq!(out.grid.len(), a.mv.tokens * a.mv.tokens);
        let total: f32 = out.value_probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-5, "value probs sum to 1");
        assert!(out.value.abs() <= 1.0);
        assert!(out.grid.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn setup_forward_prefix_drives_the_logits() {
        let a = arch();
        let net = StrategoNet::parse(&tiny_export(&a)).unwrap();
        let empty = net.setup_forward(&[]);
        assert_eq!(empty.len(), a.setup.in_dim);
        let one = vec![0.5f32; a.setup.in_dim];
        let with_one = net.setup_forward(&one);
        assert_eq!(with_one.len(), a.setup.in_dim);
        assert!(with_one.iter().all(|v| v.is_finite()));
        assert_ne!(empty, with_one, "prefix changes the next-type logits");
    }
}
