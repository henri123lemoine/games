//! A small dependency-free MLP with manual backprop: a shared two-layer ReLU
//! trunk, a policy head evaluated only on the legal-action subset (softmax
//! masking — illegal logits are never computed), and a tanh value head.
//! `f32` weights, He init (heads scaled down so the initial policy is
//! near-uniform and the value near zero), SGD with momentum and L2 decay,
//! versioned binary save/load with atomic rename.

use std::io;
use std::path::Path;

use game_core::Rng;

use super::rand::normal;

/// One training example: state encoding `x`, the search's visit distribution
/// over the legal actions as sparse `(policy index, probability)` pairs, and
/// the game outcome `z` from the perspective of the player to move at `x`.
#[derive(Clone)]
pub struct Sample {
    pub x: Vec<f32>,
    pub policy: Vec<(usize, f32)>,
    pub z: f32,
}

struct Layout {
    w1: usize,
    b1: usize,
    w2: usize,
    b2: usize,
    wp: usize,
    bp: usize,
    wv: usize,
    bv: usize,
    total: usize,
}

fn layout(input: usize, hidden: usize, policy: usize) -> Layout {
    let w1 = 0;
    let b1 = w1 + hidden * input;
    let w2 = b1 + hidden;
    let b2 = w2 + hidden * hidden;
    let wp = b2 + hidden;
    let bp = wp + policy * hidden;
    let wv = bp + policy;
    let bv = wv + hidden;
    Layout {
        w1,
        b1,
        w2,
        b2,
        wp,
        bp,
        wv,
        bv,
        total: bv + 1,
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn matvec(w: &[f32], x: &[f32], out: &mut [f32]) {
    for (row, o) in w.chunks_exact(x.len()).zip(out.iter_mut()) {
        *o = dot(row, x);
    }
}

struct Forward {
    a1: Vec<f32>,
    a2: Vec<f32>,
    probs: Vec<f32>,
    v: f32,
}

/// First-layer weights transposed to input-major order so a forward pass can
/// visit only the nonzero inputs. Built by [`Mlp::infer_cache`] against the
/// weights at that moment; stale after any parameter change.
pub struct InferCache {
    w1t: Vec<f32>,
}

/// Per-sample backprop intermediates: the error signal entering each layer
/// plus the activations needed to turn them into weight gradients.
struct Deltas {
    /// Indices and values of the nonzero inputs.
    active: Vec<(usize, f32)>,
    a1: Vec<f32>,
    a2: Vec<f32>,
    d1: Vec<f32>,
    d2: Vec<f32>,
    /// Loss gradient per policy-head logit, keyed by policy index.
    dpolicy: Vec<(usize, f32)>,
    dv: f32,
    ce: f32,
    mse: f32,
}

fn cross_entropy(policy: &[(usize, f32)], probs: &[f32]) -> f32 {
    policy
        .iter()
        .zip(probs)
        .map(|(&(_, pi), &q)| {
            if pi > 0.0 {
                -pi * q.max(1e-12).ln()
            } else {
                0.0
            }
        })
        .sum()
}

pub struct Mlp {
    input: usize,
    hidden: usize,
    policy: usize,
    params: Vec<f32>,
}

const MAGIC: &[u8; 8] = b"AZMLP\0\0\0";
const VERSION: u32 = 1;

impl Mlp {
    pub fn new(input: usize, hidden: usize, policy: usize, seed: u64) -> Mlp {
        let l = layout(input, hidden, policy);
        let mut rng = Rng::new(seed);
        let mut params = vec![0.0f32; l.total];
        let he = |fan_in: usize| (2.0 / fan_in as f64).sqrt();
        for (range, std) in [
            (l.w1..l.b1, he(input)),
            (l.w2..l.b2, he(hidden)),
            (l.wp..l.bp, he(hidden) * 0.1),
            (l.wv..l.bv, he(hidden) * 0.1),
        ] {
            for w in &mut params[range] {
                *w = (normal(&mut rng) * std) as f32;
            }
        }
        Mlp {
            input,
            hidden,
            policy,
            params,
        }
    }

    pub fn input_len(&self) -> usize {
        self.input
    }

    pub fn hidden_len(&self) -> usize {
        self.hidden
    }

    pub fn policy_len(&self) -> usize {
        self.policy
    }

    pub fn params(&self) -> &[f32] {
        &self.params
    }

    pub fn params_mut(&mut self) -> &mut [f32] {
        &mut self.params
    }

    fn l(&self) -> Layout {
        layout(self.input, self.hidden, self.policy)
    }

    /// Policy probabilities over `support` (softmax restricted to those
    /// indices) and the tanh value, for the encoded state `x`.
    pub fn policy_value(&self, x: &[f32], support: &[usize]) -> (Vec<f32>, f32) {
        let f = self.forward(x, support);
        (f.probs, f.v)
    }

    /// [`Mlp::policy_value`] skipping the zero entries of `x` in the first
    /// layer via `cache` (board encodings are mostly zeros). Results match
    /// the dense path exactly.
    pub fn policy_value_cached(
        &self,
        cache: &InferCache,
        x: &[f32],
        support: &[usize],
    ) -> (Vec<f32>, f32) {
        let f = self.forward_cached(cache, x, support);
        (f.probs, f.v)
    }

    /// Builds the transposed-first-layer cache for
    /// [`Mlp::policy_value_cached`]. The cache snapshots the current
    /// weights: rebuild it after any parameter change.
    pub fn infer_cache(&self) -> InferCache {
        let l = self.l();
        let w1 = &self.params[l.w1..l.b1];
        let mut w1t = vec![0.0f32; w1.len()];
        for (j, row) in w1.chunks_exact(self.input).enumerate() {
            for (i, &w) in row.iter().enumerate() {
                w1t[i * self.hidden + j] = w;
            }
        }
        InferCache { w1t }
    }

    fn forward(&self, x: &[f32], support: &[usize]) -> Forward {
        debug_assert_eq!(x.len(), self.input);
        let l = self.l();
        let mut a1 = vec![0.0f32; self.hidden];
        matvec(&self.params[l.w1..l.b1], x, &mut a1);
        self.finish_forward(a1, support)
    }

    fn forward_cached(&self, cache: &InferCache, x: &[f32], support: &[usize]) -> Forward {
        debug_assert_eq!(x.len(), self.input);
        debug_assert_eq!(cache.w1t.len(), self.hidden * self.input);
        let mut a1 = vec![0.0f32; self.hidden];
        for (i, &xi) in x.iter().enumerate() {
            if xi != 0.0 {
                let row = &cache.w1t[i * self.hidden..][..self.hidden];
                for (a, &w) in a1.iter_mut().zip(row) {
                    *a += xi * w;
                }
            }
        }
        self.finish_forward(a1, support)
    }

    /// Shared tail of the forward pass; `a1` holds the first-layer
    /// pre-activations without bias.
    fn finish_forward(&self, mut a1: Vec<f32>, support: &[usize]) -> Forward {
        let l = self.l();
        let p = &self.params;
        for (a, &b) in a1.iter_mut().zip(&p[l.b1..l.w2]) {
            *a = (*a + b).max(0.0);
        }
        let mut a2 = vec![0.0f32; self.hidden];
        matvec(&p[l.w2..l.b2], &a1, &mut a2);
        for (a, &b) in a2.iter_mut().zip(&p[l.b2..l.wp]) {
            *a = (*a + b).max(0.0);
        }
        let mut probs: Vec<f32> = support
            .iter()
            .map(|&i| {
                debug_assert!(i < self.policy);
                dot(&p[l.wp + i * self.hidden..][..self.hidden], &a2) + p[l.bp + i]
            })
            .collect();
        let max = probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for q in &mut probs {
            *q = (*q - max).exp();
            sum += *q;
        }
        for q in &mut probs {
            *q /= sum;
        }
        let v = (dot(&p[l.wv..l.bv], &a2) + p[l.bv]).tanh();
        Forward { a1, a2, probs, v }
    }

    /// Mean (policy cross-entropy, value MSE) over the batch.
    pub fn loss(&self, batch: &[&Sample]) -> (f32, f32) {
        let mut ce = 0.0;
        let mut mse = 0.0;
        for s in batch {
            let support: Vec<usize> = s.policy.iter().map(|&(i, _)| i).collect();
            let f = self.forward(&s.x, &support);
            ce += cross_entropy(&s.policy, &f.probs);
            mse += (f.v - s.z) * (f.v - s.z);
        }
        let k = batch.len().max(1) as f32;
        (ce / k, mse / k)
    }

    /// Mean gradient of (cross-entropy + MSE) over the batch, written into
    /// `grad` (resized to the parameter count). Returns the mean losses.
    /// L2 is applied separately by [`SgdMomentum::step`].
    pub fn grad(&self, batch: &[&Sample], grad: &mut Vec<f32>) -> (f32, f32) {
        grad.clear();
        grad.resize(self.params.len(), 0.0);
        let mut ce = 0.0;
        let mut mse = 0.0;
        for s in batch {
            let d = self.deltas(s, None);
            self.accumulate(&d, grad);
            ce += d.ce;
            mse += d.mse;
        }
        self.finish_grad(grad, batch.len(), ce, mse)
    }

    /// [`Mlp::grad`] with the per-sample backward passes computed in
    /// parallel. Accumulation stays sequential in batch order, so the result
    /// matches `grad` exactly.
    #[cfg(feature = "parallel")]
    pub fn grad_par(&self, batch: &[&Sample], grad: &mut Vec<f32>) -> (f32, f32) {
        use rayon::prelude::*;
        let cache = self.infer_cache();
        let deltas: Vec<Deltas> = batch
            .par_iter()
            .map(|s| self.deltas(s, Some(&cache)))
            .collect();
        if grad.len() != self.params.len() {
            grad.clear();
            grad.resize(self.params.len(), 0.0);
        } else {
            grad.par_chunks_mut(1 << 16).for_each(|c| c.fill(0.0));
        }
        let mut ce = 0.0;
        let mut mse = 0.0;
        for d in &deltas {
            self.accumulate(d, grad);
            ce += d.ce;
            mse += d.mse;
        }
        self.finish_grad(grad, batch.len(), ce, mse)
    }

    fn finish_grad(&self, grad: &mut [f32], batch_len: usize, ce: f32, mse: f32) -> (f32, f32) {
        let scale = 1.0 / batch_len.max(1) as f32;
        for g in grad.iter_mut() {
            *g *= scale;
        }
        (ce * scale, mse * scale)
    }

    fn deltas(&self, s: &Sample, cache: Option<&InferCache>) -> Deltas {
        let l = self.l();
        let h = self.hidden;
        let p = &self.params;
        let support: Vec<usize> = s.policy.iter().map(|&(i, _)| i).collect();
        let f = match cache {
            Some(c) => self.forward_cached(c, &s.x, &support),
            None => self.forward(&s.x, &support),
        };
        let ce = cross_entropy(&s.policy, &f.probs);
        let mse = (f.v - s.z) * (f.v - s.z);

        let mut da2 = vec![0.0f32; h];
        let mut dpolicy = Vec::with_capacity(s.policy.len());
        for (&(idx, pi), &q) in s.policy.iter().zip(&f.probs) {
            let dl = q - pi;
            dpolicy.push((idx, dl));
            let wrow = &p[l.wp + idx * h..][..h];
            for (da, &w) in da2.iter_mut().zip(wrow) {
                *da += dl * w;
            }
        }
        let dv = 2.0 * (f.v - s.z) * (1.0 - f.v * f.v);
        for (da, &w) in da2.iter_mut().zip(&p[l.wv..l.bv]) {
            *da += dv * w;
        }

        let mut d2 = vec![0.0f32; h];
        let mut da1 = vec![0.0f32; h];
        for (j, (&a2j, &d)) in f.a2.iter().zip(&da2).enumerate() {
            if a2j <= 0.0 {
                continue;
            }
            d2[j] = d;
            let wrow = &p[l.w2 + j * h..][..h];
            for (da, &w) in da1.iter_mut().zip(wrow) {
                *da += d * w;
            }
        }
        let mut d1 = vec![0.0f32; h];
        for (j, (&a1j, &d)) in f.a1.iter().zip(&da1).enumerate() {
            if a1j > 0.0 {
                d1[j] = d;
            }
        }
        let active =
            s.x.iter()
                .enumerate()
                .filter(|&(_, &v)| v != 0.0)
                .map(|(i, &v)| (i, v))
                .collect();

        Deltas {
            active,
            a1: f.a1,
            a2: f.a2,
            d1,
            d2,
            dpolicy,
            dv,
            ce,
            mse,
        }
    }

    fn accumulate(&self, d: &Deltas, grad: &mut [f32]) {
        let l = self.l();
        let h = self.hidden;
        for &(idx, dl) in &d.dpolicy {
            grad[l.bp + idx] += dl;
            let grow = &mut grad[l.wp + idx * h..][..h];
            for (g, &a) in grow.iter_mut().zip(&d.a2) {
                *g += dl * a;
            }
        }
        grad[l.bv] += d.dv;
        for (g, &a) in grad[l.wv..l.bv].iter_mut().zip(&d.a2) {
            *g += d.dv * a;
        }
        for (j, &dj) in d.d2.iter().enumerate() {
            if dj == 0.0 {
                continue;
            }
            grad[l.b2 + j] += dj;
            let grow = &mut grad[l.w2 + j * h..][..h];
            for (g, &a) in grow.iter_mut().zip(&d.a1) {
                *g += dj * a;
            }
        }
        for (j, &dj) in d.d1.iter().enumerate() {
            if dj == 0.0 {
                continue;
            }
            grad[l.b1 + j] += dj;
            let grow = &mut grad[l.w1 + j * self.input..][..self.input];
            for &(i, xv) in &d.active {
                grow[i] += dj * xv;
            }
        }
    }

    /// The versioned checkpoint encoding behind [`Mlp::save`].
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(24 + self.params.len() * 4);
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        for d in [self.input, self.hidden, self.policy] {
            buf.extend_from_slice(&(d as u32).to_le_bytes());
        }
        for w in &self.params {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        buf
    }

    /// Parses a checkpoint produced by [`Mlp::to_bytes`] / [`Mlp::save`].
    pub fn from_bytes(data: &[u8]) -> io::Result<Mlp> {
        let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());
        if data.len() < 24 {
            return Err(bad("truncated header"));
        }
        if &data[..8] != MAGIC {
            return Err(bad("not an azero MLP checkpoint"));
        }
        let u32_at = |i: usize| u32::from_le_bytes(data[i..i + 4].try_into().unwrap()) as usize;
        if u32_at(8) != VERSION as usize {
            return Err(bad("unsupported checkpoint version"));
        }
        let (input, hidden, policy) = (u32_at(12), u32_at(16), u32_at(20));
        let l = layout(input, hidden, policy);
        let body = &data[24..];
        if body.len() != l.total * 4 {
            return Err(bad("parameter count does not match dimensions"));
        }
        let params = body
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok(Mlp {
            input,
            hidden,
            policy,
            params,
        })
    }

    /// Writes a versioned binary checkpoint via a temp file + atomic rename.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        if let Some(dir) = path.parent()
            && !dir.as_os_str().is_empty()
        {
            std::fs::create_dir_all(dir)?;
        }
        let buf = self.to_bytes();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("checkpoint");
        let tmp = path.with_file_name(format!("{name}.tmp"));
        std::fs::write(&tmp, &buf)?;
        std::fs::rename(&tmp, path)
    }

    pub fn load(path: &Path) -> io::Result<Mlp> {
        let data = std::fs::read(path)?;
        Self::from_bytes(&data).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: {e}", path.display()),
            )
        })
    }
}

/// SGD with classical momentum and classical (coupled) L2 weight decay:
/// `l2 * w` is folded into the gradient *before* the momentum update, not
/// applied decoupled-AdamW-style.
pub struct SgdMomentum {
    pub lr: f32,
    pub momentum: f32,
    pub l2: f32,
    vel: Vec<f32>,
}

impl SgdMomentum {
    pub fn new(lr: f32, momentum: f32, l2: f32) -> Self {
        Self {
            lr,
            momentum,
            l2,
            vel: Vec::new(),
        }
    }

    pub fn step(&mut self, net: &mut Mlp, grad: &[f32]) {
        if self.vel.len() != grad.len() {
            self.vel = vec![0.0; grad.len()];
        }
        let (lr, momentum, l2) = (self.lr, self.momentum, self.l2);
        let update = move |ws: &mut [f32], vs: &mut [f32], gs: &[f32]| {
            for ((w, v), &g) in ws.iter_mut().zip(vs.iter_mut()).zip(gs) {
                let g = g + l2 * *w;
                *v = momentum * *v - lr * g;
                *w += *v;
            }
        };
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            const CHUNK: usize = 1 << 16;
            net.params
                .par_chunks_mut(CHUNK)
                .zip(self.vel.par_chunks_mut(CHUNK))
                .zip(grad.par_chunks(CHUNK))
                .for_each(|((ws, vs), gs)| update(ws, vs, gs));
        }
        #[cfg(not(feature = "parallel"))]
        update(&mut net.params, &mut self.vel, grad);
    }
}
