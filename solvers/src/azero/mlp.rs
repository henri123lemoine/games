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

    fn forward(&self, x: &[f32], support: &[usize]) -> Forward {
        debug_assert_eq!(x.len(), self.input);
        let l = self.l();
        let p = &self.params;
        let mut a1 = vec![0.0f32; self.hidden];
        matvec(&p[l.w1..l.b1], x, &mut a1);
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
            let (c, m) = self.backprop(s, grad);
            ce += c;
            mse += m;
        }
        let scale = 1.0 / batch.len().max(1) as f32;
        for g in grad.iter_mut() {
            *g *= scale;
        }
        (ce * scale, mse * scale)
    }

    fn backprop(&self, s: &Sample, grad: &mut [f32]) -> (f32, f32) {
        let l = self.l();
        let h = self.hidden;
        let p = &self.params;
        let support: Vec<usize> = s.policy.iter().map(|&(i, _)| i).collect();
        let f = self.forward(&s.x, &support);
        let ce = cross_entropy(&s.policy, &f.probs);
        let mse = (f.v - s.z) * (f.v - s.z);

        let mut da2 = vec![0.0f32; h];
        for (&(idx, pi), &q) in s.policy.iter().zip(&f.probs) {
            let dl = q - pi;
            grad[l.bp + idx] += dl;
            let wrow = &p[l.wp + idx * h..][..h];
            let grow = &mut grad[l.wp + idx * h..][..h];
            for (((g, da), &w), &a) in grow.iter_mut().zip(da2.iter_mut()).zip(wrow).zip(&f.a2) {
                *g += dl * a;
                *da += dl * w;
            }
        }
        let dv = 2.0 * (f.v - s.z) * (1.0 - f.v * f.v);
        grad[l.bv] += dv;
        {
            let wrow = &p[l.wv..l.bv];
            let grow = &mut grad[l.wv..l.bv];
            for (((g, da), &w), &a) in grow.iter_mut().zip(da2.iter_mut()).zip(wrow).zip(&f.a2) {
                *g += dv * a;
                *da += dv * w;
            }
        }

        let mut da1 = vec![0.0f32; h];
        for (j, (&a2j, &d)) in f.a2.iter().zip(&da2).enumerate() {
            if a2j <= 0.0 {
                continue;
            }
            grad[l.b2 + j] += d;
            let wrow = &p[l.w2 + j * h..][..h];
            let grow = &mut grad[l.w2 + j * h..][..h];
            for (((g, da), &w), &a) in grow.iter_mut().zip(da1.iter_mut()).zip(wrow).zip(&f.a1) {
                *g += d * a;
                *da += d * w;
            }
        }

        for (j, (&a1j, &d)) in f.a1.iter().zip(&da1).enumerate() {
            if a1j <= 0.0 {
                continue;
            }
            grad[l.b1 + j] += d;
            let grow = &mut grad[l.w1 + j * self.input..][..self.input];
            for (g, &xv) in grow.iter_mut().zip(&s.x) {
                *g += d * xv;
            }
        }

        (ce, mse)
    }

    /// Writes a versioned binary checkpoint via a temp file + atomic rename.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        if let Some(dir) = path.parent()
            && !dir.as_os_str().is_empty()
        {
            std::fs::create_dir_all(dir)?;
        }
        let mut buf = Vec::with_capacity(24 + self.params.len() * 4);
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        for d in [self.input, self.hidden, self.policy] {
            buf.extend_from_slice(&(d as u32).to_le_bytes());
        }
        for w in &self.params {
            buf.extend_from_slice(&w.to_le_bytes());
        }
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
        let bad = |m: &str| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: {m}", path.display()),
            )
        };
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
}

/// SGD with classical momentum and decoupled L2 weight decay.
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
        for ((w, v), &g) in net.params.iter_mut().zip(self.vel.iter_mut()).zip(grad) {
            let g = g + self.l2 * *w;
            *v = self.momentum * *v - self.lr * g;
            *w += *v;
        }
    }
}
