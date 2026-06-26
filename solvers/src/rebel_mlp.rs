//! ReBeL value-net core: a dependency-free MLP matching the paper's "Net2"
//! architecture — `n_layers` blocks of `Linear -> LayerNorm -> GeLU`, then a
//! linear head with no output activation (raw value regression). Trained with a
//! masked Huber loss (δ=1), Adam, and global L2 gradient clipping.
//!
//! Parameters are stored as `f32` (compact, exact checkpoint round-trips).
//! Inference ([`RebelMlp::forward`] / [`RebelMlp::forward_batch`]) runs as
//! batched f32 GEMMs — Apple Accelerate on macOS, the portable `gemm` crate
//! elsewhere — with LayerNorm/GeLU reductions kept in f64 so the result tracks
//! the training forward to f32 precision. Training (loss, gradients, Adam) runs
//! in `f64`: that keeps the analytic gradient tight against a finite-difference
//! check and lets the gradient-check test feed an explicit `f64` parameter
//! vector with no quantization noise.
//!
//! GeLU uses the exact form `0.5*x*(1 + erf(x/√2))`. `erf` is the
//! Abramowitz-Stegun 7.1.26 rational approximation; the backward pass
//! differentiates *that same* approximation analytically (`erf_prime`), so
//! forward and backward stay exactly consistent.
//!
//! Masked Huber, per sample: `e = pred - target`; for each output with mask 1,
//! `huber = 0.5*e²` when `|e| <= 1` else `|e| - 0.5`; the sample loss is the sum
//! of masked Huber terms divided by the number of masked outputs; the batch
//! loss is the mean over samples.
//!
//! Checkpoint layout: magic `b"REBELMLP"`, `u32` version, the four dims
//! (`input_dim`, `hidden`, `n_layers`, `output_dim`) as little-endian `u32`,
//! then every parameter as little-endian `f32` in the order produced by
//! [`layout`]: per block `weight, bias, gamma, beta`, then head `weight, bias`.
//! Optimizer state is not serialized; a loaded net starts a fresh Adam run.

use std::f64::consts::FRAC_1_SQRT_2;
use std::io;
use std::path::Path;

use game_core::Rng;
use game_core::rand::normal;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

const DEFAULT_LR: f64 = 3e-4;
const DEFAULT_GRAD_CLIP: f64 = 5.0;
const HEAD_INIT_SCALE: f64 = 0.01;
const LN_EPS: f64 = 1e-5;
const ADAM_B1: f64 = 0.9;
const ADAM_B2: f64 = 0.999;
const ADAM_EPS: f64 = 1e-8;

const ERF_P: f64 = 0.327_591_1;
const ERF_A0: f64 = 0.254_829_592;
const ERF_A1: f64 = -0.284_496_736;
const ERF_A2: f64 = 1.421_413_741;
const ERF_A3: f64 = -1.453_152_027;
const ERF_A4: f64 = 1.061_405_429;

const MAGIC: &[u8; 8] = b"REBELMLP";
const VERSION: u32 = 1;
const HEADER_LEN: usize = 28;

/// Shape of a [`RebelMlp`]: `n_layers` hidden blocks of width `hidden` mapping
/// `input_dim` inputs to `output_dim` raw values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RebelMlpConfig {
    pub input_dim: usize,
    pub hidden: usize,
    pub n_layers: usize,
    pub output_dim: usize,
}

/// One training example: encoded `input`, the regression `target` per output,
/// and a `mask` (0/1) selecting which outputs contribute to the loss.
#[derive(Clone, Debug)]
pub struct Sample {
    pub input: Vec<f32>,
    pub target: Vec<f32>,
    pub mask: Vec<f32>,
}

struct Block {
    in_dim: usize,
    weight: usize,
    bias: usize,
    gamma: usize,
    beta: usize,
}

struct Layout {
    blocks: Vec<Block>,
    head_weight: usize,
    head_bias: usize,
    total: usize,
}

fn layout(cfg: &RebelMlpConfig) -> Layout {
    let h = cfg.hidden;
    let mut off = 0usize;
    let mut blocks = Vec::with_capacity(cfg.n_layers);
    for k in 0..cfg.n_layers {
        let in_dim = if k == 0 { cfg.input_dim } else { h };
        let weight = off;
        off += h * in_dim;
        let bias = off;
        off += h;
        let gamma = off;
        off += h;
        let beta = off;
        off += h;
        blocks.push(Block {
            in_dim,
            weight,
            bias,
            gamma,
            beta,
        });
    }
    let head_weight = off;
    off += cfg.output_dim * h;
    let head_bias = off;
    off += cfg.output_dim;
    Layout {
        blocks,
        head_weight,
        head_bias,
        total: off,
    }
}

fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + ERF_P * x);
    let q = t * (ERF_A0 + t * (ERF_A1 + t * (ERF_A2 + t * (ERF_A3 + t * ERF_A4))));
    sign * (1.0 - q * (-x * x).exp())
}

fn erf_prime(x: f64) -> f64 {
    let x = x.abs();
    let t = 1.0 / (1.0 + ERF_P * x);
    let q = t * (ERF_A0 + t * (ERF_A1 + t * (ERF_A2 + t * (ERF_A3 + t * ERF_A4))));
    let qp =
        ERF_A0 + t * (2.0 * ERF_A1 + t * (3.0 * ERF_A2 + t * (4.0 * ERF_A3 + t * 5.0 * ERF_A4)));
    (ERF_P * t * t * qp + 2.0 * x * q) * (-x * x).exp()
}

fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + erf(x * FRAC_1_SQRT_2))
}

fn gelu_grad(x: f64) -> f64 {
    let u = x * FRAC_1_SQRT_2;
    0.5 * (1.0 + erf(u)) + 0.5 * x * erf_prime(u) * FRAC_1_SQRT_2
}

/// SGEMM backend for the f32 inference forward. On macOS the default is Apple
/// Accelerate (vecLib / AMX coprocessor); elsewhere it is the portable `gemm`
/// crate. The non-preferred variant is still compiled so the benchmark can
/// compare both backends on the same machine.
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // `Portable` is unconstructed in macOS release builds.
enum SgemmBackend {
    Portable,
    #[cfg(target_os = "macos")]
    Accelerate,
}

impl SgemmBackend {
    #[inline]
    fn preferred() -> Self {
        #[cfg(target_os = "macos")]
        {
            SgemmBackend::Accelerate
        }
        #[cfg(not(target_os = "macos"))]
        {
            SgemmBackend::Portable
        }
    }
}

#[cfg(target_os = "macos")]
mod accelerate {
    pub const ROW_MAJOR: i32 = 101;
    pub const NO_TRANS: i32 = 111;
    pub const TRANS: i32 = 112;

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        #[allow(clippy::too_many_arguments)]
        pub fn cblas_sgemm(
            order: i32,
            transa: i32,
            transb: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }
}

#[cfg(feature = "parallel")]
#[inline]
fn gemm_parallelism() -> gemm::Parallelism {
    gemm::Parallelism::Rayon(0)
}

#[cfg(not(feature = "parallel"))]
#[inline]
fn gemm_parallelism() -> gemm::Parallelism {
    gemm::Parallelism::None
}

/// `dst (m×n, row-major) = lhs (m×k, row-major) · weightᵀ`, where `weight` is a
/// body matrix in its natural `(n×k)` output-major row-major layout, so its
/// transpose is the `(k×n)` right operand with no copy. `dst` is fully written.
fn sgemm_into(
    backend: SgemmBackend,
    m: usize,
    k: usize,
    n: usize,
    lhs: &[f32],
    weight: &[f32],
    dst: &mut [f32],
) {
    debug_assert_eq!(lhs.len(), m * k);
    debug_assert_eq!(weight.len(), n * k);
    debug_assert_eq!(dst.len(), m * n);
    match backend {
        SgemmBackend::Portable => unsafe {
            gemm::gemm(
                m,
                n,
                k,
                dst.as_mut_ptr(),
                1,
                n as isize,
                false,
                lhs.as_ptr(),
                1,
                k as isize,
                weight.as_ptr(),
                k as isize,
                1,
                0.0,
                1.0,
                false,
                false,
                false,
                gemm_parallelism(),
            );
        },
        #[cfg(target_os = "macos")]
        SgemmBackend::Accelerate => unsafe {
            accelerate::cblas_sgemm(
                accelerate::ROW_MAJOR,
                accelerate::NO_TRANS,
                accelerate::TRANS,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                lhs.as_ptr(),
                k as i32,
                weight.as_ptr(),
                k as i32,
                0.0,
                dst.as_mut_ptr(),
                n as i32,
            );
        },
    }
}

fn matmul(
    backend: SgemmBackend,
    m: usize,
    k: usize,
    n: usize,
    lhs: &[f32],
    weight: &[f32],
) -> Vec<f32> {
    let mut dst = vec![0.0f32; m * n];
    sgemm_into(backend, m, k, n, lhs, weight, &mut dst);
    dst
}

/// In place over a row-major `(rows × h)` pre-activation: add `bias`, LayerNorm
/// (eps `LN_EPS`) with `gamma`/`beta`, then GeLU. Per-row reductions run in f64
/// so the f32 forward tracks the f64 training forward to f32 precision.
fn layer_norm_gelu(z: &mut [f32], h: usize, bias: &[f32], gamma: &[f32], beta: &[f32]) {
    let per_row = |row: &mut [f32]| {
        let mut mean = 0.0f64;
        for (v, &b) in row.iter_mut().zip(bias) {
            *v += b;
            mean += *v as f64;
        }
        mean /= h as f64;
        let var = row
            .iter()
            .map(|&v| {
                let d = v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / h as f64;
        let rstd = 1.0 / (var + LN_EPS).sqrt();
        for ((v, &g), &b) in row.iter_mut().zip(gamma).zip(beta) {
            let normalized = (*v as f64 - mean) * rstd;
            *v = gelu(g as f64 * normalized + b as f64) as f32;
        }
    };
    #[cfg(feature = "parallel")]
    z.par_chunks_mut(h).for_each(per_row);
    #[cfg(not(feature = "parallel"))]
    z.chunks_mut(h).for_each(per_row);
}

fn add_bias(dst: &mut [f32], n: usize, bias: &[f32]) {
    let per_row = |row: &mut [f32]| {
        for (d, &b) in row.iter_mut().zip(bias) {
            *d += b;
        }
    };
    #[cfg(feature = "parallel")]
    dst.par_chunks_mut(n).for_each(per_row);
    #[cfg(not(feature = "parallel"))]
    dst.chunks_mut(n).for_each(per_row);
}

/// Per-block forward activations retained for the backward pass.
struct Cache {
    block_input: Vec<Vec<f64>>,
    normalized: Vec<Vec<f64>>,
    rstd: Vec<f64>,
    pre_gelu: Vec<Vec<f64>>,
    head_input: Vec<f64>,
}

impl Cache {
    fn new(n_layers: usize) -> Self {
        Self {
            block_input: Vec::with_capacity(n_layers),
            normalized: Vec::with_capacity(n_layers),
            rstd: Vec::with_capacity(n_layers),
            pre_gelu: Vec::with_capacity(n_layers),
            head_input: Vec::new(),
        }
    }
}

pub struct RebelMlp {
    cfg: RebelMlpConfig,
    layout: Layout,
    params: Vec<f32>,
    adam_m: Vec<f64>,
    adam_v: Vec<f64>,
    step: u64,
    lr: f64,
    grad_clip: f64,
}

impl RebelMlp {
    /// Builds a net with He-initialized body weights (biases 0, LayerNorm gain
    /// 1 / bias 0) and a head whose weights and bias are He-initialized then
    /// scaled by `0.01`, so an untrained net outputs ≈ 0.
    pub fn new(cfg: RebelMlpConfig, seed: u64) -> RebelMlp {
        assert!(
            cfg.n_layers >= 1,
            "RebelMlp needs at least one hidden block"
        );
        let layout = layout(&cfg);
        let h = cfg.hidden;
        let mut rng = Rng::new(seed);
        let mut params = vec![0.0f32; layout.total];
        let he = |fan_in: usize| (2.0 / fan_in as f64).sqrt();
        for blk in &layout.blocks {
            let std = he(blk.in_dim);
            for w in &mut params[blk.weight..blk.weight + h * blk.in_dim] {
                *w = (normal(&mut rng) * std) as f32;
            }
            for g in &mut params[blk.gamma..blk.gamma + h] {
                *g = 1.0;
            }
        }
        let head_std = he(h) * HEAD_INIT_SCALE;
        let head_end = layout.head_weight + cfg.output_dim * h;
        for w in &mut params[layout.head_weight..head_end] {
            *w = (normal(&mut rng) * head_std) as f32;
        }
        for b in &mut params[layout.head_bias..layout.head_bias + cfg.output_dim] {
            *b = (normal(&mut rng) * head_std) as f32;
        }
        RebelMlp::assemble(cfg, layout, params)
    }

    fn assemble(cfg: RebelMlpConfig, layout: Layout, params: Vec<f32>) -> RebelMlp {
        let total = layout.total;
        RebelMlp {
            cfg,
            layout,
            params,
            adam_m: vec![0.0; total],
            adam_v: vec![0.0; total],
            step: 0,
            lr: DEFAULT_LR,
            grad_clip: DEFAULT_GRAD_CLIP,
        }
    }

    pub fn config(&self) -> RebelMlpConfig {
        self.cfg
    }

    pub fn input_dim(&self) -> usize {
        self.cfg.input_dim
    }

    pub fn output_dim(&self) -> usize {
        self.cfg.output_dim
    }

    pub fn params(&self) -> &[f32] {
        &self.params
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr as f64;
    }

    pub fn set_grad_clip(&mut self, max_norm: f32) {
        self.grad_clip = max_norm as f64;
    }

    fn params_f64(&self) -> Vec<f64> {
        self.params.iter().map(|&p| p as f64).collect()
    }

    /// Raw output values for a single `input` of length `input_dim`.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.cfg.input_dim, "input length mismatch");
        self.forward_batch(input, 1)
    }

    /// Raw outputs for `n` inputs laid out row-major (`n * input_dim`),
    /// returned row-major (`n * output_dim`). Each layer is a single batched
    /// f32 GEMM (Accelerate on macOS, the `gemm` crate elsewhere); LayerNorm and
    /// GeLU are applied per row.
    pub fn forward_batch(&self, inputs: &[f32], n: usize) -> Vec<f32> {
        self.forward_batch_with(inputs, n, SgemmBackend::preferred())
    }

    fn forward_batch_with(&self, inputs: &[f32], n: usize, backend: SgemmBackend) -> Vec<f32> {
        let id = self.cfg.input_dim;
        let od = self.cfg.output_dim;
        assert_eq!(inputs.len(), n * id, "batch input length mismatch");
        if n == 0 {
            return Vec::new();
        }
        let h = self.cfg.hidden;
        let mut act = inputs.to_vec();
        for blk in &self.layout.blocks {
            let k = blk.in_dim;
            let weight = &self.params[blk.weight..blk.weight + h * k];
            let bias = &self.params[blk.bias..blk.bias + h];
            let gamma = &self.params[blk.gamma..blk.gamma + h];
            let beta = &self.params[blk.beta..blk.beta + h];
            let mut z = matmul(backend, n, k, h, &act, weight);
            layer_norm_gelu(&mut z, h, bias, gamma, beta);
            act = z;
        }
        let hw = &self.params[self.layout.head_weight..self.layout.head_weight + od * h];
        let hb = &self.params[self.layout.head_bias..self.layout.head_bias + od];
        let mut out = matmul(backend, n, h, od, &act, hw);
        add_bias(&mut out, od, hb);
        out
    }

    /// Mean masked-Huber loss over `batch` (no parameter update).
    pub fn loss(&self, batch: &[Sample]) -> f32 {
        if batch.is_empty() {
            return 0.0;
        }
        let params = self.params_f64();
        let sum: f64 = batch.iter().map(|s| self.run(&params, s, None)).sum();
        (sum / batch.len() as f64) as f32
    }

    /// One Adam step on `batch`: accumulate the mean masked-Huber gradient,
    /// apply global L2 gradient clipping, then update. Returns the mean loss
    /// measured before the update.
    pub fn train_step(&mut self, batch: &[Sample]) -> f32 {
        if batch.is_empty() {
            return 0.0;
        }
        let params = self.params_f64();
        let (mut grad, loss_sum) = self.accumulate_grads(&params, batch);
        let scale = 1.0 / batch.len() as f64;
        for g in grad.iter_mut() {
            *g *= scale;
        }
        let mean_loss = loss_sum * scale;

        let norm = grad.iter().map(|&g| g * g).sum::<f64>().sqrt();
        if norm > self.grad_clip {
            let c = self.grad_clip / norm;
            for g in grad.iter_mut() {
                *g *= c;
            }
        }

        self.step += 1;
        let t = self.step as f64;
        let bias1 = 1.0 - ADAM_B1.powf(t);
        let bias2 = 1.0 - ADAM_B2.powf(t);
        for (((p, m), v), &g) in self
            .params
            .iter_mut()
            .zip(self.adam_m.iter_mut())
            .zip(self.adam_v.iter_mut())
            .zip(&grad)
        {
            *m = ADAM_B1 * *m + (1.0 - ADAM_B1) * g;
            *v = ADAM_B2 * *v + (1.0 - ADAM_B2) * g * g;
            let mhat = *m / bias1;
            let vhat = *v / bias2;
            let update = self.lr * mhat / (vhat.sqrt() + ADAM_EPS);
            *p = (*p as f64 - update) as f32;
        }
        mean_loss as f32
    }

    fn accumulate_grads(&self, params: &[f64], batch: &[Sample]) -> (Vec<f64>, f64) {
        let total = self.layout.total;
        #[cfg(feature = "parallel")]
        {
            batch
                .par_iter()
                .fold(
                    || (vec![0.0f64; total], 0.0f64),
                    |mut acc, s| {
                        let l = self.run(params, s, Some(acc.0.as_mut_slice()));
                        acc.1 += l;
                        acc
                    },
                )
                .reduce(
                    || (vec![0.0f64; total], 0.0f64),
                    |mut a, b| {
                        for (x, y) in a.0.iter_mut().zip(&b.0) {
                            *x += y;
                        }
                        a.1 += b.1;
                        a
                    },
                )
        }
        #[cfg(not(feature = "parallel"))]
        {
            let mut grad = vec![0.0f64; total];
            let mut loss = 0.0f64;
            for s in batch {
                loss += self.run(params, s, Some(grad.as_mut_slice()));
            }
            (grad, loss)
        }
    }

    fn forward_full(&self, params: &[f64], input: &[f32], cache: &mut Option<Cache>) -> Vec<f64> {
        let h = self.cfg.hidden;
        let mut act: Vec<f64> = input.iter().map(|&x| x as f64).collect();
        for blk in &self.layout.blocks {
            let in_dim = blk.in_dim;
            let w = &params[blk.weight..blk.weight + h * in_dim];
            let b = &params[blk.bias..blk.bias + h];
            let g = &params[blk.gamma..blk.gamma + h];
            let beta = &params[blk.beta..blk.beta + h];

            let mut z = vec![0.0f64; h];
            for ((zj, wrow), &bj) in z.iter_mut().zip(w.chunks_exact(in_dim)).zip(b) {
                let mut s = bj;
                for (&wij, &ai) in wrow.iter().zip(&act) {
                    s += wij * ai;
                }
                *zj = s;
            }

            let mean = z.iter().sum::<f64>() / h as f64;
            let var = z.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / h as f64;
            let rstd = 1.0 / (var + LN_EPS).sqrt();
            let mut normalized = vec![0.0f64; h];
            let mut pre_gelu = vec![0.0f64; h];
            let mut out = vec![0.0f64; h];
            for j in 0..h {
                let n = (z[j] - mean) * rstd;
                normalized[j] = n;
                let y = g[j] * n + beta[j];
                pre_gelu[j] = y;
                out[j] = gelu(y);
            }

            if let Some(c) = cache.as_mut() {
                c.block_input.push(act);
                c.normalized.push(normalized);
                c.rstd.push(rstd);
                c.pre_gelu.push(pre_gelu);
            }
            act = out;
        }

        let o = self.cfg.output_dim;
        let hw = &params[self.layout.head_weight..self.layout.head_weight + o * h];
        let hb = &params[self.layout.head_bias..self.layout.head_bias + o];
        let mut out = vec![0.0f64; o];
        for ((ok, wrow), &bk) in out.iter_mut().zip(hw.chunks_exact(h)).zip(hb) {
            let mut s = bk;
            for (&wkj, &aj) in wrow.iter().zip(&act) {
                s += wkj * aj;
            }
            *ok = s;
        }
        if let Some(c) = cache.as_mut() {
            c.head_input = act;
        }
        out
    }

    /// Forward + masked-Huber loss for one sample; if `grad` is `Some`, also
    /// backpropagates and *accumulates* into it. Returns the sample loss.
    fn run(&self, params: &[f64], s: &Sample, grad: Option<&mut [f64]>) -> f64 {
        let h = self.cfg.hidden;
        let o = self.cfg.output_dim;
        let want_grad = grad.is_some();
        let mut cache = if want_grad {
            Some(Cache::new(self.cfg.n_layers))
        } else {
            None
        };
        let pred = self.forward_full(params, &s.input, &mut cache);

        let mask_count: f64 = s.mask.iter().map(|&m| m as f64).sum();
        let denom = if mask_count > 0.0 { mask_count } else { 1.0 };
        let mut loss = 0.0f64;
        let mut dpred = vec![0.0f64; o];
        for k in 0..o {
            let m = s.mask[k] as f64;
            if m == 0.0 {
                continue;
            }
            let e = pred[k] - s.target[k] as f64;
            let ae = e.abs();
            loss += if ae <= 1.0 { 0.5 * e * e } else { ae - 0.5 };
            if want_grad {
                let de = if ae <= 1.0 { e } else { e.signum() };
                dpred[k] = m * de / denom;
            }
        }
        loss /= denom;

        let Some(grad) = grad else {
            return loss;
        };
        let cache = cache.expect("cache present when gradient requested");

        let mut da = vec![0.0f64; h];
        for k in 0..o {
            let dk = dpred[k];
            grad[self.layout.head_bias + k] += dk;
            let row = self.layout.head_weight + k * h;
            for j in 0..h {
                grad[row + j] += dk * cache.head_input[j];
                da[j] += dk * params[row + j];
            }
        }

        for li in (0..self.cfg.n_layers).rev() {
            let blk = &self.layout.blocks[li];
            let in_dim = blk.in_dim;
            let pre_gelu = &cache.pre_gelu[li];
            let normalized = &cache.normalized[li];
            let rstd = cache.rstd[li];
            let block_input = &cache.block_input[li];

            let mut dn = vec![0.0f64; h];
            let mut mean_dn = 0.0f64;
            let mut mean_dn_n = 0.0f64;
            for j in 0..h {
                let dy = da[j] * gelu_grad(pre_gelu[j]);
                grad[blk.gamma + j] += dy * normalized[j];
                grad[blk.beta + j] += dy;
                let dnj = dy * params[blk.gamma + j];
                dn[j] = dnj;
                mean_dn += dnj;
                mean_dn_n += dnj * normalized[j];
            }
            mean_dn /= h as f64;
            mean_dn_n /= h as f64;

            let mut da_prev = vec![0.0f64; in_dim];
            for j in 0..h {
                let dz = rstd * (dn[j] - mean_dn - normalized[j] * mean_dn_n);
                grad[blk.bias + j] += dz;
                let row = blk.weight + j * in_dim;
                for i in 0..in_dim {
                    grad[row + i] += dz * block_input[i];
                    da_prev[i] += dz * params[row + i];
                }
            }
            da = da_prev;
        }
        loss
    }

    /// The versioned checkpoint encoding behind [`RebelMlp::save`].
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(HEADER_LEN + self.params.len() * 4);
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        for d in [
            self.cfg.input_dim,
            self.cfg.hidden,
            self.cfg.n_layers,
            self.cfg.output_dim,
        ] {
            buf.extend_from_slice(&(d as u32).to_le_bytes());
        }
        for w in &self.params {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        buf
    }

    /// Parses a checkpoint produced by [`RebelMlp::to_bytes`] /
    /// [`RebelMlp::save`]. The returned net carries a fresh Adam state.
    pub fn from_bytes(data: &[u8]) -> io::Result<RebelMlp> {
        let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());
        if data.len() < HEADER_LEN {
            return Err(bad("truncated header"));
        }
        if &data[..8] != MAGIC {
            return Err(bad("not a rebel MLP checkpoint"));
        }
        let u32_at = |i: usize| u32::from_le_bytes(data[i..i + 4].try_into().unwrap()) as usize;
        if u32_at(8) != VERSION as usize {
            return Err(bad("unsupported checkpoint version"));
        }
        let cfg = RebelMlpConfig {
            input_dim: u32_at(12),
            hidden: u32_at(16),
            n_layers: u32_at(20),
            output_dim: u32_at(24),
        };
        if cfg.n_layers == 0 {
            return Err(bad("checkpoint has zero hidden blocks"));
        }
        let layout = layout(&cfg);
        let body = &data[HEADER_LEN..];
        if body.len() != layout.total * 4 {
            return Err(bad("parameter count does not match dimensions"));
        }
        let params = body
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok(RebelMlp::assemble(cfg, layout, params))
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

    pub fn load(path: &Path) -> io::Result<RebelMlp> {
        let data = std::fs::read(path)?;
        Self::from_bytes(&data).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: {e}", path.display()),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> RebelMlpConfig {
        RebelMlpConfig {
            input_dim: 4,
            hidden: 5,
            n_layers: 2,
            output_dim: 3,
        }
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let cfg = tiny_cfg();
        let net = RebelMlp::new(cfg, 1);
        let total = net.layout.total;
        let mut rng = Rng::new(12345);

        let params: Vec<f64> = (0..total).map(|_| normal(&mut rng) * 0.5).collect();
        let input: Vec<f32> = (0..cfg.input_dim)
            .map(|_| normal(&mut rng) as f32)
            .collect();

        let mut no_cache = None;
        let pred = net.forward_full(&params, &input, &mut no_cache);

        // Place residuals away from the Huber kink at |e| = 1 so the loss is
        // smooth: 0.3 (quadratic branch), 2.0 (linear branch); index 1 masked.
        let residuals = [0.3f64, 2.0, 2.0];
        let target: Vec<f32> = (0..cfg.output_dim)
            .map(|k| (pred[k] - residuals[k]) as f32)
            .collect();
        let mask = vec![1.0f32, 0.0, 1.0];
        let s = Sample {
            input,
            target,
            mask,
        };

        let mut grad = vec![0.0f64; total];
        net.run(&params, &s, Some(grad.as_mut_slice()));

        let h = 1e-3;
        let mut max_rel = 0.0f64;
        for i in 0..total {
            let mut perturbed = params.clone();
            perturbed[i] = params[i] + h;
            let lp = net.run(&perturbed, &s, None);
            perturbed[i] = params[i] - h;
            let lm = net.run(&perturbed, &s, None);
            let fd = (lp - lm) / (2.0 * h);
            let denom = fd.abs().max(grad[i].abs()).max(1e-6);
            max_rel = max_rel.max((fd - grad[i]).abs() / denom);
        }
        println!("grad-check max relative error = {max_rel:.3e}");
        assert!(max_rel < 1e-3, "gradient check failed: max_rel = {max_rel}");
    }

    #[test]
    fn overfits_a_fixed_batch() {
        let cfg = RebelMlpConfig {
            input_dim: 8,
            hidden: 16,
            n_layers: 2,
            output_dim: 4,
        };
        let mut net = RebelMlp::new(cfg, 7);
        let mut rng = Rng::new(99);
        let batch: Vec<Sample> = (0..16)
            .map(|_| Sample {
                input: (0..cfg.input_dim)
                    .map(|_| normal(&mut rng) as f32)
                    .collect(),
                target: (0..cfg.output_dim)
                    .map(|_| normal(&mut rng) as f32)
                    .collect(),
                mask: vec![1.0; cfg.output_dim],
            })
            .collect();

        net.set_lr(3e-3);
        let mut loss = f32::INFINITY;
        for _ in 0..1500 {
            loss = net.train_step(&batch);
        }
        assert!(loss < 1e-3, "overfit failed: final loss = {loss}");
    }

    #[test]
    fn batched_forward_matches_single() {
        let cfg = RebelMlpConfig {
            input_dim: 6,
            hidden: 10,
            n_layers: 3,
            output_dim: 5,
        };
        let net = RebelMlp::new(cfg, 3);
        let mut rng = Rng::new(55);
        let n = 7;
        let inputs: Vec<f32> = (0..n * cfg.input_dim)
            .map(|_| normal(&mut rng) as f32)
            .collect();

        let batched = net.forward_batch(&inputs, n);
        for r in 0..n {
            let single = net.forward(&inputs[r * cfg.input_dim..(r + 1) * cfg.input_dim]);
            for k in 0..cfg.output_dim {
                let diff = (batched[r * cfg.output_dim + k] - single[k]).abs();
                assert!(diff < 1e-5, "row {r} output {k} diff = {diff}");
            }
        }
    }

    #[test]
    fn f32_forward_matches_f64_reference() {
        // The fast f32 GEMM inference forward must match the f64 training
        // forward (`forward_full`) within f32 tolerance, including at the
        // deployment leaf size (input 2809, output 462).
        let cfgs = [
            RebelMlpConfig {
                input_dim: 64,
                hidden: 48,
                n_layers: 1,
                output_dim: 29,
            },
            RebelMlpConfig {
                input_dim: 128,
                hidden: 96,
                n_layers: 2,
                output_dim: 40,
            },
            RebelMlpConfig {
                input_dim: 2809,
                hidden: 256,
                n_layers: 2,
                output_dim: 462,
            },
        ];
        for cfg in cfgs {
            let net = RebelMlp::new(cfg, 100 + cfg.hidden as u64);
            let mut rng = Rng::new(2024);
            let n = 5;
            let inputs: Vec<f32> = (0..n * cfg.input_dim)
                .map(|_| normal(&mut rng) as f32)
                .collect();

            let got = net.forward_batch(&inputs, n);
            let params = net.params_f64();
            let mut max_diff = 0.0f64;
            for r in 0..n {
                let row = &inputs[r * cfg.input_dim..(r + 1) * cfg.input_dim];
                let mut no_cache = None;
                let reference = net.forward_full(&params, row, &mut no_cache);
                for k in 0..cfg.output_dim {
                    let d = (got[r * cfg.output_dim + k] as f64 - reference[k]).abs();
                    max_diff = max_diff.max(d);
                }
            }
            println!(
                "equivalence in={:5} hidden={:4} layers={}: max abs diff = {max_diff:.3e}",
                cfg.input_dim, cfg.hidden, cfg.n_layers
            );
            assert!(
                max_diff < 1e-3,
                "cfg {cfg:?}: f32-vs-f64 forward max diff = {max_diff}"
            );
        }
    }

    #[test]
    #[ignore = "benchmark; run with: cargo test --release -p solvers rebel_mlp -- --ignored --nocapture"]
    fn bench_forward_backends() {
        use std::time::Instant;
        for hidden in [512usize, 256] {
            let cfg = RebelMlpConfig {
                input_dim: 2809,
                hidden,
                n_layers: 2,
                output_dim: 462,
            };
            let n = 600;
            let net = RebelMlp::new(cfg, 1);
            let mut rng = Rng::new(7);
            let inputs: Vec<f32> = (0..n * cfg.input_dim)
                .map(|_| normal(&mut rng) as f32)
                .collect();

            let backends: &[(&str, SgemmBackend)] = &[
                ("gemm-crate", SgemmBackend::Portable),
                #[cfg(target_os = "macos")]
                ("accelerate", SgemmBackend::Accelerate),
            ];
            for &(name, backend) in backends {
                for _ in 0..3 {
                    std::hint::black_box(net.forward_batch_with(&inputs, n, backend));
                }
                let iters = 20;
                let t = Instant::now();
                for _ in 0..iters {
                    std::hint::black_box(net.forward_batch_with(&inputs, n, backend));
                }
                let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
                println!("hidden={hidden:4} backend={name:11} {ms:9.3} ms/call");
            }
        }
    }

    #[test]
    fn save_load_round_trips() {
        let cfg = RebelMlpConfig {
            input_dim: 5,
            hidden: 9,
            n_layers: 2,
            output_dim: 4,
        };
        let mut net = RebelMlp::new(cfg, 11);
        let mut rng = Rng::new(77);
        let batch: Vec<Sample> = (0..8)
            .map(|_| Sample {
                input: (0..cfg.input_dim)
                    .map(|_| normal(&mut rng) as f32)
                    .collect(),
                target: (0..cfg.output_dim)
                    .map(|_| normal(&mut rng) as f32)
                    .collect(),
                mask: vec![1.0; cfg.output_dim],
            })
            .collect();
        for _ in 0..10 {
            net.train_step(&batch);
        }

        let n = 6;
        let inputs: Vec<f32> = (0..n * cfg.input_dim)
            .map(|_| normal(&mut rng) as f32)
            .collect();
        let expected = net.forward_batch(&inputs, n);

        let bytes = net.to_bytes();
        let reloaded = RebelMlp::from_bytes(&bytes).unwrap();
        assert_eq!(reloaded.forward_batch(&inputs, n), expected);

        let path = std::env::temp_dir().join(format!("rebel_mlp_rt_{}.bin", std::process::id()));
        net.save(&path).unwrap();
        let from_disk = RebelMlp::load(&path).unwrap();
        assert_eq!(from_disk.forward_batch(&inputs, n), expected);
        let _ = std::fs::remove_file(&path);
    }
}
