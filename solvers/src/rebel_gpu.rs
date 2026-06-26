//! Metal GPU batched-inference path for the ReBeL value-net forward
//! ([`crate::rebel_mlp::RebelMlp`]), gated behind the `gpu` feature (macOS only).
//!
//! The reference ReBeL's max-scale design puts the net on the GPU behind a
//! cross-thread batched-inference server: the many independent ~300-row forwards
//! that the parallel CFR data-gen threads issue are individually too small to
//! beat the AMX `cblas_sgemm` baseline (GPU launch overhead dominates below
//! ~1200 rows), but coalescing them into one ≥2400-row matmul lets MPS win 2-4×.
//!
//! [`GpuServer`] owns a worker thread that drains pending queries from every CFR
//! thread, concatenates them (dense encoding — universal across the differing
//! per-solve active-index sets), runs the whole forward on the GPU in a single
//! command buffer (three [`MPSMatrixMultiplication`] GEMMs interleaved with a
//! fused bias+LayerNorm+GeLU compute kernel, plus a head-bias kernel), and
//! scatters the rows back. Buffers use shared (unified-memory) storage so there
//! is no host/device copy. The forward is mathematically the dense
//! [`crate::rebel_mlp::RebelMlp::forward_batch`]; LayerNorm/GeLU run in f32 (CPU
//! runs them in f64) so results match to f32 precision (~1e-8 on the deployment
//! shapes), well within the 1e-3 gate.
//!
//! NULL RESULT (M5 Max, 5p5d6f, hidden 256 & 512): this path is correct but does
//! NOT beat the Accelerate/AMX baseline, so it is **off by default** and the AMX
//! path is unchanged. An isolated GEMM microbench shows MPS only beats AMX at
//! ≥~1200 rows (2-4× at 2400-8192); below that — including the ~300-row forward a
//! single CFR solve issues — AMX wins ~2×. Beating AMX therefore *requires* the
//! cross-thread server to coalesce ~8 threads' queries into one ≥2400-row matmul.
//! But with only 18 cores shared between the synchronous CFR producers (each
//! holding ≤1 outstanding request) and this GPU-driving worker, batches coalesce
//! to just ~226-498 rows; forcing larger batches by oversubscribing the thread
//! pool thrashes the CPU-bound CFR (~53% of cost) and starves the worker. End to
//! end the server runs ~1.7-2.1× SLOWER than AMX (best GPU 143/105 samples/s vs
//! AMX 306/176 at hidden 256/512). The reference ReBeL's GPU-server design wins
//! because it has a large producer:GPU ratio (hundreds of actors per GPU); a
//! single laptop where producers and the server contend for the same cores does
//! not. Kept, gated-off, as a correct, reproducible scaffold (bench it via
//! `liars-dice --features parallel,gpu --example rebel_kcache_sweep`).

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender, bounded};
use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
};

use crate::rebel_mlp::{Layout, RebelMlpConfig, layout};

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

const KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant float ERF_P  =  0.3275911f;
constant float ERF_A0 =  0.254829592f;
constant float ERF_A1 = -0.284496736f;
constant float ERF_A2 =  1.421413741f;
constant float ERF_A3 = -1.453152027f;
constant float ERF_A4 =  1.061405429f;
constant float FRAC_1_SQRT_2 = 0.70710678118654752f;
constant float LN_EPS = 1e-5f;

inline float erf_approx(float x) {
    float s = x < 0.0f ? -1.0f : 1.0f;
    x = fabs(x);
    float t = 1.0f / (1.0f + ERF_P * x);
    float q = t * (ERF_A0 + t * (ERF_A1 + t * (ERF_A2 + t * (ERF_A3 + t * ERF_A4))));
    return s * (1.0f - q * exp(-x * x));
}

inline float gelu(float x) {
    return 0.5f * x * (1.0f + erf_approx(x * FRAC_1_SQRT_2));
}

// One thread per row: add bias, LayerNorm (eps LN_EPS) with gamma/beta, GeLU,
// in place over a row-major (rows x h) buffer.
kernel void ln_gelu(device float* z              [[buffer(0)]],
                    device const float* bias     [[buffer(1)]],
                    device const float* gamma    [[buffer(2)]],
                    device const float* beta     [[buffer(3)]],
                    constant uint& h             [[buffer(4)]],
                    constant uint& rows          [[buffer(5)]],
                    uint gid                     [[thread_position_in_grid]]) {
    if (gid >= rows) return;
    device float* r = z + (uint)gid * h;
    float mean = 0.0f;
    for (uint j = 0; j < h; j++) { float v = r[j] + bias[j]; r[j] = v; mean += v; }
    mean /= (float)h;
    float var = 0.0f;
    for (uint j = 0; j < h; j++) { float d = r[j] - mean; var += d * d; }
    var /= (float)h;
    float rstd = rsqrt(var + LN_EPS);
    for (uint j = 0; j < h; j++) {
        float nrm = (r[j] - mean) * rstd;
        r[j] = gelu(gamma[j] * nrm + beta[j]);
    }
}

// One thread per row: add the output bias to a row-major (rows x od) buffer.
kernel void add_bias(device float* out           [[buffer(0)]],
                     device const float* bias     [[buffer(1)]],
                     constant uint& od            [[buffer(2)]],
                     constant uint& rows          [[buffer(3)]],
                     uint gid                     [[thread_position_in_grid]]) {
    if (gid >= rows) return;
    device float* r = out + (uint)gid * od;
    for (uint j = 0; j < od; j++) r[j] += bias[j];
}
"#;

struct BlockBuffers {
    in_dim: usize,
    weight: Buffer,
    bias: Buffer,
    gamma: Buffer,
    beta: Buffer,
}

/// All Metal state for one forward path. Lives entirely on the worker thread
/// (the wrapped objc objects are `!Send`); never crosses a thread boundary.
struct GpuEngine {
    device: Device,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    ln_gelu: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    add_bias: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    cfg: RebelMlpConfig,
    blocks: Vec<BlockBuffers>,
    head_weight: Buffer,
    head_bias: Buffer,
    capacity: usize,
    in_buf: Buffer,
    act_a: Buffer,
    act_b: Buffer,
    out_buf: Buffer,
}

fn shared_buffer(device: &Device, len_bytes: usize) -> Buffer {
    device
        .newBufferWithLength_options(len_bytes.max(4), MTLResourceOptions::StorageModeShared)
        .expect("failed to allocate MTLBuffer")
}

fn upload(device: &Device, data: &[f32]) -> Buffer {
    let buf = shared_buffer(device, data.len() * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            buf.contents().as_ptr() as *mut f32,
            data.len(),
        );
    }
    buf
}

fn pipeline(
    device: &Device,
    lib: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
    let func = lib
        .newFunctionWithName(&NSString::from_str(name))
        .unwrap_or_else(|| panic!("kernel function `{name}` not found"));
    device
        .newComputePipelineStateWithFunction_error(&func)
        .expect("failed to build compute pipeline")
}

impl GpuEngine {
    fn new(params: &[f32], cfg: RebelMlpConfig) -> GpuEngine {
        let device = MTLCreateSystemDefaultDevice().expect("no Metal device");
        let queue = device.newCommandQueue().expect("no command queue");
        let lib = device
            .newLibraryWithSource_options_error(&NSString::from_str(KERNEL_SRC), None)
            .expect("failed to compile Metal kernel source");
        let ln_gelu = pipeline(&device, &lib, "ln_gelu");
        let add_bias = pipeline(&device, &lib, "add_bias");

        let lay: Layout = layout(&cfg);
        let h = cfg.hidden;
        let blocks = lay
            .blocks
            .iter()
            .map(|b| BlockBuffers {
                in_dim: b.in_dim,
                weight: upload(&device, &params[b.weight..b.weight + h * b.in_dim]),
                bias: upload(&device, &params[b.bias..b.bias + h]),
                gamma: upload(&device, &params[b.gamma..b.gamma + h]),
                beta: upload(&device, &params[b.beta..b.beta + h]),
            })
            .collect();
        let head_weight = upload(
            &device,
            &params[lay.head_weight..lay.head_weight + cfg.output_dim * h],
        );
        let head_bias = upload(
            &device,
            &params[lay.head_bias..lay.head_bias + cfg.output_dim],
        );

        let cap = 1;
        let in_buf = shared_buffer(&device, cap * cfg.input_dim * 4);
        let act_a = shared_buffer(&device, cap * h * 4);
        let act_b = shared_buffer(&device, cap * h * 4);
        let out_buf = shared_buffer(&device, cap * cfg.output_dim * 4);
        GpuEngine {
            device,
            queue,
            ln_gelu,
            add_bias,
            cfg,
            blocks,
            head_weight,
            head_bias,
            capacity: cap,
            in_buf,
            act_a,
            act_b,
            out_buf,
        }
    }

    fn ensure_capacity(&mut self, n: usize) {
        if n <= self.capacity {
            return;
        }
        let h = self.cfg.hidden;
        self.in_buf = shared_buffer(&self.device, n * self.cfg.input_dim * 4);
        self.act_a = shared_buffer(&self.device, n * h * 4);
        self.act_b = shared_buffer(&self.device, n * h * 4);
        self.out_buf = shared_buffer(&self.device, n * self.cfg.output_dim * 4);
        self.capacity = n;
    }

    fn mtx(&self, buf: &Buffer, rows: usize, cols: usize) -> Retained<MPSMatrix> {
        let desc = unsafe {
            MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                rows,
                cols,
                cols * 4,
                MPSDataType::Float32,
            )
        };
        unsafe { MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), buf, &desc) }
    }

    /// `result(rows x out) = left(rows x interior) * right(out x interior)ᵀ`,
    /// encoded into `cb`. `right` is a weight in its natural `(out x interior)`
    /// row-major layout, so `transposeRight` makes it the `(interior x out)`
    /// operand with no copy — matching the CPU `NO_TRANS`/`TRANS` GEMM.
    #[allow(clippy::too_many_arguments)]
    fn matmul(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        left: &Buffer,
        right: &Buffer,
        result: &Buffer,
        rows: usize,
        interior: usize,
        out: usize,
    ) {
        let l = self.mtx(left, rows, interior);
        let r = self.mtx(right, out, interior);
        let res = self.mtx(result, rows, out);
        let mm = unsafe {
            MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                MPSMatrixMultiplication::alloc(),
                &self.device,
                false,
                true,
                rows,
                out,
                interior,
                1.0,
                0.0,
            )
        };
        unsafe { mm.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(cb, &l, &r, &res) };
    }

    fn dispatch_rows(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        pipe: &ProtocolObject<dyn MTLComputePipelineState>,
        bufs: &[(&Buffer, usize)],
        scalars: &[u32],
        rows: usize,
    ) {
        let enc = cb.computeCommandEncoder().expect("compute encoder");
        enc.setComputePipelineState(pipe);
        for &(buf, idx) in bufs {
            unsafe { enc.setBuffer_offset_atIndex(Some(buf), 0, idx) };
        }
        for (i, s) in scalars.iter().enumerate() {
            let idx = bufs.len() + i;
            unsafe {
                enc.setBytes_length_atIndex(
                    std::ptr::NonNull::new(s as *const u32 as *mut std::ffi::c_void).unwrap(),
                    4,
                    idx,
                );
            }
        }
        let grid = MTLSize {
            width: rows,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 64.min(rows.max(1)),
            height: 1,
            depth: 1,
        };
        enc.dispatchThreads_threadsPerThreadgroup(grid, tg);
        enc.endEncoding();
    }

    /// Full dense forward for `n` row-major inputs (`n * input_dim`), returning
    /// row-major outputs (`n * output_dim`).
    fn forward_dense(&mut self, inputs: &[f32], n: usize) -> Vec<f32> {
        let id = self.cfg.input_dim;
        let h = self.cfg.hidden;
        let od = self.cfg.output_dim;
        debug_assert_eq!(inputs.len(), n * id);
        self.ensure_capacity(n);
        unsafe {
            std::ptr::copy_nonoverlapping(
                inputs.as_ptr(),
                self.in_buf.contents().as_ptr() as *mut f32,
                inputs.len(),
            );
        }

        let cb = self.queue.commandBuffer().expect("command buffer");
        let rows_u = n as u32;
        let h_u = h as u32;

        let mut src = &self.in_buf;
        let mut src_cols = id;
        // Ping-pong block outputs between act_a and act_b.
        for (k, blk) in self.blocks.iter().enumerate() {
            let dst = if k % 2 == 0 { &self.act_a } else { &self.act_b };
            self.matmul(&cb, src, &blk.weight, dst, n, src_cols, h);
            self.dispatch_rows(
                &cb,
                &self.ln_gelu,
                &[(dst, 0), (&blk.bias, 1), (&blk.gamma, 2), (&blk.beta, 3)],
                &[h_u, rows_u],
                n,
            );
            src = dst;
            src_cols = h;
            debug_assert_eq!(blk.in_dim, if k == 0 { id } else { h });
        }

        self.matmul(&cb, src, &self.head_weight, &self.out_buf, n, h, od);
        self.dispatch_rows(
            &cb,
            &self.add_bias,
            &[(&self.out_buf, 0), (&self.head_bias, 1)],
            &[od as u32, rows_u],
            n,
        );

        cb.commit();
        cb.waitUntilCompleted();

        let mut out = vec![0.0f32; n * od];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.out_buf.contents().as_ptr() as *const f32,
                out.as_mut_ptr(),
                out.len(),
            );
        }
        out
    }
}

struct Job {
    input: Vec<f32>,
    n: usize,
    resp: Sender<Vec<f32>>,
}

/// Cross-thread batched-inference server. Holds only the `Send + Sync` job
/// queue; the GPU state lives on the spawned worker thread.
pub struct GpuServer {
    jobs: Sender<Job>,
    output_dim: usize,
    batches: AtomicU64,
    rows: AtomicU64,
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

impl GpuServer {
    /// Spawns the worker thread, which builds the engine (uploading `params`
    /// once) and serves coalesced batches until the queue is dropped.
    pub fn new(params: Vec<f32>, cfg: RebelMlpConfig) -> GpuServer {
        let cap = env_usize("REBEL_GPU_CAP", 4096);
        let target = env_usize("REBEL_GPU_TARGET", 2048).min(cap);
        let window_us = env_usize("REBEL_GPU_WINDOW_US", 150) as u64;
        let (tx, rx) = bounded::<Job>(1024);
        std::thread::Builder::new()
            .name("rebel-gpu".into())
            .spawn(move || worker(params, cfg, rx, cap, target, window_us))
            .expect("spawn gpu worker");
        GpuServer {
            jobs: tx,
            output_dim: cfg.output_dim,
            batches: AtomicU64::new(0),
            rows: AtomicU64::new(0),
        }
    }

    /// Submit one dense query (`n * input_dim` row-major) and block until the
    /// worker returns this query's `n * output_dim` rows.
    pub fn forward(&self, input: Vec<f32>, n: usize) -> Vec<f32> {
        thread_local! {
            static RESP: (Sender<Vec<f32>>, Receiver<Vec<f32>>) = bounded(1);
        }
        self.batches.fetch_add(1, Ordering::Relaxed);
        self.rows.fetch_add(n as u64, Ordering::Relaxed);
        RESP.with(|(tx, rx)| {
            self.jobs
                .send(Job {
                    input,
                    n,
                    resp: tx.clone(),
                })
                .expect("gpu worker gone");
            rx.recv().expect("gpu worker dropped response")
        })
    }

    /// `(submitted_queries, submitted_rows)` — lets a bench report the mean
    /// per-submission size (not the coalesced batch size).
    pub fn submit_stats(&self) -> (u64, u64) {
        (
            self.batches.load(Ordering::Relaxed),
            self.rows.load(Ordering::Relaxed),
        )
    }

    pub fn output_dim(&self) -> usize {
        self.output_dim
    }
}

fn worker(
    params: Vec<f32>,
    cfg: RebelMlpConfig,
    rx: Receiver<Job>,
    cap: usize,
    target: usize,
    window_us: u64,
) {
    let mut engine = GpuEngine::new(&params, cfg);
    let od = cfg.output_dim;
    let report = std::env::var("REBEL_GPU_REPORT").is_ok();
    let mut coalesced_batches = 0u64;
    let mut coalesced_rows = 0u64;
    loop {
        let first = match rx.recv() {
            Ok(j) => j,
            Err(_) => break,
        };
        let mut inputs = first.input;
        let mut parts: Vec<(usize, Sender<Vec<f32>>)> = vec![(first.n, first.resp)];
        let mut rows = first.n;

        // Coalesce: drain everything queued, then optionally wait a short window
        // for stragglers from the other CFR threads, up to the row target/cap.
        let deadline = Instant::now() + Duration::from_micros(window_us);
        loop {
            match rx.try_recv() {
                Ok(j) => {
                    inputs.extend_from_slice(&j.input);
                    rows += j.n;
                    parts.push((j.n, j.resp));
                    if rows >= cap {
                        break;
                    }
                }
                Err(_) => {
                    if rows >= target || window_us == 0 || Instant::now() >= deadline {
                        break;
                    }
                    std::hint::spin_loop();
                }
            }
        }

        let out = engine.forward_dense(&inputs, rows);
        coalesced_batches += 1;
        coalesced_rows += rows as u64;

        let mut off = 0usize;
        for (cnt, resp) in parts {
            let _ = resp.send(out[off * od..(off + cnt) * od].to_vec());
            off += cnt;
        }
    }
    if report && coalesced_batches > 0 {
        eprintln!(
            "[rebel-gpu] coalesced {coalesced_batches} batches, {coalesced_rows} rows, mean {:.0} rows/batch",
            coalesced_rows as f64 / coalesced_batches as f64
        );
    }
}
