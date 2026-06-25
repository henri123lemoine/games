# MLX vs PyTorch-MPS for the Stratego move-net — Apple M5 Max

**Question:** Does MLX (Python) train a transformer the size of Ataraxos's Stratego
"move net" materially faster than PyTorch-MPS (Python) on *this* M5 Max, for a real
training **step**? The answer decides Rust+tch(MPS) vs hybrid Rust-sim + MLX-Python.

## Environment (this machine)

| | |
|---|---|
| Machine | Apple **M5 Max**, GPU `applegpu_g17s`, 18 cores |
| macOS | **26.5.1** (build 25F80) — **above** the 26.2 floor MLX needs for the Metal 4 Tensor API |
| Unified memory | **64 GB** (NOT 128 GB — prompt assumed 128; this unit reports 64). `max_recommended_working_set_size` = **55.66 GB** |
| Python | 3.12.5 (fresh `uv` venv, isolated) |
| MLX | **0.31.2** (mlx-metal 0.31.2) — installs clean; bf16 GEMM + `async_eval` present |
| PyTorch | **2.12.1**, `mps` available & built; bf16 autocast works |

Both frameworks installed without error. MLX 0.31.2 > the 0.30.0 in the prompt, so the
M5 tensor-unit path is present and the OS is new enough — no OS blocker.

## Network (faithfully matched in both)

Encoder-only, pre-LN, ReLU FF. `Linear(643→d)` embed over 93 tokens (92 board + 1 value),
learned positional embedding. Move head: `q,k=Linear(d,d)`, `attn=(q@kᵀ)/√d → (B,92,92)`.
Value head: `Linear(d,3)` on the value token. Loss = policy CE (**fp32** softmax over the
92×92=8464 logits) + value CE; **fwd + bwd + AdamW.step()**; bf16 (torch autocast / MLX
native bf16). Warmup ≥20, ≥100 timed steps (fewer where a config was too slow to finish).
Params match: **big = 14.78M (torch) / 14.76M (MLX)**, **small = 5.06M / 5.05M** (the
small delta is MLX MHA bias defaults — negligible).

## Results — median step time / throughput / peak mem

| framework | model | params | batch | step (ms) | samples/s | peak GB |
|---|---|---|---|---|---|---|
| **mlx (sync)** | small | 5.05M | 1024 | **192** | 5321 | 7.0 |
| torch-mps (MHA) | small | 5.06M | 1024 | 521 | ~1965 | — |
| torch-mps (SDPA) | small | 5.06M | 1024 | 397 | 2579 | 11.1 |
| **mlx (sync)** | small | 5.05M | 4096 | **740** | 5538 | 26.8 |
| torch-mps (MHA) | small | 5.06M | 4096 | 1745 | ~2348 | — |
| **mlx (sync)** | big | 14.76M | 1024 | **448** | 2284 | 12.5 |
| torch-mps (MHA) | big | 14.78M | 1024 | 991 | ~1033 | — |
| torch-mps (SDPA) | big | 14.78M | 1024 | 718 | 1427 | 17.6 |
| mlx (sync) | big | 14.76M | 4096 | **~60–65 s** ⚠ | ~65 | ~49 |
| torch-mps (MHA) | big | 14.78M | 4096 | **~192 s** ⚠ | ~21 | — |

`async_eval`: no material change (small b1024 198 ms, small b4096 739 ms, big b1024 432 ms
vs sync 192/740/448). The step is **GPU-compute-bound**, not launch-overhead-bound, so the
async pipeline trick buys nothing here.

### MLX speedup over the *best* torch path (lower step time wins)

| config | MLX | best torch | **MLX is** |
|---|---|---|---|
| small b=1024 | 192 ms | 397 ms (SDPA) | **2.07× faster** |
| small b=4096 | 740 ms | 1745 ms (MHA)* | **2.36× faster** |
| big b=1024 | 448 ms | 718 ms (SDPA) | **1.60× faster** |
| big b=4096 | ~60 s ⚠ | ~192 s ⚠ | ~3× (both unusable) |

\* torch-SDPA not separately run at b=4096; MHA shown. SDPA would shave ~25%, so the real
small-b4096 margin is ≈1.7–2.4×. Either way MLX wins.

## VERDICT

**Yes — MLX materially beats PyTorch-MPS for our move-net size on this M5 Max: ~1.6× (big,
b=1024) to ~2.4× (small) faster per training step, even after giving PyTorch its fastest
fused-attention (SDPA) path.** The small-matmul GEMM deficit from MLX issue #3196 does NOT
dominate a full training step here: the step is more than isolated GEMMs (LayerNorm,
attention, the 8464-way fp32 softmax+backward, AdamW), and across that whole step MLX is
the clear winner at every batch/size that fits in memory. PyTorch-MPS was not just slower
but *pathological* on the big config at b=4096 (~192 s/step steady-state, ~124 s first-step
compile) — a 4-config sweep couldn't finish a single timed big-b4096 step in 46 minutes.

For the trainer decision: **the hybrid Rust-sim + MLX-Python trainer is the faster compute
path** on Apple silicon at this model size. (Caveat: this is a Python-vs-Python kernel
comparison; tch is libtorch's *same* MPS backend, so Rust+tch would inherit PyTorch-MPS's
~1.6–2.4× disadvantage, not beat it.)

## Caveats — read these

1. **b=4096 is memory-bound on THIS 64 GB unit, not compute-bound.** MLX big-b4096 peaks
   ~49 GB vs the 55.66 GB recommended working set, so it throttles to ~60 s/step — far worse
   than the ~1.8 s that big-b1024×4 would predict. On the **128 GB** machine the prompt
   assumed, big-b4096 has ample headroom and should run ~1.8–2 s/step; the MLX>torch ordering
   would hold (likely widen). **Don't read the b=4096 rows as fundamental** — they're this
   box's RAM ceiling. The decision-relevant rows are b=1024 (and small b=4096), where MLX
   wins cleanly. Plenty of 128 GB headroom for activations/replay on the target hardware.
2. **torch fairness.** Default `nn.MultiheadAttention` hits a slow unfused MPS path;
   `F.scaled_dot_product_attention` (fused) is ~24–28 % faster and is what the SDPA rows use.
   The verdict uses torch's *best* path, so MLX's win is not an artifact of a weak torch impl.
3. **First-step compile.** Both frameworks pay a one-time Metal compile (torch big-b4096
   first step 124 s; MLX 29 s). Amortized over a real run it's negligible, and warmup
   excludes it from the medians — but torch's compile cost is also much larger.
4. **bf16 parity.** torch uses autocast (bf16 matmuls, fp32 master/softmax); MLX casts the
   module to bf16 (`set_dtype`) with fp32 loss/softmax. Slightly different bf16 surfaces, but
   the same intent; this is an unavoidable framework difference, noted for honesty.
5. Timing: torch via `mps.synchronize()` bracketing; MLX via `mx.eval` per step (and an
   async-pipelined variant). Medians over 100 steps (30/8 for the slow big-b4096 probes).

## Files
- `spec.py` — shared model/loss spec
- `bench_torch.py` — PyTorch-MPS benchmark (nn.MultiheadAttention)
- `bench_torch_sdpa.py` — PyTorch-MPS benchmark (fused SDPA)
- `bench_mlx.py` — MLX benchmark (sync + `--async-eval`)
- `diag_torch.py` / `probe_torch.py` — per-phase / per-step torch diagnostics
- `probe_mlx.py` — per-step MLX diagnostic (big-b4096)
- `results_raw.txt` — raw numbers
