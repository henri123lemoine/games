// The AlphaZero snake bot. The wasm engine runs the park/resume PUCT search;
// this driver supplies the leaf evaluations. With WebGPU it answers each parked
// leaf batch with the GPU net (weights from the AZNET1 export); without it, it
// hands the same weights to the wasm engine and lets snakePlayCpu run the whole
// search in-wasm against nn-infer's reference forward.
//
// Why GPU is the primary path for this tiny 4×64 net (measured, not assumed):
// the conv forward is ~120M MACs, which the wasm CPU does in ~25 ms/leaf even
// with SIMD — so a real search is hopeless on the CPU (128 sims ≈ 3 s). On the
// GPU the compute is effectively free and the only cost is the per-batch
// mapAsync round-trip; gathering wide batches keeps a 64–128-sim move to a
// handful of round-trips (~140–200 ms headless, less in-browser). The CPU path
// therefore exists only as a correctness fallback for visitors with no WebGPU,
// pinned to a tiny budget so it stays responsive (and labelled as degraded via
// the telemetry seam).
//
// Snake's food placement is a chance node the engine resolves with its own rng,
// so the bot can't mirror moves like the go/chess bots; instead it reconstructs
// its search root from the engine's view JSON before each move (snakeSetState),
// which also resets the tree.

import type { EngineHost } from '../engine/host';
import type { ViewState } from '../engine/protocol';
import { SnakeGpu, softmaxOver } from '../frontends/snake/azgpu';
import { reportMove } from '../frontends/snake/telemetry';
import { isCpuFallback } from '../shell/azero';
import { gpuLoader, weightsLoader } from './azero-net';
import type { ClientBot } from './index';

// Each parked leaf batch is one GPU round-trip (a mapAsync readback), and on
// this net that round-trip — not the conv compute — sets the latency, so the
// search gathers batches up to the evaluator's MAX_BATCH to spend as few
// round-trips per move as possible.
const LEAVES = 32;
/** GPU default: ~3 round-trips/move, ~140 ms headless (faster in-browser), and
 * strong — the policy is already sharp, so this much search plays well. */
const GPU_DEFAULT_SIMS = 64;
/** GPU ceiling: ~5 round-trips, ~200 ms headless. */
const GPU_MAX_SIMS = 128;
/** CPU fallback ceiling. The wasm forward is ~25 ms/leaf, so even this is
 * ~200 ms/move; anything higher is unplayable without a GPU. */
const CPU_DEFAULT_SIMS = 4;
const CPU_MAX_SIMS = 8;
const getWeights = weightsLoader(`${import.meta.env.BASE_URL}azero/azero-snake.azweb`);
const getGpu = gpuLoader(SnakeGpu.init, getWeights);

/** WebGPU: the search runs in the wasm worker, the leaves on the GPU. The bot
 * resets its root from the view each move, so applied moves need no mirroring. */
class AzeroSnakeGpu implements ClientBot {
  private cancelled = false;
  constructor(
    private host: EngineHost,
    private gpu: SnakeGpu,
  ) {}

  onMove(): Promise<void> {
    return Promise.resolve();
  }

  async chooseMove(st: ViewState): Promise<string> {
    if (this.cancelled) throw new Error('cancelled');
    const t0 = performance.now();
    await this.host.snakeSetState(JSON.stringify(st.viewData));
    let priors = new Float32Array(0);
    let values = new Float32Array(0);
    let trips = 0;
    for (;;) {
      if (this.cancelled) throw new Error('cancelled');
      const batch = await this.host.snakeAdvance(priors, values);
      if (batch.n === 0) break;
      if (this.cancelled) throw new Error('cancelled');
      trips++;
      const { logits, values: v } = await this.gpu.forward(batch.features, batch.n);
      const flat: number[] = [];
      for (let i = 0; i < batch.n; i++) {
        const support = batch.support.subarray(batch.offsets[i], batch.offsets[i + 1]);
        flat.push(...softmaxOver(logits, support, i * 4));
      }
      priors = Float32Array.from(flat);
      values = v.slice(0, batch.n);
    }
    const { uci, stats } = await this.host.snakeBest();
    reportMove({ backend: 'gpu', ms: performance.now() - t0, sims: stats.sims, trips });
    return uci;
  }

  cancel(): void {
    this.cancelled = true;
  }
}

/** No WebGPU: the search and the reference forward both run in the wasm worker.
 * One round-trip per move (the search is atomic worker-side), so no advance
 * loop to cancel — just a guard so a torn-down match drops its move. */
class AzeroSnakeCpu implements ClientBot {
  private cancelled = false;
  constructor(private host: EngineHost) {}

  onMove(): Promise<void> {
    return Promise.resolve();
  }

  async chooseMove(st: ViewState): Promise<string> {
    if (this.cancelled) throw new Error('cancelled');
    const t0 = performance.now();
    const { uci, stats } = await this.host.snakePlayCpu(JSON.stringify(st.viewData));
    if (this.cancelled) throw new Error('cancelled');
    reportMove({ backend: 'cpu', ms: performance.now() - t0, sims: stats.sims, trips: 0 });
    return uci;
  }

  cancel(): void {
    this.cancelled = true;
  }
}

export async function createAzeroSnake(
  host: EngineHost,
  opts: Record<string, string>,
): Promise<ClientBot> {
  const seed = Number(opts.seed) >>> 0 || 1;
  const wantSims = Number(opts.sims) > 0 ? Number(opts.sims) : 0;
  // Prefer WebGPU; it is the only backend that can run a real search on this
  // net responsively. Fall back to the CPU forward only if there is no device
  // or it fails to come up — and say so loudly, since the fallback is far
  // slower and weaker (the user's overlay shows which backend actually ran).
  if (!isCpuFallback()) {
    try {
      const gpu = await getGpu();
      const sims = Math.min(wantSims || GPU_DEFAULT_SIMS, GPU_MAX_SIMS);
      // GPU path evaluates leaves page-side, so the wasm bot needs no weights.
      await host.snakeNew(sims, LEAVES, seed);
      console.info(`[snake] WebGPU backend, ${sims} sims`);
      return new AzeroSnakeGpu(host, gpu);
    } catch (e) {
      console.warn('[snake] WebGPU init failed, falling back to the slow CPU forward:', e);
    }
  } else {
    console.warn('[snake] no WebGPU (navigator.gpu absent); using the slow CPU forward');
  }
  // CPU: pinned to a tiny budget so moves stay responsive without a GPU.
  const sims = Math.min(wantSims || CPU_DEFAULT_SIMS, CPU_MAX_SIMS);
  await host.snakeNew(sims, LEAVES, seed, await getWeights());
  console.info(`[snake] CPU fallback backend, ${sims} sims (degraded — no GPU)`);
  return new AzeroSnakeCpu(host);
}
