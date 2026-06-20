// The AlphaZero snake bot. The wasm engine runs the park/resume PUCT search;
// this driver supplies the leaf evaluations. With WebGPU it answers each parked
// leaf batch with the GPU net (weights from the AZSNK1 export); without it, it
// hands the same weights to the wasm engine and lets snakePlayCpu run the whole
// search in-wasm against snakeinfer's reference forward — same net, so anyone
// can play, GPU or not.
//
// Snake's food placement is a chance node the engine resolves with its own rng,
// so the bot can't mirror moves like the go/chess bots; instead it reconstructs
// its search root from the engine's view JSON before each move (snakeSetState),
// which also resets the tree.

import type { EngineHost } from '../engine/host';
import type { ViewState } from '../engine/protocol';
import { SnakeGpu, softmaxOver } from '../frontends/snake/azgpu';
import { CPU_MAX_SIMS, isCpuFallback, TRIVIAL_SIMS } from '../shell/azero';
import type { ClientBot } from './index';

// Each parked leaf batch is one GPU round-trip (a mapAsync readback), and on
// the snake net that round-trip — not the conv compute — sets the latency. So
// gather wide batches (capped at SnakeGpu's MAX_BATCH) to keep moves to a
// handful of round-trips; 32 leaves halves the per-move latency vs 8.
const LEAVES = 32;
/** A strong budget that stays under ~300 ms/move even on the slowest WebGPU
 * round-trip we measured headless; in-browser it is well faster. ~128 sims of
 * the 4×64 net is genuinely strong play. */
const DEFAULT_SIMS = 128;
const MAX_SIMS = 256;
const WEIGHTS_URL = `${import.meta.env.BASE_URL}azero/azero-snake.azweb`;

/** The raw export bytes, fetched once per page and shared by both backends. */
let weightsOnce: Promise<ArrayBuffer> | null = null;
function getWeights(): Promise<ArrayBuffer> {
  weightsOnce ??= (async () => {
    const resp = await fetch(WEIGHTS_URL);
    if (!resp.ok) throw new Error(`weights ${WEIGHTS_URL} missing (HTTP ${resp.status})`);
    return resp.arrayBuffer();
  })();
  weightsOnce.catch(() => {
    weightsOnce = null;
  });
  return weightsOnce;
}

/** One device + weight upload per page, not per match. */
let gpuOnce: Promise<SnakeGpu> | null = null;
function getGpu(): Promise<SnakeGpu> {
  gpuOnce ??= (async () => {
    const gpu = await SnakeGpu.init(await getWeights());
    void gpu.lost.then(() => {
      gpuOnce = null;
    });
    return gpu;
  })();
  gpuOnce.catch(() => {
    gpuOnce = null;
  });
  return gpuOnce;
}

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
    await this.host.snakeSetState(JSON.stringify(st.viewData));
    let priors = new Float32Array(0);
    let values = new Float32Array(0);
    for (;;) {
      if (this.cancelled) throw new Error('cancelled');
      const batch = await this.host.snakeAdvance(priors, values);
      if (batch.n === 0) break;
      if (this.cancelled) throw new Error('cancelled');
      const { logits, values: v } = await this.gpu.forward(batch.features, batch.n);
      const flat: number[] = [];
      for (let i = 0; i < batch.n; i++) {
        const support = batch.support.subarray(batch.offsets[i], batch.offsets[i + 1]);
        flat.push(...softmaxOver(logits, support, i * 4));
      }
      priors = Float32Array.from(flat);
      values = v.slice(0, batch.n);
    }
    return (await this.host.snakeBest()).uci;
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
    const { uci } = await this.host.snakePlayCpu(JSON.stringify(st.viewData));
    if (this.cancelled) throw new Error('cancelled');
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
  // Prefer WebGPU; if the device fails to come up even where it is advertised,
  // fall through to CPU rather than failing the match.
  if (!isCpuFallback()) {
    try {
      const gpu = await getGpu();
      const sims = Math.min(Number(opts.sims) > 0 ? Number(opts.sims) : DEFAULT_SIMS, MAX_SIMS);
      // GPU path evaluates leaves page-side, so the wasm bot needs no weights.
      await host.snakeNew(sims, LEAVES, seed);
      return new AzeroSnakeGpu(host, gpu);
    } catch {
      // fall through to the CPU forward
    }
  }
  // CPU: the chosen level, capped so moves stay responsive without a GPU.
  const sims = Math.min(Number(opts.sims) > 0 ? Number(opts.sims) : TRIVIAL_SIMS, CPU_MAX_SIMS);
  await host.snakeNew(sims, LEAVES, seed, await getWeights());
  return new AzeroSnakeCpu(host);
}
