// The AlphaZero go bot. The wasm engine runs the park/resume PUCT search and
// mirrors the game; this driver supplies the leaf evaluations. With WebGPU it
// answers each parked leaf batch with the GPU net (weights from the AZWEBGO2
// export); without it, it hands the same weights to the wasm engine and lets
// `play_cpu` run the whole search in-wasm against goinfer's reference forward —
// same net, so anyone can play, GPU or not.

import { isCpuFallback, TRIVIAL_SIMS } from '../shell/azero';
import type { EngineHost } from '../engine/host';
import type { MatchEventData, ViewState } from '../engine/protocol';
import { GoGpu, policyLen, softmaxOver } from '../frontends/go/azgpu';
import type { ClientBot } from './index';

const DEFAULT_SIMS = 400;
const LEAVES = 8;
const WEIGHTS_URL = `${import.meta.env.BASE_URL}azero/azero-go.azweb`;

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
let gpuOnce: Promise<GoGpu> | null = null;
function getGpu(): Promise<GoGpu> {
  gpuOnce ??= (async () => {
    const gpu = await GoGpu.init(await getWeights());
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

class AzeroGoGpu implements ClientBot {
  private cancelled = false;
  private readonly stride: number;

  constructor(
    private host: EngineHost,
    private gpu: GoGpu,
    private size: number,
  ) {
    this.stride = policyLen(size);
  }

  onMove(ev: MatchEventData): Promise<void> {
    return this.host.azPush(ev.label);
  }

  async chooseMove(_st: ViewState): Promise<string> {
    let priors = new Float32Array(0);
    let values = new Float32Array(0);
    for (;;) {
      if (this.cancelled) throw new Error('cancelled');
      const batch = await this.host.azAdvance(priors, values);
      if (batch.n === 0) break;
      if (this.cancelled) throw new Error('cancelled');
      const { logits, values: v } = await this.gpu.forward(batch.features, batch.n, this.size);
      const flat: number[] = [];
      for (let i = 0; i < batch.n; i++) {
        const support = batch.support.subarray(batch.offsets[i], batch.offsets[i + 1]);
        flat.push(...softmaxOver(logits, support, i * this.stride));
      }
      priors = Float32Array.from(flat);
      values = v.slice(0, batch.n);
    }
    return (await this.host.azBest()).uci;
  }

  cancel(): void {
    this.cancelled = true;
  }
}

/** No WebGPU: the search and the reference forward both run in the wasm
 * worker. One round-trip per move (the search is atomic worker-side), so no
 * advance loop to cancel — just a guard so a torn-down match drops its move. */
class AzeroGoCpu implements ClientBot {
  private cancelled = false;
  constructor(private host: EngineHost) {}

  onMove(ev: MatchEventData): Promise<void> {
    return this.host.azPush(ev.label);
  }

  async chooseMove(_st: ViewState): Promise<string> {
    if (this.cancelled) throw new Error('cancelled');
    const { uci } = await this.host.azPlayCpu();
    if (this.cancelled) throw new Error('cancelled');
    return uci;
  }

  cancel(): void {
    this.cancelled = true;
  }
}

export async function createAzeroGo(
  host: EngineHost,
  opts: Record<string, string>,
): Promise<ClientBot> {
  const seed = Number(opts.seed) >>> 0 || 1;
  // Prefer WebGPU; if the device fails to come up even where it is advertised,
  // fall through to CPU rather than failing the match.
  if (!isCpuFallback()) {
    try {
      const gpu = await getGpu();
      const sims = Number(opts.sims) > 0 ? Number(opts.sims) : DEFAULT_SIMS;
      // The pooled net is board-size-agnostic; play at the requested size (≤
      // the export's max), no per-size weights needed.
      const size = Number(opts.size) > 0 ? Number(opts.size) : gpu.model.size;
      await host.goNew(sims, LEAVES, seed, size);
      return new AzeroGoGpu(host, gpu, size);
    } catch {
      // fall through to the CPU forward
    }
  }
  // CPU: locked to the trivial visit budget so moves stay responsive.
  const size = Number(opts.size) > 0 ? Number(opts.size) : 19;
  await host.goNew(TRIVIAL_SIMS, LEAVES, seed, size, await getWeights());
  return new AzeroGoCpu(host);
}
