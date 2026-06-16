// The azero-gpu go bot: the wasm engine runs the park/resume PUCT search and
// mirrors the game; this driver evaluates each parked leaf batch with the
// WebGPU net (weights from the AZWEBGO2 export) and feeds the results back
// until the search is done.

import type { EngineHost } from '../engine/host';
import type { MatchEventData, ViewState } from '../engine/protocol';
import { GoGpu, policyLen, softmaxOver } from '../frontends/go/azgpu';
import type { ClientBot } from './index';

const DEFAULT_SIMS = 400;
const LEAVES = 8;

/** One device + weight upload per page, not per match. */
let gpuOnce: Promise<GoGpu> | null = null;

function getGpu(): Promise<GoGpu> {
  gpuOnce ??= (async () => {
    const url = `${import.meta.env.BASE_URL}azero/azero-go.azweb`;
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`weights ${url} missing (HTTP ${resp.status})`);
    const gpu = await GoGpu.init(await resp.arrayBuffer());
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

class AzeroGo implements ClientBot {
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

export async function createAzeroGo(
  host: EngineHost,
  opts: Record<string, string>,
): Promise<ClientBot> {
  const gpu = await getGpu();
  const sims = Number(opts.sims) > 0 ? Number(opts.sims) : DEFAULT_SIMS;
  const seed = Number(opts.seed) >>> 0 || 1;
  // The pooled net is board-size-agnostic; play at the requested size (≤ the
  // export's max), no per-size weights needed.
  const size = Number(opts.size) > 0 ? Number(opts.size) : gpu.model.size;
  await host.goNew(sims, LEAVES, seed, size);
  return new AzeroGo(host, gpu, size);
}
