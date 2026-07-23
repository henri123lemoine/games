// The AlphaZero go bot. The wasm engine runs the park/resume PUCT search and
// mirrors the game; this driver supplies the leaf evaluations. With WebGPU it
// answers each parked leaf batch with the GPU net (weights from the AZNET1
// export); without it, it hands the same weights to the wasm engine and lets
// `play_cpu` run the whole search in-wasm against nn-infer's reference forward —
// same net, so anyone can play, GPU or not.

import { CPU_MAX_SIMS, cpuFallbackMessage, isCpuFallback } from '../shell/azero';
import type { EngineHost } from '../engine/host';
import type { MatchEventData, ViewState } from '../engine/protocol';
import { GoGpu, policyLen, softmaxOver } from '../frontends/go/azgpu';
import { setGoEval } from '../frontends/go/eval-bridge';
import { assetUrl } from '../assets';
import { gpuLoader, weightsLoader } from './azero-net';
import type { ClientBot } from './index';
import { errorMessage, requiredU32 } from './options';

const LEAVES = 8;
// Shared with the shell's per-visitor forward self-check so it validates the
// same device + weights the bot booted, with no extra fetch or device init.
export const getGoWeights = weightsLoader(assetUrl('azero/azero-go.azweb'));
export const getGoGpu = gpuLoader(GoGpu.init, getGoWeights);
const getWeights = getGoWeights;
const getGpu = getGoGpu;

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

  finalResult(): Promise<string> {
    return this.host.azFinalResult();
  }

  cancel(): void {
    this.cancelled = true;
    setGoEval(null);
  }
}

/** No WebGPU: the search and the reference forward both run in the wasm
 * worker. One round-trip per move (the search is atomic worker-side), so no
 * advance loop to cancel — just a guard so a torn-down match drops its move. */
class AzeroGoCpu implements ClientBot {
  private cancelled = false;
  constructor(
    private host: EngineHost,
    readonly cpuFallback: string,
  ) {}

  onMove(ev: MatchEventData): Promise<void> {
    return this.host.azPush(ev.label);
  }

  async chooseMove(_st: ViewState): Promise<string> {
    if (this.cancelled) throw new Error('cancelled');
    const { uci } = await this.host.azPlayCpu();
    if (this.cancelled) throw new Error('cancelled');
    return uci;
  }

  finalResult(): Promise<string> {
    return this.host.azFinalResult();
  }

  cancel(): void {
    this.cancelled = true;
    setGoEval(null);
  }
}

export async function createAzeroGo(
  host: EngineHost,
  opts: Record<string, string>,
): Promise<ClientBot> {
  const seed = requiredU32(opts, 'seed');
  const requestedSims = requiredU32(opts, 'sims');
  const size = requiredU32(opts, 'size');
  let cpuReason = 'No compatible WebGPU device was detected';
  // Prefer WebGPU; if the device fails to come up even where it is advertised,
  // fall through to CPU rather than failing the match.
  if (!isCpuFallback()) {
    let gpu: GoGpu | null = null;
    try {
      gpu = await getGpu();
    } catch (error) {
      cpuReason = `WebGPU initialization failed: ${errorMessage(error)}`;
    }
    if (gpu) {
      // The pooled net is board-size-agnostic; play at the requested size (≤
      // the export's max), no per-size weights needed.
      // Weights also go to the wasm bot so it can run the ownership head for
      // the pass decision; leaf evaluation still happens on the GPU.
      await host.goNew(requestedSims, LEAVES, seed, size, await getWeights());
      setGoEval(() => host.goEval());
      return new AzeroGoGpu(host, gpu, size);
    }
  }
  // CPU: the chosen level, capped so moves stay responsive without a GPU.
  const sims = Math.min(requestedSims, CPU_MAX_SIMS);
  await host.goNew(sims, LEAVES, seed, size, await getWeights());
  setGoEval(() => host.goEval());
  return new AzeroGoCpu(host, cpuFallbackMessage(cpuReason, sims));
}
