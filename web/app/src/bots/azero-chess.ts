// The AlphaZero chess bot. The wasm engine runs the park/resume PUCT search
// and mirrors the game; this driver supplies the leaf evaluations. With WebGPU
// it answers each parked leaf batch with the GPU net (weights from the AZNET1
// export); without it, it hands the same weights to the wasm engine and lets
// `play_cpu` run the whole search in-wasm against nn-infer's reference forward —
// same net, so anyone can play, GPU or not.

import { CPU_MAX_SIMS, cpuFallbackMessage, isCpuFallback, TRIVIAL_SIMS } from '../shell/azero';
import type { EngineHost } from '../engine/host';
import type { MatchEventData, ViewState } from '../engine/protocol';
import { AzGpu, POLICY_LEN, softmaxOver } from '../frontends/chess/azgpu';
import { setChessEval } from '../frontends/chess/eval-bridge';
import { gpuLoader, weightsLoader } from './azero-net';
import type { ClientBot } from './index';

const DEFAULT_SIMS = 600;
const LEAVES = 8;
const getWeights = weightsLoader(`${import.meta.env.BASE_URL}azero/azero-chess.azweb`);
const getGpu = gpuLoader(AzGpu.init, getWeights);

class AzeroChessGpu implements ClientBot {
  private cancelled = false;

  constructor(
    private host: EngineHost,
    private gpu: AzGpu,
  ) {}

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
      const { logits, values: v } = await this.gpu.forward(batch.features, batch.n);
      const flat: number[] = [];
      for (let i = 0; i < batch.n; i++) {
        const support = batch.support.subarray(batch.offsets[i], batch.offsets[i + 1]);
        flat.push(...softmaxOver(logits, support, i * POLICY_LEN));
      }
      priors = Float32Array.from(flat);
      values = v.slice(0, batch.n);
    }
    return (await this.host.azBest()).uci;
  }

  cancel(): void {
    this.cancelled = true;
    setChessEval(null);
  }
}

/** No WebGPU: the search and the reference forward both run in the wasm
 * worker. One round-trip per move (the search is atomic worker-side), so no
 * advance loop to cancel — just a guard so a torn-down match drops its move. */
class AzeroChessCpu implements ClientBot {
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

  cancel(): void {
    this.cancelled = true;
    setChessEval(null);
  }
}

export async function createAzeroChess(
  host: EngineHost,
  opts: Record<string, string>,
): Promise<ClientBot> {
  const seed = Number(opts.seed) >>> 0 || 1;
  let cpuReason = 'No compatible WebGPU device was detected';
  // Prefer WebGPU; if the device fails to come up even where it is advertised,
  // fall through to CPU rather than failing the match.
  if (!isCpuFallback()) {
    try {
      const gpu = await getGpu();
      const sims = Number(opts.sims) > 0 ? Number(opts.sims) : DEFAULT_SIMS;
      // GPU search evaluates leaves page-side, so the wasm bot needs no weights
      // to play — but the debug position readout (`chessEval`) runs the net's
      // value head in-wasm, so load them anyway (the bytes are already fetched).
      await host.azNew(sims, LEAVES, seed, await getWeights());
      setChessEval(() => host.chessEval());
      return new AzeroChessGpu(host, gpu);
    } catch {
      cpuReason = 'WebGPU was detected, but initialization failed';
      // fall through to the CPU forward
    }
  }
  // CPU: the chosen level, capped so moves stay responsive without a GPU.
  const sims = Math.min(Number(opts.sims) > 0 ? Number(opts.sims) : TRIVIAL_SIMS, CPU_MAX_SIMS);
  await host.azNew(sims, LEAVES, seed, await getWeights());
  setChessEval(() => host.chessEval());
  return new AzeroChessCpu(host, cpuFallbackMessage(cpuReason, sims));
}
