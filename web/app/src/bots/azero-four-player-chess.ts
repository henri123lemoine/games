// Four-player chess AlphaZero driver. Every external bot seat owns an
// independent wasm search tree; all four share the page-lifetime weights/GPU.

import { CPU_MAX_SIMS, cpuFallbackMessage, isCpuFallback } from '../shell/azero';
import type { EngineHost } from '../engine/host';
import type { MatchEventData, ViewState } from '../engine/protocol';
import {
  FourPlayerAzGpu,
  POLICY_LEN,
  softmaxOver,
  VALUE_SEATS,
} from '../frontends/four-player-chess/azgpu';
import { gpuLoader, weightsLoader } from './azero-net';
import type { ClientBot } from './index';
import { errorMessage, requiredU32 } from './options';

const LEAVES = 8;
const getWeights = weightsLoader(
  `${import.meta.env.BASE_URL}azero/four-player-chess.azweb`,
);
const getGpu = gpuLoader(FourPlayerAzGpu.init, getWeights);

function seatSoftmax(logits: Float32Array, batch: number): Float32Array {
  const values = new Float32Array(batch * VALUE_SEATS);
  for (let row = 0; row < batch; row++) {
    const base = row * VALUE_SEATS;
    const raw = Array.from(logits.subarray(base, base + VALUE_SEATS));
    const max = Math.max(...raw);
    const exps = raw.map((value) => Math.exp(value - max));
    const total = exps.reduce((sum, value) => sum + value, 0);
    for (let seat = 0; seat < VALUE_SEATS; seat++) values[base + seat] = exps[seat] / total;
  }
  return values;
}

class FourPlayerGpuBot implements ClientBot {
  private cancelled = false;

  constructor(
    private host: EngineHost,
    private gpu: FourPlayerAzGpu,
  ) {}

  onMove(event: MatchEventData): Promise<void> {
    return this.host.azPush(event.label);
  }

  async chooseMove(_state: ViewState): Promise<string> {
    let priors = new Float32Array(0);
    let values = new Float32Array(0);
    for (;;) {
      if (this.cancelled) throw new Error('cancelled');
      const batch = await this.host.azAdvance(priors, values);
      if (batch.n === 0) break;
      const output = await this.gpu.forward(batch.features, batch.n);
      const flat: number[] = [];
      for (let row = 0; row < batch.n; row++) {
        const support = batch.support.subarray(batch.offsets[row], batch.offsets[row + 1]);
        flat.push(...softmaxOver(output.logits, support, row * POLICY_LEN));
      }
      priors = Float32Array.from(flat);
      values = new Float32Array(seatSoftmax(output.values, batch.n));
    }
    return (await this.host.azBest()).uci;
  }

  cancel(): void {
    this.cancelled = true;
  }
}

class FourPlayerCpuBot implements ClientBot {
  private cancelled = false;

  constructor(
    private host: EngineHost,
    readonly cpuFallback: string,
  ) {}

  onMove(event: MatchEventData): Promise<void> {
    return this.host.azPush(event.label);
  }

  async chooseMove(_state: ViewState): Promise<string> {
    if (this.cancelled) throw new Error('cancelled');
    const move = await this.host.azPlayCpu();
    if (this.cancelled) throw new Error('cancelled');
    return move.uci;
  }

  cancel(): void {
    this.cancelled = true;
  }
}

export async function createAzeroFourPlayerChess(
  host: EngineHost,
  opts: Record<string, string>,
): Promise<ClientBot> {
  const seed = requiredU32(opts, 'seed');
  const requestedSims = requiredU32(opts, 'sims');
  let cpuReason = 'No compatible WebGPU device was detected';
  if (!isCpuFallback()) {
    try {
      const gpu = await getGpu();
      await host.fourNew(requestedSims, LEAVES, seed);
      return new FourPlayerGpuBot(host, gpu);
    } catch (error) {
      cpuReason = `WebGPU initialization failed: ${errorMessage(error)}`;
    }
  }
  const sims = Math.min(requestedSims, CPU_MAX_SIMS);
  await host.fourNew(sims, LEAVES, seed, await getWeights());
  return new FourPlayerCpuBot(host, cpuFallbackMessage(cpuReason, sims));
}
