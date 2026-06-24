// The AlphaZero Pente bot. Like the go bot, the wasm engine runs the park/resume
// PUCT search and mirrors the game; this driver supplies the leaf evaluations.
// With WebGPU it answers each parked leaf batch with the GPU net (weights from
// the AZNET1 export); without it, it hands the same weights to the wasm engine
// and lets play_cpu run the whole search in-wasm against nn-infer's reference
// forward — same net, so anyone can play, GPU or not.
//
// The wasm bot also runs the same move-time VCF hybrid the native lab bot does:
// before deferring to the searched move it plays a sound, capture-aware forced
// win when the bounded forcing solver proves one. That is pure Rust inside the
// engine, so it happens transparently here — the advance/best loop is identical
// to go's; a VCF win just makes advance return 0 immediately with best ready.

import { CPU_MAX_SIMS, isCpuFallback, TRIVIAL_SIMS } from '../shell/azero';
import type { EngineHost } from '../engine/host';
import type { MatchEventData, ViewState } from '../engine/protocol';
import { PenteGpu, policyLen, softmaxOver } from '../frontends/pente/azgpu';
import { setPenteEval } from '../frontends/pente/eval-bridge';
import { gpuLoader, weightsLoader } from './azero-net';
import type { ClientBot } from './index';

// The shipped net is trained at 19×19; the arcade pins the AZ matchup there.
const SIZE = 19;
const DEFAULT_SIMS = 400;
const LEAVES = 8;
// The native bot's move-time forcing budget (depth 8, ~4000 nodes): enough to
// prove the short forcing wins at the root while returning well within a move.
const VCF_DEPTH = 8;
const VCF_NODES = 4000;
const getWeights = weightsLoader(`${import.meta.env.BASE_URL}azero/azero-pente.azweb`);
const getGpu = gpuLoader(PenteGpu.init, getWeights);

class AzeroPenteGpu implements ClientBot {
  private cancelled = false;
  private readonly stride = policyLen(SIZE);

  constructor(
    private host: EngineHost,
    private gpu: PenteGpu,
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
      const { logits, values: v } = await this.gpu.forward(batch.features, batch.n, SIZE);
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
    setPenteEval(null);
  }
}

/** No WebGPU: the search and the reference forward both run in the wasm worker.
 * One round-trip per move (the search is atomic worker-side), so no advance loop
 * to cancel — just a guard so a torn-down match drops its move. */
class AzeroPenteCpu implements ClientBot {
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
    setPenteEval(null);
  }
}

export async function createAzeroPente(
  host: EngineHost,
  opts: Record<string, string>,
): Promise<ClientBot> {
  const seed = Number(opts.seed) >>> 0 || 1;
  const vcfDepth = Number(opts['vcf-depth']) > 0 ? Number(opts['vcf-depth']) : VCF_DEPTH;
  const vcfNodes = Number(opts['vcf-nodes']) > 0 ? Number(opts['vcf-nodes']) : VCF_NODES;
  // Prefer WebGPU; if the device fails to come up even where it is advertised,
  // fall through to CPU rather than failing the match.
  if (!isCpuFallback()) {
    try {
      const gpu = await getGpu();
      const sims = Number(opts.sims) > 0 ? Number(opts.sims) : DEFAULT_SIMS;
      // GPU path evaluates leaves page-side, so the wasm bot needs no weights;
      // the VCF hybrid is net-free and runs regardless.
      await host.penteNew(sims, LEAVES, seed, SIZE, vcfDepth, vcfNodes, await getWeights());
      setPenteEval(() => host.penteEval());
      return new AzeroPenteGpu(host, gpu);
    } catch {
      // fall through to the CPU forward
    }
  }
  // CPU: the chosen level, capped so moves stay responsive without a GPU.
  const sims = Math.min(Number(opts.sims) > 0 ? Number(opts.sims) : TRIVIAL_SIMS, CPU_MAX_SIMS);
  await host.penteNew(sims, LEAVES, seed, SIZE, vcfDepth, vcfNodes, await getWeights());
  setPenteEval(() => host.penteEval());
  return new AzeroPenteCpu(host);
}
