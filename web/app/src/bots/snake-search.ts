// The snake bot's search, run COMPLETELY OFF the player's critical path.
//
// Why this exists: snake is real-time. The player's snake must advance on a
// fixed game clock and turn within one tick of a keypress, NEVER waiting on the
// bot's ~190ms PUCT search. So the search does not run inside the game tick at
// all — it runs here as a perpetual background loop on its OWN engine worker
// (separate from the worker that advances the game), continuously re-searching
// the latest board and publishing a single "current best move". The game tick
// reads that best move synchronously (no await) and moves on; if a fresh search
// isn't ready, the tick uses the last one (a touch staler / weaker, which is
// explicitly fine — the player never waits for the bot).
//
// Decoupling guarantees:
//   - SEPARATE worker: the search's `snakeAdvance` round-trips can't queue ahead
//     of the game worker's `apply` calls, so a slow search can't delay a tick.
//   - The loop yields to the event loop every batch (await), so the main thread
//     stays free for input + the rAF glide every frame.
//   - The tick never calls into this class with an await that blocks on a
//     search; it only reads `best` (a plain field).

import { EngineHost } from '../engine/host';
import { SnakeGpu, softmaxOver } from '../frontends/snake/azgpu';
import { reportMove } from '../frontends/snake/telemetry';
import { cpuFallbackMessage, isCpuFallback } from '../shell/azero';
import { gpuLoader, weightsLoader } from './azero-net';

const LEAVES = 32;
const GPU_DEFAULT_SIMS = 128;
const GPU_MAX_SIMS = 128;
const CPU_DEFAULT_SIMS = 4;
const CPU_MAX_SIMS = 8;
/** Per-search wall-clock cap (anytime). The search is off the critical path, so
 * this only bounds how stale a published best move can get, not the game pace. */
const SEARCH_BUDGET_MS = 190;

/** Test hook: `?snakeSlowBot=<ms>` injects a delay into each search batch to
 * make the bot artificially slow — the validation harness uses it to PROVE the
 * player's input latency and tick rate are independent of the bot's compute.
 * Off (0) in normal play. */
function slowBotMs(): number {
  try {
    const v = Number(new URLSearchParams(window.location.search).get('snakeSlowBot'));
    return Number.isFinite(v) && v > 0 ? v : 0;
  } catch {
    return 0;
  }
}

const getWeights = weightsLoader(`${import.meta.env.BASE_URL}azero/azero-snake.azweb`);
const getGpu = gpuLoader(SnakeGpu.init, getWeights);

export interface SnakeSearchHandle {
  /** Point the search at the latest board (the seat-1-to-move view JSON). The
   * published best is cleared, so `best()` only returns a result computed for
   * THIS root — never a stale move for an old board. */
  setRoot(viewJson: string): void;
  /** The completed best heading label for the CURRENT root, or null if the
   * search hasn't finished one yet (the common case at the fast clock, when the
   * driver uses the policy floor instead). Read synchronously by the tick. */
  best(): string | null;
  /** Backend actually running, for the HUD. */
  backend(): 'gpu' | 'cpu';
  stop(): void;
}

/** GPU background search: PUCT in its own worker, leaves on the shared GPU. */
class GpuSearch implements SnakeSearchHandle {
  private stopped = false;
  private root: string | null = null;
  private current: string | null = null;
  private rev = 0; // bumped whenever a new root arrives, to abandon a stale search

  constructor(
    private host: EngineHost,
    private gpu: SnakeGpu,
  ) {
    void this.loop();
  }

  setRoot(viewJson: string): void {
    if (viewJson !== this.root) {
      this.root = viewJson;
      this.current = null; // the old best was for an old board — never reuse it
      this.rev++;
    }
  }

  best(): string | null {
    return this.current;
  }

  backend(): 'gpu' | 'cpu' {
    return 'gpu';
  }

  stop(): void {
    this.stopped = true;
  }

  /** Perpetual: take the latest root, search it to the budget (yielding every
   * batch), publish the best move, repeat. Abandons a search whose root went
   * stale so a new board is picked up promptly. */
  private async loop(): Promise<void> {
    while (!this.stopped) {
      const root = this.root;
      if (root === null) {
        await sleep(8);
        continue;
      }
      const myRev = this.rev;
      try {
        await this.searchOnce(root, myRev);
      } catch {
        // A transient worker/GPU hiccup: pause briefly and retry; the tick keeps
        // using the last good move meanwhile.
        await sleep(16);
      }
    }
  }

  private async searchOnce(root: string, myRev: number): Promise<void> {
    const t0 = performance.now();
    const deadline = t0 + SEARCH_BUDGET_MS;
    await this.host.snakeSetState(root);
    let priors = new Float32Array(0);
    let values = new Float32Array(0);
    let trips = 0;
    const slow = slowBotMs();
    for (;;) {
      if (this.stopped || this.rev !== myRev) return; // root went stale
      if (slow > 0) await sleep(slow); // test hook: artificially slow the bot
      const batch = await this.host.snakeAdvance(priors, values);
      if (batch.n === 0) break;
      trips++;
      const { logits, values: v } = await this.gpu.forward(batch.features, batch.n);
      const flat: number[] = [];
      for (let i = 0; i < batch.n; i++) {
        const support = batch.support.subarray(batch.offsets[i], batch.offsets[i + 1]);
        flat.push(...softmaxOver(logits, support, i * 4));
      }
      priors = Float32Array.from(flat);
      values = v.slice(0, batch.n);
      if (performance.now() >= deadline) break;
    }
    if (this.stopped || this.rev !== myRev) return;
    const { uci, stats } = await this.host.snakeBest();
    if (this.stopped || this.rev !== myRev) return; // root changed during the readback
    this.current = uci;
    reportMove({ backend: 'gpu', ms: performance.now() - t0, sims: stats.sims, trips });
  }
}

/** CPU background search (no WebGPU): the whole search runs in the worker; we
 * just call it in a loop on the latest root. Tiny budget so each pass is short. */
class CpuSearch implements SnakeSearchHandle {
  private stopped = false;
  private root: string | null = null;
  private current: string | null = null;

  constructor(private host: EngineHost) {
    void this.loop();
  }

  setRoot(viewJson: string): void {
    if (viewJson !== this.root) {
      this.root = viewJson;
      this.current = null;
    }
  }

  best(): string | null {
    return this.current;
  }

  backend(): 'gpu' | 'cpu' {
    return 'cpu';
  }

  stop(): void {
    this.stopped = true;
  }

  private async loop(): Promise<void> {
    while (!this.stopped) {
      const root = this.root;
      if (root === null) {
        await sleep(8);
        continue;
      }
      try {
        const slow = slowBotMs();
        if (slow > 0) await sleep(slow); // test hook: artificially slow the bot
        const t0 = performance.now();
        const { uci, stats } = await this.host.snakePlayCpu(root);
        if (this.stopped) return;
        if (this.root === root) {
          // Only publish if still the current board (else it's stale).
          this.current = uci;
          reportMove({ backend: 'cpu', ms: performance.now() - t0, sims: stats.sims, trips: 0 });
        }
      } catch {
        await sleep(16);
      }
      await sleep(0); // yield so the loop never hogs the microtask queue
    }
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

/** The always-available CPU policy floor: one net forward + 1-ply safety per
 * call (~1-2ms), on its OWN worker that runs NOTHING ELSE — so it is never
 * blocked by the heavy search and answers every tick promptly, even with no GPU
 * and a throttled CPU. This is the guaranteed non-suicidal move. */
export class SnakePolicy {
  constructor(private host: EngineHost) {}

  /** A safe, competent move for the given board (seat-1-to-move view JSON). */
  policyMove(viewJson: string): Promise<string> {
    return this.host.snakePolicyMove(viewJson);
  }
}

export interface SnakeBot {
  policy: SnakePolicy;
  /** The background refinement search, or null if it couldn't start (the policy
   * floor alone still plays). */
  search: SnakeSearchHandle | null;
  cpuFallback?: string;
  stop(): void;
}

/** Spin up the snake bot: a CPU policy FLOOR on its own worker (always loaded
 * with weights, so it works with no WebGPU) plus a background refinement search
 * on a SECOND worker (GPU when available, else the slow CPU MCTS). Each runs on
 * its own worker so neither can ever contend with the game tick. The caller owns
 * the bot and must `stop()` it on teardown. */
export async function createSnakeBot(opts: Record<string, string>): Promise<SnakeBot> {
  const seed = Number(opts.seed) >>> 0 || 1;
  const wantSims = Number(opts.sims) > 0 ? Number(opts.sims) : 0;
  const weights = await getWeights();

  // The policy floor: its own worker, weights loaded, sims=1 (it never searches,
  // only policy_move — but snakeNew wants a budget).
  const policyHost = new EngineHost();
  await policyHost.snakeNew(1, LEAVES, seed, weights);
  const policy = new SnakePolicy(policyHost);

  // The refinement search on a second worker.
  const searchHost = new EngineHost();
  let search: SnakeSearchHandle | null = null;
  let cpuFallback = '';
  let cpuReason = 'No compatible WebGPU device was detected';
  if (!isCpuFallback()) {
    try {
      const gpu = await getGpu();
      const sims = Math.min(wantSims || GPU_DEFAULT_SIMS, GPU_MAX_SIMS);
      await searchHost.snakeNew(sims, LEAVES, seed);
      console.info(`[snake] policy floor + background WebGPU search (${sims} sims)`);
      search = new GpuSearch(searchHost, gpu);
    } catch (e) {
      cpuReason = 'WebGPU was detected, but initialization failed';
      console.warn('[snake] WebGPU init failed; policy floor + slow CPU search:', e);
    }
  } else {
    console.warn('[snake] no WebGPU; policy floor + slow CPU search');
  }
  if (!search) {
    const sims = Math.min(wantSims || CPU_DEFAULT_SIMS, CPU_MAX_SIMS);
    await searchHost.snakeNew(sims, LEAVES, seed, weights);
    search = new CpuSearch(searchHost);
    cpuFallback = cpuFallbackMessage(cpuReason, sims);
  }

  return {
    policy,
    search,
    cpuFallback: cpuFallback || undefined,
    stop() {
      search?.stop();
      searchHost.terminate();
      policyHost.terminate();
    },
  };
}
