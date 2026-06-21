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
import { isCpuFallback } from '../shell/azero';
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
  /** Point the search at the latest board (the seat-1-to-move view JSON). */
  setRoot(viewJson: string): void;
  /** The most recent completed best heading label ("up"/"right"/"down"/"left"),
   * or null before the first search completes. Read synchronously by the tick. */
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
    this.root = viewJson;
    this.rev++;
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
    this.root = viewJson;
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
        this.current = uci;
        reportMove({ backend: 'cpu', ms: performance.now() - t0, sims: stats.sims, trips: 0 });
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

/** Spin up the snake bot's background search on its OWN engine worker (so it can
 * never contend with the game tick). Prefers WebGPU; falls back to the in-wasm
 * CPU forward. Caller owns the returned handle and must `stop()` it on teardown.
 * The returned `host` is also owned by the caller (terminate on teardown). */
export async function createSnakeSearch(
  opts: Record<string, string>,
): Promise<{ search: SnakeSearchHandle; host: EngineHost }> {
  const seed = Number(opts.seed) >>> 0 || 1;
  const wantSims = Number(opts.sims) > 0 ? Number(opts.sims) : 0;
  const host = new EngineHost();
  if (!isCpuFallback()) {
    try {
      const gpu = await getGpu();
      const sims = Math.min(wantSims || GPU_DEFAULT_SIMS, GPU_MAX_SIMS);
      await host.snakeNew(sims, LEAVES, seed);
      console.info(`[snake] background WebGPU search, ${sims} sims, ${SEARCH_BUDGET_MS}ms budget`);
      return { search: new GpuSearch(host, gpu), host };
    } catch (e) {
      console.warn('[snake] WebGPU init failed, background search on the slow CPU forward:', e);
    }
  } else {
    console.warn('[snake] no WebGPU; background search on the slow CPU forward');
  }
  const sims = Math.min(wantSims || CPU_DEFAULT_SIMS, CPU_MAX_SIMS);
  await host.snakeNew(sims, LEAVES, seed, await getWeights());
  console.info(`[snake] background CPU search, ${sims} sims (degraded — no GPU)`);
  return { search: new CpuSearch(host), host };
}
