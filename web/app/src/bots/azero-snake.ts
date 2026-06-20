// The AlphaZero snake bot. The snake net is tiny (4×64), so the whole search
// runs in-wasm against snakeinfer's reference forward — no WebGPU path. Snake's
// food placement is a chance node the engine resolves with its own rng, so the
// bot can't mirror moves like the go/chess bots; instead it reconstructs its
// search root from the engine's view JSON before each move (snakePlayCpu).

import type { EngineHost } from '../engine/host';
import type { ViewState } from '../engine/protocol';
import type { ClientBot } from './index';

const LEAVES = 8;
const SNAKE_DEFAULT_SIMS = 3;
const SNAKE_MAX_SIMS = 6;
const WEIGHTS_URL = `${import.meta.env.BASE_URL}azero/azero-snake.azweb`;

/** The raw export bytes, fetched once per page. */
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

// One round-trip per move (the search is atomic worker-side), so there is no
// advance loop to cancel — just a guard so a torn-down match drops its move.
class AzeroSnakeCpu implements ClientBot {
  private cancelled = false;
  constructor(private host: EngineHost) {}

  // The bot reconstructs its root from the view each move, so applied moves
  // need no mirroring.
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
  // The chosen level, capped so moves stay responsive on the CPU forward.
  const sims = Math.min(Number(opts.sims) > 0 ? Number(opts.sims) : SNAKE_DEFAULT_SIMS, SNAKE_MAX_SIMS);
  await host.snakeNew(sims, LEAVES, seed, await getWeights());
  return new AzeroSnakeCpu(host);
}
