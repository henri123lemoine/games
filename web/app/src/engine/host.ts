// Promise-RPC over the engine worker. One host per worker; the tournament
// screen spawns several hosts to use multiple cores.

import type {
  AzBatch,
  AzBest,
  EngineRequest,
  EngineResponse,
  Manifest,
  MatchEventData,
  ViewState,
  Wdl,
} from './protocol';

type Pending = { resolve: (v: unknown) => void; reject: (e: Error) => void };

/** `Omit` that distributes over a discriminated union. */
type DistributiveOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never;

export class EngineHost {
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<number, Pending>();

  constructor() {
    this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e: MessageEvent<EngineResponse>) => {
      const p = this.pending.get(e.data.id);
      if (!p) return;
      this.pending.delete(e.data.id);
      if (e.data.ok) p.resolve(e.data.data);
      else p.reject(new Error(e.data.error));
    };
  }

  private call(req: DistributiveOmit<EngineRequest, 'id'>): Promise<unknown> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ ...req, id });
    });
  }

  manifest(): Promise<Manifest> {
    return this.call({ op: 'manifest' }) as Promise<Manifest>;
  }

  create(game: string, opts: Record<string, string | number>): Promise<ViewState> {
    return this.call({ op: 'create', game, opts }) as Promise<ViewState>;
  }

  /** One bot move, or null at the human's turn / game over. */
  step(): Promise<MatchEventData | null> {
    return this.call({ op: 'step' }) as Promise<MatchEventData | null>;
  }

  state(): Promise<ViewState> {
    return this.call({ op: 'state' }) as Promise<ViewState>;
  }

  apply(input: string): Promise<MatchEventData> {
    return this.call({ op: 'apply', input }) as Promise<MatchEventData>;
  }

  artifact(key: string, bytes: ArrayBuffer): Promise<void> {
    return this.call({ op: 'artifact', key, bytes }) as Promise<void>;
  }

  pairs(
    game: string,
    opts: Record<string, string | number>,
    a: string,
    b: string,
    seed: number,
    lo: number,
    hi: number,
  ): Promise<Wdl> {
    return this.call({ op: 'pairs', game, opts, a, b, seed, lo, hi }) as Promise<Wdl>;
  }

  field(
    game: string,
    opts: Record<string, string | number>,
    a: string,
    b: string,
    seed: number,
    lo: number,
    hi: number,
  ): Promise<{ wins: number; losses: number }> {
    return this.call({ op: 'field', game, opts, a, b, seed, lo, hi }) as Promise<{
      wins: number;
      losses: number;
    }>;
  }

  elo(w: number, d: number, l: number): Promise<{ elo: number; margin: number }> {
    return this.call({ op: 'elo', w, d, l }) as Promise<{ elo: number; margin: number }>;
  }

  fitElo(records: [number, number, number][][]): Promise<number[]> {
    return this.call({ op: 'fitElo', records }) as Promise<number[]>;
  }

  azNew(sims: number, leaves: number, seed: number): Promise<void> {
    return this.call({ op: 'azNew', sims, leaves, seed }) as Promise<void>;
  }

  azPush(uci: string): Promise<void> {
    return this.call({ op: 'azPush', uci }) as Promise<void>;
  }

  azAdvance(priors: Float32Array, values: Float32Array): Promise<AzBatch> {
    return this.call({ op: 'azAdvance', priors, values }) as Promise<AzBatch>;
  }

  azBest(): Promise<AzBest> {
    return this.call({ op: 'azBest' }) as Promise<AzBest>;
  }

  terminate(): void {
    this.worker.terminate();
    for (const p of this.pending.values()) p.reject(new Error('engine terminated'));
    this.pending.clear();
  }
}
