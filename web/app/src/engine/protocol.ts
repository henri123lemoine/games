// The worker protocol: a thin promise-RPC mirror of the wasm engine API.
// Everything game-specific stays inside opaque JSON (view data, event data).

export interface GameOpt {
  key: string;
  value: string;
  note: string;
}

export interface GameInfo {
  id: string;
  /** Display name from the registry. */
  name: string;
  summary: string;
  /** Single-player: no seat option; `bot=` decides play vs watch. */
  solo: boolean;
  /** Bot spec watch mode uses on solo games (empty for versus games). */
  watchBot: string;
  /** Human-readable help line (derived from the schema engine-side). */
  opts: string;
  /** The structured option schema — what the settings drawer renders. */
  optsSchema: GameOpt[];
}

export interface CompareInfo {
  id: string;
  bots: string;
  field: boolean;
}

export interface Manifest {
  games: GameInfo[];
  compare: CompareInfo[];
}

export interface MatchEventData {
  seat: number;
  label: string;
  text: string;
  detail: string | null;
  data: unknown;
}

/** Snapshot of the match as the human sees it. */
export interface ViewState {
  view: string;
  viewData: unknown;
  labels: string[];
  toAct: number;
  isOver: boolean;
  result: string | null;
  humanSeat: number;
  numSeats: number;
}

export interface Wdl {
  w: number;
  d: number;
  l: number;
}

export type EngineRequest =
  | { id: number; op: 'manifest' }
  | { id: number; op: 'create'; game: string; opts: Record<string, string | number> }
  | { id: number; op: 'step' }
  | { id: number; op: 'state' }
  | { id: number; op: 'apply'; input: string }
  | { id: number; op: 'artifact'; key: string; bytes: ArrayBuffer }
  | {
      id: number;
      op: 'pairs';
      game: string;
      opts: Record<string, string | number>;
      a: string;
      b: string;
      seed: number;
      lo: number;
      hi: number;
    }
  | {
      id: number;
      op: 'field';
      game: string;
      opts: Record<string, string | number>;
      a: string;
      b: string;
      seed: number;
      lo: number;
      hi: number;
    }
  | { id: number; op: 'elo'; w: number; d: number; l: number }
  | { id: number; op: 'fitElo'; records: [number, number, number][][] }
  | { id: number; op: 'azNew'; sims: number; leaves: number; seed: number }
  | { id: number; op: 'azPush'; uci: string }
  | { id: number; op: 'azAdvance'; priors: Float32Array; values: Float32Array }
  | { id: number; op: 'azBest' };

/** One gathered leaf batch from the wasm AZ search (empty when done). */
export interface AzBatch {
  n: number;
  /** Flat board planes, `[n × 18·64]`. */
  features: Float32Array<ArrayBuffer>;
  /** Flat legal policy indices; `offsets` delimits the per-leaf runs. */
  support: Uint16Array<ArrayBuffer>;
  offsets: Uint32Array<ArrayBuffer>;
}

export interface AzBest {
  uci: string;
  stats: { value: number; sims: number };
}

export type EngineResponse =
  | { id: number; ok: true; data: unknown }
  | { id: number; ok: false; error: string };
