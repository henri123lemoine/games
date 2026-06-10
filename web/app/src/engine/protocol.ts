// The worker protocol: a thin promise-RPC mirror of the wasm engine API.
// Everything game-specific stays inside opaque JSON (view data, event data).

export interface GameInfo {
  id: string;
  summary: string;
  opts: string;
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
  | { id: number; op: 'fitElo'; records: [number, number, number][][] };

export type EngineResponse =
  | { id: number; ok: true; data: unknown }
  | { id: number; ok: false; error: string };
