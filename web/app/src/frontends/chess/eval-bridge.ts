// A one-slot bridge from the active chess bot driver (which holds the
// EngineHost) to the chess frontend (which has no host handle of its own). The
// driver publishes a bound `chessEval` when a match starts and clears it on
// teardown; the frontend reads through it for the debug position-quality
// readout. Scoped to the single live match — the shell tears down and rebuilds
// both ends per game. Mirrors frontends/go/eval-bridge.ts.

import type { ChessEval } from '../../engine/protocol';

export type ChessEvalFn = () => Promise<ChessEval | null>;

let current: ChessEvalFn | null = null;

export function setChessEval(fn: ChessEvalFn | null): void {
  current = fn;
}

export function chessEval(): Promise<ChessEval | null> {
  return current ? current() : Promise.resolve(null);
}
