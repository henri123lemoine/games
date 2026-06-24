// A one-slot bridge from the active Pente bot driver (which holds the
// EngineHost) to the Pente frontend (which has no host handle of its own). The
// driver publishes a bound `penteEval` when a match starts and clears it on
// teardown; the frontend reads through it for the debug position-quality
// readout. Scoped to the single live match — the shell tears down and rebuilds
// both ends per game.

import type { PenteEval } from '../../engine/protocol';

export type PenteEvalFn = () => Promise<PenteEval | null>;

let current: PenteEvalFn | null = null;

export function setPenteEval(fn: PenteEvalFn | null): void {
  current = fn;
}

export function penteEval(): Promise<PenteEval | null> {
  return current ? current() : Promise.resolve(null);
}
