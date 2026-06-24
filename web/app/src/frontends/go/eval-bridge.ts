// A one-slot bridge from the active Go bot driver (which holds the EngineHost)
// to the Go frontend (which has no host handle of its own). The driver
// publishes a bound `goEval` when a match starts and clears it on teardown; the
// frontend reads through it for the debug position-quality readout. Scoped to
// the single live match — the shell tears down and rebuilds both ends per game.

import type { GoEval } from '../../engine/protocol';

export type GoEvalFn = () => Promise<GoEval | null>;

let current: GoEvalFn | null = null;

export function setGoEval(fn: GoEvalFn | null): void {
  current = fn;
}

export function goEval(): Promise<GoEval | null> {
  return current ? current() : Promise.resolve(null);
}
