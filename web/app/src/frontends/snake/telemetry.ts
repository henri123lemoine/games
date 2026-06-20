// Per-move telemetry for the snake card's debug overlay. The bot driver
// (bots/azero-snake.ts) writes one record per move with `reportMove`; the
// frontend's render loop reads the latest with `lastMove` each frame and paints
// the HUD. A lock-free shared record — no callback into the frontend, so the
// two files stay decoupled and a remount can't leave a dangling reference.

export interface MoveTelemetry {
  /** The backend that produced the move. */
  backend: 'gpu' | 'cpu';
  /** Wall-clock time the whole move took (set_state → best), in ms. */
  ms: number;
  /** Simulations actually run for the move (the achieved visit budget). */
  sims: number;
  /** GPU leaf-batch round-trips spent on the move; 0 on the CPU backend. */
  trips: number;
  /** `performance.now()` when the record was written (overlay staleness). */
  at: number;
}

let last: MoveTelemetry | null = null;

/** Bot driver → telemetry: record the move just chosen. */
export function reportMove(t: Omit<MoveTelemetry, 'at'>): void {
  last = { ...t, at: performance.now() };
}

/** Overlay → telemetry: the most recent move, or null before the first move. */
export function lastMove(): MoveTelemetry | null {
  return last;
}

/** Drop the record (on match teardown) so a new match starts clean. */
export function resetTelemetry(): void {
  last = null;
}
