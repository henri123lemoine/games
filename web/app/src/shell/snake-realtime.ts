// Real-time driver for snake PLAY (human seat 0 vs the AlphaZero bot).
//
// Snake is real-time, but the shell's generic match loop is serial: it awaits
// the bot's ~190ms search before the human can act, so the player's snake was
// gated by the bot's compute (presses landed hundreds of ms late). This driver
// replaces that loop for snake play and fully DECOUPLES the player from the bot:
//
//   - A FIXED game clock advances the world every TICK_MS, independent of the
//     bot. The player's snake always moves at this constant rate.
//   - The player's latest heading is sampled from the frontend at each tick
//     (instant — no await), so a keypress turns the snake within ≤ one tick.
//   - The bot runs a perpetual background search on its OWN worker (see
//     bots/snake-search.ts) and publishes a "current best move"; the tick reads
//     that synchronously. If a fresh search isn't ready, the tick uses the last
//     one (weaker-when-rushed is explicitly acceptable). The tick NEVER awaits
//     the search.
//
// The only awaits in the tick are the GAME worker's `step`/`apply` (fast state
// updates, no search), on a host separate from the bot's search worker — so a
// slow bot can never delay a tick.

import type { EngineHost } from '../engine/host';
import type { ViewState } from '../engine/protocol';
import type { SnakeBot } from '../bots/snake-search';

/** The slice of the snake frontend the driver needs. */
export interface RealtimeBoard {
  pollHeading(): string;
  pushState(state: ViewState, durMs: number): void;
  render(state: ViewState): void;
}

/** Fixed cell period — Google-Snake pace. The player's snake advances one cell
 * per this, ALWAYS, regardless of the bot. */
const TICK_MS = 120;

const DIR_LABEL: Record<string, string> = { n: 'up', e: 'right', s: 'down', w: 'left' };

export class SnakeRealtime {
  private timer = 0;
  private running = false;
  private alive = true;
  /** The bot snake's current heading, the last-ditch fallback if even the policy
   * floor fails (it should not). */
  private botHeading = 'left';

  constructor(
    private host: EngineHost,
    private bot: SnakeBot,
    private board: RealtimeBoard,
    private isCurrent: () => boolean,
    private onOver: (st: ViewState) => void,
  ) {}

  /** Begin the fixed-clock loop from the initial state. */
  start(initial: ViewState): void {
    this.board.render(initial);
    this.bot.search?.setRoot(JSON.stringify(initial.viewData));
    this.running = true;
    this.scheduleNext(performance.now());
  }

  stop(): void {
    this.running = false;
    if (this.timer) clearTimeout(this.timer);
    this.timer = 0;
  }

  /** Schedule the next tick so ticks land on a FIXED grid (drift-free): the next
   * tick fires at the previous target + TICK_MS, not "TICK_MS after this tick
   * finished", so a slow apply doesn't slow the player's cadence. */
  private scheduleNext(prevTarget: number): void {
    if (!this.running) return;
    const target = prevTarget + TICK_MS;
    const wait = Math.max(0, target - performance.now());
    this.timer = window.setTimeout(() => {
      this.timer = 0;
      void this.tick(target);
    }, wait);
  }

  private async tick(target: number): Promise<void> {
    if (!this.running || !this.isCurrent()) return;
    try {
      // Resolve a food chance node if one is pending (no-op otherwise), so a
      // driven seat is up. This is a fast state op on the GAME worker.
      await this.host.step();
      if (!this.running || !this.isCurrent()) return;

      // Seat 0 = player: sample the held/pressed heading NOW, instantly, and
      // commit it. The player's move never waits on the bot.
      const playerHeading = this.board.pollHeading();
      await this.host.apply(playerHeading);
      if (!this.running || !this.isCurrent()) return;

      // Seat 1 = bot. The board is now seat-1-to-move (seat 0's pending is set);
      // point the background search at it and get the bot's move for THIS board:
      //   - a fresh search best if one is ready (the strong bonus), else
      //   - the always-available CPU POLICY FLOOR (one net forward + 1-ply
      //     safety, ~1-2ms on its own free worker) — a real, non-suicidal move.
      // The bot NEVER coasts straight; the policy floor guarantees a played move
      // even with no GPU and a throttled CPU.
      const seat1View = await this.host.state();
      if (!this.running || !this.isCurrent()) return;
      const seat1Json = JSON.stringify(seat1View.viewData);
      this.bot.search?.setRoot(seat1Json);
      let botHeading = this.bot.search?.best() ?? null;
      if (!botHeading) {
        try {
          botHeading = await this.bot.policy.policyMove(seat1Json);
        } catch {
          botHeading = this.botHeading; // last-ditch; should not happen
        }
      }
      if (!this.running || !this.isCurrent()) return;
      await this.host.apply(botHeading);
      if (!this.running || !this.isCurrent()) return;

      // The tick is resolved; read the new world and glide to it over the fixed
      // clock period.
      const st = await this.host.state();
      if (!this.running || !this.isCurrent()) return;
      const view = st.viewData as { snakes?: { dir?: string }[] } | null;
      const dir = view?.snakes?.[1]?.dir;
      if (dir && DIR_LABEL[dir]) this.botHeading = DIR_LABEL[dir];
      this.board.pushState(st, TICK_MS);

      if (st.isOver) {
        this.alive = false;
        this.running = false;
        this.board.render(st);
        this.onOver(st);
        return;
      }
    } catch {
      // A transient engine hiccup: keep the clock going so the player's snake
      // never freezes; the next tick re-reads state.
    }
    this.scheduleNext(target);
  }

  isAlive(): boolean {
    return this.alive;
  }
}
