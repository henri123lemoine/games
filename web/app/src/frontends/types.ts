// The per-game frontend contract (see WEB.md). Each game ships its own
// package with its own visual identity; the shell only drives this interface.
// View/event `data` payloads are game-private JSON shared with the game's
// Rust crate — the shell never interprets them.

import type { MatchEventData, ViewState } from '../engine/protocol';

export interface FrontendCtx {
  gameId: string;
  opts: Record<string, string | number>;
  humanSeat: number;
  numSeats: number;
  /** Submit the human's move: a legal-action index as a string, or
   * game-native text (e.g. `e2e4`). */
  submit(input: string): void;
  /** Honor `prefers-reduced-motion` and the spectate speed setting. */
  animationScale(): number;
}

export interface GameFrontend {
  mount(host: HTMLElement, ctx: FrontendCtx): void;
  /** Full redraw from the current state. Called after mount, after every
   * animated event, and on the human's turn. */
  render(state: ViewState): void;
  /** Animate one applied move; resolve when done. The shell awaits this
   * before stepping further — spectate pacing falls out of it. */
  animate(event: MatchEventData, after: ViewState): Promise<void>;
  /** It is the human's turn: enable board-native input for these actions. */
  promptAction(labels: string[]): void;
  unmount(): void;
}

export type FrontendFactory = () => GameFrontend;

export function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}
