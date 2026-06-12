// Client-side bots: per-game drivers for externally driven match seats
// (registry bots whose moves the page computes — e.g. WebGPU evaluation the
// sync wasm engine cannot do). The shell consults this registry the same way
// it consults the frontends one; a game without a matching driver here plays
// its bots inside the wasm engine as usual.

import type { EngineHost } from '../engine/host';
import type { MatchEventData, ViewState } from '../engine/protocol';
import { createAzeroChess } from './azero-chess';

export interface ClientBot {
  /** Mirror every applied move (any seat's), in order. */
  onMove(ev: MatchEventData): Promise<void>;
  /** Compute the move for the external seat to act (a submit-able input). */
  chooseMove(st: ViewState): Promise<string>;
  /** Abandon any in-flight work; the bot will not be called again. A stale
   * chooseMove loop left running would corrupt the next match's search. */
  cancel(): void;
}

export type ClientBotFactory = (
  host: EngineHost,
  opts: Record<string, string>,
) => Promise<ClientBot>;

const factories = new Map<string, ClientBotFactory>([['chess/azero-gpu', createAzeroChess]]);

export function clientBotFor(gameId: string, bot: string | undefined): ClientBotFactory | null {
  return (bot && factories.get(`${gameId}/${bot}`)) || null;
}
