// Client-side bots: per-game drivers for externally driven match seats
// (registry bots whose moves the page computes — e.g. WebGPU evaluation the
// sync wasm engine cannot do). The shell consults this registry the same way
// it consults the frontends one; a game without a matching driver here plays
// its bots inside the wasm engine as usual.

import { EngineHost } from '../engine/host';
import type { MatchEventData, ViewState } from '../engine/protocol';
import { createAzeroChess } from './azero-chess';
import { createAzeroFourPlayerChess } from './azero-four-player-chess';
import { createAzeroGo } from './azero-go';
import { createAzeroPente } from './azero-pente';

export interface ClientBot {
  /** Mirror every applied move (any seat's), in order. */
  onMove(ev: MatchEventData): Promise<void>;
  /** Compute the move for the external seat to act (a submit-able input). */
  chooseMove(st: ViewState): Promise<string>;
  /** Abandon any in-flight work; the bot will not be called again. A stale
   * chooseMove loop left running would corrupt the next match's search. */
  cancel(): void;
  /** A result string to show in place of the engine's at game over (e.g. an
   * ownership-adjudicated go score), or `''`/absent to keep the engine's. */
  finalResult?(): Promise<string>;
  /** Non-empty when a requested GPU bot is actually running on CPU. */
  cpuFallback?: string;
}

export type ClientBotFactory = (
  host: EngineHost,
  opts: Record<string, string>,
) => Promise<ClientBot>;

const factories = new Map<string, ClientBotFactory>([
  ['chess/azero-gpu', createAzeroChess],
  ['four-player-chess/azero-gpu', createAzeroFourPlayerChess],
  ['go/azero-gpu', createAzeroGo],
  ['pente/azero-gpu', createAzeroPente],
]);

export function clientBotFor(gameId: string, bot: string | undefined): ClientBotFactory | null {
  return (bot && factories.get(`${gameId}/${bot}`)) || null;
}

export interface ClientBotConfig {
  seat: number;
  bot: string;
  opts: Record<string, string>;
}

/** One independently configured client-side bot per externally driven seat.
 * Dedicated workers keep distinct visit/forcing budgets genuinely independent
 * and let a superseded async model boot be torn down without touching a match. */
export async function createClientBots(
  gameId: string,
  configs: ClientBotConfig[],
): Promise<ClientBot | null> {
  if (configs.length === 0) return null;
  const entries: { seat: number; bot: ClientBot; host: EngineHost }[] = [];
  try {
    for (const config of configs) {
      const factory = clientBotFor(gameId, config.bot);
      if (!factory)
        throw new Error(`no client-side driver for ${gameId}/${config.bot} at seat ${config.seat}`);
      // Search state never shares the match worker. A superseded async model
      // boot can then be terminated without posting stale config into the new
      // match, and every external seat owns an independent tree.
      const host = new EngineHost();
      try {
        entries.push({
          seat: config.seat,
          bot: await factory(host, config.opts),
          host,
        });
      } catch (error) {
        host.terminate();
        throw error;
      }
    }
  } catch (error) {
    for (const entry of entries) {
      entry.bot.cancel();
      entry.host.terminate();
    }
    throw error;
  }

  const bySeat = new Map(entries.map((entry) => [entry.seat, entry.bot]));
  const fallbackMessages = [
    ...new Set(entries.map((entry) => entry.bot.cpuFallback).filter((x): x is string => !!x)),
  ];
  return {
    cpuFallback: fallbackMessages.length ? fallbackMessages.join(' ') : undefined,
    async onMove(ev: MatchEventData): Promise<void> {
      await Promise.all(entries.map((entry) => entry.bot.onMove(ev)));
    },
    async chooseMove(st: ViewState): Promise<string> {
      const bot = bySeat.get(st.toAct);
      if (!bot) throw new Error(`no client-side bot configured for seat ${st.toAct}`);
      return bot.chooseMove(st);
    },
    async finalResult(): Promise<string> {
      for (const entry of entries) {
        const result = (await entry.bot.finalResult?.()) ?? '';
        if (result) return result;
      }
      return '';
    },
    cancel(): void {
      for (const entry of entries) {
        entry.bot.cancel();
        entry.host.terminate();
      }
    },
  };
}
