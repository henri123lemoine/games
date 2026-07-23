// Shared presentation config for bots: human-facing names and difficulty
// presets, used by both the match settings drawer and the tournament lab so
// the two never drift. The registry stays the source of truth for which bots
// exist; this only decides how they're shown and what a difficulty maps to.

import type { GameInfo } from "../engine/protocol";

/** Human-facing names for the registry's terse bot ids. */
export const BOT_LABELS: Record<string, string> = {
  alphabeta: "Alpha-Beta",
  "alphabeta-rich": "Alpha-Beta (rich)",
  azero: "AlphaZero (CPU)",
  "azero-gpu": "AlphaZero",
  mcts: "MCTS",
  "mcts-eval": "MCTS (eval)",
  "mcts-spec": "MCTS (spec)",
  rollout: "Rollout",
  history: "Neural",
  ataraxios: "Ataraxios",
  belief: "Belief",
  bns: "Best-node search",
  random: "Random",
};

export function botLabel(bot: string): string {
  return BOT_LABELS[bot] ?? bot;
}

/** Provenance blurbs per `game/bot`: what the opponent is, how it was built,
 * and what it cost to train — real numbers from the actual runs, all on one
 * Apple M5 Max MacBook. Shown from the ⓘ beside a bot's seat picker. */
export const BOT_INFO: Record<string, string> = {
  "stratego/ataraxios":
    "Follows Ataraxos AI's implementation. A 27M-parameter transformer pair — an 8-layer move net plus a 4-layer setup net that arranges its own army. Trained from scratch by self-play in MLX on a Apple M5 Max MacBook: 6.5 days, 7,600 iterations, ~1.5 billion moves.",
  "chess/azero-gpu":
    "AlphaZero conv-resnet trained by self-play with MCTS in one overnight run on a MacBook.",
  "four-player-chess/azero-gpu":
    "A 3×24 four-seat AlphaZero conv-resnet trained locally from scratch on an Apple M5 Max CPU: 70 iterations, 4,200 self-play games (3,923 against frozen league checkpoints), 303,107 positions, and 4h05m of measured work. Its value head predicts all four players at once; play-time search is multiplayer MCTS.",
  "go/azero-gpu":
    "AlphaZero self-play net with board-size-agnostic global-pool heads, similar to KataGo, trained for about two days on a MacBook. Play-time search is MCTS.",
  "pente/azero-gpu": "AlphaZero self-play net.",
  "snake/bns":
    "A simultaneous best-node search with bitboard territory, collision-aware quiescence, transpositions, and a phase-aware Battlesnake evaluation.",
  "liars-dice/history":
    "PPO over a history-attention encoder with a belief head, trained in a multi-round self-play league with an exploiter pool across about ten days. The shipped net is the league's round-21 head-to-head champion.",
  "liars-dice/rollout":
    "A Monte-Carlo rollout bot: samples the hidden dice, plays out candidate bids, and picks the best average.",
  "poker/equity":
    "A Monte-Carlo equity bot: samples hole cards and runouts to estimate win probability and bets accordingly.",
  "othello/alphabeta":
    "Classic alpha-beta search over a hand-tuned positional evaluation.",
  "connect4/alphabeta":
    "Classic alpha-beta search with a hand-tuned evaluation.",
  "pente/alphabeta": "Alpha-beta search with a VCF hybrid.",
  "twentyone/__solver__":
    "The game solved offline into exact lookup tables, playing perfectly within its heart budget.",
};

export function botInfo(gameId: string, bot: string): string | undefined {
  return BOT_INFO[`${gameId}/${bot}`];
}

export interface Difficulty {
  /** The bot knob this difficulty drives (depth / sims / rollouts). */
  key: string;
  /** Easy → Hard, as `[label, value]`; the value is what the knob is set to. */
  levels: [string, string][];
}

/** Difficulty presets per `game/bot`: the roster (or the tournament) picks the
 * bot, this picks how hard it plays — so a visitor never types a raw search
 * depth. A bot with no entry here has no tunable difficulty. */
export const DIFFICULTY: Record<string, Difficulty> = {
  "chess/alphabeta": {
    key: "depth",
    levels: [
      ["Easy", "2"],
      ["Medium", "4"],
      ["Hard", "6"],
    ],
  },
  "chess/alphabeta-rich": {
    key: "depth",
    levels: [
      ["Easy", "2"],
      ["Medium", "4"],
      ["Hard", "6"],
    ],
  },
  "chess/azero": {
    key: "sims",
    levels: [
      ["Trivial", "1"],
      ["Easy", "64"],
      ["Medium", "256"],
      ["Hard", "800"],
    ],
  },
  "chess/azero-gpu": {
    key: "sims",
    levels: [
      ["Trivial", "1"],
      ["Easy", "1200"],
      ["Medium", "4800"],
      ["Hard", "12000"],
    ],
  },
  "four-player-chess/azero-gpu": {
    key: "sims",
    levels: [
      ["Trivial", "1"],
      ["Easy", "64"],
      ["Medium", "256"],
      ["Hard", "800"],
    ],
  },
  "othello/alphabeta": {
    key: "depth",
    levels: [
      ["Easy", "3"],
      ["Medium", "5"],
      ["Hard", "7"],
    ],
  },
  "othello/mcts": {
    key: "sims",
    levels: [
      ["Easy", "500"],
      ["Medium", "2000"],
      ["Hard", "6000"],
    ],
  },
  "connect4/alphabeta": {
    key: "depth",
    levels: [
      ["Easy", "5"],
      ["Medium", "7"],
      ["Hard", "9"],
    ],
  },
  "connect4/mcts": {
    key: "sims",
    levels: [
      ["Easy", "500"],
      ["Medium", "2000"],
      ["Hard", "6000"],
    ],
  },
  "pente/alphabeta": {
    key: "depth",
    levels: [
      ["Easy", "2"],
      ["Medium", "4"],
      ["Hard", "5"],
    ],
  },
  "pente/mcts": {
    key: "sims",
    levels: [
      ["Easy", "1000"],
      ["Medium", "4000"],
      ["Hard", "10000"],
    ],
  },
  "pente/azero-gpu": {
    key: "sims",
    levels: [
      ["Trivial", "1"],
      ["Easy", "16"],
      ["Medium", "64"],
      ["Hard", "256"],
    ],
  },
  "go/mcts": {
    key: "sims",
    levels: [
      ["Easy", "400"],
      ["Medium", "1500"],
      ["Hard", "4000"],
    ],
  },
  "go/mcts-eval": {
    key: "sims",
    levels: [
      ["Easy", "400"],
      ["Medium", "1500"],
      ["Hard", "4000"],
    ],
  },
  "go/mcts-spec": {
    key: "sims",
    levels: [
      ["Easy", "400"],
      ["Medium", "1500"],
      ["Hard", "4000"],
    ],
  },
  "go/azero-gpu": {
    key: "sims",
    levels: [
      ["Trivial", "1"],
      ["Easy", "400"],
      ["Medium", "1200"],
      ["Hard", "2400"],
    ],
  },
  "liars-dice/rollout": {
    key: "rollouts",
    levels: [
      ["Easy", "100"],
      ["Medium", "400"],
      ["Hard", "1000"],
    ],
  },
  "poker/equity": {
    key: "samples",
    levels: [
      ["Easy", "300"],
      ["Medium", "1200"],
      ["Hard", "3000"],
    ],
  },
  "poker/rollout": {
    key: "rollouts",
    levels: [
      ["Easy", "60"],
      ["Medium", "150"],
      ["Hard", "400"],
    ],
  },
  "snake/bns": {
    key: "millis",
    levels: [
      ["Easy", "25"],
      ["Medium", "120"],
      ["Hard", "440"],
    ],
  },
};

export function difficultyFor(
  gameId: string,
  bot: string,
): Difficulty | undefined {
  return DIFFICULTY[`${gameId}/${bot}`];
}

/** Discrete choices for the small count options, so the settings drawer offers
 * a dropdown rather than a free-text field. Hearts is limited to the shipped
 * solver artifacts. */
export const OPT_CHOICES: Record<string, string[]> = {
  players: ["2", "3", "4", "5", "6"],
  dice: ["3", "4", "5", "6"],
  hearts: ["3", "6"],
  size: ["9", "13", "15", "19"],
};

/** Per-game narrowings of [`OPT_CHOICES`]: a game that should expose fewer
 * values than the generic list (pente is 19×19 only), or a game-specific
 * option that deserves a dropdown (stratego's setup mode). */
const OPT_CHOICES_BY_GAME: Record<string, Record<string, string[]>> = {
  pente: { size: ["19"] },
  snake: {
    players: ["2", "3", "4"],
    mode: [
      "standard",
      "royale",
      "constrictor",
      "wrapped",
      "wrapped-constrictor",
    ],
    food: ["standard", "one"],
    model: ["mcs", "brs+", "full"],
  },
  stratego: { setup: ["random", "manual"] },
};

/** The choices a game offers for an option — its own narrowing, else the
 * generic list. A single-element result means the option is fixed (the shell
 * hides its dropdown). */
export function optChoicesFor(
  gameId: string,
  key: string,
): string[] | undefined {
  return OPT_CHOICES_BY_GAME[gameId]?.[key] ?? OPT_CHOICES[key];
}

/** Bots that can't run in the tournament (they need a GPU or a trained
 * artifact the tournament pool doesn't load) — plus stratego's heuristic,
 * which the site never publishes anywhere by policy. */
const TOURNEY_EXCLUDE = new Set([
  "azero",
  "azero-gpu",
  "ataraxios",
  "heuristic",
]);

/** Bot ids a game can field in the tournament — its play bots minus the ones
 * that need a GPU/artifact. Empty for games without bot-vs-bot evaluation. */
export function tourneyBots(game: GameInfo): string[] {
  const spec = game.optsSchema.find((o) => o.key === "bot");
  if (!spec) return [];
  return spec.value.split("|").filter((b) => !TOURNEY_EXCLUDE.has(b));
}

export interface ParsedBotSpec {
  bot: string;
  opts: Record<string, string>;
}

/** Parse the lab's canonical `bot:key=value,key=value` wire format. Invalid
 * specs fail here instead of quietly turning into a differently configured
 * client-side bot. */
export function parseBotSpec(text: string): ParsedBotSpec {
  const colon = text.indexOf(":");
  const bot = colon < 0 ? text : text.slice(0, colon);
  if (!bot) throw new Error(`bot spec has no bot name: '${text}'`);
  const opts: Record<string, string> = {};
  if (colon >= 0) {
    const rest = text.slice(colon + 1);
    for (const part of rest.split(",")) {
      const equals = part.indexOf("=");
      if (equals <= 0)
        throw new Error(
          `bot option must be key=value, got '${part}' in '${text}'`,
        );
      opts[part.slice(0, equals)] = part.slice(equals + 1);
    }
  }
  return { bot, opts };
}

/** Serialize one bot and all of its explicit per-seat options. */
export function formatBotSpec(
  bot: string,
  opts: Record<string, string> = {},
): string {
  const entries = Object.entries(opts).filter(([, value]) => value !== "");
  return entries.length
    ? `${bot}:${entries.map(([key, value]) => `${key}=${value}`).join(",")}`
    : bot;
}

/** The middle difficulty value for a bot, or '' if it has no difficulty knob.
 * Used as the default when a bot enters a heterogeneous match via the roster. */
export function mediumLevel(gameId: string, bot: string): string {
  const d = difficultyFor(gameId, bot);
  return d ? (d.levels[1] ?? d.levels[0])[1] : "";
}

/** JS twin of the lab's `split_specs`: split a `bots=` list on commas while
 * keeping commas that belong to one spec's own options (a segment with `=`
 * but no `:` continues the previous spec). */
export function splitSpecs(s: string): string[] {
  const out: string[] = [];
  for (const seg of s.split(",")) {
    if (seg.includes("=") && !seg.includes(":") && out.length)
      out[out.length - 1] += `,${seg}`;
    else out.push(seg);
  }
  return out;
}
