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
  belief: "Belief",
  random: "Random",
};

export function botLabel(bot: string): string {
  return BOT_LABELS[bot] ?? bot;
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
  "chess/alphabeta": { key: "depth", levels: [["Easy", "2"], ["Medium", "4"], ["Hard", "6"]] },
  "chess/alphabeta-rich": { key: "depth", levels: [["Easy", "2"], ["Medium", "4"], ["Hard", "6"]] },
  "chess/azero": { key: "sims", levels: [["Easy", "64"], ["Medium", "256"], ["Hard", "800"]] },
  "chess/azero-gpu": { key: "sims", levels: [["Easy", "64"], ["Medium", "256"], ["Hard", "800"]] },
  "othello/alphabeta": { key: "depth", levels: [["Easy", "3"], ["Medium", "5"], ["Hard", "7"]] },
  "othello/mcts": { key: "sims", levels: [["Easy", "500"], ["Medium", "2000"], ["Hard", "6000"]] },
  "connect4/alphabeta": { key: "depth", levels: [["Easy", "5"], ["Medium", "7"], ["Hard", "9"]] },
  "connect4/mcts": { key: "sims", levels: [["Easy", "500"], ["Medium", "2000"], ["Hard", "6000"]] },
  "go/mcts": { key: "sims", levels: [["Easy", "400"], ["Medium", "1500"], ["Hard", "4000"]] },
  "go/mcts-eval": { key: "sims", levels: [["Easy", "400"], ["Medium", "1500"], ["Hard", "4000"]] },
  "go/mcts-spec": { key: "sims", levels: [["Easy", "400"], ["Medium", "1500"], ["Hard", "4000"]] },
  "go/azero-gpu": { key: "sims", levels: [["Easy", "400"], ["Medium", "1500"], ["Hard", "4000"]] },
  "liars-dice/rollout": { key: "rollouts", levels: [["Easy", "100"], ["Medium", "400"], ["Hard", "1000"]] },
  "2048/mcts": { key: "sims", levels: [["Easy", "100"], ["Medium", "200"], ["Hard", "600"]] },
  "2048/mcts-eval": { key: "sims", levels: [["Easy", "100"], ["Medium", "200"], ["Hard", "600"]] },
  "snake/mcts": { key: "sims", levels: [["Easy", "60"], ["Medium", "150"], ["Hard", "400"]] },
  "snake/mcts-eval": { key: "sims", levels: [["Easy", "60"], ["Medium", "150"], ["Hard", "400"]] },
};

export function difficultyFor(gameId: string, bot: string): Difficulty | undefined {
  return DIFFICULTY[`${gameId}/${bot}`];
}

/** Discrete choices for the small count options, so the settings drawer offers
 * a dropdown rather than a free-text field. Hearts is limited to the shipped
 * solver artifacts. */
export const OPT_CHOICES: Record<string, string[]> = {
  players: ["2", "3", "4", "5", "6"],
  dice: ["3", "4", "5", "6"],
  hearts: ["3", "6"],
  size: ["9", "13", "19"],
};

/** Bots that can't run in the tournament: they need a GPU or a trained
 * artifact the tournament pool doesn't load. */
const TOURNEY_EXCLUDE = new Set(["azero", "azero-gpu"]);

/** Bot ids a game can field in the tournament — its play bots minus the ones
 * that need a GPU/artifact. Empty for games without bot-vs-bot evaluation. */
export function tourneyBots(game: GameInfo): string[] {
  const spec = game.optsSchema.find((o) => o.key === "bot");
  if (!spec) return [];
  return spec.value.split("|").filter((b) => !TOURNEY_EXCLUDE.has(b));
}

/** A compare/tourney spec string, e.g. `alphabeta:depth=4`; a bot with no
 * difficulty knob is just its bare name. */
export function botSpec(gameId: string, bot: string, levelValue?: string): string {
  const diff = difficultyFor(gameId, bot);
  return diff && levelValue ? `${bot}:${diff.key}=${levelValue}` : bot;
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
