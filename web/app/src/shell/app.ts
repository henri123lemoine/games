// The arcade shell: pick a game and you are immediately playing it against
// the lab's bot — configuration lives in a quiet settings drawer, not between
// you and the board. One engine worker drives play; the shell owns the loop
// and narration, frontends own the board.

import {
  type ClientBot,
  type ClientBotConfig,
  clientBotFor,
  createClientBots,
} from "../bots";
import { assetUrl } from "../assets";
import { EngineHost } from "../engine/host";
import type {
  GameInfo,
  GameOpt,
  Manifest,
  MatchEventData,
  ViewState,
} from "../engine/protocol";
import { frontendFor, hasFrontend } from "../frontends";
import type { SlitherScreen } from "../frontends/slither";
import { RANK_ICONS } from "../frontends/stratego/sprites";
import type { FrontendCtx, GameFrontend } from "../frontends/types";
import { CPU_LEVELS, isCpuFallback, TRIVIAL_SIMS } from "./azero";
import {
  DIFFICULTY,
  botInfo,
  botLabel,
  formatBotSpec,
  mediumLevel,
  optChoicesFor,
  parseBotSpec,
  splitSpecs,
} from "./config";
import { TournamentScreen } from "./tournament";
import { getGoGpu, getGoWeights } from "../bots/azero-go";
import { type ConformanceResult, runGoConformance } from "../frontends/go/conformance";

/** What clicking a card starts: browser-tuned, no questions asked. Chess and
 * Go open against AlphaZero (Medium); with no WebGPU the driver runs the same
 * net on the CPU at the trivial budget. `sims` here is the AlphaZero budget. */
/** The Pente AlphaZero net is trained at 19×19; single-sourced pin so the
 * default and the buildOpts guard below can't drift. */
const PENTE_AZ_SIZE = "19";

const DEFAULT_OPTS: Record<string, Record<string, string>> = {
  chess: { bot: "azero-gpu", sims: "4800" },
  "liars-dice": { players: "5", dice: "5", faces: "6", rollouts: "400", bot: "history" },
  twentyone: { hearts: "3" },
  othello: { depth: "5" },
  connect4: { depth: "7" },
  // AlphaZero Pente is trained at 19×19 and the human opens (Black, seat 0); a
  // ~88% first-player game, so the human gets the edge. The net plays the same
  // VCF hybrid the native bot does.
  pente: { size: PENTE_AZ_SIZE, bot: "azero-gpu", sims: "64" },
  go: { size: "19", bot: "azero-gpu", sims: "1200" },
  snake: { players: "2", food: "one", bot: "bns", millis: "25" },
  // The site ships only the trained net (ataraxios) — never the heuristic.
  stratego: { bot: "ataraxios", setup: "manual" },
};

/** Games registered in the lab but not surfaced on the site. */
const HIDDEN_GAMES = new Set<string>([]);

/** Board-native real-time games: their frontend owns the whole screen and draws
 * its own status/result (chips, HP, win/lose overlay). The shell's generic side
 * panel — a per-move log and a "Thinking…" status — would just be a scrolling
 * wall of narration over a game that needs none, so these get the full-width
 * board and no side panel (no `.log`/`.status` elements, so the shell's
 * per-move narration and status calls quietly no-op). */
const BOARD_NATIVE_REALTIME = new Set<string>(["snake"]);

/** Trained artifacts fetched as static assets, keyed by the path the
 * registry asks for. */
const ARTIFACTS: Record<string, string> = {
  "data/azero/chess.bin": "artifacts/azero-chess.bin",
  "data/twentyone/solver-h3.bin": "artifacts/t21-solver-h3.bin",
  "data/twentyone/solver-h6.bin": "artifacts/t21-solver-h6.bin",
  "runs/ld_history/best.bin": "artifacts/ld-history-champion.bin",
  "runs/stratego/ataraxios.bin": "artifacts/ataraxios.bin",
};

/** The shipped artifacts a match with these opts will ask the registry for.
 * A config whose artifact is not shipped fails loudly at create — the
 * browser never trains. */
function artifactsFor(gameId: string, opts: Record<string, string>): string[] {
  const seatBots = opts.bots
    ? splitSpecs(opts.bots).map((spec) => parseBotSpec(spec).bot)
    : [];
  const wanted: string[] = [];
  if (gameId === "chess") {
    const usesAzero =
      opts.bot === "azero" ||
      seatBots.includes("azero");
    const net = opts.net ?? (usesAzero ? "data/azero/chess.bin" : null);
    if (net) wanted.push(net);
  }
  if (gameId === "twentyone")
    wanted.push(`data/twentyone/solver-h${opts.hearts ?? "6"}.bin`);
  if (gameId === "liars-dice") {
    const usesHistory =
      opts.bot === "history" ||
      seatBots.includes("history");
    if (usesHistory) wanted.push("runs/ld_history/best.bin");
  }
  if (gameId === "stratego") {
    const usesNet =
      opts.bot === "ataraxios" ||
      seatBots.includes("ataraxios");
    if (usesNet) wanted.push("runs/stratego/ataraxios.bin");
  }
  return wanted.filter((w) => w in ARTIFACTS);
}

interface OptField {
  key: string;
  value: string;
  note: string;
  bots: string[];
}

/** The drawer's fields come from the engine's structured option schema;
 * seed, seat, and bot get dedicated rows, and native-only options (training
 * knobs) do not exist on the web. */
function optFields(
  schema: GameOpt[],
  current: Record<string, string>,
): OptField[] {
  return schema
    .filter(
      (o) =>
        o.key !== "seed" &&
        o.key !== "seat" &&
        o.key !== "bot" &&
        !o.nativeOnly,
    )
    .map((o) => ({
      key: o.key,
      value: current[o.key] ?? o.value.split("|")[0].replace(/\.{3}$/, ""),
      note: o.note,
      bots: o.bots,
    }));
}

/** The bot the engine will seat for these opts — explicit choice, or the
 * schema's first listed bot (the registry default) for versus games. */
function effectiveBot(game: GameInfo, opts: Record<string, string>): string {
  const spec = game.optsSchema.find((o) => o.key === "bot");
  if (!spec) return "";
  return opts.bot ?? (game.solo ? "" : spec.value.split("|")[0]);
}

/** Seat names per game, matching what each board draws; `Seat N` otherwise. */
const SEAT_LABELS: Record<string, string[]> = {
  chess: ["White", "Black"],
  othello: ["Black", "White"],
  go: ["Black", "White"],
  pente: ["Black", "White"],
  connect4: ["Red", "Yellow"],
  twentyone: ["Player 1", "Player 2"],
};

function seatLabelFor(gameId: string, i: number): string {
  return SEAT_LABELS[gameId]?.[i] ?? `Seat ${i + 1}`;
}

/** The synthetic roster value for the only opponent a game offers when it
 * has no selectable `bot` (Twenty-One's solver). `sendsBot:false` means the
 * engine seats that opponent itself, so no `bot` option is sent. */
const SOLVER_OPPONENT = { value: "__solver__", label: "CFR solver", sendsBot: false };

interface RosterBot {
  value: string;
  label: string;
  sendsBot: boolean;
}

/** Opt-in allowlist of the opponents each game publishes on the site, keyed by
 * `game.id`. The lab registry declares *every* bot (research variants, weak
 * baselines, the obsolete CPU `azero` net); the public arcade lists only the
 * ones curated here — each game's opponent(s) strong enough to be a genuine
 * challenge, omitting random/weak/research bots. A game or bot absent here is
 * hidden by default, so nothing leaks onto the site until deliberately
 * published. (`azero-gpu` is the trained net: WebGPU when the browser has it,
 * the identical in-wasm CPU forward otherwise — one opponent, never a GPU/CPU
 * choice; the superseded `azero` net is never listed.) */
const SHOWN_BOTS: Record<string, readonly string[]> = {
  chess: ["azero-gpu"],
  "liars-dice": ["history", "rollout"],
  poker: ["equity"],
  othello: ["alphabeta"],
  connect4: ["alphabeta"],
  go: ["azero-gpu"],
  pente: ["azero-gpu", "alphabeta"],
  snake: ["bns"],
  // Only the trained net is published — the heuristic and random baselines
  // stay lab-only by policy.
  stratego: ["ataraxios"],
};

/** Opponents a seat can be filled with: the game's published [`SHOWN_BOTS`]
 * allowlist intersected with the bots the engine actually declares, or the
 * synthetic solver for games without a `bot` schema. */
function rosterBots(game: GameInfo): RosterBot[] {
  const spec = game.optsSchema.find((o) => o.key === "bot");
  if (!spec) return [SOLVER_OPPONENT];
  const available = new Set(spec.value.split("|"));
  return (SHOWN_BOTS[game.id] ?? [])
    .filter((b) => available.has(b))
    .map((b) => ({ value: b, label: botLabel(b), sendsBot: true }));
}

/** The roster value currently filling the bot seats. */
function currentBotValue(game: GameInfo, opts: Record<string, string>): string {
  const spec = game.optsSchema.find((o) => o.key === "bot");
  return spec ? effectiveBot(game, opts) : SOLVER_OPPONENT.value;
}

/** Seats in the current configuration: 1 (solo), the `players` count (e.g.
 * Liar's Dice), or 2 for the head-to-head games. */
function seatCount(game: GameInfo, opts: Record<string, string>): number {
  if (game.solo) return 1;
  const players = game.optsSchema.find((o) => o.key === "players");
  if (players)
    return Number(opts.players ?? players.value.split("|")[0]) || 2;
  return 2;
}

function randomSeed(): number {
  return (Math.floor(Math.random() * 0x7fff_ffff) | 1) >>> 0;
}

/** First concrete value in a manifest option, or undefined for generated
 * placeholders such as `...`. */
function optionDefault(opt: GameOpt): string | undefined {
  const value = opt.value.split("|")[0];
  return !value || value.endsWith("...") ? undefined : value;
}

function sameOptions(a: Record<string, string>, b: Record<string, string>): boolean {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  return [...keys].every((key) => a[key] === b[key]);
}

/** For interpolating user-editable values into markup (drawer fields). */
function esc(s: string): string {
  return s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
}

/** `<option>` markup for `[label, value]` pairs, marking `cur` selected and
 * prepending it as a "Custom" option when it isn't one of the pairs. Shared by
 * the drawer and the quick-controls strip. */
function optionList(pairs: [string, string][], cur: string): string {
  const opts = pairs.map(
    ([l, v]) =>
      `<option value="${esc(v)}"${v === cur ? " selected" : ""}>${esc(l)}</option>`,
  );
  if (!pairs.some(([, v]) => v === cur))
    opts.unshift(`<option value="${esc(cur)}" selected>Custom (${esc(cur)})</option>`);
  return opts.join("");
}

/** Inline single-color glyphs (currentColor) for the icon-only controls —
 * settings, the tournament lab, and the source link. Kept as crisp SVG rather
 * than emoji/font glyphs so they sharpen at any size and follow the theme. */
const ICON_GEAR =
  '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="3.2"/><path d="M19.4 13.5a7.7 7.7 0 0 0 0-3l1.7-1.3-1.8-3.1-2 .8a7.6 7.6 0 0 0-2.6-1.5L14.3 2h-3.6l-.4 2.1a7.6 7.6 0 0 0-2.6 1.5l-2-.8L3.9 8l1.7 1.3a7.7 7.7 0 0 0 0 3L3.9 13.5l1.8 3.1 2-.8a7.6 7.6 0 0 0 2.6 1.5l.4 2.1h3.6l.4-2.1a7.6 7.6 0 0 0 2.6-1.5l2 .8 1.8-3.1z"/></svg>';
const ICON_BEAKER =
  '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M9 3h6M10 3v6.2L5.3 17a2 2 0 0 0 1.8 3h9.8a2 2 0 0 0 1.8-3L14 9.2V3"/><path d="M7.2 14h9.6"/></svg>';
const ICON_GITHUB =
  '<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor" aria-hidden="true"><path d="M12 .5a11.5 11.5 0 0 0-3.64 22.42c.58.1.79-.25.79-.56v-2c-3.2.7-3.88-1.36-3.88-1.36-.53-1.34-1.3-1.7-1.3-1.7-1.05-.72.08-.71.08-.71 1.17.08 1.78 1.2 1.78 1.2 1.04 1.78 2.73 1.26 3.4.96.1-.75.4-1.27.73-1.56-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.18 1.18a11 11 0 0 1 5.79 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.83 1.19 3.09 0 4.42-2.69 5.39-5.25 5.68.41.35.78 1.05.78 2.12v3.14c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .5z"/></svg>';

/** Static mini-board previews on the home cards — each game introduces
 * itself with its own board, not an icon. */
function miniFor(id: string): string {
  switch (id) {
    case "chess":
      return `<div class="mini mini-chess"><span class="mini-pc" style="left:30%;top:30%">♞</span><span class="mini-pc mini-pc-w" style="left:70%;top:70%">♙</span></div>`;
    case "liars-dice":
      return `<div class="mini mini-dice">
        <span class="mini-die"><i style="left:25%;top:25%"></i><i style="left:65%;top:65%"></i></span>
        <span class="mini-die mini-die-2"><i style="left:45%;top:45%"></i><i style="left:18%;top:18%"></i><i style="left:72%;top:72%"></i></span>
        <span class="mini-cup"></span></div>`;
    case "twentyone":
      return `<div class="mini mini-t21"><span class="mini-card">7♠</span><span class="mini-card mini-card-2">9♦</span><span class="mini-heart">♥♥♥</span></div>`;
    case "poker":
      return `<div class="mini mini-poker"><span class="mini-pcard mini-pcard-r">A♥</span><span class="mini-pcard">K♠</span><span class="mini-chip mini-chip-1"></span><span class="mini-chip mini-chip-2"></span><span class="mini-chip mini-chip-3"></span></div>`;
    case "othello":
      // A full 8×8 reversi board zoomed in so the grid bleeds past the frame
      // (slice crop), with a believable mid-game cluster of discs around the
      // centre. Cell C=40, disc centres at 20+40·k.
      return `<div class="mini mini-othello">
        <svg class="mini-ot-svg" viewBox="0 0 320 320" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
          <defs>
            <radialGradient id="ot-w" cx="0.35" cy="0.3" r="0.8"><stop offset="0" stop-color="#fff"/><stop offset="1" stop-color="#cfc9b8"/></radialGradient>
            <radialGradient id="ot-b" cx="0.35" cy="0.3" r="0.8"><stop offset="0" stop-color="#5a5a5a"/><stop offset="1" stop-color="#0a0a0a"/></radialGradient>
          </defs>
          <rect width="320" height="320" fill="#2f6b46"/>
          ${[40, 80, 120, 160, 200, 240, 280]
            .map((p) => `<line x1="${p}" y1="0" x2="${p}" y2="320" stroke="rgba(0,0,0,0.32)" stroke-width="2"/><line x1="0" y1="${p}" x2="320" y2="${p}" stroke="rgba(0,0,0,0.32)" stroke-width="2"/>`)
            .join("")}
          ${[[80, 80], [80, 240], [240, 80], [240, 240]]
            .map(([x, y]) => `<circle cx="${x}" cy="${y}" r="3.5" fill="rgba(0,0,0,0.42)"/>`)
            .join("")}
          ${[
            ["w", 3, 1], ["b", 4, 1],
            ["b", 2, 2], ["w", 3, 2], ["b", 4, 2], ["w", 5, 2],
            ["w", 2, 3], ["b", 3, 3], ["b", 4, 3], ["b", 5, 3], ["w", 6, 3],
            ["b", 1, 4], ["w", 2, 4], ["w", 3, 4], ["b", 4, 4], ["w", 5, 4],
            ["b", 3, 5], ["w", 4, 5], ["b", 5, 5],
            ["b", 4, 6],
          ]
            .map(([c, col, row]) => `<circle cx="${20 + 40 * Number(col)}" cy="${20 + 40 * Number(row)}" r="16" fill="url(#ot-${c})" stroke="rgba(0,0,0,0.35)" stroke-width="0.8"/>`)
            .join("")}
        </svg>
      </div>`;
    case "connect4":
      // A full 7-wide Connect-4 board zoomed in so the rows continue below the
      // frame (slice crop): the top sits at the top, the bottom row is cut off,
      // implying a taller board. Varying column heights read as a real mid-game
      // (any visible disc is gravity-consistent — the hidden rows below it are
      // full).
      return `<div class="mini mini-c4">
        <svg class="mini-c4-svg" viewBox="0 0 274 110" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
          <defs>
            <radialGradient id="c4-r" cx="0.36" cy="0.3" r="0.85"><stop offset="0" stop-color="#ff8a7a"/><stop offset="0.55" stop-color="#e23b2e"/><stop offset="1" stop-color="#a01a12"/></radialGradient>
            <radialGradient id="c4-y" cx="0.36" cy="0.3" r="0.85"><stop offset="0" stop-color="#ffe89a"/><stop offset="0.55" stop-color="#f2c037"/><stop offset="1" stop-color="#b8860b"/></radialGradient>
          </defs>
          <rect x="2" y="2" width="270" height="250" rx="10" fill="#2256b6"/>
          <rect x="2" y="2" width="270" height="26" rx="10" fill="#2c63c9"/>
          ${[23, 61, 99, 137, 175, 213, 251]
            .flatMap((x, c) =>
              [20, 58, 96].map((y, r) => {
                const fills: Record<string, string> = { "0-2": "c4-r", "1-1": "c4-y", "1-2": "c4-r", "2-0": "c4-r", "2-1": "c4-y", "2-2": "c4-r", "3-0": "c4-y", "3-1": "c4-r", "3-2": "c4-y", "4-1": "c4-r", "4-2": "c4-y", "5-2": "c4-r", "6-1": "c4-y", "6-2": "c4-r" };
                const fill = fills[`${c}-${r}`];
                return `<circle cx="${x}" cy="${y}" r="15.5" fill="${fill ? `url(#${fill})` : "#16335f"}"/>`;
              }),
            )
            .join("")}
        </svg>
      </div>`;
    case "go":
      // Cell size C=22 (.mini-go), stones ON intersections (k·C).
      return `<div class="mini mini-go"><span class="mini-stone mini-stone-b" style="left:44px;top:44px"></span><span class="mini-stone mini-stone-w" style="left:66px;top:66px"></span><span class="mini-stone mini-stone-b" style="left:44px;top:88px"></span><span class="mini-stone mini-stone-w" style="left:88px;top:44px"></span></div>`;
    case "pente":
      // A contested endgame: black completes five-in-a-row along y=66, white
      // bracketing both ends, the rest interlocked above and below like a real
      // game.
      return `<div class="mini mini-pente">${[
        ["b", 66, 66], ["b", 88, 66], ["b", 110, 66], ["b", 132, 66], ["b", 154, 66],
        ["b", 88, 88], ["b", 154, 44],
        ["w", 44, 66], ["w", 176, 66], ["w", 66, 44], ["w", 110, 44], ["w", 132, 88], ["w", 66, 88],
      ]
        .map(([c, x, y]) => `<span class="mini-pstone mini-pstone-${c}" style="left:${x}px;top:${y}px"></span>`)
        .join("")}</div>`;
    case "snake":
      // The Snake-game snake: a glossy tube that follows the grid with
      // right-angle (rounded) turns, ending in a detailed head — two eyes,
      // nostrils, and a forked tongue — beside an apple.
      return `<div class="mini mini-snake">
        <svg class="mini-snake-svg" viewBox="0 0 220 110" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
          <defs>
            <linearGradient id="snk-body" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0" stop-color="#3fb950"/><stop offset="1" stop-color="#1f7a34"/>
            </linearGradient>
            <radialGradient id="snk-head" cx="0.36" cy="0.3" r="0.85">
              <stop offset="0" stop-color="#86efa0"/><stop offset="0.6" stop-color="#46c45c"/><stop offset="1" stop-color="#1f7a34"/>
            </radialGradient>
            <radialGradient id="snk-food" cx="0.36" cy="0.3" r="0.85">
              <stop offset="0" stop-color="#ffc7be"/><stop offset="0.5" stop-color="#f85149"/><stop offset="1" stop-color="#b21f17"/>
            </radialGradient>
          </defs>
          <circle cx="40" cy="28" r="9" fill="url(#snk-food)"/>
          <path class="snk-rim" d="M22 77 H88 V33 H132 V77 H176"/>
          <path class="snk-tube" d="M22 77 H88 V33 H132 V77 H176"/>
          <path class="snk-gloss" d="M22 77 H88 V33 H132 V77 H176"/>
          <circle cx="176" cy="77" r="14.5" fill="url(#snk-head)" stroke="#0c3a1c" stroke-width="1.5"/>
          <circle cx="171" cy="70" r="4.2" fill="#fff"/><circle cx="172.2" cy="70" r="2.1" fill="#0a1f12"/>
          <circle cx="182" cy="71" r="4.2" fill="#fff"/><circle cx="183.2" cy="71" r="2.1" fill="#0a1f12"/>
          <circle cx="186" cy="74.5" r="0.9" fill="#0a1f12"/><circle cx="186" cy="80" r="0.9" fill="#0a1f12"/>
          <path d="M189 77 H201 M201 77 L207 73 M201 77 L207 81" fill="none" stroke="#e5484d" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>`;
    case "coil":
      // A fat slither.io serpent: a thick gradient body with banded segment
      // rings and a top gloss, a domed head with two eyes, and glowing pellets
      // scattered on the dark arena.
      return `<div class="mini mini-slither">
        <svg class="mini-slither-svg" viewBox="0 0 220 110" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
          <defs>
            <linearGradient id="cl-body" x1="0" y1="0" x2="1" y2="0.3">
              <stop offset="0" stop-color="#33d6ff"/><stop offset="0.5" stop-color="#1f8bff"/><stop offset="1" stop-color="#7b5bff"/>
            </linearGradient>
            <radialGradient id="cl-head" cx="0.36" cy="0.3" r="0.85">
              <stop offset="0" stop-color="#bdf0ff"/><stop offset="0.55" stop-color="#33b6ff"/><stop offset="1" stop-color="#1463d8"/>
            </radialGradient>
            <radialGradient id="cl-p1" cx="0.4" cy="0.35" r="0.8"><stop offset="0" stop-color="#fff2a8"/><stop offset="1" stop-color="#f5b400"/></radialGradient>
            <radialGradient id="cl-p2" cx="0.4" cy="0.35" r="0.8"><stop offset="0" stop-color="#c8ffd0"/><stop offset="1" stop-color="#34d058"/></radialGradient>
          </defs>
          <circle cx="40" cy="30" r="5" fill="url(#cl-p1)"/>
          <circle cx="150" cy="86" r="5" fill="url(#cl-p2)"/>
          <circle cx="92" cy="20" r="4" fill="url(#cl-p1)"/>
          <path class="cl-rim" d="M18 78 C 64 92, 70 40, 116 46 S 176 84, 202 50"/>
          <path class="cl-tube" d="M18 78 C 64 92, 70 40, 116 46 S 176 84, 202 50"/>
          <path class="cl-bands" d="M18 78 C 64 92, 70 40, 116 46 S 176 84, 202 50"/>
          <path class="cl-gloss" d="M18 71 C 64 85, 70 33, 116 39 S 176 77, 202 44"/>
          <circle cx="202" cy="50" r="17" fill="url(#cl-head)" stroke="#0d3a7a" stroke-width="1.5"/>
          <circle cx="206" cy="43" r="5" fill="#fff" stroke="#0a1230" stroke-width="0.8"/><circle cx="208.4" cy="43" r="2.6" fill="#0a1230"/><circle cx="206.6" cy="41.4" r="1" fill="#fff"/>
          <circle cx="206" cy="56" r="5" fill="#fff" stroke="#0a1230" stroke-width="0.8"/><circle cx="208.4" cy="56" r="2.6" fill="#0a1230"/><circle cx="206.6" cy="54.4" r="1" fill="#fff"/>
        </svg>
      </div>`;
    case "stratego":
      return `<div class="mini mini-stratego">
        <svg viewBox="0 0 220 110" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
          <defs>
            <linearGradient id="sgm-red" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0" stop-color="#c24a42"/><stop offset="1" stop-color="#932e27"/>
            </linearGradient>
            <linearGradient id="sgm-blue" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0" stop-color="#47689f"/><stop offset="1" stop-color="#2b4778"/>
            </linearGradient>
            <pattern id="sgm-rib" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <rect width="7" height="7" fill="#33589e"/>
              <line x1="0" y1="0" x2="0" y2="7" stroke="#16294f" stroke-width="2.2"/>
            </pattern>
            <filter id="sgm-grain" x="0" y="0" width="100%" height="100%">
              <feTurbulence type="fractalNoise" baseFrequency="0.5" numOctaves="2" seed="3" result="n"/>
              <feColorMatrix in="n" type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0.7 0.7 0 0 0"/>
              <feComposite operator="in" in2="SourceGraphic"/>
            </filter>
          </defs>
          <rect width="220" height="110" fill="#b3c68c"/>
          <rect width="220" height="110" fill="#3f5a2b" opacity="0.16" filter="url(#sgm-grain)"/>
          <g stroke="#55673c" stroke-width="1" opacity="0.4">
            ${Array.from({ length: 9 }, (_, i) => `<line x1="${(i + 1) * 22}" y1="0" x2="${(i + 1) * 22}" y2="110"/>`).join("")}
            <line x1="0" y1="33" x2="220" y2="33"/><line x1="0" y1="77" x2="220" y2="77"/>
          </g>
          <rect x="40" y="30" width="52" height="52" rx="12" fill="#cfc191"/>
          <rect x="43" y="33" width="46" height="46" rx="9" fill="#8db8d8"/>
          <path d="M50 48 h32 M54 60 h24 M50 70 h32" fill="none" stroke="#eaf3f9" stroke-width="2" opacity="0.55" stroke-linecap="round"/>
          <g transform="rotate(-4 132 62)">
            <rect x="106" y="20" width="52" height="84" rx="10" fill="#6e1b16"/>
            <rect x="110" y="25" width="44" height="74" rx="7" fill="url(#sgm-red)"/>
            <g transform="translate(111 40) scale(0.122)"><path d="${RANK_ICONS["10"]}" fill="#e9c97e"/></g>
            <circle cx="121" cy="34" r="8.5" fill="#5e150f" stroke="#caa552" stroke-width="1.2"/>
            <text x="121" y="38" text-anchor="middle" font-family="ui-monospace,monospace" font-weight="800" font-size="11" letter-spacing="-1" fill="#e9c97e">10</text>
          </g>
          <g transform="rotate(5 184 66)">
            <rect x="160" y="24" width="52" height="84" rx="10" fill="#16294f"/>
            <rect x="164" y="29" width="44" height="74" rx="7" fill="url(#sgm-rib)"/>
            <rect x="168" y="33" width="36" height="66" rx="5" fill="none" stroke="#e9c97e" stroke-width="1.4" opacity="0.65"/>
            <g transform="translate(163 43) scale(0.09)"><path d="${RANK_ICONS.back}" fill="#e9c97e" opacity="0.75"/></g>
          </g>
        </svg>
      </div>`;
    case "dood":
      return `<div class="mini mini-doom"><span class="mini-doom-word">DOOD</span></div>`;
    default:
      return `<div class="mini"></div>`;
  }
}

type Mode = "play" | "watch";

export class App {
  private host = new EngineHost();
  private manifest!: Manifest;
  private frontend: GameFrontend | null = null;
  private clientBot: ClientBot | null = null;
  private tourney: TournamentScreen | null = null;
  private slither: SlitherScreen | null = null;
  private gen = 0;
  private speedScale = 1;
  private submitResolve: ((input: string) => void) | null = null;
  private logEl: HTMLElement | null = null;
  private statusEl: HTMLElement | null = null;
  private sideEl: HTMLElement | null = null;
  private readoutEl: HTMLElement | null = null;
  private debugOn = localStorage.getItem("arcadeDebug") === "1";
  private debugSubs = new Set<(on: boolean) => void>();

  constructor(private root: HTMLElement) {
    window.addEventListener("hashchange", () => this.route());
  }

  async start(): Promise<void> {
    this.root.innerHTML = '<div class="boot">Waking the engine…</div>';
    this.manifest = await this.host.manifest();
    this.manifest.games = this.manifest.games.filter(
      (g) => !HIDDEN_GAMES.has(g.id),
    );
    this.route();
  }

  // ---------- routing ----------
  // The screen is a function of the URL hash: `#/` home, `#/g/<id>` a match
  // (`?mode=watch` to spectate), `#/lab` the tournament. Navigation sets the
  // hash; the hashchange listener renders — so the browser back button just
  // works, and matches are deep-linkable.

  private route(): void {
    const [path, query] = location.hash.replace(/^#/, "").split("?");
    const segs = path.split("/").filter(Boolean);
    const params = new URLSearchParams(query ?? "");
    if (segs[0] === "lab") {
      this.renderTournament();
      return;
    }
    if (segs[0] === "dood") {
      this.renderDoom();
      return;
    }
    if (segs[0] === "coil") {
      void this.renderSlither();
      return;
    }
    if (segs[0] === "g" && segs[1]) {
      const game = this.manifest.games.find((g) => g.id === segs[1]);
      if (game) {
        const mode: Mode = params.get("mode") === "watch" ? "watch" : "play";
        void this.startMatch(game, mode);
        return;
      }
      history.replaceState(null, "", "#/");
    }
    this.renderHome();
  }

  /** Navigate by setting the hash (adds a history entry; renders via the
   * hashchange listener). Re-renders in place when already on the target. */
  private navTo(path: string): void {
    const target = `#${path}`;
    if (location.hash === target) this.route();
    else location.hash = target;
  }

  /** Keep the URL in step with the live match without adding history — an
   * in-place rematch or mode flip should not stack back-button entries. */
  private syncMatchUrl(game: GameInfo, mode: Mode): void {
    const q = mode === "watch" ? "?mode=watch" : "";
    history.replaceState(null, "", `#/g/${game.id}${q}`);
  }

  /** Show the corner "‹ Games" crumb (beside the site link) on every screen
   * except the home grid, where it would point back to where you already are. */
  private setGamesLink(visible: boolean): void {
    const el = document.querySelector<HTMLElement>("[data-games-link]");
    if (el) el.hidden = !visible;
  }

  // ---------- home ----------

  private renderHome(): void {
    this.teardown();
    this.setGamesLink(false);
    const cards = this.manifest.games
      .map(
        (g) => `
        <div class="card" data-game="${g.id}" role="button" tabindex="0">
          ${miniFor(g.id)}
          <div class="card-text">
            <span class="card-name">${esc(g.name || g.id)}</span>
          </div>
          <button type="button" class="card-watch" title="Watch bots play">watch</button>
        </div>`,
      )
      .join("");
    // DOOD and Coil are not engine games (real-time, not Game-trait matches);
    // their cards open standalone screens instead of starting a match.
    const doomCard = `
        <div class="card card-doom" data-special="dood" role="button" tabindex="0">
          ${miniFor("dood")}
          <div class="card-text">
            <span class="card-name">DOOD</span>
          </div>
        </div>`;
    this.root.innerHTML = `
      <div class="home">
        <header class="home-head">
          <h1>Games Room</h1>
        </header>
        <div class="card-grid">${cards}${doomCard}</div>
        <footer class="home-footer">
          <button type="button" class="icon-btn tourney-link" title="Tournament lab" aria-label="Tournament lab">${ICON_BEAKER}</button>
          <a class="icon-btn" href="https://github.com/henri123lemoine/games" title="GitHub" aria-label="GitHub">${ICON_GITHUB}</a>
        </footer>
      </div>`;
    for (const el of this.root.querySelectorAll<HTMLElement>(".card")) {
      const game = this.manifest.games.find((g) => g.id === el.dataset.game);
      if (!game) continue;
      const play = () => this.navTo(`/g/${game.id}`);
      el.onclick = play;
      el.onkeydown = (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          play();
        }
      };
      el.querySelector<HTMLButtonElement>(".card-watch")!.onclick = (e) => {
        e.stopPropagation();
        this.navTo(`/g/${game.id}?mode=watch`);
      };
    }
    const doomEl = this.root.querySelector<HTMLElement>('.card[data-special="dood"]');
    if (doomEl) {
      const openDoom = () => this.navTo("/dood");
      doomEl.onclick = openDoom;
      doomEl.onkeydown = (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          openDoom();
        }
      };
    }
    const slitherEl = this.root.querySelector<HTMLElement>(
      '.card[data-special="coil"]',
    );
    if (slitherEl) {
      const openSlither = () => this.navTo("/coil");
      slitherEl.onclick = openSlither;
      slitherEl.onkeydown = (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          openSlither();
        }
      };
    }
    this.root.querySelector<HTMLButtonElement>(".tourney-link")!.onclick = () =>
      this.navTo("/lab");
  }

  /** DOOM: the strategic FFA WebAssembly build runs in its own page (it is not
   * a Game-trait match), mounted in an iframe. */
  private renderDoom(): void {
    this.teardown();
    const src = `${import.meta.env.BASE_URL}doom-ai/index.html`;
    this.root.innerHTML = `
      <div class="match doom-screen">
        <header class="match-bar">
          <span class="match-title">DOOM · THE FOUNDRY FFA</span>
          <span class="spacer"></span>
          <span class="muted doom-note">1–3 tactical opponents · click Fight, then use mouse + WASD</span>
        </header>
        <div class="doom-frame-wrap">
          <iframe class="doom-frame" src="${src}" title="DOOM"
            allow="autoplay; fullscreen"></iframe>
        </div>
      </div>`;
    this.setGamesLink(true);
  }

  /** Coil: the lab's own real-time game, played against the PPO-trained
   * encircle bot. Not a Game-trait match, so it runs standalone on its own
   * screen (its wasm steps the world and drives the bots each frame). */
  private async renderSlither(): Promise<void> {
    this.teardown();
    this.root.innerHTML = `
      <div class="match slither-screen">
        <header class="match-bar">
          <span class="match-title">Coil</span>
          <span class="spacer"></span>
          <span class="muted">vs. the trained encircle bot · runs in your browser</span>
        </header>
        <div class="slither-mount"></div>
      </div>`;
    this.setGamesLink(true);
    const mount = this.root.querySelector<HTMLElement>(".slither-mount")!;
    const { SlitherScreen } = await import("../frontends/slither");
    // A navigation during the dynamic import bumps `gen`; bail rather than
    // mounting onto a torn-down screen.
    const gen = this.gen;
    this.slither = new SlitherScreen();
    await this.slither.mount(mount);
    if (gen !== this.gen) {
      this.slither.destroy();
      this.slither = null;
    }
  }

  private renderTournament(): void {
    this.teardown();
    this.setGamesLink(true);
    this.tourney = new TournamentScreen(
      this.root,
      this.manifest.compare,
      this.manifest.games,
      this.host,
      () => this.navTo("/"),
    );
    this.tourney.render();
  }

  // ---------- match ----------

  private buildOpts(
    game: GameInfo,
    mode: Mode,
    overrides: Record<string, string>,
  ): Record<string, string> {
    const opts: Record<string, string> = {
      ...DEFAULT_OPTS[game.id],
      ...overrides,
    };
    if (mode === "watch") {
      if (game.solo) opts.bot ||= game.watchBot;
      else opts.seat = "watch";
    } else if (game.solo) {
      delete opts.bot;
    } else if (opts.seat === "watch") {
      opts.seat = "0";
    }
    if (opts.bots) {
      // Heterogeneous bots: each seat's knobs live in its own spec, so the
      // shared map carries no `bot` and no bot-knob keys (they'd be rejected
      // as unused by the engine).
      delete opts.bot;
      for (const o of game.optsSchema) if (o.bots.length > 0) delete opts[o.key];
      // A browser with no WebGPU must not keep displaying a GPU-only level in
      // a per-seat spec while the driver silently caps it. Normalize every
      // active external seat to the canonical trivial CPU budget.
      if (isCpuFallback()) {
        const human = opts.seat === "watch" ? -1 : Number(opts.seat ?? "0");
        opts.bots = splitSpecs(opts.bots)
          .map((text, seat) => {
            const spec = parseBotSpec(text);
            if (seat !== human && spec.bot === "azero-gpu") {
              const allowed = new Set(CPU_LEVELS.map(([, value]) => value));
              if (!allowed.has(spec.opts.sims ?? "")) spec.opts.sims = String(TRIVIAL_SIMS);
            }
            return formatBotSpec(spec.bot, spec.opts);
          })
          .join(",");
      }
    } else {
      const bot = effectiveBot(game, opts);
      for (const o of game.optsSchema) {
        if (o.bots.length > 0 && !o.bots.includes(bot)) delete opts[o.key];
      }
      // AlphaZero on the CPU forward (no WebGPU) offers only the responsive
      // levels — snap anything off that list (e.g. the GPU default) to Trivial,
      // resolved once here so the drawer, the quick controls, and the bot agree.
      const diff = DIFFICULTY[`${game.id}/${bot}`];
      if (bot === "azero-gpu" && isCpuFallback() && diff) {
        const allowed = new Set(CPU_LEVELS.map(([, v]) => v));
        if (!allowed.has(opts[diff.key] ?? "")) opts[diff.key] = String(TRIVIAL_SIMS);
      }
    }
    // The Pente AlphaZero net is trained at 19×19 and the client driver pins the
    // board there, so force the engine match to 19 too whenever that bot plays —
    // a stale size from the drawer would desync the page-side search from the
    // engine board.
    if (
      game.id === "pente" &&
      this.seatStates(game, opts).some((bot) => bot === "azero-gpu")
    ) {
      opts.size = PENTE_AZ_SIZE;
    }
    opts.seed ||= String(randomSeed());
    return opts;
  }

  private async startMatch(
    game: GameInfo,
    mode: Mode,
    overrides: Record<string, string> = {},
  ): Promise<void> {
    const gen = ++this.gen;
    this.teardownMatch();
    try {
      const opts = this.buildOpts(game, mode, overrides);
      this.syncMatchUrl(game, mode);
      this.renderMatchSkeleton(game, mode, opts);
      // Each externally driven seat gets its own resolved config and search
      // instance. This is what makes the per-seat controls true at runtime.
      const clientConfigs = this.clientBotConfigs(game, opts);
      if (clientConfigs.length && isCpuFallback()) this.showCpuNote();
      await this.loadArtifacts(game, opts);
      const st = await this.host.create(game.id, opts);
      if (gen !== this.gen) return;
      const boardEl = this.root.querySelector<HTMLElement>(".board")!;
      this.frontend = frontendFor(game.id);
      const ctx: FrontendCtx = {
        gameId: game.id,
        opts,
        humanSeat: st.humanSeat,
        numSeats: st.numSeats,
        submit: (input) => this.submit(input),
        animationScale: () => this.animationScale(),
        debug: () => this.debugOn,
        onDebugChange: (cb) => this.onDebugChange(cb),
        setDebugReadout: (lines) => this.setDebugReadout(lines),
        debugLog: (text) => this.debugLog(text),
      };
      this.frontend.mount(boardEl, ctx);
      this.frontend.render(st);
      this.fillSeatSlots(game, opts);

      const clientBot = await createClientBots(game.id, clientConfigs);
      if (gen !== this.gen) {
        clientBot?.cancel();
        return;
      }
      this.clientBot = clientBot;
      if (this.clientBot?.cpuFallback) this.showCpuNote(this.clientBot.cpuFallback);
      // The GPU bot booted with no CPU-fallback note, so WebGPU really ran:
      // validate this device's forward against the reference, non-blocking.
      if (game.id === "go" && clientConfigs.length && !this.clientBot?.cpuFallback)
        void this.checkGoConformance();
      this.setStatus(st.humanSeat < 0 ? "Bots playing…" : "Thinking…");
      void this.runLoop(gen);
    } catch (e) {
      if (gen === this.gen)
        this.setStatus(`Could not start: ${message(e)}`, "error");
    }
  }

  private renderMatchSkeleton(
    game: GameInfo,
    mode: Mode,
    opts: Record<string, string>,
  ): void {
    // Normal is the default. Battlesnake humans may opt into a different visual
    // pace, but input itself is never held behind an extra cadence timer.
    this.speedScale = 1;
    const speedControl =
      game.id === "snake" && mode === "play"
        ? `<label class="speed-label">pace
            <select class="speed">
              <option value="1.25">relaxed</option>
              <option value="1" selected>normal</option>
              <option value="0.7">fast</option>
            </select>
          </label>`
        : mode === "watch"
        ? `<label class="speed-label">speed
            <select class="speed">
              <option value="2">slow</option>
              <option value="1" selected>normal</option>
              <option value="0.4">fast</option>
              <option value="0">instant</option>
            </select>
          </label>`
        : "";
    this.root.innerHTML = `
      <div class="match">
        <header class="match-bar">
          <span class="match-title">${esc(game.name || game.id)}</span>
          <span class="spacer"></span>
          ${speedControl}
          <button type="button" class="link again">rematch</button>
          <button type="button" class="icon-btn gear" title="Match settings" aria-label="Match settings">${ICON_GEAR}</button>
        </header>
        ${this.quickControlsHtml(game, opts)}
        <div class="cpu-note" hidden></div>
        <div class="cpu-note gpu-mismatch-note" hidden></div>
        <div class="match-body${game.solo || BOARD_NATIVE_REALTIME.has(game.id) ? " match-body--solo" : ""}">
          <section class="board"></section>
          ${this.sideHtml(game)}
        </div>
        <div class="drawer" hidden>
          <div class="drawer-panel">
            <h3>Match settings</h3>
            <div class="drawer-fields"></div>
            <div class="drawer-actions">
              <button type="button" class="primary drawer-apply">Restart with these</button>
              <button type="button" class="link drawer-close">cancel</button>
            </div>
          </div>
        </div>
      </div>`;
    this.logEl = this.root.querySelector(".log");
    this.statusEl = this.root.querySelector(".status");
    this.sideEl = this.root.querySelector(".side");
    this.readoutEl = this.root.querySelector(".debug-readout");
    const debugCheck = this.root.querySelector<HTMLInputElement>(".debug-check");
    if (debugCheck)
      debugCheck.onchange = () => this.setDebug(debugCheck.checked);
    this.setGamesLink(true);
    this.root.querySelector<HTMLButtonElement>(".again")!.onclick = () =>
      void this.startMatch(game, mode, { ...opts, seed: String(randomSeed()) });
    const speed = this.root.querySelector<HTMLSelectElement>(".speed");
    if (speed)
      speed.onchange = (e) => {
        this.speedScale = Number((e.target as HTMLSelectElement).value);
      };
    const form = this.root.querySelector<HTMLFormElement>(".free-input");
    if (form)
      form.onsubmit = (e) => {
        e.preventDefault();
        const input = form.querySelector("input")!;
        if (input.value.trim()) {
          this.submit(input.value.trim());
          input.value = "";
        }
      };
    this.wireDrawer(game, opts);
    this.wireQuickControls(game, opts);
  }

  /** The always-visible controls strip: the game-level dropdowns a visitor is
   * most likely to change (board size, player count) — the genuine game
   * characteristics. Difficulty is NOT here; it belongs to the opponent and
   * rides next to that seat's selector (see `fillSeatSlots`). Empty for solo /
   * heterogeneous matches (their settings stay in the drawer). */
  private quickControlsHtml(
    game: GameInfo,
    opts: Record<string, string>,
  ): string {
    if (game.solo || opts.bots) return "";
    const cells: string[] = [];
    const cell = (key: string, label: string, pairs: [string, string][], cur: string, locked: boolean) =>
      `<label class="qc"><span class="qc-name">${esc(label)}</span><select class="qc-select" data-key="${esc(key)}"${locked ? " disabled" : ""}>${optionList(pairs, cur)}</select></label>`;
    for (const o of game.optsSchema) {
      if (o.bots.length || o.key === "seat" || o.key === "seed" || o.nativeOnly)
        continue;
      const choices = optChoicesFor(game.id, o.key);
      // A single fixed value (e.g. Pente's 19×19) needs no picker.
      if (!choices || choices.length <= 1) continue;
      const cur = opts[o.key] ?? o.value.split("|")[0];
      cells.push(cell(o.key, o.key, choices.map((c) => [c, c]), cur, false));
    }
    return cells.length ? `<div class="match-controls">${cells.join("")}</div>` : "";
  }

  private wireQuickControls(
    game: GameInfo,
    opts: Record<string, string>,
  ): void {
    for (const sel of this.root.querySelectorAll<HTMLSelectElement>(
      ".match-controls .qc-select",
    )) {
      sel.onchange = () => {
        const overrides: Record<string, string> = {};
        if (opts.seat !== undefined) overrides.seat = opts.seat;
        if (opts.bot !== undefined) overrides.bot = opts.bot;
        for (const el of this.root.querySelectorAll<HTMLSelectElement>(
          ".match-controls .qc-select",
        )) {
          const key = el.dataset.key;
          if (key && el.value.trim() !== "") overrides[key] = el.value.trim();
        }
        const mode: Mode = opts.seat === "watch" ? "watch" : "play";
        void this.startMatch(game, mode, overrides);
      };
    }
  }

  /** Surfaces, in-match, that AlphaZero is on the CPU forward. */
  private showCpuNote(text?: string): void {
    const note = this.root.querySelector<HTMLElement>(".cpu-note");
    if (!note) return;
    note.textContent =
      text ??
      "CPU FALLBACK ACTIVE: No compatible WebGPU device was detected. AlphaZero is running on the CPU, so only the Trivial and Light levels are offered. Open it in a WebGPU browser (recent Chrome/Edge) for the full difficulty ladder.";
    note.hidden = false;
  }

  /** Persistent warning that this device's WebGPU forward diverges from the
   * reference. Warn-only: the bot keeps running on the GPU it booted. */
  private showGpuMismatchNote(msg: string): void {
    const note = this.root.querySelector<HTMLElement>(".gpu-mismatch-note");
    if (!note) return;
    note.textContent = msg;
    note.hidden = false;
  }

  /** Confirms this device computes the go net the same as the reference,
   * warning the visitor if it does not. Reuses the GPU device + weights the bot
   * just booted, runs once per device per net version (cached in localStorage),
   * and never blocks the bot's first move. */
  private async checkGoConformance(): Promise<void> {
    const gen = this.gen;
    try {
      const [gpu, weights] = await Promise.all([getGoGpu(), getGoWeights()]);
      const key = goSelfcheckKey(weights);
      const cached = readSelfcheck(key);
      if (cached?.pass) {
        console.info("[go-selfcheck] cached pass; skipping re-run");
        return;
      }
      const result = await runGoConformance(gpu, weights, { limit: 10 });
      if (gen !== this.gen) return;
      writeSelfcheck(key, result);
      console.info(
        `[go-selfcheck] pass=${result.pass} maxDp=${result.maxDp.toExponential(2)} ` +
          `maxDv=${result.maxDv.toExponential(2)} over ${result.count} fixtures`,
      );
      if (!result.pass) this.reportGoMismatch(result);
    } catch (e) {
      console.info(`[go-selfcheck] skipped: ${message(e)}`);
    }
  }

  private reportGoMismatch(result: ConformanceResult): void {
    const where = result.worst
      ? ` at ply ${result.worst.plies}, ${result.worst.size}×${result.worst.size}`
      : "";
    this.showGpuMismatchNote(
      `Inference check: this browser's GPU computes the AlphaZero network ` +
        `differently from the reference (max policy Δ ${result.maxDp.toExponential(2)}, ` +
        `value Δ ${result.maxDv.toExponential(2)}${where}). Move quality may be ` +
        `degraded on this device.`,
    );
  }

  /** The match side panel: status + move log for the turn-based versus games,
   * and a type-a-move input only for the generic fallback (custom frontends
   * have board-native input). Solo games render their own score and game-over
   * overlay, so they get no side panel — the board takes the full width. */
  private sideHtml(game: GameInfo): string {
    if (game.solo || BOARD_NATIVE_REALTIME.has(game.id)) return "";
    const freeInput = hasFrontend(game.id)
      ? ""
      : `<form class="free-input">
          <input placeholder="or type a move…" autocomplete="off" />
          <button type="submit">send</button>
        </form>`;
    return `<aside class="side${this.debugOn ? " debug-on" : ""}">
        <div class="status">Starting…</div>
        <div class="log-head">
          <span class="log-title">Log</span>
          <label class="debug-toggle">
            <input type="checkbox" class="debug-check"${this.debugOn ? " checked" : ""} />
            <span class="debug-pill"></span>
            <span class="debug-word">debug</span>
          </label>
        </div>
        <div class="debug-readout" aria-live="polite"></div>
        <div class="log" aria-live="polite"></div>
        ${freeInput}
      </aside>`;
  }

  /** Per-seat selection. Each board places `<span class="seat-slot" data-seat>`
   * markers in its player panel; the shell fills them with a wired `<select>`
   * (You / a bot). A board that places no slots gets a shell-drawn fallback
   * bar instead, so every game has working controls. Changing a seat restarts
   * the match: two different in-engine bots become a `bots=` match (e.g.
   * AlphaBeta vs MCTS); a uniform choice uses the simpler `bot=`. */
  private fillSeatSlots(game: GameInfo, opts: Record<string, string>): void {
    let slots = [
      ...this.root.querySelectorAll<HTMLElement>(".board .seat-slot[data-seat]"),
    ];
    if (slots.length === 0) {
      const n = seatCount(game, opts);
      const bar = document.createElement("div");
      bar.className = "roster";
      bar.innerHTML = Array.from(
        { length: n },
        (_, i) =>
          `<label class="seat"><span class="seat-name">${esc(this.seatName(game, i))}</span><span class="seat-slot" data-seat="${i}"></span></label>`,
      ).join("");
      this.root
        .querySelector(".match-bar")!
        .insertAdjacentElement("afterend", bar);
      slots = [...bar.querySelectorAll<HTMLElement>(".seat-slot[data-seat]")];
    }
    const states = this.seatStates(game, opts);
    const bots = rosterBots(game);
    for (const slot of slots) {
      const i = Number(slot.dataset.seat);
      const sel = document.createElement("select");
      sel.className = "seat-select";
      const choices = [{ value: "__you__", label: "You" }].concat(
        bots.map((b) => ({ value: b.value, label: b.label })),
      );
      for (const c of choices) {
        const o = document.createElement("option");
        o.value = c.value;
        o.textContent = c.label;
        o.selected = c.value === states[i];
        sel.append(o);
      }
      sel.onchange = () => this.applySeatChange(game, opts, i, sel.value);
      const level = this.seatLevelSelect(game, opts, i, states[i]);
      const info = this.seatInfoButton(game, states[i]);
      const parts = [sel, level, info].filter((n): n is HTMLElement => n !== null);
      slot.replaceChildren(parts.length > 1 ? this.fragment(parts) : sel);
    }
  }

  /** The ⓘ beside a bot seat: a provenance popover — what this opponent is
   * and how (and for how long) it was trained. */
  private seatInfoButton(game: GameInfo, seatValue: string): HTMLElement | null {
    const text = botInfo(game.id, seatValue);
    if (!text) return null;
    const wrap = document.createElement("span");
    wrap.className = "seat-info";
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "seat-info-btn";
    btn.title = "About this opponent";
    btn.setAttribute("aria-label", "About this opponent");
    btn.textContent = "i";
    const pop = document.createElement("div");
    pop.className = "bot-info-pop";
    pop.hidden = true;
    pop.innerHTML = `<strong>${esc(botLabel(seatValue))}</strong><span>${esc(text)}</span>`;
    const close = (e: MouseEvent) => {
      if (!wrap.contains(e.target as Node)) {
        pop.hidden = true;
        document.removeEventListener("pointerdown", close);
      }
    };
    btn.onclick = () => {
      pop.hidden = !pop.hidden;
      if (!pop.hidden) {
        // Fixed positioning, clamped to the viewport: immune to any
        // overflow-clipping ancestor and to seats near a window edge.
        const b = btn.getBoundingClientRect();
        const width = Math.min(320, window.innerWidth - 24);
        pop.style.width = `${width}px`;
        pop.style.left = `${Math.max(12, Math.min(b.left, window.innerWidth - width - 12))}px`;
        pop.style.top = `${b.bottom + 8}px`;
        document.addEventListener("pointerdown", close);
        const below = window.innerHeight - b.bottom - 20;
        pop.style.maxHeight = `${Math.max(120, below)}px`;
      }
    };
    wrap.append(btn, pop);
    return wrap;
  }

  /** A `[sel, level]` pair wrapped so `replaceChildren` takes one node. */
  private fragment(nodes: Node[]): DocumentFragment {
    const f = document.createDocumentFragment();
    for (const n of nodes) f.append(n);
    return f;
  }

  /** Difficulty is a property of the opponent, independent per seat: when seat
   * `i` holds a bot with a difficulty ladder, render a level `<select>` beside
   * that seat's picker showing that seat's OWN level (from its `bots=` spec
   * when heterogeneous, else the shared/default value). Changing it restarts
   * with only that seat's knob changed. Returns null for "You" and ladder-less
   * bots. */
  private seatLevelSelect(
    game: GameInfo,
    opts: Record<string, string>,
    i: number,
    seatValue: string,
  ): HTMLSelectElement | null {
    if (seatValue === "__you__") return null;
    const diff = DIFFICULTY[`${game.id}/${seatValue}`];
    if (!diff) return null;
    const cpu = seatValue === "azero-gpu" && isCpuFallback();
    const levels = cpu ? CPU_LEVELS : diff.levels;
    const cur = this.seatLevel(game, opts, i, seatValue);
    const sel = document.createElement("select");
    sel.className = "seat-level";
    sel.setAttribute("aria-label", "Difficulty");
    for (const [label, value] of levels) {
      const o = document.createElement("option");
      o.value = value;
      o.textContent = label;
      o.selected = value === cur;
      sel.append(o);
    }
    if (!levels.some(([, v]) => v === cur)) {
      const o = document.createElement("option");
      o.value = cur;
      o.textContent = `Custom (${cur})`;
      o.selected = true;
      sel.prepend(o);
    }
    sel.onchange = () => this.applyLevelChange(game, opts, i, sel.value);
    return sel;
  }

  /** Seat `i`'s current difficulty value: its own spec knob in a `bots=` match,
   * else the shared `opts[diff.key]`, else the CPU/medium default. */
  private seatLevel(
    game: GameInfo,
    opts: Record<string, string>,
    i: number,
    seatValue: string,
  ): string {
    const diff = DIFFICULTY[`${game.id}/${seatValue}`];
    if (!diff) return "";
    if (opts.bots) {
      const spec = splitSpecs(opts.bots)[i];
      if (spec) {
        const value = parseBotSpec(spec).opts[diff.key];
        if (value !== undefined) return value;
      }
    }
    const cpu = seatValue === "azero-gpu" && isCpuFallback();
    return opts[diff.key] ?? (cpu ? String(TRIVIAL_SIMS) : mediumLevel(game.id, seatValue));
  }

  /** Change ONLY seat `i`'s difficulty and restart. Composes a per-seat
   * `bots=` spec from every seat's current (bot, difficulty), overriding the
   * edited seat's knob — so two bots hold genuinely independent levels. */
  private applyLevelChange(
    game: GameInfo,
    opts: Record<string, string>,
    i: number,
    value: string,
  ): void {
    const states = this.seatStates(game, opts);
    const human = states.indexOf("__you__");
    const botFiller = states.find((s) => s !== "__you__") ?? rosterBots(game)[0]?.value ?? "";
    const specs = states.map((seat, j) => {
      const bot = seat === "__you__" ? botFiller : seat;
      const level = j === i ? value : this.seatLevel(game, opts, j, bot);
      const botOpts = this.seatBotOptions(game, opts, j, bot);
      const diff = DIFFICULTY[`${game.id}/${bot}`];
      if (diff && level) botOpts[diff.key] = level;
      return formatBotSpec(bot, botOpts);
    });
    const carry = this.gameLevelCarry(game, opts);
    const config = { ...carry, bots: specs.join(",") };
    if (human >= 0)
      void this.startMatch(game, "play", { ...config, seat: String(human) });
    else void this.startMatch(game, "watch", config);
  }

  private seatName(game: GameInfo, i: number): string {
    return game.solo ? "Player" : seatLabelFor(game.id, i);
  }

  /** Who fills each seat right now, as roster values (`__you__` or a bot). */
  private seatStates(game: GameInfo, opts: Record<string, string>): string[] {
    const n = seatCount(game, opts);
    if (game.solo) return [opts.bot ?? "__you__"];
    let states: string[];
    if (opts.bots) {
      const specs = splitSpecs(opts.bots);
      states = Array.from(
        { length: n },
        (_, i) => (specs[i] ? parseBotSpec(specs[i]).bot : currentBotValue(game, opts)),
      );
    } else {
      const cur = currentBotValue(game, opts);
      states = Array.from({ length: n }, () => cur);
    }
    const human = opts.seat === "watch" ? -1 : Number(opts.seat ?? "0");
    if (human >= 0 && human < n) states[human] = "__you__";
    return states;
  }

  /** Explicit bot-scoped options currently belonging to one seat. In simple
   * `bot=` form they live at the top level; in `bots=` form they live only in
   * that seat's spec. Options from a different bot are never inherited. */
  private seatBotOptions(
    game: GameInfo,
    opts: Record<string, string>,
    i: number,
    bot: string,
  ): Record<string, string> {
    if (opts.bots) {
      const text = splitSpecs(opts.bots)[i];
      if (text) {
        const spec = parseBotSpec(text);
        if (spec.bot === bot) return { ...spec.opts };
      }
      return {};
    }
    const out: Record<string, string> = {};
    for (const opt of game.optsSchema) {
      if (opt.bots.includes(bot) && opts[opt.key] !== undefined)
        out[opt.key] = opts[opt.key];
    }
    return out;
  }

  /** Resolve the actual page-side driver configuration for every external
   * seat. Browser-tuned defaults win over registry defaults; explicit per-seat
   * values win over both. Missing required values are then rejected by the
   * driver instead of silently becoming a private constant. */
  private clientBotConfigs(
    game: GameInfo,
    opts: Record<string, string>,
  ): ClientBotConfig[] {
    const states = this.seatStates(game, opts);
    const configs: ClientBotConfig[] = [];
    for (const [seat, bot] of states.entries()) {
      if (bot === "__you__" || !clientBotFor(game.id, bot)) continue;
      const explicit = this.seatBotOptions(game, opts, seat, bot);
      const allowed = new Set(
        game.optsSchema
          .filter((opt) => !opt.nativeOnly && opt.bots.includes(bot))
          .map((opt) => opt.key),
      );
      const unknown = Object.keys(explicit).filter((key) => !allowed.has(key));
      if (unknown.length)
        throw new Error(
          `unused option(s) for client bot '${bot}' at seat ${seat}: ${unknown.join(", ")}`,
        );
      const resolved: Record<string, string> = { ...opts, ...explicit, bot };
      delete resolved.bots;
      delete resolved.seat;
      for (const opt of game.optsSchema) {
        if (opt.nativeOnly || !opt.bots.includes(bot) || resolved[opt.key] !== undefined)
          continue;
        const value = DEFAULT_OPTS[game.id]?.[opt.key] ?? optionDefault(opt);
        if (value !== undefined) resolved[opt.key] = value;
      }
      configs.push({ seat, bot, opts: resolved });
    }
    return configs;
  }

  /** Game-level options only (board size, player count, …) — never the seat,
   * the bot, or a bot's difficulty knob. Carried across a heterogeneous
   * restart, where per-bot knobs live in the specs instead. */
  private gameLevelCarry(
    game: GameInfo,
    opts: Record<string, string>,
  ): Record<string, string> {
    const out: Record<string, string> = {};
    for (const o of game.optsSchema) {
      if (
        o.bots.length === 0 &&
        o.key !== "seat" &&
        o.key !== "seed" &&
        !o.nativeOnly &&
        opts[o.key] !== undefined
      )
        out[o.key] = opts[o.key];
    }
    return out;
  }

  /** Apply one seat's new value and restart with the derived configuration. */
  private applySeatChange(
    game: GameInfo,
    opts: Record<string, string>,
    i: number,
    value: string,
  ): void {
    if (game.solo) {
      if (value === "__you__") void this.startMatch(game, "play", {});
      else void this.startMatch(game, "watch", { bot: value });
      return;
    }
    const n = seatCount(game, opts);
    const bots = rosterBots(game);
    const sendsBot = (name: string) =>
      bots.find((b) => b.value === name)?.sendsBot ?? false;
    const previous = this.seatStates(game, opts);
    const next = [...previous];
    next[i] = value;
    if (value === "__you__") {
      // At most one human; bump any other "You" back to a bot.
      const fallback = bots[0]?.value ?? "__solver__";
      for (let j = 0; j < n; j++)
        if (j !== i && next[j] === "__you__") next[j] = fallback;
    }
    const human = next.indexOf("__you__");
    const vals = next.filter((x) => x !== "__you__");
    const botFiller = vals[0] ?? bots[0]?.value ?? "";
    // Preserve every explicit option on unchanged seats. A newly selected bot
    // starts from its medium difficulty and its own declared defaults, never
    // stale knobs belonging to the bot it replaced.
    const seatConfigs = next.map((valueAtSeat, j) => {
      const bot = valueAtSeat === "__you__" ? botFiller : valueAtSeat;
      const changedBot = valueAtSeat !== "__you__" && bot !== previous[j];
      const botOpts = changedBot ? {} : this.seatBotOptions(game, opts, j, bot);
      const diff = DIFFICULTY[`${game.id}/${bot}`];
      if (diff) {
        const level = changedBot
          ? mediumLevel(game.id, bot)
          : this.seatLevel(game, opts, j, bot);
        if (level) botOpts[diff.key] = level;
      }
      return { bot, botOpts, human: valueAtSeat === "__you__" };
    });
    // The simple `bot=` form is valid only when every actual bot seat has the
    // same bot and the same complete option map.
    const botSeats = seatConfigs.filter((config) => !config.human);
    const uniform =
      botSeats.length > 0 &&
      botSeats.every(
        (config) =>
          config.bot === botSeats[0].bot && sameOptions(config.botOpts, botSeats[0].botOpts),
      );
    if (uniform) {
      const carry = this.gameLevelCarry(game, opts);
      const changes: Record<string, string> = sendsBot(botSeats[0].bot)
        ? { bot: botSeats[0].bot, ...botSeats[0].botOpts }
        : {};
      if (human >= 0)
        void this.startMatch(game, "play", {
          ...carry,
          ...changes,
          seat: String(human),
        });
      else void this.startMatch(game, "watch", { ...carry, ...changes });
    } else {
      const specs = seatConfigs.map((config) => formatBotSpec(config.bot, config.botOpts));
      const carry = this.gameLevelCarry(game, opts);
      if (human >= 0)
        void this.startMatch(game, "play", {
          ...carry,
          bots: specs.join(","),
          seat: String(human),
        });
      else
        void this.startMatch(game, "watch", { ...carry, bots: specs.join(",") });
    }
  }

  private wireDrawer(game: GameInfo, opts: Record<string, string>): void {
    const drawer = this.root.querySelector<HTMLElement>(".drawer")!;
    const fieldsEl = drawer.querySelector<HTMLElement>(".drawer-fields")!;
    const note = (text: string) =>
      text ? `<small class="opt-note">${esc(text)}</small>` : "";
    const row = (label: string, control: string, hint = "") =>
      `<label class="opt-row"><span>${esc(label)}</span>${control}${note(hint)}</label>`;
    const selectRow = (key: string, label: string, pairs: [string, string][], cur: string) =>
      row(label, `<select name="d-${esc(key)}">${optionList(pairs, cur)}</select>`);

    // Who plays each seat lives in the roster; the drawer holds game settings
    // and difficulty. Knobs become levels/dropdowns — no raw search depths,
    // and no seed (matches are always randomly seeded).
    const open = () => {
      const curBot = effectiveBot(game, opts);
      // Heterogeneous matches carry per-bot knobs in their specs, so the drawer
      // shows only game-level settings — no single difficulty applies.
      const diff = opts.bots ? undefined : DIFFICULTY[`${game.id}/${curBot}`];
      const fields = optFields(game.optsSchema, opts).filter(
        (f) =>
          (f.bots.length === 0 || (!opts.bots && f.bots.includes(curBot))) &&
          !(diff && f.key === diff.key),
      );
      // Without a GPU the difficulty offers only the responsive CPU levels,
      // matching the always-visible level control.
      const cpu = curBot === "azero-gpu" && isCpuFallback();
      const diffRow = !diff
        ? ""
        : selectRow(
            "difficulty-target",
            "difficulty",
            cpu ? CPU_LEVELS : diff.levels,
            opts[diff.key] ?? (cpu ? String(TRIVIAL_SIMS) : diff.levels[1][1]),
          );
      const fieldRows = fields.map((f) => {
        const choices = optChoicesFor(game.id, f.key);
        return choices
          ? selectRow(f.key, f.key, choices.map((c) => [c, c]), f.value)
          : row(f.key, `<input name="d-${esc(f.key)}" value="${esc(f.value)}" autocomplete="off" />`, f.note);
      });
      const body = diffRow + fieldRows.join("");
      fieldsEl.innerHTML = body || `<p class="muted">No settings for this game.</p>`;
      // The difficulty control writes the bot's actual knob on apply.
      fieldsEl.dataset.diffKey = diff ? diff.key : "";
      drawer.hidden = false;
    };
    this.root.querySelector<HTMLButtonElement>(".gear")!.onclick = open;
    drawer.querySelector<HTMLButtonElement>(".drawer-close")!.onclick = () => {
      drawer.hidden = true;
    };
    drawer.onclick = (e) => {
      if (e.target === drawer) drawer.hidden = true;
    };
    drawer.querySelector<HTMLButtonElement>(".drawer-apply")!.onclick = () => {
      // Keep the roster's seat/bot; the drawer only edits settings and knobs.
      const overrides: Record<string, string> = {};
      if (opts.seat !== undefined) overrides.seat = opts.seat;
      if (opts.bot !== undefined) overrides.bot = opts.bot;
      if (opts.bots !== undefined) overrides.bots = opts.bots;
      const diffKey = fieldsEl.dataset.diffKey ?? "";
      const controls = fieldsEl.querySelectorAll<
        HTMLInputElement | HTMLSelectElement
      >("input, select");
      for (const el of controls) {
        let key = el.name.replace(/^d-/, "");
        if (key === "difficulty-target") {
          if (!diffKey) continue;
          key = diffKey;
        }
        if (el.value.trim() !== "") overrides[key] = el.value.trim();
      }
      const mode: Mode = game.solo
        ? opts.bot
          ? "watch"
          : "play"
        : opts.seat === "watch"
          ? "watch"
          : "play";
      void this.startMatch(game, mode, overrides);
    };
  }

  private async runLoop(gen: number): Promise<void> {
    const fail = (e: unknown) => {
      if (gen === this.gen) this.setStatus(message(e), "error");
    };
    while (gen === this.gen) {
      let ev: MatchEventData | null;
      try {
        ev = await this.host.step();
      } catch (e) {
        fail(e);
        return;
      }
      if (gen !== this.gen) return;
      if (ev) {
        try {
          this.log(ev);
          await this.clientBot?.onMove(ev);
          const st = await this.host.state();
          if (gen !== this.gen) return;
          const nextPreparation = st.isOver
            ? Promise.resolve()
            : this.host.prepare();
          await this.frontend!.animate(ev, st);
          await nextPreparation;
        } catch (e) {
          fail(e);
          return;
        }
        continue;
      }
      const st = await this.host.state();
      if (gen !== this.gen) return;
      this.frontend!.render(st);
      if (st.isOver) {
        const adjudicated = (await this.clientBot?.finalResult?.()) || "";
        if (gen !== this.gen) return;
        const result = adjudicated || st.result || "Game over";
        this.setStatus(result, "result");
        this.logText(`— ${result}`);
        return;
      }
      if (this.clientBot && st.toAct >= 0 && st.toAct !== st.humanSeat) {
        this.setStatus("Thinking…");
        try {
          const t0 = performance.now();
          const input = await this.clientBot.chooseMove(st);
          const thinkMs = performance.now() - t0;
          if (gen !== this.gen) return;
          const mev = await this.host.apply(input);
          if (gen !== this.gen) return;
          this.log(mev, thinkMs);
          await this.clientBot.onMove(mev);
          const after = await this.host.state();
          if (gen !== this.gen) return;
          await this.frontend!.animate(mev, after);
        } catch (e) {
          fail(e);
          return;
        }
        continue;
      }
      this.setStatus("Your turn");
      // In simultaneous games the opponents choose from this same immutable
      // state, so their search can run now instead of starting after input.
      const preparation = this.host.prepare();
      this.frontend!.promptAction(st.labels);
      const input = await new Promise<string>(
        (res) => (this.submitResolve = res),
      );
      if (gen !== this.gen) return;
      if (st.numSeats > 1) this.setStatus("Thinking…");
      try {
        await preparation;
        const mev = await this.host.apply(input);
        if (gen !== this.gen) return;
        this.log(mev);
        await this.clientBot?.onMove(mev);
        const after = await this.host.state();
        if (gen !== this.gen) return;
        // Begin the next simultaneous search while the current real move is
        // gliding. By the next cell boundary it is normally already cached.
        const nextPreparation = after.isOver
          ? Promise.resolve()
          : this.host.prepare();
        await this.frontend!.animate(mev, after);
        await nextPreparation;
      } catch (e) {
        fail(e);
      }
    }
  }

  private async loadArtifacts(
    game: GameInfo,
    opts: Record<string, string>,
  ): Promise<void> {
    for (const id of artifactsFor(game.id, opts)) {
      const url = assetUrl(ARTIFACTS[id]);
      const resp = await fetch(url);
      if (!resp.ok)
        throw new Error(`artifact ${url} missing (HTTP ${resp.status})`);
      await this.host.artifact(id, await resp.arrayBuffer());
    }
  }

  private submit(input: string): void {
    const resolve = this.submitResolve;
    if (!resolve) return;
    this.submitResolve = null;
    resolve(input);
  }

  private animationScale(): number {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return 0;
    return this.speedScale;
  }

  /** One move's line in the log: `ev.text` always, with `ev.detail` and the
   * shell-added meta (seat, raw label, optional think time) folded in as
   * debug-only lines that CSS reveals when the panel is in debug mode. */
  private log(ev: MatchEventData, thinkMs?: number): void {
    this.logText(ev.text);
    if (ev.detail) this.logText(ev.detail, "detail");
    const meta = [`seat ${ev.seat}`, `label ${ev.label}`];
    if (thinkMs !== undefined) meta.push(`think ${Math.round(thinkMs)}ms`);
    this.logText(meta.join(" · "), "meta");
  }

  /** A debug-only line in the log; shown only when the panel is in debug mode. */
  private debugLog(text: string): void {
    this.logText(text, "meta");
  }

  private logText(text: string, kind?: "detail" | "meta"): void {
    if (!this.logEl) return;
    const line = document.createElement("div");
    line.className = kind
      ? `log-line log-${kind} log-debug`
      : "log-line";
    line.textContent = text;
    this.logEl.append(line);
    this.logEl.scrollTop = this.logEl.scrollHeight;
  }

  /** Read the debug flag, flip the side panel's class so debug-only lines and
   * the readout reveal/hide, persist, and fan out to subscribers. */
  private setDebug(on: boolean): void {
    if (on === this.debugOn) return;
    this.debugOn = on;
    localStorage.setItem("arcadeDebug", on ? "1" : "0");
    this.sideEl?.classList.toggle("debug-on", on);
    for (const cb of this.debugSubs) cb(on);
  }

  private onDebugChange(cb: (on: boolean) => void): () => void {
    this.debugSubs.add(cb);
    cb(this.debugOn);
    return () => this.debugSubs.delete(cb);
  }

  /** Replace the side panel's persistent debug readout. No-op without a side
   * panel (solo/snake) so frontends can call it unconditionally. */
  private setDebugReadout(lines: string[]): void {
    if (!this.readoutEl) return;
    this.readoutEl.replaceChildren(
      ...lines.map((text) => {
        const row = document.createElement("div");
        row.className = "readout-row";
        row.textContent = text;
        return row;
      }),
    );
  }

  private setStatus(
    text: string,
    kind: "info" | "error" | "result" = "info",
  ): void {
    if (!this.statusEl) return;
    this.statusEl.textContent = text;
    this.statusEl.className = `status status-${kind}`;
  }

  /** Stops the live match's machinery: the client bot's in-flight search,
   * the frontend's listeners/timers, and any pending human prompt. Runs at
   * the top of every startMatch — a rematch must never leave the previous
   * match's chooseMove loop driving the worker's search. */
  private teardownMatch(): void {
    this.clientBot?.cancel();
    this.clientBot = null;
    this.frontend?.unmount();
    this.frontend = null;
    this.submitResolve = null;
    // The next match registers fresh subscribers after mounting; drop the old
    // ones so a toggle never drives a torn-down panel's readout.
    this.debugSubs.clear();
  }

  private teardown(): void {
    this.gen++;
    this.tourney?.destroy();
    this.tourney = null;
    this.slither?.destroy();
    this.slither = null;
    this.teardownMatch();
    this.logEl = null;
    this.statusEl = null;
    this.sideEl = null;
    this.readoutEl = null;
  }
}

function message(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** Cheap net-version signature: byte length plus a few sampled bytes, so the
 * cached verdict is invalidated whenever the shipped weights change. */
function goSelfcheckKey(weights: ArrayBuffer): string {
  const b = new Uint8Array(weights);
  const n = b.length;
  const sample = n
    ? [b[0], b[(n >> 2) | 0], b[(n >> 1) | 0], b[((3 * n) >> 2) | 0], b[n - 1]].join(".")
    : "0";
  return `azeroGoSelfcheck:${n}:${sample}`;
}

/** A passing cached verdict skips the re-run; a cached FAIL is overwritten so a
 * driver update that fixes the divergence can clear the warning. */
function readSelfcheck(key: string): ConformanceResult | null {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as ConformanceResult) : null;
  } catch {
    return null;
  }
}

function writeSelfcheck(key: string, result: ConformanceResult): void {
  try {
    localStorage.setItem(key, JSON.stringify(result));
  } catch {
    // localStorage may be unavailable/full; the check still warns this session.
  }
}

export type { Manifest, ViewState };
