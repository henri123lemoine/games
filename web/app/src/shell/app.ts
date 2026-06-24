// The arcade shell: pick a game and you are immediately playing it against
// the lab's bot — configuration lives in a quiet settings drawer, not between
// you and the board. One engine worker drives play; the shell owns the loop
// and narration, frontends own the board.

import { type ClientBot, clientBotFor } from "../bots";
import { createSnakeBot, type SnakeBot } from "../bots/snake-search";
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
import type { RealtimeBoard } from "./snake-realtime";
import { SnakeRealtime } from "./snake-realtime";
import type { FrontendCtx, GameFrontend } from "../frontends/types";
import { CPU_LEVELS, isCpuFallback, TRIVIAL_SIMS } from "./azero";
import {
  DIFFICULTY,
  OPT_CHOICES,
  botLabel,
  botSpec,
  mediumLevel,
  splitSpecs,
} from "./config";
import { TournamentScreen } from "./tournament";

/** What clicking a card starts: browser-tuned, no questions asked. Chess and
 * Go open against AlphaZero (Medium); with no WebGPU the driver runs the same
 * net on the CPU at the trivial budget. `sims` here is the AlphaZero budget. */
const DEFAULT_OPTS: Record<string, Record<string, string>> = {
  chess: { bot: "azero-gpu", sims: "256" },
  "liars-dice": { players: "5", dice: "5", rollouts: "400" },
  twentyone: { hearts: "3" },
  othello: { depth: "5" },
  connect4: { depth: "7" },
  pente: { size: "13", depth: "4" },
  go: { size: "9", bot: "azero-gpu", sims: "1500" },
  snake: { bot: "azero-gpu", sims: "128" },
  "2048": {},
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
};

/** The shipped artifacts a match with these opts will ask the registry for.
 * A config whose artifact is not shipped fails loudly at create — the
 * browser never trains. */
function artifactsFor(gameId: string, opts: Record<string, string>): string[] {
  const wanted: string[] = [];
  if (gameId === "chess") {
    const usesAzero =
      opts.bot === "azero" ||
      (opts.bots ?? "").split(",").some((s) => s.split(":")[0] === "azero");
    const net = opts.net ?? (usesAzero ? "data/azero/chess.bin" : null);
    if (net) wanted.push(net);
  }
  if (gameId === "twentyone")
    wanted.push(`data/twentyone/solver-h${opts.hearts ?? "6"}.bin`);
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
  "liars-dice": ["rollout"],
  poker: ["equity"],
  othello: ["alphabeta"],
  connect4: ["alphabeta"],
  go: ["azero-gpu"],
  pente: ["alphabeta"],
  "2048": ["mcts"],
  snake: ["azero-gpu"],
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

/** Static mini-board previews on the home cards — each game introduces
 * itself with its own board, not an icon. */
function miniFor(id: string): string {
  switch (id) {
    case "chess":
      return `<div class="mini mini-chess"><span class="mini-pc" style="left:12%;top:8%">♞</span><span class="mini-pc mini-pc-w" style="left:58%;top:52%">♙</span></div>`;
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
      return `<div class="mini mini-othello"><span class="mini-disc mini-disc-b" style="left:28%;top:28%"></span><span class="mini-disc mini-disc-w" style="left:52%;top:28%"></span><span class="mini-disc mini-disc-w" style="left:28%;top:52%"></span><span class="mini-disc mini-disc-b" style="left:52%;top:52%"></span></div>`;
    case "connect4":
      return `<div class="mini mini-c4"></div>`;
    case "go":
      return `<div class="mini mini-go"><span class="mini-stone mini-stone-b" style="left:30%;top:30%"></span><span class="mini-stone mini-stone-w" style="left:55%;top:47%"></span><span class="mini-stone mini-stone-b" style="left:38%;top:63%"></span></div>`;
    case "pente":
      return `<div class="mini mini-pente"><span class="mini-pstone mini-pstone-b" style="left:18%;top:50%"></span><span class="mini-pstone mini-pstone-w" style="left:40%;top:50%"></span><span class="mini-pstone mini-pstone-w" style="left:60%;top:50%"></span><span class="mini-pstone mini-pstone-b" style="left:82%;top:50%"></span></div>`;
    case "2048":
      return `<div class="mini mini-2048"><span>2</span><span class="v4">4</span><span class="v8">8</span><span class="v16">16</span></div>`;
    case "snake":
      return `<div class="mini mini-snake"><span class="mini-seg mini-seg-a" style="left:18%;top:50%"></span><span class="mini-seg mini-seg-a" style="left:34%;top:50%"></span><span class="mini-seg mini-seg-a mini-head-a" style="left:50%;top:50%"></span><span class="mini-seg mini-seg-b mini-head-b" style="left:74%;top:28%"></span><span class="mini-seg mini-seg-b" style="left:74%;top:44%"></span><span class="mini-food" style="left:62%;top:70%"></span></div>`;
    case "slither":
      return `<div class="mini mini-slither"><span class="mini-worm mini-worm-b" style="left:20%;top:34%"></span><span class="mini-worm mini-worm-b" style="left:34%;top:30%"></span><span class="mini-worm mini-worm-b mini-worm-head-b" style="left:48%;top:30%"></span><span class="mini-worm mini-worm-a" style="left:58%;top:66%"></span><span class="mini-worm mini-worm-a" style="left:70%;top:60%"></span><span class="mini-worm mini-worm-a mini-worm-head-a" style="left:80%;top:50%"></span><span class="mini-pellet" style="left:40%;top:62%"></span><span class="mini-pellet" style="left:66%;top:32%"></span></div>`;
    case "doom":
      return `<div class="mini mini-doom"><span class="mini-doom-word">DOOM</span></div>`;
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
  /** Snake play runs on a dedicated real-time driver (fixed clock, bot's policy
   * floor + search off the critical path) instead of the serial match loop. */
  private snakeRealtime: SnakeRealtime | null = null;
  private snakeBot: SnakeBot | null = null;

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
    if (segs[0] === "doom" || segs[0] === "doom-ai") {
      this.renderDoom();
      return;
    }
    if (segs[0] === "slither") {
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

  // ---------- home ----------

  private renderHome(): void {
    this.teardown();
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
    // DOOM and Slither are not engine games (real-time, not Game-trait matches);
    // their cards open standalone screens instead of starting a match.
    const slitherCard = `
        <div class="card card-slither" data-special="slither" role="button" tabindex="0">
          ${miniFor("slither")}
          <div class="card-text">
            <span class="card-name">Slither</span>
          </div>
        </div>`;
    const doomCard = `
        <div class="card card-doom" data-special="doom" role="button" tabindex="0">
          ${miniFor("doom")}
          <div class="card-text">
            <span class="card-name">DOOM</span>
          </div>
        </div>`;
    this.root.innerHTML = `
      <div class="home">
        <header class="home-head">
          <h1>Games Room</h1>
        </header>
        <div class="card-grid">${cards}${slitherCard}${doomCard}</div>
        <footer class="home-footer">
          <nav>
            <a href="https://github.com/henri123lemoine/games">GitHub</a>
            <a href="https://henrilemoine.com/">henrilemoine.com</a>
            <button type="button" class="link tourney-link">tournament lab</button>
          </nav>
          <span class="muted">Runs entirely in your browser.</span>
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
    const doomEl = this.root.querySelector<HTMLElement>('.card[data-special="doom"]');
    if (doomEl) {
      const openDoom = () => this.navTo("/doom");
      doomEl.onclick = openDoom;
      doomEl.onkeydown = (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          openDoom();
        }
      };
    }
    const slitherEl = this.root.querySelector<HTMLElement>(
      '.card[data-special="slither"]',
    );
    if (slitherEl) {
      const openSlither = () => this.navTo("/slither");
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

  /** DOOM: the doomgeneric RL-deathmatch build. The human plays one seat; the
   * other is driven by a PPO net trained by self-play, run tch-free in the
   * browser from the same LOS-gated state it trained on. Standalone page in an
   * iframe. */
  private renderDoom(): void {
    this.teardown();
    const src = `${import.meta.env.BASE_URL}doom-ai/index.html`;
    this.root.innerHTML = `
      <div class="match doom-screen">
        <header class="match-bar">
          <button type="button" class="link back">&larr; games</button>
          <span class="match-title">DOOM</span>
          <span class="spacer"></span>
          <span class="muted doom-note">self-play bot · click the frame, then fight</span>
        </header>
        <div class="doom-frame-wrap">
          <iframe class="doom-frame" src="${src}" title="DOOM"
            allow="autoplay; fullscreen"></iframe>
        </div>
      </div>`;
    this.root.querySelector<HTMLButtonElement>(".back")!.onclick = () =>
      this.navTo("/");
  }

  /** Slither: the lab's own real-time game, played against the PPO-trained
   * encircle bot. Not a Game-trait match, so it runs standalone on its own
   * screen (its wasm steps the world and drives the bots each frame). */
  private async renderSlither(): Promise<void> {
    this.teardown();
    this.root.innerHTML = `
      <div class="match slither-screen">
        <header class="match-bar">
          <button type="button" class="link back">&larr; games</button>
          <span class="match-title">Slither</span>
          <span class="spacer"></span>
          <span class="muted">vs. the trained encircle bot · runs in your browser</span>
        </header>
        <div class="slither-mount"></div>
      </div>`;
    this.root.querySelector<HTMLButtonElement>(".back")!.onclick = () =>
      this.navTo("/");
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
    const opts = this.buildOpts(game, mode, overrides);
    this.syncMatchUrl(game, mode);
    this.renderMatchSkeleton(game, mode, opts);
    // An AlphaZero seat (single bot, or one seat of a heterogeneous board) is
    // driven page-side: WebGPU when present, otherwise the in-wasm CPU forward.
    // Snake's `azero` is CPU-only (`azero-gpu` is the GPU-capable id).
    const isAzeroSpec = (b: string) => b === "azero-gpu" || b === "azero";
    const azeroSeat =
      isAzeroSpec(opts.bot ?? "") ||
      splitSpecs(opts.bots ?? "").some((s) => isAzeroSpec(s.split(":")[0]));
    const usesAzeroGpu =
      opts.bot === "azero-gpu" ||
      splitSpecs(opts.bots ?? "").some((s) => s.split(":")[0] === "azero-gpu");
    if (azeroSeat && isCpuFallback()) this.showCpuNote();
    // Snake PLAY (human seated, AlphaZero opponent) runs real-time: a dedicated
    // fixed-clock driver with the bot's search off the critical path, so the
    // player's snake is never gated by the bot's compute. Watch mode and the
    // CPU/keyboard fallback keep the generic serial loop.
    const snakePlay =
      game.id === "snake" && usesAzeroGpu && opts.seat !== "watch";
    try {
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
      };
      this.frontend.mount(boardEl, ctx);
      this.frontend.render(st);
      this.fillSeatSlots(game, opts);

      if (snakePlay) {
        await this.startSnakeRealtime(gen, opts, st);
        return;
      }

      const makeBot = clientBotFor(game.id, usesAzeroGpu ? "azero-gpu" : opts.bot);
      this.clientBot = makeBot ? await makeBot(this.host, opts) : null;
      if (gen !== this.gen) return;
      this.setStatus(st.humanSeat < 0 ? "Bots playing…" : "Thinking…");
      void this.runLoop(gen);
    } catch (e) {
      if (gen === this.gen)
        this.setStatus(`Could not start: ${message(e)}`, "error");
    }
  }

  /** Wire snake's real-time driver: the bot (CPU policy floor + background
   * search, each on its own worker) plus the fixed-clock driver that reads the
   * player's input every tick without ever awaiting the heavy search. */
  private async startSnakeRealtime(
    gen: number,
    opts: Record<string, string>,
    initial: ViewState,
  ): Promise<void> {
    const bot = await createSnakeBot(opts);
    if (gen !== this.gen) {
      bot.stop();
      return;
    }
    this.snakeBot = bot;
    const board = this.frontend as unknown as RealtimeBoard;
    this.snakeRealtime = new SnakeRealtime(
      this.host,
      bot,
      board,
      () => gen === this.gen,
      () => {
        /* the frontend draws its own game-over overlay */
      },
    );
    this.snakeRealtime.start(initial);
  }

  private renderMatchSkeleton(
    game: GameInfo,
    mode: Mode,
    opts: Record<string, string>,
  ): void {
    // Pacing only matters while spectating; a human-vs-bot game has nothing to
    // pace. Reset to normal each match and show the control only when watching.
    this.speedScale = 1;
    const speedControl =
      mode === "watch"
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
          <button type="button" class="link back">&larr; games</button>
          <span class="match-title">${esc(game.name || game.id)}</span>
          <span class="spacer"></span>
          ${speedControl}
          <button type="button" class="link again">rematch</button>
          <button type="button" class="link gear" title="Match settings">⚙ settings</button>
        </header>
        ${this.quickControlsHtml(game, opts)}
        <div class="cpu-note" hidden></div>
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
    this.root.querySelector<HTMLButtonElement>(".back")!.onclick = () =>
      this.navTo("/");
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
   * most likely to change (board size, player count) and, for a single-bot
   * match, a difficulty selector — so the common settings are one click away
   * on the board, not hidden in the drawer. Empty for solo / heterogeneous
   * matches (their settings stay in the drawer). */
  private quickControlsHtml(
    game: GameInfo,
    opts: Record<string, string>,
  ): string {
    if (game.solo || opts.bots) return "";
    const bot = effectiveBot(game, opts);
    const cells: string[] = [];
    const cell = (key: string, label: string, pairs: [string, string][], cur: string, locked: boolean) =>
      `<label class="qc"><span class="qc-name">${esc(label)}</span><select class="qc-select" data-key="${esc(key)}"${locked ? " disabled" : ""}>${optionList(pairs, cur)}</select></label>`;
    for (const o of game.optsSchema) {
      if (o.bots.length || o.key === "seat" || o.key === "seed" || o.nativeOnly)
        continue;
      const choices = OPT_CHOICES[o.key];
      if (!choices) continue;
      const cur = opts[o.key] ?? o.value.split("|")[0];
      cells.push(cell(o.key, o.key, choices.map((c) => [c, c]), cur, false));
    }
    const diff = DIFFICULTY[`${game.id}/${bot}`];
    if (diff) {
      // Without a GPU only the two responsive CPU levels are offered; otherwise
      // the full ladder.
      const cpu = bot === "azero-gpu" && isCpuFallback();
      const levels = cpu ? CPU_LEVELS : diff.levels;
      const cur = opts[diff.key] ?? (cpu ? String(TRIVIAL_SIMS) : mediumLevel(game.id, bot));
      cells.push(cell(diff.key, "level", levels, cur, false));
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

  /** Surfaces, in-match, that AlphaZero is on the CPU forward (no WebGPU) — the
   * honest "it'll be slower, fewer levels" note. */
  private showCpuNote(): void {
    const note = this.root.querySelector<HTMLElement>(".cpu-note");
    if (!note) return;
    note.textContent =
      "No GPU detected — AlphaZero is running on the CPU, which is much slower, so only the Trivial and Light levels are offered. Open it in a WebGPU browser (recent Chrome/Edge) for the full difficulty ladder.";
    note.hidden = false;
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
    return `<aside class="side">
        <div class="status">Starting…</div>
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
      slot.replaceChildren(sel);
    }
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
        (_, i) => specs[i]?.split(":")[0] ?? currentBotValue(game, opts),
      );
    } else {
      const cur = currentBotValue(game, opts);
      states = Array.from({ length: n }, () => cur);
    }
    const human = opts.seat === "watch" ? -1 : Number(opts.seat ?? "0");
    if (human >= 0 && human < n) states[human] = "__you__";
    return states;
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
    const next = this.seatStates(game, opts);
    next[i] = value;
    if (value === "__you__") {
      // At most one human; bump any other "You" back to a bot.
      const fallback = bots[0]?.value ?? "__solver__";
      for (let j = 0; j < n; j++)
        if (j !== i && next[j] === "__you__") next[j] = fallback;
    }
    const human = next.indexOf("__you__");
    const vals = next.filter((x) => x !== "__you__");
    const uniform = vals.length > 0 && vals.every((x) => x === vals[0]);
    if (uniform) {
      const carry = { ...opts };
      delete carry.seat;
      delete carry.bot;
      delete carry.bots;
      delete carry.seed;
      const changes: Record<string, string> = sendsBot(vals[0])
        ? { bot: vals[0] }
        : {};
      if (human >= 0)
        void this.startMatch(game, "play", {
          ...carry,
          ...changes,
          seat: String(human),
        });
      else void this.startMatch(game, "watch", { ...carry, ...changes });
    } else {
      const specs = next.map((v) => {
        const bot = v === "__you__" ? vals[0] : v;
        return botSpec(game.id, bot, mediumLevel(game.id, bot));
      });
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
        const choices = OPT_CHOICES[f.key];
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
          await this.frontend!.animate(ev, st);
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
          const input = await this.clientBot.chooseMove(st);
          if (gen !== this.gen) return;
          const mev = await this.host.apply(input);
          if (gen !== this.gen) return;
          this.log(mev);
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
      this.frontend!.promptAction(st.labels);
      const input = await new Promise<string>(
        (res) => (this.submitResolve = res),
      );
      if (gen !== this.gen) return;
      if (st.numSeats > 1) this.setStatus("Thinking…");
      try {
        const mev = await this.host.apply(input);
        if (gen !== this.gen) return;
        this.log(mev);
        await this.clientBot?.onMove(mev);
        const after = await this.host.state();
        if (gen !== this.gen) return;
        await this.frontend!.animate(mev, after);
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
      const url = `${import.meta.env.BASE_URL}${ARTIFACTS[id]}`;
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

  private log(ev: MatchEventData): void {
    this.logText(ev.text);
    if (ev.detail) this.logText(ev.detail, true);
  }

  private logText(text: string, detail = false): void {
    if (!this.logEl) return;
    const line = document.createElement("div");
    line.className = detail ? "log-line log-detail" : "log-line";
    line.textContent = text;
    this.logEl.append(line);
    this.logEl.scrollTop = this.logEl.scrollHeight;
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
    this.snakeRealtime?.stop();
    this.snakeRealtime = null;
    this.snakeBot?.stop();
    this.snakeBot = null;
    this.frontend?.unmount();
    this.frontend = null;
    this.submitResolve = null;
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
  }
}

function message(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

export type { Manifest, ViewState };
