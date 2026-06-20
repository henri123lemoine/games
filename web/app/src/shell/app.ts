// The arcade shell: pick a game and you are immediately playing it against
// the lab's bot — configuration lives in a quiet settings drawer, not between
// you and the board. One engine worker drives play; the shell owns the loop
// and narration, frontends own the board.

import { type ClientBot, clientBotFor } from "../bots";
import { EngineHost } from "../engine/host";
import type {
  GameInfo,
  GameOpt,
  Manifest,
  MatchEventData,
  ViewState,
} from "../engine/protocol";
import { frontendFor, hasFrontend } from "../frontends";
import type { FrontendCtx, GameFrontend } from "../frontends/types";
import {
  DIFFICULTY,
  OPT_CHOICES,
  botLabel,
  botSpec,
  mediumLevel,
  splitSpecs,
} from "./config";
import { TournamentScreen } from "./tournament";

/** What clicking a card starts: browser-tuned, no questions asked. */
const DEFAULT_OPTS: Record<string, Record<string, string>> = {
  chess: { depth: "4" },
  "liars-dice": { players: "5", dice: "5", rollouts: "400" },
  twentyone: { hearts: "3" },
  othello: { depth: "5" },
  connect4: { depth: "7" },
  go: { size: "9", sims: "1500" },
  "2048": {},
};

/** Games registered in the lab but not surfaced on the site. Snake is solo
 * and too easy to fit the "play the lab's bots" thesis; it returns once it
 * is a competitive 1v1 game — see games/snake/REDESIGN.md. The Rust crate
 * and CLI keep it. */
const HIDDEN_GAMES = new Set(["snake"]);

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

/** Bots the web hides from selection: the wasm CPU `azero` plays at random
 * strength (the validated inference is the GPU path; only `azero-gpu` is kept),
 * so it is dropped everywhere on the site. */
const HIDDEN_BOTS = new Set(["azero"]);

/** Opponents a seat can be filled with. Reads the game's `bot` schema (real
 * bots), or the synthetic solver for games without one. Hidden bots and, where
 * WebGPU is missing, GPU-only bots drop out so the roster never offers a dead
 * choice. */
function rosterBots(game: GameInfo): RosterBot[] {
  const spec = game.optsSchema.find((o) => o.key === "bot");
  if (!spec) return [SOLVER_OPPONENT];
  const gpu = "gpu" in navigator;
  return spec.value
    .split("|")
    .filter((b) => !HIDDEN_BOTS.has(b) && (gpu || b !== "azero-gpu"))
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
    case "othello":
      return `<div class="mini mini-othello"><span class="mini-disc mini-disc-b" style="left:28%;top:28%"></span><span class="mini-disc mini-disc-w" style="left:52%;top:28%"></span><span class="mini-disc mini-disc-w" style="left:28%;top:52%"></span><span class="mini-disc mini-disc-b" style="left:52%;top:52%"></span></div>`;
    case "connect4":
      return `<div class="mini mini-c4"></div>`;
    case "go":
      return `<div class="mini mini-go"><span class="mini-stone mini-stone-b" style="left:30%;top:30%"></span><span class="mini-stone mini-stone-w" style="left:55%;top:47%"></span><span class="mini-stone mini-stone-b" style="left:38%;top:63%"></span></div>`;
    case "2048":
      return `<div class="mini mini-2048"><span>2</span><span class="v4">4</span><span class="v8">8</span><span class="v16">16</span></div>`;
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
  private gen = 0;
  private speedScale = 1;
  private submitResolve: ((input: string) => void) | null = null;
  private logEl: HTMLElement | null = null;
  private statusEl: HTMLElement | null = null;

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
    if (segs[0] === "doom") {
      this.renderDoom();
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
    // DOOM is not an engine game (a real-time FPS, not a Game-trait match); its
    // card links to the vendored standalone port instead of starting a match.
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
        <div class="card-grid">${cards}${doomCard}</div>
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
    this.root.querySelector<HTMLButtonElement>(".tourney-link")!.onclick = () =>
      this.navTo("/lab");
  }

  /** DOOM: the vendored WebAssembly port runs in its own page (it is not a
   * Game-trait match), mounted in an iframe with a back bar. */
  private renderDoom(): void {
    this.teardown();
    const src = `${import.meta.env.BASE_URL}doom/doom.html`;
    this.root.innerHTML = `
      <div class="match doom-screen">
        <header class="match-bar">
          <button type="button" class="link back">&larr; games</button>
          <span class="match-title">DOOM</span>
          <span class="spacer"></span>
          <span class="muted doom-note">shareware · click the frame, then play</span>
        </header>
        <div class="doom-frame-wrap">
          <iframe class="doom-frame" src="${src}" title="DOOM"
            allow="autoplay; fullscreen"></iframe>
        </div>
      </div>`;
    this.root.querySelector<HTMLButtonElement>(".back")!.onclick = () =>
      this.navTo("/");
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
    // A GPU AlphaZero seat (single bot, or one seat of a heterogeneous board)
    // is driven page-side via WebGPU.
    const usesGpu =
      opts.bot === "azero-gpu" ||
      splitSpecs(opts.bots ?? "").some((s) => s.split(":")[0] === "azero-gpu");
    if (usesGpu && !("gpu" in navigator)) {
      this.setStatus(
        "AlphaZero needs WebGPU, which this browser doesn't have — pick another bot.",
        "error",
      );
      return;
    }
    try {
      await this.loadArtifacts(game, opts);
      const st = await this.host.create(game.id, opts);
      if (gen !== this.gen) return;
      const makeBot = clientBotFor(game.id, usesGpu ? "azero-gpu" : opts.bot);
      this.clientBot = makeBot ? await makeBot(this.host, opts) : null;
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
          <button type="button" class="link gear" title="Match settings">⚙</button>
        </header>
        <div class="match-body${game.solo ? " match-body--solo" : ""}">
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
  }

  /** The match side panel: status + move log for the turn-based versus games,
   * and a type-a-move input only for the generic fallback (custom frontends
   * have board-native input). Solo games render their own score and game-over
   * overlay, so they get no side panel — the board takes the full width. */
  private sideHtml(game: GameInfo): string {
    if (game.solo) return "";
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
    const option = (value: string, label: string, sel: boolean) =>
      `<option value="${esc(value)}"${sel ? " selected" : ""}>${esc(label)}</option>`;
    const selectRow = (key: string, label: string, pairs: [string, string][], cur: string) => {
      const known = pairs.some(([, v]) => v === cur);
      const opts_ = pairs.map(([l, v]) => option(v, l, v === cur));
      if (!known) opts_.unshift(option(cur, `Custom (${cur})`, true));
      return row(label, `<select name="d-${esc(key)}">${opts_.join("")}</select>`);
    };

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
      const diffRow = diff
        ? selectRow("difficulty-target", "difficulty", diff.levels, opts[diff.key] ?? diff.levels[1][1])
        : "";
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
        this.setStatus(st.result ?? "Game over", "result");
        this.logText(`— ${st.result ?? "game over"}`);
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
    this.frontend?.unmount();
    this.frontend = null;
    this.submitResolve = null;
  }

  private teardown(): void {
    this.gen++;
    this.tourney?.destroy();
    this.tourney = null;
    this.teardownMatch();
    this.logEl = null;
    this.statusEl = null;
  }
}

function message(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

export type { Manifest, ViewState };
