// The arcade shell: game picker → match setup → match screen. One engine
// worker drives play; the shell owns the loop and narration, frontends own
// the board.

import { EngineHost } from '../engine/host';
import type { GameInfo, Manifest, MatchEventData, ViewState } from '../engine/protocol';
import { frontendFor } from '../frontends';
import type { FrontendCtx, GameFrontend } from '../frontends/types';
import { TournamentScreen } from './tournament';

const GAME_NAMES: Record<string, string> = {
  chess: 'Chess',
  'liars-dice': "Liar's Dice",
  twentyone: 'Twenty-One',
  othello: 'Othello',
  connect4: 'Connect Four',
  go: 'Go',
  '2048': '2048',
  snake: 'Snake',
};

const GAME_GLYPHS: Record<string, string> = {
  chess: '♞',
  'liars-dice': '⚄',
  twentyone: '♥',
  othello: '◐',
  connect4: '◍',
  go: '◉',
  '2048': '⌗',
  snake: '⌇',
};

/** Single-player games spectate via a bot option, not a seat. */
const SINGLE_PLAYER = new Set(['2048', 'snake']);

/** Browser-friendly default overrides (single-threaded wasm). */
const WEB_DEFAULTS: Record<string, Record<string, string>> = {
  go: { sims: '2000' },
  'liars-dice': { rollouts: '400' },
  twentyone: { iters: '20000' },
};

/** Trained artifacts fetched as static assets, keyed by the path the
 * registry asks for. */
const ARTIFACTS: Record<string, string> = {
  'data/azero/chess.bin': 'artifacts/azero-chess.bin',
};

interface OptField {
  key: string;
  value: string;
}

function parseOptFields(help: string, gameId: string): OptField[] {
  const fields: OptField[] = [];
  const seen = new Set<string>();
  for (const m of help.matchAll(/([A-Za-z0-9_-]+)=([^\s]+)/g)) {
    const key = m[1];
    if (key === 'seed' || key === 'seat' || /^\d+$/.test(key) || seen.has(key)) continue;
    seen.add(key);
    const value = WEB_DEFAULTS[gameId]?.[key] ?? m[2].split('|')[0].replace(/\.{3}$/, '');
    fields.push({ key, value });
  }
  return fields;
}

function randomSeed(): number {
  return (Math.floor(Math.random() * 0x7fff_ffff) | 1) >>> 0;
}

export class App {
  private host = new EngineHost();
  private manifest!: Manifest;
  private frontend: GameFrontend | null = null;
  private gen = 0;
  private speedScale = 1;
  private submitResolve: ((input: string) => void) | null = null;
  private logEl: HTMLElement | null = null;
  private statusEl: HTMLElement | null = null;

  constructor(private root: HTMLElement) {}

  async start(): Promise<void> {
    this.root.innerHTML = '<div class="boot">Loading the engine…</div>';
    this.manifest = await this.host.manifest();
    this.renderHome();
  }

  // ---------- home ----------

  private tourney: TournamentScreen | null = null;

  private renderTournament(): void {
    this.teardownMatch();
    this.tourney = new TournamentScreen(this.root, this.manifest.compare, this.host, () =>
      this.renderHome(),
    );
    this.tourney.render();
  }

  private renderHome(): void {
    this.teardownMatch();
    const cards = this.manifest.games
      .map(
        (g) => `
        <button class="card" data-game="${g.id}">
          <span class="card-glyph">${GAME_GLYPHS[g.id] ?? '∎'}</span>
          <span class="card-name">${GAME_NAMES[g.id] ?? g.id}</span>
          <span class="card-summary">${g.summary}</span>
        </button>`,
      )
      .join('');
    this.root.innerHTML = `
      <div class="home">
        <header class="hero">
          <h1>Games Arcade</h1>
          <p>Play against the lab's bots — search and learned policies in Rust,
             compiled to WebAssembly, running entirely in your browser.</p>
        </header>
        <div class="card-grid">${cards}</div>
        <div class="home-foot">
          <button type="button" class="link tourney-link">Tournament lab &rarr;</button>
        </div>
      </div>`;
    for (const el of this.root.querySelectorAll<HTMLButtonElement>('.card')) {
      el.onclick = () => {
        const game = this.manifest.games.find((g) => g.id === el.dataset.game);
        if (game) this.renderSetup(game);
      };
    }
    this.root.querySelector<HTMLButtonElement>('.tourney-link')!.onclick = () =>
      this.renderTournament();
  }

  // ---------- setup ----------

  private renderSetup(game: GameInfo): void {
    this.teardownMatch();
    const fields = parseOptFields(game.opts, game.id);
    const fieldRows = fields
      .map(
        (f) => `
        <label class="opt-row">
          <span>${f.key}</span>
          <input name="opt-${f.key}" value="${f.value}" autocomplete="off" />
        </label>`,
      )
      .join('');
    const seatChoices = SINGLE_PLAYER.has(game.id)
      ? ''
      : `<label class="opt-row"><span>seat</span>
           <input name="opt-seat" value="0" autocomplete="off" /></label>`;
    this.root.innerHTML = `
      <div class="setup">
        <button type="button" class="link back">&larr; all games</button>
        <h2>${GAME_NAMES[game.id] ?? game.id}</h2>
        <p class="muted">${game.summary}</p>
        <div class="mode-row">
          <button type="button" class="mode selected" data-mode="play">Play</button>
          <button type="button" class="mode" data-mode="watch">Watch bots</button>
        </div>
        <div class="opts">
          ${seatChoices}
          ${fieldRows}
          <label class="opt-row"><span>seed</span>
            <input name="opt-seed" value="${randomSeed()}" autocomplete="off" /></label>
        </div>
        <button type="button" class="primary start">Start match</button>
      </div>`;
    let mode: 'play' | 'watch' = 'play';
    for (const el of this.root.querySelectorAll<HTMLButtonElement>('.mode')) {
      el.onclick = () => {
        mode = el.dataset.mode as 'play' | 'watch';
        this.root
          .querySelectorAll('.mode')
          .forEach((m) => m.classList.toggle('selected', m === el));
      };
    }
    this.root.querySelector<HTMLButtonElement>('.back')!.onclick = () => this.renderHome();
    this.root.querySelector<HTMLButtonElement>('.start')!.onclick = () => {
      const opts: Record<string, string> = {};
      for (const input of this.root.querySelectorAll<HTMLInputElement>('.opts input')) {
        const key = input.name.replace(/^opt-/, '');
        if (input.value.trim() !== '') opts[key] = input.value.trim();
      }
      if (mode === 'watch') {
        if (SINGLE_PLAYER.has(game.id)) opts.bot ||= 'mcts-eval';
        else opts.seat = 'watch';
      } else if (SINGLE_PLAYER.has(game.id)) {
        delete opts.bot;
      }
      void this.startMatch(game, opts);
    };
  }

  // ---------- match ----------

  private async startMatch(game: GameInfo, opts: Record<string, string>): Promise<void> {
    const gen = ++this.gen;
    this.renderMatchSkeleton(game, opts);
    try {
      await this.loadArtifacts(opts);
      const st = await this.host.create(game.id, opts);
      if (gen !== this.gen) return;
      const boardEl = this.root.querySelector<HTMLElement>('.board')!;
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
      this.setStatus(st.humanSeat < 0 ? 'Bots playing…' : 'Thinking…');
      void this.runLoop(gen);
    } catch (e) {
      if (gen === this.gen) this.setStatus(`Failed to start: ${message(e)}`, 'error');
    }
  }

  private renderMatchSkeleton(game: GameInfo, opts: Record<string, string>): void {
    this.root.innerHTML = `
      <div class="match">
        <header class="match-bar">
          <button type="button" class="link back">&larr; all games</button>
          <span class="match-title">${GAME_NAMES[game.id] ?? game.id}</span>
          <span class="spacer"></span>
          <label class="speed-label">speed
            <select class="speed">
              <option value="2">slow</option>
              <option value="1" selected>normal</option>
              <option value="0.4">fast</option>
              <option value="0">instant</option>
            </select>
          </label>
          <button type="button" class="link again">new match</button>
        </header>
        <div class="match-body">
          <section class="board"></section>
          <aside class="side">
            <div class="status">Starting…</div>
            <div class="log" aria-live="polite"></div>
            <form class="free-input">
              <input placeholder="or type a move…" autocomplete="off" />
              <button type="submit">send</button>
            </form>
          </aside>
        </div>
      </div>`;
    this.logEl = this.root.querySelector('.log');
    this.statusEl = this.root.querySelector('.status');
    this.root.querySelector<HTMLButtonElement>('.back')!.onclick = () => this.renderHome();
    this.root.querySelector<HTMLButtonElement>('.again')!.onclick = () => {
      void this.startMatch(game, { ...opts, seed: String(randomSeed()) });
    };
    this.root.querySelector<HTMLSelectElement>('.speed')!.onchange = (e) => {
      this.speedScale = Number((e.target as HTMLSelectElement).value);
    };
    const form = this.root.querySelector<HTMLFormElement>('.free-input')!;
    form.onsubmit = (e) => {
      e.preventDefault();
      const input = form.querySelector('input')!;
      if (input.value.trim()) {
        this.submit(input.value.trim());
        input.value = '';
      }
    };
  }

  private async runLoop(gen: number): Promise<void> {
    while (gen === this.gen) {
      let ev: MatchEventData | null;
      try {
        ev = await this.host.step();
      } catch (e) {
        this.setStatus(message(e), 'error');
        return;
      }
      if (gen !== this.gen) return;
      if (ev) {
        this.log(ev);
        const st = await this.host.state();
        if (gen !== this.gen) return;
        await this.frontend!.animate(ev, st);
        continue;
      }
      const st = await this.host.state();
      if (gen !== this.gen) return;
      this.frontend!.render(st);
      if (st.isOver) {
        this.setStatus(st.result ?? 'Game over', 'result');
        this.logText(`— ${st.result ?? 'game over'}`);
        return;
      }
      this.setStatus('Your turn');
      this.frontend!.promptAction(st.labels);
      const input = await new Promise<string>((res) => (this.submitResolve = res));
      if (gen !== this.gen) return;
      this.setStatus('Thinking…');
      try {
        const mev = await this.host.apply(input);
        if (gen !== this.gen) return;
        this.log(mev);
        const after = await this.host.state();
        if (gen !== this.gen) return;
        await this.frontend!.animate(mev, after);
      } catch (e) {
        this.setStatus(message(e), 'error');
      }
    }
  }

  private async loadArtifacts(opts: Record<string, string>): Promise<void> {
    const wanted = opts.net ?? (opts.bot === 'azero' ? 'data/azero/chess.bin' : null);
    if (!wanted || !(wanted in ARTIFACTS)) return;
    const url = `${import.meta.env.BASE_URL}${ARTIFACTS[wanted]}`;
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`artifact ${url} missing (HTTP ${resp.status})`);
    await this.host.artifact(wanted, await resp.arrayBuffer());
  }

  private submit(input: string): void {
    const resolve = this.submitResolve;
    if (!resolve) return;
    this.submitResolve = null;
    resolve(input);
  }

  private animationScale(): number {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return 0;
    return this.speedScale;
  }

  private log(ev: MatchEventData): void {
    this.logText(ev.text);
    if (ev.detail) this.logText(ev.detail, true);
  }

  private logText(text: string, detail = false): void {
    if (!this.logEl) return;
    const line = document.createElement('div');
    line.className = detail ? 'log-line log-detail' : 'log-line';
    line.textContent = text;
    this.logEl.append(line);
    this.logEl.scrollTop = this.logEl.scrollHeight;
  }

  private setStatus(text: string, kind: 'info' | 'error' | 'result' = 'info'): void {
    if (!this.statusEl) return;
    this.statusEl.textContent = text;
    this.statusEl.className = `status status-${kind}`;
  }

  private teardownMatch(): void {
    this.gen++;
    this.tourney?.destroy();
    this.tourney = null;
    this.frontend?.unmount();
    this.frontend = null;
    this.submitResolve = null;
    this.logEl = null;
    this.statusEl = null;
  }
}

function message(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

export type { Manifest, ViewState };
