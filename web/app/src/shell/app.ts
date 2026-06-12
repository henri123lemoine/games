// The arcade shell: pick a game and you are immediately playing it against
// the lab's bot — configuration lives in a quiet settings drawer, not between
// you and the board. One engine worker drives play; the shell owns the loop
// and narration, frontends own the board.

import { type ClientBot, clientBotFor } from '../bots';
import { EngineHost } from '../engine/host';
import type { GameInfo, GameOpt, Manifest, MatchEventData, ViewState } from '../engine/protocol';
import { frontendFor } from '../frontends';
import type { FrontendCtx, GameFrontend } from '../frontends/types';
import { TournamentScreen } from './tournament';

const GAME_TAGLINES: Record<string, string> = {
  chess: 'alpha-beta search, perft-validated rules',
  'liars-dice': 'belief-tracking bots that bluff back',
  twentyone: 'a CFR+ solver trained as you watch',
  othello: 'weighted squares and mobility',
  connect4: 'deep tactical search',
  go: 'Monte-Carlo tree search, 9×9',
  '2048': 'an MCTS bot, or your own arrows',
  snake: "the classic, and it won't wait for you",
};

/** What clicking a card starts: browser-tuned, no questions asked. */
const DEFAULT_OPTS: Record<string, Record<string, string>> = {
  chess: { depth: '4' },
  'liars-dice': { players: '5', dice: '5', rollouts: '400' },
  twentyone: { hearts: '3', iters: '1500' },
  othello: { depth: '5' },
  connect4: { depth: '7' },
  go: { size: '9', sims: '1500' },
  '2048': {},
  snake: {},
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

/** The drawer's fields come from the engine's structured option schema;
 * seed and seat get dedicated rows, so they are filtered out here. */
function optFields(schema: GameOpt[], current: Record<string, string>): OptField[] {
  return schema
    .filter((o) => o.key !== 'seed' && o.key !== 'seat')
    .map((o) => ({
      key: o.key,
      value: current[o.key] ?? o.value.split('|')[0].replace(/\.{3}$/, ''),
    }));
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
    case 'chess':
      return `<div class="mini mini-chess"><span class="mini-pc" style="left:12%;top:8%">♞</span><span class="mini-pc mini-pc-w" style="left:58%;top:52%">♙</span></div>`;
    case 'liars-dice':
      return `<div class="mini mini-dice">
        <span class="mini-die"><i style="left:25%;top:25%"></i><i style="left:65%;top:65%"></i></span>
        <span class="mini-die mini-die-2"><i style="left:45%;top:45%"></i><i style="left:18%;top:18%"></i><i style="left:72%;top:72%"></i></span>
        <span class="mini-cup"></span></div>`;
    case 'twentyone':
      return `<div class="mini mini-t21"><span class="mini-card">7♠</span><span class="mini-card mini-card-2">9♦</span><span class="mini-heart">♥♥♥</span></div>`;
    case 'othello':
      return `<div class="mini mini-othello"><span class="mini-disc mini-disc-b" style="left:28%;top:28%"></span><span class="mini-disc mini-disc-w" style="left:52%;top:28%"></span><span class="mini-disc mini-disc-w" style="left:28%;top:52%"></span><span class="mini-disc mini-disc-b" style="left:52%;top:52%"></span></div>`;
    case 'connect4':
      return `<div class="mini mini-c4"></div>`;
    case 'go':
      return `<div class="mini mini-go"><span class="mini-stone mini-stone-b" style="left:30%;top:30%"></span><span class="mini-stone mini-stone-w" style="left:55%;top:47%"></span><span class="mini-stone mini-stone-b" style="left:38%;top:63%"></span></div>`;
    case '2048':
      return `<div class="mini mini-2048"><span>2</span><span class="v4">4</span><span class="v8">8</span><span class="v16">16</span></div>`;
    case 'snake':
      return `<div class="mini mini-snake"><span class="mini-seg" style="left:18%;top:55%"></span><span class="mini-seg" style="left:33%;top:55%"></span><span class="mini-seg" style="left:48%;top:55%"></span><span class="mini-seg mini-head" style="left:48%;top:38%"></span><span class="mini-food" style="left:72%;top:25%"></span></div>`;
    default:
      return `<div class="mini"></div>`;
  }
}

type Mode = 'play' | 'watch';

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

  constructor(private root: HTMLElement) {}

  async start(): Promise<void> {
    this.root.innerHTML = '<div class="boot">Waking the engine…</div>';
    this.manifest = await this.host.manifest();
    this.renderHome();
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
            <span class="card-summary">${esc(GAME_TAGLINES[g.id] ?? g.summary)}</span>
          </div>
          <button type="button" class="card-watch" title="Watch bots play">watch</button>
        </div>`,
      )
      .join('');
    this.root.innerHTML = `
      <div class="home">
        <header class="hero">
          <p class="eyebrow">rust · webassembly · runs on your device</p>
          <h1>The Games Room</h1>
          <p>Eight games, one engine. Sit down against the lab's bots —
             or let them play each other.</p>
        </header>
        <div class="card-grid">${cards}</div>
        <div class="home-foot">
          <button type="button" class="link tourney-link">Bot tournament lab &rarr;</button>
        </div>
      </div>`;
    for (const el of this.root.querySelectorAll<HTMLElement>('.card')) {
      const game = this.manifest.games.find((g) => g.id === el.dataset.game);
      if (!game) continue;
      const play = () => void this.startMatch(game, 'play');
      el.onclick = play;
      el.onkeydown = (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          play();
        }
      };
      el.querySelector<HTMLButtonElement>('.card-watch')!.onclick = (e) => {
        e.stopPropagation();
        void this.startMatch(game, 'watch');
      };
    }
    this.root.querySelector<HTMLButtonElement>('.tourney-link')!.onclick = () =>
      this.renderTournament();
  }

  private renderTournament(): void {
    this.teardown();
    this.tourney = new TournamentScreen(this.root, this.manifest.compare, this.host, () =>
      this.renderHome(),
    );
    this.tourney.render();
  }

  // ---------- match ----------

  private buildOpts(game: GameInfo, mode: Mode, overrides: Record<string, string>): Record<string, string> {
    const opts: Record<string, string> = { ...DEFAULT_OPTS[game.id], ...overrides };
    if (mode === 'watch') {
      if (game.solo) opts.bot ||= game.watchBot;
      else opts.seat = 'watch';
    } else if (game.solo) {
      delete opts.bot;
    } else if (opts.seat === 'watch') {
      opts.seat = '0';
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
    let gpuNote = '';
    if (opts.bot === 'azero-gpu' && !('gpu' in navigator)) {
      opts.bot = 'azero';
      gpuNote = 'WebGPU unavailable — playing the CPU azero net instead.';
    }
    this.renderMatchSkeleton(game, mode, opts);
    if (gpuNote) this.logText(gpuNote, true);
    if (game.id === 'twentyone') {
      this.setStatus('Training the CFR+ solver in your browser…');
    }
    try {
      await this.loadArtifacts(opts);
      const st = await this.host.create(game.id, opts);
      if (gen !== this.gen) return;
      const makeBot = clientBotFor(game.id, opts.bot);
      this.clientBot = makeBot ? await makeBot(this.host, opts) : null;
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
      if (gen === this.gen) this.setStatus(`Could not start: ${message(e)}`, 'error');
    }
  }

  private renderMatchSkeleton(game: GameInfo, mode: Mode, opts: Record<string, string>): void {
    const modeLabel = mode === 'watch' ? 'take a seat' : 'watch bots';
    this.root.innerHTML = `
      <div class="match">
        <header class="match-bar">
          <button type="button" class="link back">&larr; games</button>
          <span class="match-title">${esc(game.name || game.id)}</span>
          <span class="spacer"></span>
          <label class="speed-label">speed
            <select class="speed">
              <option value="2">slow</option>
              <option value="1" selected>normal</option>
              <option value="0.4">fast</option>
              <option value="0">instant</option>
            </select>
          </label>
          <button type="button" class="link mode-toggle">${modeLabel}</button>
          <button type="button" class="link again">rematch</button>
          <button type="button" class="link gear" title="Match settings">⚙</button>
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
    this.logEl = this.root.querySelector('.log');
    this.statusEl = this.root.querySelector('.status');
    this.root.querySelector<HTMLButtonElement>('.back')!.onclick = () => this.renderHome();
    this.root.querySelector<HTMLButtonElement>('.again')!.onclick = () =>
      void this.startMatch(game, mode, { ...opts, seed: String(randomSeed()) });
    this.root.querySelector<HTMLButtonElement>('.mode-toggle')!.onclick = () =>
      void this.startMatch(game, mode === 'watch' ? 'play' : 'watch', {
        seed: String(randomSeed()),
      });
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
    this.wireDrawer(game, mode, opts);
  }

  private wireDrawer(game: GameInfo, mode: Mode, opts: Record<string, string>): void {
    const drawer = this.root.querySelector<HTMLElement>('.drawer')!;
    const fieldsEl = drawer.querySelector<HTMLElement>('.drawer-fields')!;
    const open = () => {
      const fields = optFields(game.optsSchema, opts);
      const seatRow = game.solo
        ? ''
        : `<label class="opt-row"><span>seat</span>
             <input name="d-seat" value="${esc(opts.seat ?? '0')}" autocomplete="off" /></label>`;
      fieldsEl.innerHTML = `
        ${seatRow}
        ${fields
          .map(
            (f) => `<label class="opt-row"><span>${esc(f.key)}</span>
              <input name="d-${esc(f.key)}" value="${esc(f.value)}" autocomplete="off" /></label>`,
          )
          .join('')}
        <label class="opt-row"><span>seed</span>
          <input name="d-seed" value="${esc(String(opts.seed ?? randomSeed()))}" autocomplete="off" /></label>`;
      drawer.hidden = false;
    };
    this.root.querySelector<HTMLButtonElement>('.gear')!.onclick = open;
    drawer.querySelector<HTMLButtonElement>('.drawer-close')!.onclick = () => {
      drawer.hidden = true;
    };
    drawer.onclick = (e) => {
      if (e.target === drawer) drawer.hidden = true;
    };
    drawer.querySelector<HTMLButtonElement>('.drawer-apply')!.onclick = () => {
      const overrides: Record<string, string> = {};
      for (const input of fieldsEl.querySelectorAll<HTMLInputElement>('input')) {
        const key = input.name.replace(/^d-/, '');
        if (input.value.trim() !== '') overrides[key] = input.value.trim();
      }
      const nextMode: Mode = overrides.seat === 'watch' ? 'watch' : mode;
      void this.startMatch(game, nextMode, overrides);
    };
  }

  private async runLoop(gen: number): Promise<void> {
    const fail = (e: unknown) => {
      if (gen === this.gen) this.setStatus(message(e), 'error');
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
        this.setStatus(st.result ?? 'Game over', 'result');
        this.logText(`— ${st.result ?? 'game over'}`);
        return;
      }
      if (this.clientBot && st.toAct >= 0 && st.toAct !== st.humanSeat) {
        this.setStatus('Thinking…');
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
      this.setStatus('Your turn');
      this.frontend!.promptAction(st.labels);
      const input = await new Promise<string>((res) => (this.submitResolve = res));
      if (gen !== this.gen) return;
      if (st.numSeats > 1) this.setStatus('Thinking…');
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
