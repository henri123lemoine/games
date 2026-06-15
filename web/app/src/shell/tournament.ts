// The tournament lab: round-robin bot tournaments running on a pool of
// engine workers (one wasm instance per core), with a live pairing matrix
// and a Bradley-Terry Elo table that updates as results stream in.

import { EngineHost } from '../engine/host';
import type { CompareInfo, GameInfo, Wdl } from '../engine/protocol';
import { botLabel, botSpec, difficultyFor, tourneyBots } from './config';

/** Fixed per-game options the tournament needs. It is head-to-head, so
 * Liar's Dice is pinned to two seats. */
const TOURNEY_OPTS: Record<string, Record<string, string>> = {
  go: { size: '9' },
  'liars-dice': { players: '2', dice: '5' },
};

interface Entrant {
  bot: string;
  /** The chosen difficulty knob value, or '' for bots with no difficulty. */
  level: string;
}

interface PairTask {
  i: number;
  j: number;
  k: number;
}

export class TournamentScreen {
  private hosts: EngineHost[] = [];
  private running = false;
  private gen = 0;
  private entrants: Entrant[] = [];

  constructor(
    private root: HTMLElement,
    private compare: CompareInfo[],
    private games: GameInfo[],
    private statsHost: EngineHost,
    private onBack: () => void,
  ) {}

  render(): void {
    const options = this.compare
      .map((c) => `<option value="${c.id}">${c.id}</option>`)
      .join('');
    this.root.innerHTML = `
      <div class="tourney">
        <button type="button" class="link back">&larr; arcade</button>
        <h2>Tournament lab</h2>
        <p class="muted">Round-robin between bots, paired seat-swapped games on a pool of engine
           workers, Bradley-Terry Elo fitted live. Same statistics as the lab's CLI.</p>
        <div class="tourney-form">
          <label class="opt-row"><span>game</span>
            <select class="t-game">${options}</select></label>
          <div class="t-entrants"></div>
          <button type="button" class="link t-add">+ add bot</button>
          <label class="opt-row"><span>games / pairing</span>
            <select class="t-games">
              <option value="4">4</option>
              <option value="8" selected>8</option>
              <option value="16">16</option>
              <option value="32">32</option>
            </select></label>
          <button type="button" class="primary t-run">Run tournament</button>
        </div>
        <div class="t-progress"></div>
        <div class="t-standings"></div>
        <div class="t-matrix"></div>
      </div>`;
    this.root.querySelector<HTMLButtonElement>('.back')!.onclick = () => {
      this.destroy();
      this.onBack();
    };
    const gameSel = this.root.querySelector<HTMLSelectElement>('.t-game')!;
    gameSel.onchange = () => {
      this.entrants = this.defaultEntrants(gameSel.value);
      this.renderEntrants();
    };
    this.root.querySelector<HTMLButtonElement>('.t-add')!.onclick = () => {
      const bots = this.botsFor(gameSel.value);
      const bot = bots[0] ?? '';
      this.entrants.push({ bot, level: this.mediumLevel(gameSel.value, bot) });
      this.renderEntrants();
    };
    this.entrants = this.defaultEntrants(gameSel.value);
    this.renderEntrants();
    this.root.querySelector<HTMLButtonElement>('.t-run')!.onclick = () => void this.run();
  }

  private botsFor(gameId: string): string[] {
    const g = this.games.find((x) => x.id === gameId);
    return g ? tourneyBots(g) : [];
  }

  private mediumLevel(gameId: string, bot: string): string {
    const d = difficultyFor(gameId, bot);
    return d ? (d.levels[1] ?? d.levels[0])[1] : '';
  }

  private defaultEntrants(gameId: string): Entrant[] {
    return this.botsFor(gameId).map((bot) => ({
      bot,
      level: this.mediumLevel(gameId, bot),
    }));
  }

  /** Renders the entrant rows (bot + difficulty) and wires their controls.
   * Re-rendered on every change so a bot switch reflows its difficulty list. */
  private renderEntrants(): void {
    const gameId = this.root.querySelector<HTMLSelectElement>('.t-game')!.value;
    const bots = this.botsFor(gameId);
    const container = this.root.querySelector<HTMLElement>('.t-entrants')!;
    container.innerHTML = this.entrants
      .map((e, idx) => {
        const botOpts = bots
          .map(
            (b) =>
              `<option value="${b}"${b === e.bot ? ' selected' : ''}>${escapeHtml(botLabel(b))}</option>`,
          )
          .join('');
        const diff = difficultyFor(gameId, e.bot);
        const level = diff
          ? `<select class="t-level" data-i="${idx}">${diff.levels
              .map(
                ([l, v]) =>
                  `<option value="${v}"${v === e.level ? ' selected' : ''}>${l}</option>`,
              )
              .join('')}</select>`
          : `<span class="t-nolevel">—</span>`;
        return `<div class="t-entrant">
          <select class="t-bot" data-i="${idx}">${botOpts}</select>
          ${level}
          <button type="button" class="link t-remove" data-i="${idx}" title="remove">×</button>
        </div>`;
      })
      .join('');
    for (const sel of container.querySelectorAll<HTMLSelectElement>('.t-bot')) {
      sel.onchange = () => {
        const i = Number(sel.dataset.i);
        this.entrants[i] = { bot: sel.value, level: this.mediumLevel(gameId, sel.value) };
        this.renderEntrants();
      };
    }
    for (const sel of container.querySelectorAll<HTMLSelectElement>('.t-level')) {
      sel.onchange = () => {
        this.entrants[Number(sel.dataset.i)].level = sel.value;
      };
    }
    for (const btn of container.querySelectorAll<HTMLButtonElement>('.t-remove')) {
      btn.onclick = () => {
        this.entrants.splice(Number(btn.dataset.i), 1);
        this.renderEntrants();
      };
    }
  }

  private async run(): Promise<void> {
    if (this.running) {
      this.gen++;
      this.stopPool();
      this.running = false;
      this.root.querySelector('.t-run')!.textContent = 'Run tournament';
      return;
    }
    const gen = ++this.gen;
    const game = this.root.querySelector<HTMLSelectElement>('.t-game')!.value;
    const bots = this.entrants.map((e) => botSpec(game, e.bot, e.level || undefined));
    const gamesPer = Math.max(
      2,
      Number(this.root.querySelector<HTMLSelectElement>('.t-games')!.value) || 8,
    );
    const seed = (Math.floor(Math.random() * 0x7fffffff) | 1) >>> 0;
    const opts = TOURNEY_OPTS[game] ?? {};
    if (bots.length < 2) {
      this.progress('Add at least two bots.');
      return;
    }
    this.running = true;
    this.root.querySelector('.t-run')!.textContent = 'Stop';

    const n = bots.length;
    const pairsPer = Math.max(1, Math.floor(gamesPer / 2));
    const records: Wdl[][] = Array.from({ length: n }, () =>
      Array.from({ length: n }, () => ({ w: 0, d: 0, l: 0 })),
    );
    const tasks: PairTask[] = [];
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++)
        for (let k = 0; k < pairsPer; k++) tasks.push({ i, j, k });
    let done = 0;
    const total = tasks.length;
    this.renderTables(bots, records, null);
    this.progress(`0 / ${total} pairs`);

    const cores = navigator.hardwareConcurrency || 4;
    const poolSize = Math.max(1, Math.min(4, cores - 2, total));
    this.hosts = Array.from({ length: poolSize }, () => new EngineHost());

    let cursor = 0;
    let eloRefresh: Promise<void> = Promise.resolve();
    const runOn = async (host: EngineHost): Promise<void> => {
      while (this.gen === gen && cursor < tasks.length) {
        const t = tasks[cursor++];
        const pairSeed = (seed ^ ((t.i * n + t.j) << 16)) >>> 0;
        try {
          const r = await host.pairs(game, opts, bots[t.i], bots[t.j], pairSeed, t.k, t.k + 1);
          if (this.gen !== gen) return;
          const cell = records[t.i][t.j];
          cell.w += r.w;
          cell.d += r.d;
          cell.l += r.l;
          const mirror = records[t.j][t.i];
          mirror.w += r.l;
          mirror.d += r.d;
          mirror.l += r.w;
          done++;
          this.progress(`${done} / ${total} pairs`);
          eloRefresh = eloRefresh.then(async () => {
            if (this.gen !== gen) return;
            const matrix = records.map((row) =>
              row.map((c) => [c.w, c.d, c.l] as [number, number, number]),
            );
            const elos = await this.statsHost.fitElo(matrix);
            if (this.gen === gen) this.renderTables(bots, records, elos);
          });
        } catch (e) {
          if (this.gen !== gen) return;
          this.progress(`error: ${e instanceof Error ? e.message : e}`);
          this.gen++;
          this.running = false;
          const btn = this.root.querySelector('.t-run');
          if (btn) btn.textContent = 'Run tournament';
          return;
        }
      }
    };
    await Promise.all(this.hosts.map(runOn));
    await eloRefresh.catch(() => {});
    if (this.gen === gen) {
      this.progress(`done — ${done * 2} games across ${total} pairs on ${poolSize} workers`);
      this.running = false;
      const btn = this.root.querySelector('.t-run');
      if (btn) btn.textContent = 'Run tournament';
    }
    this.stopPool();
  }

  private renderTables(bots: string[], records: Wdl[][], elos: number[] | null): void {
    const n = bots.length;
    const totals = bots.map((_, i) =>
      records[i].reduce(
        (acc, c) => ({ w: acc.w + c.w, d: acc.d + c.d, l: acc.l + c.l }),
        { w: 0, d: 0, l: 0 },
      ),
    );
    const order = bots.map((_, i) => i);
    if (elos) order.sort((a, b) => elos[b] - elos[a]);
    const standings = order
      .map((i, rank) => {
        const t = totals[i];
        const e = elos ? `${elos[i] >= 0 ? '+' : ''}${elos[i].toFixed(0)}` : '—';
        return `<tr><td>${rank + 1}</td><td class="t-spec">${escapeHtml(bots[i])}</td>
                <td class="t-elo">${e}</td><td>${t.w}-${t.d}-${t.l}</td></tr>`;
      })
      .join('');
    this.root.querySelector('.t-standings')!.innerHTML = `
      <table class="t-table">
        <thead><tr><th>#</th><th>bot</th><th>elo</th><th>W-D-L</th></tr></thead>
        <tbody>${standings}</tbody>
      </table>`;
    let matrix = '<table class="t-table t-grid"><thead><tr><th></th>';
    for (let j = 0; j < n; j++) matrix += `<th>${j + 1}</th>`;
    matrix += '</tr></thead><tbody>';
    for (let i = 0; i < n; i++) {
      matrix += `<tr><th>${i + 1}. ${escapeHtml(shorten(bots[i]))}</th>`;
      for (let j = 0; j < n; j++) {
        const c = records[i][j];
        matrix +=
          i === j
            ? '<td class="t-self">·</td>'
            : `<td>${c.w + c.d + c.l ? `${c.w}-${c.d}-${c.l}` : ''}</td>`;
      }
      matrix += '</tr>';
    }
    matrix += '</tbody></table>';
    this.root.querySelector('.t-matrix')!.innerHTML = matrix;
  }

  private progress(text: string): void {
    const el = this.root.querySelector('.t-progress');
    if (el) el.textContent = text;
  }

  private stopPool(): void {
    for (const h of this.hosts) h.terminate();
    this.hosts = [];
  }

  destroy(): void {
    this.gen++;
    this.stopPool();
  }
}

function shorten(spec: string): string {
  return spec.length > 24 ? `${spec.slice(0, 22)}…` : spec;
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
