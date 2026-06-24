// No-Limit Hold'em frontend: a felt table with the bots seated around it, the
// community board and pot center-table, playing-card hole cards, chips that
// slide in as bets, and a showdown reveal. The view/transition JSON schemas
// are the private contract with games/poker/src/ui.rs.

import type { MatchEventData, ViewState } from '../../engine/protocol';
import type { FrontendCtx, GameFrontend } from '../types';
import { sleep } from '../types';
import { STYLE, STYLE_ID } from './style';

interface PkPlayer {
  seat: number;
  stack: number;
  committed: number;
  streetBet: number;
  folded: boolean;
  allIn: boolean;
  toAct: boolean;
  net: number | null;
  /** Two card strings (e.g. `["As","Td"]`), or null when hidden. */
  hole: string[] | null;
}

interface PkView {
  seats: number;
  viewer: number;
  spectator: boolean;
  phase: 'preflop' | 'flop' | 'turn' | 'river' | 'showdown' | 'over';
  button: number;
  pot: number;
  currentBet: number;
  toCall: number;
  bigBlind: number;
  board: string[];
  players: PkPlayer[];
}

interface PkReveal {
  seat: number;
  hole: string[];
  net: number;
}
interface PkTransition {
  kind: 'fold' | 'check' | 'call' | 'raise' | 'allin';
  seat: number;
  amount: number;
  pot: number;
  gameOver: boolean;
  showdown: PkReveal[] | null;
}

const SUIT_GLYPH: Record<string, string> = { c: '♣', d: '♦', h: '♥', s: '♠' };
const RED_SUITS = new Set(['d', 'h']);

function isView(v: unknown): v is PkView {
  return typeof v === 'object' && v !== null && Array.isArray((v as PkView).players);
}
function isTransition(d: unknown): d is PkTransition {
  return typeof d === 'object' && d !== null && typeof (d as PkTransition).kind === 'string';
}

/** A face-up playing card from a `"Td"`-style string. */
function cardHtml(card: string, extra = ''): string {
  const rank = card[0] === 'T' ? '10' : card[0];
  const suit = card[1];
  const red = RED_SUITS.has(suit) ? ' red' : '';
  return `<div class="pk-card${red}${extra}">
    <span class="pk-rank">${rank}</span>
    <span class="pk-suit">${SUIT_GLYPH[suit] ?? suit}</span>
  </div>`;
}
function backHtml(extra = ''): string {
  return `<div class="pk-card back${extra}"></div>`;
}

/** Seat coordinates (percent of the table) around an oval, with the human (or
 * seat 0 when spectating) at bottom center and seats proceeding clockwise. */
function seatPos(displayIndex: number, n: number): { x: number; y: number; below: boolean } {
  const a = (Math.PI / 180) * (90 + (360 * displayIndex) / n);
  const x = 50 + 42 * Math.cos(a);
  const y = 50 + 40 * Math.sin(a);
  return { x, y, below: y > 52 };
}

/** A running session bankroll, so chips feel persistent across hands even
 * though each engine match is one hand. Keyed per table shape. */
function bankKey(ctx: FrontendCtx): string {
  return `pk-bank-${ctx.gameId}-${ctx.numSeats}`;
}

class PokerFrontend implements GameFrontend {
  private ctx!: FrontendCtx;
  private seatsEl!: HTMLElement;
  private centerEl!: HTMLElement;
  private bannerEl!: HTMLElement;
  private controlsEl!: HTMLElement;
  private bankEl!: HTMLElement;
  private view: PkView | null = null;
  private boardShown = 0;
  private dead = false;
  private bank = 0;
  /** Pods are built once; only their dynamic parts re-render, so the shell's
   * per-seat picker (placed in the pod, under the name) is never wiped. */
  private seatsBuilt = false;

  mount(host: HTMLElement, ctx: FrontendCtx): void {
    this.ctx = ctx;
    if (!document.getElementById(STYLE_ID)) {
      const style = document.createElement('style');
      style.id = STYLE_ID;
      style.textContent = STYLE;
      document.head.append(style);
    }
    this.bank = Number(sessionStorage.getItem(bankKey(ctx)) ?? '0') || 0;
    host.innerHTML = `
      <div class="pk-root">
        <div class="pk-table">
          <div class="pk-felt"></div>
          <div class="pk-center">
            <div class="pk-street"></div>
            <div class="pk-board"></div>
            <div class="pk-pot"></div>
          </div>
          <div class="pk-seats"></div>
          <div class="pk-banner"></div>
        </div>
        <div class="pk-bank"></div>
        <div class="pk-controls"></div>
      </div>`;
    this.seatsEl = host.querySelector('.pk-seats')!;
    this.centerEl = host.querySelector('.pk-center')!;
    this.bannerEl = host.querySelector('.pk-banner')!;
    this.controlsEl = host.querySelector('.pk-controls')!;
    this.bankEl = host.querySelector('.pk-bank')!;
    this.renderBank();
  }

  render(state: ViewState): void {
    if (!isView(state.viewData)) {
      const pre = document.createElement('pre');
      pre.className = 'pk-fallback';
      pre.textContent = state.view;
      this.seatsEl.replaceChildren(pre);
      return;
    }
    this.view = state.viewData;
    this.boardShown = this.view.board.length;
    this.renderSeats(this.view);
    this.renderCenter(this.view);
    if (state.toAct !== state.humanSeat || state.isOver) this.controlsEl.replaceChildren();
  }

  async animate(event: MatchEventData, after: ViewState): Promise<void> {
    const scale = this.ctx.animationScale();
    const data = event.data;
    if (isTransition(data) && data.gameOver && data.showdown) {
      this.render(after);
      if (scale > 0 && !this.dead) await this.playShowdown(data, scale);
      return;
    }
    // A board card just turned: show the previous board, then deal the new
    // street so the flop/turn/river visibly lands.
    const prevBoard = this.boardShown;
    this.render(after);
    if (scale > 0 && !this.dead && isView(after.viewData)) {
      if (after.viewData.board.length > prevBoard) {
        await this.dealBoard(prevBoard, scale);
      } else if (isTransition(data)) {
        await this.flashAction(data, scale);
      }
    }
    if (isTransition(data) && data.gameOver) {
      this.render(after);
      if (scale > 0 && !this.dead) await this.playShowdown(data, scale);
    }
  }

  promptAction(labels: string[]): void {
    if (this.ctx.humanSeat < 0 || !this.view) return;
    this.renderControls(labels);
  }

  unmount(): void {
    this.dead = true;
  }

  // ---------- rendering ----------

  private name(seat: number): string {
    return seat === this.ctx.humanSeat ? 'You' : `Bot ${seat + 1}`;
  }

  private renderBank(): void {
    const v = this.bank;
    const sign = v > 0 ? '+' : '';
    const tone = v > 0 ? '#3a8a52' : v < 0 ? '#b04a3a' : 'inherit';
    this.bankEl.innerHTML = `<span style="font:600 12px ui-monospace,Menlo,monospace;color:${tone}">session: ${sign}${v.toFixed(1)} bb</span>`;
  }

  private renderCenter(view: PkView): void {
    const street = this.centerEl.querySelector('.pk-street')!;
    const board = this.centerEl.querySelector('.pk-board')!;
    const pot = this.centerEl.querySelector('.pk-pot')!;
    street.textContent = view.phase === 'over' ? 'showdown' : view.phase;
    const winners = view.phase === 'over' ? this.winnerSeats(view) : new Set<number>();
    const winCards = new Set<string>();
    if (winners.size) {
      for (const p of view.players)
        if (winners.has(p.seat) && p.hole) for (const c of p.hole) winCards.add(c);
    }
    board.innerHTML = view.board
      .map((c) => cardHtml(c, winCards.has(c) ? ' win-card' : ''))
      .join('');
    pot.innerHTML = `pot <b>${view.pot}</b>`;
  }

  private winnerSeats(view: PkView): Set<number> {
    const live = view.players.filter((p) => !p.folded && p.net !== null);
    const best = Math.max(...live.map((p) => p.net ?? -Infinity));
    return new Set(live.filter((p) => (p.net ?? -Infinity) >= 0 && (p.net ?? 0) === best).map((p) => p.seat));
  }

  /** Build the positioned pods once: their seat position, name, and the
   * shell-filled opponent/difficulty picker (under the name) are stable, so the
   * per-action re-render of the dynamic parts never disturbs the picker. */
  private buildSeats(view: PkView): void {
    if (this.seatsBuilt) return;
    this.seatsBuilt = true;
    const n = view.seats;
    const anchor = this.ctx.humanSeat >= 0 ? this.ctx.humanSeat : 0;
    this.seatsEl.innerHTML = view.players
      .map((p) => {
        const pos = seatPos((p.seat - anchor + n) % n, n);
        return `
        <div class="pk-seat${pos.below ? ' below' : ''}" data-seat="${p.seat}"
             style="left:${pos.x.toFixed(2)}%;top:${pos.y.toFixed(2)}%">
          <div class="pk-pod">
            <span class="pk-deco"></span>
            <div class="pk-holes"></div>
            <div class="pk-name">${this.name(p.seat)}</div>
            <span class="seat-slot" data-seat="${p.seat}"></span>
            <div class="pk-stack"></div>
          </div>
          <span class="pk-bet"></span>
        </div>`;
      })
      .join('');
  }

  /** Refresh only the dynamic parts of each pod (state classes, cards, chips,
   * tags); the pod shell and its picker stay put. */
  private renderSeats(view: PkView): void {
    this.buildSeats(view);
    const winners = view.phase === 'over' ? this.winnerSeats(view) : new Set<number>();
    for (const p of view.players) {
      const seatEl = this.seatsEl.querySelector<HTMLElement>(`.pk-seat[data-seat="${p.seat}"]`);
      if (!seatEl) continue;
      seatEl.classList.toggle('folded', p.folded);
      seatEl.classList.toggle('turn', p.toAct);
      seatEl.classList.toggle('winner', winners.has(p.seat));
      const tag = p.folded
        ? '<span class="pk-tag folded">FOLD</span>'
        : p.allIn
          ? '<span class="pk-tag allin">ALL-IN</span>'
          : '';
      const dealer =
        p.seat === view.button ? '<span class="pk-dealer" style="right:-4px;bottom:-4px">D</span>' : '';
      seatEl.querySelector('.pk-deco')!.innerHTML = tag + dealer;
      seatEl.querySelector('.pk-holes')!.innerHTML = this.holesHtml(view, p);
      const bust = p.stack <= 0 && !p.allIn ? ' pk-bust' : '';
      seatEl.querySelector('.pk-stack')!.innerHTML = `<span class="${bust}">${p.stack} bb</span>`;
      seatEl.querySelector('.pk-bet')!.textContent = p.streetBet > 0 ? String(p.streetBet) : '';
    }
  }

  private holesHtml(view: PkView, p: PkPlayer): string {
    if (p.folded && view.phase !== 'over') {
      return p.seat === view.viewer && p.hole
        ? p.hole.map((c) => cardHtml(c, ' muck')).join('')
        : `${backHtml(' muck')}${backHtml(' muck')}`;
    }
    if (p.hole) return p.hole.map((c) => cardHtml(c)).join('');
    // Hidden opponent: face-down until shown.
    if (p.folded) return '';
    return `${backHtml()}${backHtml()}`;
  }

  // ---------- animation ----------

  private async dealBoard(prev: number, scale: number): Promise<void> {
    const cards = this.centerEl.querySelectorAll<HTMLElement>('.pk-board .pk-card');
    for (let i = prev; i < cards.length; i++) {
      cards[i].classList.add('deal-in');
      await sleep(120 * scale);
      if (this.dead) return;
    }
    await sleep(160 * scale);
  }

  private async flashAction(t: PkTransition, scale: number): Promise<void> {
    const seatEl = this.seatsEl.querySelector<HTMLElement>(`[data-seat="${t.seat}"] .pk-pod`);
    if (!seatEl) return;
    const verb =
      t.kind === 'fold'
        ? 'folds'
        : t.kind === 'check'
          ? 'checks'
          : t.kind === 'call'
            ? `calls ${t.amount}`
            : t.kind === 'allin'
              ? `all-in ${t.amount}`
              : `raises ${t.amount}`;
    this.banner(`${this.name(t.seat)} ${verb}`, 'info', false);
    seatEl.animate(
      [{ transform: 'scale(1)' }, { transform: 'scale(1.05)' }, { transform: 'scale(1)' }],
      { duration: 240 * scale, easing: 'ease-out' },
    );
    // Let a bot's move linger so the table doesn't blitz; the human's own
    // action stays snappy (no need to slow your own confirmation).
    const botPace = t.seat !== this.ctx.humanSeat ? 2 : 1;
    await sleep((t.kind === 'fold' || t.kind === 'check' ? 240 : 380) * botPace * scale);
    this.hideBanner();
  }

  private async playShowdown(t: PkTransition, scale: number): Promise<void> {
    if (!t.showdown) return;
    const human = this.ctx.humanSeat;
    // Reveal contested hands (multiway), then the chip results.
    if (t.showdown.length > 1) {
      this.banner('Showdown', 'info', true);
      await sleep(700 * scale);
      if (this.dead) return;
    }
    for (const p of this.view?.players ?? []) {
      const net = p.net ?? 0;
      if (Math.abs(net) < 0.001) continue;
      const seatEl = this.seatsEl.querySelector<HTMLElement>(`[data-seat="${p.seat}"] .pk-pod`);
      if (!seatEl) continue;
      const float = document.createElement('span');
      float.className = `pk-float ${net > 0 ? 'win' : 'lose'}`;
      float.textContent = `${net > 0 ? '+' : ''}${net.toFixed(net % 1 === 0 ? 0 : 1)}`;
      seatEl.append(float);
    }
    const mine = this.view?.players.find((p) => p.seat === human)?.net ?? 0;
    if (human >= 0) {
      const msg =
        mine > 0.001
          ? `You win ${mine.toFixed(1)} bb`
          : mine < -0.001
            ? `You lose ${Math.abs(mine).toFixed(1)} bb`
            : 'You break even';
      this.banner(msg, mine >= 0 ? 'good' : 'bad', true);
    } else {
      const top = Math.max(...(this.view?.players.map((p) => p.net ?? 0) ?? [0]));
      const w = this.view?.players.find((p) => (p.net ?? 0) === top);
      this.banner(`${this.name(w?.seat ?? 0)} takes ${(top || 0).toFixed(0)} bb`, 'good', true);
    }
    // Persist the running bankroll across hands.
    if (human >= 0) {
      this.bank += mine;
      sessionStorage.setItem(bankKey(this.ctx), String(this.bank));
      this.renderBank();
    }
    await sleep(1500 * scale);
    if (this.dead) return;
    this.hideBanner();
  }

  private banner(text: string, tone: 'info' | 'good' | 'bad', sticky: boolean): void {
    this.bannerEl.textContent = text;
    this.bannerEl.className = `pk-banner show ${tone === 'info' ? '' : tone}`;
    void sticky;
  }
  private hideBanner(): void {
    this.bannerEl.classList.remove('show');
  }

  // ---------- controls ----------

  private submit(index: number): void {
    for (const b of this.controlsEl.querySelectorAll('button')) b.disabled = true;
    this.ctx.submit(String(index));
  }

  /** Build fold / check|call / raise-with-a-slider from the legal labels. The
   * labels carry amounts (`"call 6"`, `"raise to 18"`, `"all-in 200"`). */
  private renderControls(labels: string[]): void {
    const buttons: HTMLElement[] = [];
    const raises: { idx: number; to: number; label: string }[] = [];
    let allInIdx = -1;
    let allInTo = 0;

    labels.forEach((label, i) => {
      if (label === 'fold') {
        buttons.push(this.btn('Fold', 'fold', i));
      } else if (label === 'check') {
        buttons.push(this.btn('Check', 'call', i));
      } else if (label.startsWith('call')) {
        const amt = label.split(' ')[1] ?? '';
        buttons.push(this.btn(`Call ${amt}`, 'call', i));
      } else if (label.startsWith('raise to')) {
        const to = Number(label.replace(/[^0-9]/g, ''));
        raises.push({ idx: i, to, label });
      } else if (label.startsWith('all-in')) {
        allInIdx = i;
        allInTo = Number(label.replace(/[^0-9]/g, ''));
      }
    });

    this.controlsEl.replaceChildren(...buttons);

    if (raises.length || allInIdx >= 0) {
      this.controlsEl.append(this.raiserWidget(raises, allInIdx, allInTo));
    }
  }

  private btn(text: string, kind: string, index: number): HTMLButtonElement {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = `pk-btn ${kind}`;
    b.textContent = text;
    b.onclick = () => this.submit(index);
    return b;
  }

  /** A slider over the offered raise sizes plus all-in; the button submits the
   * selected size. Snaps to the nearest offered `Raise(to)` (or all-in). */
  private raiserWidget(
    raises: { idx: number; to: number; label: string }[],
    allInIdx: number,
    allInTo: number,
  ): HTMLElement {
    // Build the ladder of selectable sizes (sorted), each mapped to a submit
    // index. All-in is the top rung.
    const rungs = [...raises].sort((a, b) => a.to - b.to);
    if (allInIdx >= 0 && !rungs.some((r) => r.to === allInTo)) {
      rungs.push({ idx: allInIdx, to: allInTo, label: 'all-in' });
    } else if (allInIdx >= 0) {
      // Replace the equal-sized raise with all-in (same chips, clearer label).
      const k = rungs.findIndex((r) => r.to === allInTo);
      rungs[k] = { idx: allInIdx, to: allInTo, label: 'all-in' };
    }
    rungs.sort((a, b) => a.to - b.to);

    const wrap = document.createElement('div');
    wrap.className = 'pk-raiser';
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = '0';
    slider.max = String(rungs.length - 1);
    slider.step = '1';
    slider.value = String(Math.min(rungs.length - 1, Math.max(0, Math.floor(rungs.length / 2))));
    const amt = document.createElement('span');
    amt.className = 'pk-amt';
    const go = document.createElement('button');
    go.type = 'button';
    go.className = 'pk-btn raise';

    const quick = document.createElement('div');
    quick.className = 'pk-quick';
    const setTo = (i: number) => {
      slider.value = String(i);
      update();
    };
    [
      ['min', 0],
      ['½', Math.max(0, Math.round((rungs.length - 1) * 0.25))],
      ['pot', Math.max(0, Math.round((rungs.length - 1) * 0.6))],
      ['max', rungs.length - 1],
    ].forEach(([lbl, idx]) => {
      const q = document.createElement('button');
      q.type = 'button';
      q.textContent = String(lbl);
      q.onclick = () => setTo(idx as number);
      quick.append(q);
    });

    const update = () => {
      const rung = rungs[Number(slider.value)];
      const isShove = rung.label === 'all-in';
      amt.textContent = `${rung.to}`;
      go.textContent = isShove ? `All-in ${rung.to}` : `Raise to ${rung.to}`;
    };
    slider.oninput = update;
    go.onclick = () => this.submit(rungs[Number(slider.value)].idx);
    update();

    wrap.append(quick, slider, amt, go);
    return wrap;
  }
}

export function createPokerFrontend(): GameFrontend {
  return new PokerFrontend();
}
