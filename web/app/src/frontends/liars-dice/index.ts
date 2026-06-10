// Liar's Dice frontend: players around a felt table, CSS pip dice, the
// current bid center-table with the round's bid ladder, and a choreographed
// LIAR/EXACT reveal. The view/transition JSON schemas are the private
// contract with games/liars-dice/src/ui.rs.

import type { MatchEventData, ViewState } from '../../engine/protocol';
import type { FrontendCtx, GameFrontend } from '../types';
import { sleep } from '../types';
import { STYLE, STYLE_ID } from './style';

interface LdBid {
  qty: number;
  face: number;
  by: number;
  forced: boolean;
}

interface LdSeatView {
  seat: number;
  alive: boolean;
  count: number;
  /** Die values, or `null` when this hand is hidden from the viewer. */
  dice: number[] | null;
}

interface LdHistoryEntry {
  seat: number;
  qty: number;
  face: number;
}

interface LdView {
  players: number;
  dice: number;
  faces: number;
  viewer: number;
  spectator: boolean;
  phase: 'rolling' | 'bidding' | 'over';
  round: number;
  totalDice: number;
  turn: number;
  winner: number | null;
  bid: LdBid | null;
  history: LdHistoryEntry[];
  hands: LdSeatView[];
}

interface LdReveal {
  kind: 'liar' | 'exact';
  caller: number;
  bidder: number;
  bid: { qty: number; face: number };
  actual: number;
  hands: number[][];
  loser: number | null;
  diceLeft: number[];
  gameOver: boolean;
  winner: number | null;
  nextRound: number;
}

type BannerTone = 'liar' | 'exact' | 'good' | 'info';

function isView(v: unknown): v is LdView {
  return typeof v === 'object' && v !== null && Array.isArray((v as LdView).hands);
}

function isReveal(d: unknown): d is LdReveal {
  if (typeof d !== 'object' || d === null) return false;
  const k = (d as LdReveal).kind;
  return k === 'liar' || k === 'exact';
}

function dieHtml(value: number, extra = ''): string {
  return `<span class="ld-die${extra}" data-v="${value}">${'<i></i>'.repeat(9)}</span>`;
}

function cupHtml(count: number): string {
  return `<span class="ld-cup"><span class="ld-cup-count">${count}</span></span>`;
}

/** Seat coordinates (percent of the table) around an oval, with display
 * index 0 — the human, or seat 0 when spectating — at bottom center and
 * turn order proceeding clockwise. */
function seatPos(displayIndex: number, n: number): { x: number; y: number } {
  const a = (Math.PI / 180) * (90 + (360 * displayIndex) / n);
  return { x: 50 + 39 * Math.cos(a), y: 50 + 36 * Math.sin(a) };
}

class LiarsDiceFrontend implements GameFrontend {
  private ctx!: FrontendCtx;
  private tableEl!: HTMLElement;
  private seatsEl!: HTMLElement;
  private centerEl!: HTMLElement;
  private bannerEl!: HTMLElement;
  private controlsEl!: HTMLElement;
  private view: LdView | null = null;
  private ladder: LdHistoryEntry[] = [];
  private ladderRound = -1;
  private dead = false;
  private openQty = 1;
  private openFace = 1;

  mount(host: HTMLElement, ctx: FrontendCtx): void {
    this.ctx = ctx;
    if (!document.getElementById(STYLE_ID)) {
      const style = document.createElement('style');
      style.id = STYLE_ID;
      style.textContent = STYLE;
      document.head.append(style);
    }
    host.innerHTML = `
      <div class="ld-root">
        <div class="ld-table">
          <div class="ld-felt"></div>
          <div class="ld-center">
            <div class="ld-round"></div>
            <div class="ld-bid-box"></div>
            <div class="ld-ladder"></div>
          </div>
          <div class="ld-seats"></div>
          <div class="ld-banner"></div>
        </div>
        <div class="ld-controls"></div>
      </div>`;
    this.tableEl = host.querySelector('.ld-table')!;
    this.seatsEl = host.querySelector('.ld-seats')!;
    this.centerEl = host.querySelector('.ld-center')!;
    this.bannerEl = host.querySelector('.ld-banner')!;
    this.controlsEl = host.querySelector('.ld-controls')!;
  }

  render(state: ViewState): void {
    if (!isView(state.viewData)) {
      const pre = document.createElement('pre');
      pre.className = 'ld-fallback';
      pre.textContent = state.view;
      this.seatsEl.replaceChildren(pre);
      return;
    }
    const view = state.viewData;
    this.view = view;
    this.syncLadder(view);
    this.renderSeats(view);
    this.renderCenter(view);
    if (state.toAct !== state.humanSeat || state.isOver) this.controlsEl.replaceChildren();
  }

  async animate(event: MatchEventData, after: ViewState): Promise<void> {
    const scale = this.ctx.animationScale();
    if (isReveal(event.data)) {
      if (scale > 0 && !this.dead) await this.playReveal(event.data, scale);
      this.render(after);
      return;
    }
    if (scale > 0 && !this.dead) await this.animateBid(event.seat, after, scale);
    this.render(after);
    if (scale > 0 && !this.dead) {
      this.centerEl
        .querySelector('.ld-bid-main')
        ?.animate(
          [{ transform: 'scale(0.8)' }, { transform: 'scale(1.12)' }, { transform: 'scale(1)' }],
          { duration: 300 * scale, easing: 'ease-out' },
        );
      await sleep(150 * scale);
    }
  }

  promptAction(labels: string[]): void {
    if (this.ctx.humanSeat < 0) return;
    if (labels.some((l) => l.startsWith('open '))) this.renderOpenControls(labels);
    else this.renderResponseControls(labels);
  }

  unmount(): void {
    this.dead = true;
  }

  // ---------- rendering ----------

  private name(seat: number): string {
    return seat === this.ctx.humanSeat ? 'You' : `Player ${seat}`;
  }

  private syncLadder(view: LdView): void {
    if (view.round !== this.ladderRound) {
      this.ladderRound = view.round;
      this.ladder = [...view.history];
    } else if (view.history.length > this.ladder.length) {
      this.ladder = [...view.history];
    } else if (view.bid && !view.bid.forced) {
      const last = this.ladder[this.ladder.length - 1];
      if (!last || last.qty !== view.bid.qty || last.face !== view.bid.face) {
        this.ladder.push({ seat: view.bid.by, qty: view.bid.qty, face: view.bid.face });
      }
    }
  }

  private handHtml(hand: LdSeatView): string {
    if (!hand.alive) return '<span class="ld-out-mark">×</span>';
    if (hand.dice === null || hand.dice.length === 0) return cupHtml(hand.count);
    return hand.dice.map((d) => dieHtml(d)).join('');
  }

  private renderSeats(view: LdView): void {
    const n = view.players;
    const anchor = this.ctx.humanSeat >= 0 ? this.ctx.humanSeat : 0;
    const parts: string[] = [];
    for (const hand of view.hands) {
      const pos = seatPos((hand.seat - anchor + n) % n, n);
      const classes = ['ld-seat'];
      if (!hand.alive) classes.push('ld-out');
      if (hand.alive && view.phase === 'bidding' && view.turn === hand.seat)
        classes.push('ld-turn');
      if (hand.alive && view.phase === 'rolling') classes.push('ld-roll');
      const won = view.phase === 'over' && view.winner === hand.seat;
      if (won) classes.push('ld-winner');
      const bubble =
        view.bid && !view.bid.forced && view.phase === 'bidding' && view.bid.by === hand.seat
          ? `<span class="ld-bubble">${view.bid.qty}×${dieHtml(view.bid.face)}</span>`
          : '';
      const crown = won ? '<span class="ld-crown">★</span>' : '';
      const tag = hand.alive
        ? `<span class="ld-tag">${hand.count} ${hand.count === 1 ? 'die' : 'dice'}</span>`
        : '<span class="ld-out-tag">OUT</span>';
      parts.push(`
        <div class="${classes.join(' ')}" data-seat="${hand.seat}"
             style="left:${pos.x.toFixed(2)}%;top:${pos.y.toFixed(2)}%">
          <div class="ld-pod">
            ${bubble}
            <div class="ld-hand">${this.handHtml(hand)}</div>
            <div class="ld-name">${crown}${this.name(hand.seat)} ${tag}</div>
          </div>
        </div>`);
    }
    this.seatsEl.innerHTML = parts.join('');
  }

  private renderCenter(view: LdView): void {
    const round = this.centerEl.querySelector('.ld-round')!;
    const bidBox = this.centerEl.querySelector('.ld-bid-box')!;
    const ladder = this.centerEl.querySelector('.ld-ladder')!;
    round.textContent = `round ${view.round} · ${view.totalDice} dice in play`;
    if (view.phase === 'over' && view.winner !== null) {
      const verb = view.winner === this.ctx.humanSeat ? 'win' : 'wins';
      bidBox.innerHTML = `<span class="ld-win-text">★ ${this.name(view.winner)} ${verb}</span>`;
    } else if (view.phase === 'rolling') {
      bidBox.innerHTML = '<span class="ld-open-hint">shaking the cups…</span>';
    } else if (view.bid) {
      const verb = view.bid.by === this.ctx.humanSeat ? 'bid' : 'bids';
      const sub = view.bid.forced ? 'forced opening bid' : `${this.name(view.bid.by)} ${verb}`;
      bidBox.innerHTML = `
        <div class="ld-bid-main">${view.bid.qty}<span class="ld-x">×</span>${dieHtml(view.bid.face)}</div>
        <div class="ld-bid-sub">${sub}</div>`;
    } else {
      const verb = view.turn === this.ctx.humanSeat ? 'open' : 'opens';
      bidBox.innerHTML = `<span class="ld-open-hint">${this.name(view.turn)} ${verb} the round…</span>`;
    }
    const rungs = this.ladder.slice(-6);
    ladder.innerHTML = rungs
      .map((r, i) => {
        const who = r.seat === this.ctx.humanSeat ? 'you' : `P${r.seat}`;
        const now = i === rungs.length - 1 ? ' ld-rung-now' : '';
        return `<div class="ld-rung${now}"><span>${who}</span> ${r.qty}×${dieHtml(r.face)}</div>`;
      })
      .join('');
  }

  // ---------- controls ----------

  private submit(index: number): void {
    for (const b of this.controlsEl.querySelectorAll('button')) b.disabled = true;
    this.ctx.submit(String(index));
  }

  private renderResponseControls(labels: string[]): void {
    const bid = this.view?.bid;
    const faces = this.view?.faces ?? 6;
    const buttons = labels.map((label, i) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'ld-btn';
      if (label === 'raise quantity' && bid) {
        b.innerHTML = `Raise to ${bid.qty + 1}×${dieHtml(bid.face)}`;
      } else if (label === 'raise face' && bid) {
        const [q, f] = bid.face < faces ? [bid.qty, bid.face + 1] : [bid.qty + 1, 1];
        b.innerHTML = `Raise to ${q}×${dieHtml(f)}`;
      } else if (label === 'call LIAR') {
        b.classList.add('ld-btn-liar');
        b.textContent = 'LIAR!';
      } else if (label === 'call EXACT') {
        b.classList.add('ld-btn-exact');
        b.textContent = 'EXACT';
      } else {
        b.textContent = label;
      }
      b.onclick = () => this.submit(i);
      return b;
    });
    this.controlsEl.replaceChildren(...buttons);
  }

  private renderOpenControls(labels: string[]): void {
    const byBid = new Map<string, number>();
    let maxQty = 1;
    for (const [i, label] of labels.entries()) {
      const m = /^open (\d+)x(\d+)$/.exec(label);
      if (!m) continue;
      byBid.set(`${m[1]}x${m[2]}`, i);
      maxQty = Math.max(maxQty, Number(m[1]));
    }
    const faces = this.view?.faces ?? 6;
    const own = this.view?.hands.find((h) => h.seat === this.ctx.humanSeat)?.dice ?? [];
    const tally = new Array<number>(faces + 1).fill(0);
    for (const d of own) tally[d]++;
    let bestFace = 1;
    for (let f = 1; f <= faces; f++) if (tally[f] >= tally[bestFace]) bestFace = f;
    this.openFace = bestFace;
    this.openQty = Math.min(maxQty, Math.max(1, tally[bestFace]));

    const panel = document.createElement('div');
    panel.className = 'ld-open';
    panel.innerHTML = `
      <span class="ld-open-label">open the round</span>
      <div class="ld-qty">
        <button type="button" class="ld-step ld-minus">−</button>
        <span class="ld-qty-n"></span>
        <button type="button" class="ld-step ld-plus">+</button>
      </div>
      <div class="ld-faces"></div>
      <button type="button" class="ld-btn ld-go"></button>`;
    const qtyEl = panel.querySelector<HTMLElement>('.ld-qty-n')!;
    const facesEl = panel.querySelector<HTMLElement>('.ld-faces')!;
    const goBtn = panel.querySelector<HTMLButtonElement>('.ld-go')!;
    const minus = panel.querySelector<HTMLButtonElement>('.ld-minus')!;
    const plus = panel.querySelector<HTMLButtonElement>('.ld-plus')!;
    const faceButtons: HTMLButtonElement[] = [];
    const update = () => {
      qtyEl.textContent = String(this.openQty);
      minus.disabled = this.openQty <= 1;
      plus.disabled = this.openQty >= maxQty;
      faceButtons.forEach((fb, i) => fb.classList.toggle('ld-sel', i + 1 === this.openFace));
      goBtn.innerHTML = `Bid ${this.openQty}×${dieHtml(this.openFace)}`;
      goBtn.disabled = !byBid.has(`${this.openQty}x${this.openFace}`);
    };
    for (let f = 1; f <= faces; f++) {
      const fb = document.createElement('button');
      fb.type = 'button';
      fb.className = 'ld-face-btn';
      fb.innerHTML = dieHtml(f);
      fb.onclick = () => {
        this.openFace = f;
        update();
      };
      faceButtons.push(fb);
      facesEl.append(fb);
    }
    minus.onclick = () => {
      this.openQty = Math.max(1, this.openQty - 1);
      update();
    };
    plus.onclick = () => {
      this.openQty = Math.min(maxQty, this.openQty + 1);
      update();
    };
    goBtn.onclick = () => {
      const idx = byBid.get(`${this.openQty}x${this.openFace}`);
      if (idx !== undefined) this.submit(idx);
    };
    update();
    this.controlsEl.replaceChildren(panel);
  }

  // ---------- animation ----------

  private showBanner(text: string, tone: BannerTone): void {
    this.bannerEl.textContent = text;
    this.bannerEl.className = `ld-banner ld-banner-${tone} ld-show`;
  }

  private hideBanner(): void {
    this.bannerEl.classList.remove('ld-show');
  }

  /** A bid chip flies from the bidder's seat to center table. */
  private async animateBid(seat: number, after: ViewState, scale: number): Promise<void> {
    const bid = isView(after.viewData) ? after.viewData.bid : null;
    const seatEl = this.seatsEl.querySelector<HTMLElement>(`[data-seat="${seat}"]`);
    if (!bid || !seatEl) return;
    const chip = document.createElement('div');
    chip.className = 'ld-fly';
    chip.innerHTML = `${bid.qty}×${dieHtml(bid.face)}`;
    chip.style.left = seatEl.style.left;
    chip.style.top = seatEl.style.top;
    this.tableEl.append(chip);
    const table = this.tableEl.getBoundingClientRect();
    const dx = ((50 - parseFloat(seatEl.style.left)) / 100) * table.width;
    const dy = ((46 - parseFloat(seatEl.style.top)) / 100) * table.height;
    const anim = chip.animate(
      [
        { transform: 'translate(-50%, -50%)', opacity: 1 },
        { transform: `translate(calc(-50% + ${dx}px), calc(-50% + ${dy}px))`, opacity: 0.15 },
      ],
      { duration: 480 * scale, easing: 'cubic-bezier(0.3, 0.7, 0.4, 1)', fill: 'forwards' },
    );
    await anim.finished.catch(() => undefined);
    chip.remove();
  }

  /** Replace the center bid with a running count of the called face. */
  private setTally(count: number | null, face: number, target: number): void {
    const bidBox = this.centerEl.querySelector('.ld-bid-box');
    if (!bidBox) return;
    bidBox.innerHTML = `
      <div class="ld-bid-main">
        <span class="ld-tally-n">${count ?? '?'}</span><span class="ld-x">/</span>${target}<span class="ld-x">×</span>${dieHtml(face)}
      </div>
      <div class="ld-bid-sub">counting ${face}s across the table…</div>`;
  }

  /** The LIAR/EXACT moment: announce the call, flip every cup open around
   * the table while the count tallies up, then the verdict — the loser
   * flashes red and drops a die, eliminations and the winner get their own
   * beat. */
  private async playReveal(r: LdReveal, scale: number): Promise<void> {
    const t = (ms: number) => ms * scale;
    const n = r.hands.length;
    const face = r.bid.face;
    const human = this.ctx.humanSeat;

    const callVerb = r.caller === human ? 'call' : 'calls';
    const callName = r.kind === 'liar' ? 'LIAR' : 'EXACT';
    this.showBanner(
      `${this.name(r.caller)} ${callVerb} ${callName} on ${r.bid.qty}×${face}!`,
      r.kind === 'liar' ? 'liar' : 'exact',
    );
    this.setTally(null, face, r.bid.qty);
    await sleep(t(900));
    if (this.dead) return;

    let tally = 0;
    for (let off = 0; off < n; off++) {
      const seat = (r.caller + off) % n;
      const dice = r.hands[seat];
      if (!dice.length) continue;
      const handEl = this.seatsEl.querySelector<HTMLElement>(`[data-seat="${seat}"] .ld-hand`);
      if (handEl) {
        handEl.innerHTML = dice
          .map((d) => dieHtml(d, d === face ? ' ld-hit ld-flip' : ' ld-flip'))
          .join('');
      }
      tally += dice.filter((d) => d === face).length;
      this.setTally(tally, face, r.bid.qty);
      await sleep(t(380));
      if (this.dead) return;
    }
    await sleep(t(250));
    if (this.dead) return;

    const loseText =
      r.loser === null
        ? ''
        : ` ${this.name(r.loser)} ${r.loser === human ? 'lose' : 'loses'} a die.`;
    let verdict: string;
    if (r.kind === 'liar') {
      verdict =
        r.actual < r.bid.qty
          ? `A lie — only ${r.actual}!${loseText}`
          : `The bid was good — ${r.actual} on the table.${loseText}`;
    } else {
      verdict =
        r.loser === null
          ? `EXACT — dead on ${r.actual}! Nobody loses a die.`
          : `Not exact — ${r.actual}, not ${r.bid.qty}.${loseText}`;
    }
    this.showBanner(verdict, r.loser === null ? 'good' : 'liar');
    if (r.loser !== null) {
      const seatEl = this.seatsEl.querySelector<HTMLElement>(`[data-seat="${r.loser}"]`);
      seatEl?.classList.add('ld-lose');
      const float = document.createElement('span');
      float.className = 'ld-float';
      float.textContent = '−1 die';
      seatEl?.querySelector('.ld-pod')?.append(float);
    } else {
      this.seatsEl.querySelector(`[data-seat="${r.caller}"]`)?.classList.add('ld-safe');
    }
    await sleep(t(1100));
    if (this.dead) return;

    if (r.loser !== null && r.diceLeft[r.loser] === 0 && !r.gameOver) {
      this.showBanner(
        `${this.name(r.loser)} ${r.loser === human ? 'are' : 'is'} out of the game!`,
        'liar',
      );
      await sleep(t(900));
      if (this.dead) return;
    }
    if (r.gameOver && r.winner !== null) {
      this.seatsEl.querySelector(`[data-seat="${r.winner}"]`)?.classList.add('ld-winner');
      this.showBanner(
        `★ ${this.name(r.winner)} ${r.winner === human ? 'win' : 'wins'} the game!`,
        'good',
      );
      await sleep(t(1200));
    }
    this.hideBanner();
  }
}

export function createLiarsDiceFrontend(): GameFrontend {
  return new LiarsDiceFrontend();
}
