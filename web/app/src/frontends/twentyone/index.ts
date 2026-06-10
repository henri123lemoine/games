// Twenty-One frontend: a card-table for the hearts duel. Each round both
// players get one open card and one hole card, then alternate draw/stand;
// closest to 21 wins and the loser burns hearts equal to the round number.
//
// View schema (games/twentyone/src/ui.rs::view_data):
//   {round, hearts: [h0,h1], maxHearts, roundActive, deckCount, toAct,
//    players: [{up: [..], down: n|null, total: n|null, stood}, ..],
//    lastReveal: {downs: [d0,d1], up: [[..],[..]]} | null}
// Transition schema (transition_data):
//   {kind: "draw"|"stand", seat} |
//   {kind: "roundEnd", seat, downs, totals, winner, damage, hearts}
// Drawn cards resolve at a chance node after the draw event, so card reveals
// are animated from view diffs (new cards get an enter animation).

import type { MatchEventData, ViewState } from '../../engine/protocol';
import type { FrontendCtx, GameFrontend } from '../types';
import { sleep } from '../types';

interface T21Player {
  up: number[];
  down: number | null;
  total: number | null;
  stood: boolean;
}

interface T21View {
  round: number;
  hearts: [number, number];
  maxHearts: number;
  roundActive: boolean;
  deckCount: number;
  toAct: number | null;
  players: [T21Player, T21Player];
  lastReveal: { downs: [number, number]; up: [number[], number[]] } | null;
}

type T21Event =
  | { kind: 'draw'; seat: number }
  | { kind: 'stand'; seat: number }
  | {
      kind: 'roundEnd';
      seat: number;
      downs: [number, number];
      totals: [number, number];
      winner: number | null;
      damage: number;
      hearts: [number, number];
    };

interface SeatEls {
  bar: HTMLElement;
  name: HTMLElement;
  hearts: HTMLElement;
  badge: HTMLElement;
  cards: HTMLElement;
  total: HTMLElement;
}

const STYLE_ID = 'twentyone-frontend-style';

const CSS = `
.t21 { display: flex; flex-direction: column; gap: 14px; }
.t21-table { position: relative; display: flex; flex-direction: column; gap: 12px;
  border-radius: calc(var(--radius) + 4px); padding: 18px 20px;
  background: radial-gradient(120% 95% at 50% 0%, #17463a 0%, #0f3328 55%, #0a2419 100%);
  border: 1px solid var(--border);
  box-shadow: inset 0 0 70px rgba(0, 0, 0, .45), 0 14px 40px rgba(0, 0, 0, .45); }
.t21-seat { display: flex; flex-direction: column; gap: 10px; }
.t21-seat[data-pos="bottom"] { flex-direction: column-reverse; }
.t21-seat-bar { display: flex; align-items: center; gap: 12px; width: fit-content;
  padding: 6px 14px; border-radius: 999px; background: rgba(1, 4, 9, .4);
  border: 1px solid transparent; transition: border-color .2s, box-shadow .2s; }
.t21-seat-bar.t21-active { border-color: var(--accent);
  box-shadow: 0 0 14px rgba(88, 166, 255, .28); }
.t21-name { font-weight: 600; }
.t21-hearts { letter-spacing: 2px; font-size: 15px; line-height: 1; }
.t21-heart { color: #ff5d6c; text-shadow: 0 0 6px rgba(255, 93, 108, .45);
  display: inline-block; }
.t21-heart.t21-lost { color: rgba(230, 237, 243, .16); text-shadow: none; }
.t21-heart-break { animation: t21-heart-break .8s ease-in forwards; }
@keyframes t21-heart-break {
  40% { transform: scale(1.45); }
  to { transform: scale(.3); opacity: 0; }
}
.t21-badge { font-size: 10px; font-weight: 700; letter-spacing: .14em;
  text-transform: uppercase; color: var(--accent-2); padding: 2px 8px;
  border: 1px solid var(--accent-2); border-radius: 999px; }
.t21-cards { display: flex; gap: 10px; flex-wrap: wrap; align-items: center;
  min-height: 88px; }
.t21-card { position: relative; width: clamp(46px, 9vw, 60px); aspect-ratio: 5 / 7;
  border-radius: 8px; background: linear-gradient(160deg, #fdfcf7, #e9e6d8);
  color: #222b3a; display: grid; place-items: center; font-weight: 700;
  font-size: clamp(18px, 3.4vw, 24px);
  box-shadow: 0 4px 10px rgba(0, 0, 0, .4), inset 0 0 0 1px rgba(0, 0, 0, .1); }
.t21-pip { position: absolute; top: 3px; left: 6px; font-size: 11px; font-weight: 600; }
.t21-pip-br { top: auto; left: auto; bottom: 3px; right: 6px; transform: rotate(180deg); }
.t21-back { background:
  repeating-linear-gradient(45deg, rgba(88, 166, 255, .16) 0 5px, transparent 5px 10px),
  repeating-linear-gradient(-45deg, rgba(188, 140, 255, .12) 0 5px, transparent 5px 10px),
  linear-gradient(#233048, #1b2740);
  box-shadow: 0 4px 10px rgba(0, 0, 0, .4), inset 0 0 0 2px rgba(88, 166, 255, .25);
  color: rgba(88, 166, 255, .75); font-size: clamp(15px, 2.6vw, 19px); }
.t21-hole { outline: 2px dashed rgba(88, 166, 255, .5); outline-offset: 2px; }
.t21-hole::after { content: 'hole'; position: absolute; bottom: 2px; left: 50%;
  transform: translateX(-50%); font-size: 8px; font-weight: 600;
  letter-spacing: .14em; text-transform: uppercase; color: var(--accent); }
.t21-spent { opacity: .55; filter: saturate(.7); }
.t21-deal { animation: t21-deal .32s cubic-bezier(.2, .8, .3, 1.15) backwards; }
@keyframes t21-deal {
  from { transform: translateY(-20px) rotate(4deg) scale(.72); opacity: 0; }
}
.t21-flip { animation: t21-flip .5s ease both; }
@keyframes t21-flip { from { transform: rotateY(90deg); } }
.t21-placeholder { color: var(--text-dim); font-size: 13px; font-style: italic; }
.t21-total { width: fit-content; min-height: 27px; padding: 4px 12px;
  border-radius: 999px; background: rgba(1, 4, 9, .45); border: 1px solid var(--border);
  font-size: 13px; color: var(--text-dim); }
.t21-total b { color: var(--text); font-size: 15px; }
.t21-total.t21-bust b { color: var(--bad); }
.t21-total.t21-sweet b { color: #ffd566; text-shadow: 0 0 8px rgba(255, 213, 102, .5); }
.t21-mid { display: flex; align-items: center; justify-content: space-between;
  gap: 14px; padding: 2px 2px; }
.t21-round b { font-size: 15px; display: block; }
.t21-stake { font-size: 12px; color: var(--text-dim); }
.t21-deck { display: flex; align-items: center; gap: 10px; }
.t21-deck-pile { position: relative; width: 34px; height: 46px; }
.t21-deck-pile i { position: absolute; inset: 0; border-radius: 6px;
  background: linear-gradient(#233048, #1b2740);
  box-shadow: inset 0 0 0 1.5px rgba(88, 166, 255, .3), 0 2px 5px rgba(0, 0, 0, .4); }
.t21-deck-pile i:nth-child(1) { transform: translate(-3px, 2px) rotate(-4deg); }
.t21-deck-pile i:nth-child(3) { transform: translate(3px, -2px) rotate(3deg); }
.t21-deck-pulse { animation: t21-deck-pulse .26s ease; }
@keyframes t21-deck-pulse { 50% { transform: scale(1.12); } }
.t21-deck-count { font-size: 12px; color: var(--text-dim); white-space: nowrap; }
.t21-banner { position: absolute; inset: 0; display: grid; place-items: center;
  pointer-events: none; z-index: 3; }
.t21-banner[hidden] { display: none; }
.t21-banner-chip { max-width: 82%; text-align: center; padding: 12px 26px;
  border-radius: 14px; background: rgba(1, 4, 9, .85); backdrop-filter: blur(4px);
  border: 1px solid var(--border); font-weight: 700; font-size: 17px;
  animation: t21-pop .35s cubic-bezier(.2, .9, .3, 1.3) backwards; }
.t21-banner-good .t21-banner-chip { border-color: var(--good); color: var(--good); }
.t21-banner-bad .t21-banner-chip { border-color: var(--bad); color: var(--bad); }
@keyframes t21-pop { from { transform: scale(.7); opacity: 0; } }
.t21-actions { display: flex; gap: 12px; justify-content: center; min-height: 64px; }
.t21-btn { flex: 1 1 0; max-width: 230px; padding: 12px 18px; border-radius: 14px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--text);
  font-weight: 700; font-size: 17px; letter-spacing: .07em; text-transform: uppercase;
  transition: transform .08s, filter .15s, box-shadow .15s; }
.t21-btn:hover:not(:disabled) { filter: brightness(1.15); }
.t21-btn:active:not(:disabled) { transform: translateY(2px) scale(.985); }
.t21-btn:disabled { opacity: .45; cursor: default; }
.t21-btn-draw { background: linear-gradient(135deg, var(--accent), var(--accent-2));
  border: none; color: #04111f; box-shadow: 0 6px 18px rgba(88, 166, 255, .3); }
.t21-btn span { display: block; font-size: 11px; font-weight: 500; letter-spacing: 0;
  text-transform: none; opacity: .8; }
.t21-stand-flash { animation: t21-stand-flash .4s ease; }
@keyframes t21-stand-flash { 30% { box-shadow: 0 0 0 2px var(--accent-2); } }
@media (max-width: 520px) {
  .t21-table { padding: 12px; }
  .t21-cards { gap: 6px; min-height: 72px; }
}
`;

function ensureStyle(): void {
  if (document.getElementById(STYLE_ID)) return;
  const el = document.createElement('style');
  el.id = STYLE_ID;
  el.textContent = CSS;
  document.head.append(el);
}

const SEAT_HTML = `
  <div class="t21-seat-bar">
    <span class="t21-name"></span>
    <span class="t21-hearts"></span>
    <span class="t21-badge" hidden>stood</span>
  </div>
  <div class="t21-cards"></div>
  <div class="t21-total"></div>`;

function sum(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0);
}

class TwentyOneFrontend implements GameFrontend {
  private ctx!: FrontendCtx;
  private seatEls: SeatEls[] = [];
  private roundEl!: HTMLElement;
  private stakeEl!: HTMLElement;
  private deckEl!: HTMLElement;
  private deckPile!: HTMLElement;
  private deckCountEl!: HTMLElement;
  private bannerEl!: HTMLElement;
  private bannerChip!: HTMLElement;
  private actionsEl!: HTMLElement;

  private prevCounts = [0, 0];
  private lastRound = 0;
  private prevHearts: [number, number] | null = null;
  private roundEndSeen = false;

  mount(host: HTMLElement, ctx: FrontendCtx): void {
    this.ctx = ctx;
    ensureStyle();
    host.innerHTML = `
      <div class="t21">
        <div class="t21-table">
          <section class="t21-seat" data-pos="top">${SEAT_HTML}</section>
          <div class="t21-mid">
            <div class="t21-round"><b></b><span class="t21-stake"></span></div>
            <div class="t21-deck">
              <span class="t21-deck-count"></span>
              <div class="t21-deck-pile"><i></i><i></i><i></i></div>
            </div>
          </div>
          <section class="t21-seat" data-pos="bottom">${SEAT_HTML}</section>
          <div class="t21-banner" hidden><div class="t21-banner-chip"></div></div>
        </div>
        <div class="t21-actions"></div>
      </div>`;
    const grab = (pos: string): SeatEls => {
      const s = host.querySelector<HTMLElement>(`[data-pos="${pos}"]`)!;
      return {
        bar: s.querySelector('.t21-seat-bar')!,
        name: s.querySelector('.t21-name')!,
        hearts: s.querySelector('.t21-hearts')!,
        badge: s.querySelector('.t21-badge')!,
        cards: s.querySelector('.t21-cards')!,
        total: s.querySelector('.t21-total')!,
      };
    };
    const bottomSeat = ctx.humanSeat >= 0 ? ctx.humanSeat : 0;
    this.seatEls = [];
    this.seatEls[bottomSeat] = grab('bottom');
    this.seatEls[1 - bottomSeat] = grab('top');
    this.roundEl = host.querySelector('.t21-round b')!;
    this.stakeEl = host.querySelector('.t21-stake')!;
    this.deckEl = host.querySelector('.t21-deck')!;
    this.deckPile = host.querySelector('.t21-deck-pile')!;
    this.deckCountEl = host.querySelector('.t21-deck-count')!;
    this.bannerEl = host.querySelector('.t21-banner')!;
    this.bannerChip = host.querySelector('.t21-banner-chip')!;
    this.actionsEl = host.querySelector('.t21-actions')!;
    if (ctx.humanSeat < 0) this.actionsEl.style.display = 'none';
  }

  private seatName(seat: number): string {
    if (seat === this.ctx.humanSeat) return 'You';
    return this.ctx.humanSeat >= 0 ? 'Bot' : `Player ${seat}`;
  }

  private cardEl(value: number | null, hole: boolean): HTMLElement {
    const el = document.createElement('div');
    if (value === null) {
      el.className = 't21-card t21-back t21-hole';
      el.textContent = '?';
    } else {
      el.className = hole ? 't21-card t21-hole' : 't21-card';
      el.innerHTML = `<span class="t21-pip">${value}</span><b>${value}</b><span class="t21-pip t21-pip-br">${value}</span>`;
    }
    return el;
  }

  private setTotal(el: HTMLElement, label: string, n: number | null): void {
    if (n === null) {
      el.innerHTML = '&nbsp;';
      el.classList.remove('t21-bust', 't21-sweet');
      return;
    }
    el.innerHTML = `${label} <b>${n}</b>`;
    el.classList.toggle('t21-bust', n > 21);
    el.classList.toggle('t21-sweet', n === 21);
  }

  private renderHearts(el: HTMLElement, have: number, max: number): void {
    const spans: HTMLElement[] = [];
    for (let i = 0; i < max; i++) {
      const s = document.createElement('span');
      s.className = i < have ? 't21-heart' : 't21-heart t21-lost';
      s.textContent = '♥';
      spans.push(s);
    }
    el.replaceChildren(...spans);
  }

  private renderSeat(seat: number, v: T21View, state: ViewState): void {
    const els = this.seatEls[seat];
    const p = v.players[seat];
    const scale = this.ctx.animationScale();
    els.name.textContent = this.seatName(seat);
    this.renderHearts(els.hearts, v.hearts[seat], v.maxHearts);
    els.badge.hidden = !(v.roundActive && p.stood);
    els.bar.classList.toggle(
      't21-active',
      v.roundActive && !state.isOver && v.toAct === seat,
    );
    if (v.roundActive) {
      let pos = 0;
      const dealt = this.prevCounts[seat];
      const mk = (value: number | null, hole: boolean): HTMLElement => {
        const el = this.cardEl(value, hole);
        if (pos >= dealt && scale > 0) {
          el.classList.add('t21-deal');
          el.style.animationDuration = `${320 * scale}ms`;
          el.style.animationDelay = `${(pos - dealt) * 80 * scale}ms`;
        }
        pos++;
        return el;
      };
      const cards = [mk(p.up[0] ?? null, false), mk(p.down, true)];
      for (const c of p.up.slice(1)) cards.push(mk(c, false));
      els.cards.replaceChildren(...cards);
      if (p.total !== null) this.setTotal(els.total, 'total', p.total);
      else this.setTotal(els.total, 'showing', sum(p.up));
      this.prevCounts[seat] = p.up.length + 1;
    } else if (v.lastReveal) {
      const ups = v.lastReveal.up[seat];
      const down = v.lastReveal.downs[seat];
      const cards = [this.cardEl(ups[0] ?? null, false), this.cardEl(down, true)];
      for (const c of ups.slice(1)) cards.push(this.cardEl(c, false));
      for (const c of cards) c.classList.add('t21-spent');
      els.cards.replaceChildren(...cards);
      this.setTotal(els.total, 'total', sum(ups) + down);
      this.prevCounts[seat] = 0;
    } else {
      const ph = document.createElement('div');
      ph.className = 't21-placeholder';
      ph.textContent = 'waiting for the deal…';
      els.cards.replaceChildren(ph);
      this.setTotal(els.total, '', null);
      this.prevCounts[seat] = 0;
    }
  }

  private showBanner(text: string, cls: '' | 'good' | 'bad'): void {
    this.bannerChip.textContent = text;
    this.bannerEl.className = `t21-banner${cls ? ` t21-banner-${cls}` : ''}`;
    this.bannerEl.hidden = false;
  }

  private endText(winner: number | null, damage: number, round: number): string {
    if (winner === null) return `Round ${round}: push — no damage`;
    const who =
      winner === this.ctx.humanSeat
        ? 'You win'
        : `${this.seatName(winner)} wins`;
    return `${who} round ${round} · −${damage} ♥`;
  }

  private endClass(winner: number | null): '' | 'good' | 'bad' {
    if (this.ctx.humanSeat < 0 || winner === null) return '';
    return winner === this.ctx.humanSeat ? 'good' : 'bad';
  }

  render(state: ViewState): void {
    const v = state.viewData as T21View | null;
    if (!v) return;
    const roundChanged = v.round !== this.lastRound;
    if (roundChanged) this.prevCounts = [0, 0];
    if (state.isOver) {
      const cls =
        this.ctx.humanSeat >= 0
          ? v.hearts[this.ctx.humanSeat] > 0
            ? 'good'
            : 'bad'
          : '';
      this.showBanner(state.result ?? 'Game over', cls);
    } else if (v.roundActive) {
      this.bannerEl.hidden = true;
      this.roundEndSeen = false;
    } else if (roundChanged && this.lastRound > 0 && !this.roundEndSeen && this.prevHearts) {
      const loser =
        v.hearts[0] < this.prevHearts[0] ? 0 : v.hearts[1] < this.prevHearts[1] ? 1 : null;
      const winner = loser === null ? null : 1 - loser;
      const damage = loser === null ? 0 : this.prevHearts[loser] - v.hearts[loser];
      this.showBanner(this.endText(winner, damage, this.lastRound), this.endClass(winner));
      this.roundEndSeen = true;
    }
    this.roundEl.textContent = `Round ${v.round}`;
    this.stakeEl.textContent = `${v.round} ♥ at stake`;
    this.deckCountEl.textContent = v.roundActive ? `${v.deckCount} in deck` : '';
    this.deckEl.style.visibility = v.roundActive ? 'visible' : 'hidden';
    this.renderSeat(0, v, state);
    this.renderSeat(1, v, state);
    if (state.toAct !== state.humanSeat) this.actionsEl.replaceChildren();
    this.lastRound = v.round;
    this.prevHearts = [v.hearts[0], v.hearts[1]];
  }

  private async showdown(
    d: Extract<T21Event, { kind: 'roundEnd' }>,
    scale: number,
  ): Promise<void> {
    for (const seat of [0, 1]) {
      const els = this.seatEls[seat];
      const back = els.cards.querySelector('.t21-back');
      if (back) {
        const reveal = this.cardEl(d.downs[seat], true);
        reveal.classList.add('t21-flip');
        reveal.style.animationDuration = `${500 * scale}ms`;
        back.replaceWith(reveal);
      }
      this.setTotal(els.total, 'total', d.totals[seat]);
    }
    await sleep(700 * scale);
    this.showBanner(this.endText(d.winner, d.damage, this.lastRound), this.endClass(d.winner));
    if (d.winner !== null) {
      const loser = 1 - d.winner;
      const alive = [
        ...this.seatEls[loser].hearts.querySelectorAll<HTMLElement>('.t21-heart:not(.t21-lost)'),
      ];
      for (const h of alive.slice(Math.max(0, alive.length - d.damage))) {
        h.style.animationDuration = `${800 * scale}ms`;
        h.classList.add('t21-heart-break');
      }
    }
    await sleep(900 * scale);
  }

  async animate(event: MatchEventData, after: ViewState): Promise<void> {
    const scale = this.ctx.animationScale();
    const d = (event.data ?? null) as T21Event | null;
    if (d?.kind === 'draw') {
      if (scale > 0) {
        this.deckPile.classList.add('t21-deck-pulse');
        await sleep(260 * scale);
        this.deckPile.classList.remove('t21-deck-pulse');
      }
      this.render(after);
      await sleep(160 * scale);
    } else if (d?.kind === 'stand') {
      this.render(after);
      if (scale > 0) {
        const bar = this.seatEls[d.seat].bar;
        bar.classList.add('t21-stand-flash');
        await sleep(340 * scale);
        bar.classList.remove('t21-stand-flash');
      }
    } else if (d?.kind === 'roundEnd') {
      this.roundEndSeen = true;
      if (scale > 0) await this.showdown(d, scale);
      else this.showBanner(this.endText(d.winner, d.damage, this.lastRound), this.endClass(d.winner));
      this.render(after);
      await sleep(250 * scale);
    } else {
      this.render(after);
      await sleep(200 * scale);
    }
  }

  promptAction(labels: string[]): void {
    const subs: Record<string, string> = {
      draw: 'take a card',
      stand: 'hold your total',
    };
    const btns = labels.map((label, i) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = label === 'draw' ? 't21-btn t21-btn-draw' : 't21-btn';
      const sub = subs[label];
      b.innerHTML = sub ? `${label}<span>${sub}</span>` : label;
      b.onclick = () => {
        for (const x of btns) x.disabled = true;
        this.ctx.submit(String(i));
      };
      return b;
    });
    this.actionsEl.replaceChildren(...btns);
  }

  unmount(): void {}
}

export function createTwentyOneFrontend(): GameFrontend {
  return new TwentyOneFrontend();
}
