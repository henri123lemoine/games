// The fallback frontend: terminal view + action buttons. Every registered
// game is playable through this the moment it exists; a polished per-game
// frontend replaces it by registering in `frontends/index.ts`.

import type { MatchEventData, ViewState } from '../engine/protocol';
import type { FrontendCtx, GameFrontend } from './types';
import { sleep } from './types';

export class GenericFrontend implements GameFrontend {
  private ctx!: FrontendCtx;
  private viewEl!: HTMLPreElement;
  private actionsEl!: HTMLDivElement;

  mount(host: HTMLElement, ctx: FrontendCtx): void {
    this.ctx = ctx;
    host.innerHTML = `
      <div class="generic">
        <pre class="generic-view"></pre>
        <div class="generic-actions"></div>
      </div>`;
    this.viewEl = host.querySelector('.generic-view')!;
    this.actionsEl = host.querySelector('.generic-actions')!;
  }

  render(state: ViewState): void {
    this.viewEl.textContent = state.view;
    if (state.toAct !== state.humanSeat) this.actionsEl.replaceChildren();
  }

  async animate(event: MatchEventData, after: ViewState): Promise<void> {
    void event;
    this.render(after);
    await sleep(250 * this.ctx.animationScale());
  }

  promptAction(labels: string[]): void {
    const buttons = labels.map((label, i) => {
      const b = document.createElement('button');
      b.className = 'action-btn';
      b.textContent = label;
      b.onclick = () => this.ctx.submit(String(i));
      return b;
    });
    this.actionsEl.replaceChildren(...buttons);
  }

  unmount(): void {}
}
