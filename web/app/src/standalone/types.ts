// Standalone arcade games: real-time canvas games that do not run through the
// wasm engine's turn-based match protocol. The shell launches them by handing
// over the board element and a context, and tearing them down on navigation.

export interface StandaloneCtx {
  /** Honor `prefers-reduced-motion` and pause when the tab is hidden. */
  reducedMotion: boolean;
}

export interface StandaloneGame {
  mount(host: HTMLElement, ctx: StandaloneCtx): void;
  unmount(): void;
}

export interface StandaloneInfo {
  id: string;
  name: string;
  /** Inline markup for the home-card preview, mirroring `miniFor` in the shell. */
  mini: string;
  create(): StandaloneGame;
}
