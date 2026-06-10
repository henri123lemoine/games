import { App } from './shell/app';

new App(document.getElementById('app')!).start().catch((e) => {
  document.getElementById('app')!.innerHTML =
    `<div class="boot">Failed to start the engine: ${e instanceof Error ? e.message : e}</div>`;
});
