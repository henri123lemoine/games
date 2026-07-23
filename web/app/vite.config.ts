import { resolve } from 'node:path';
import { defineConfig } from 'vite';
import { computeManifest } from './scripts/r2-assets.mjs';

// base './' keeps the build embeddable at any path on the host site.
// The heavyweight payloads live in R2, not in the repo, so every build —
// dev included — bakes their URLs into `assetUrl` (src/assets.ts).
export default defineConfig({
  base: './',
  define: {
    __ARCADE_ASSETS__: JSON.stringify(computeManifest()),
  },
  build: {
    target: 'esnext',
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, 'index.html'),
        azeroTest: resolve(import.meta.dirname, 'azero-test.html'),
        fourPlayerChessAzeroTest: resolve(
          import.meta.dirname,
          'four-player-chess-azero-test.html',
        ),
        goAzeroTest: resolve(import.meta.dirname, 'go-azero-test.html'),
        bench: resolve(import.meta.dirname, 'bench.html'),
      },
    },
  },
  worker: { format: 'es' },
});
