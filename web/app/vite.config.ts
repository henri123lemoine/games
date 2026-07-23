import { resolve } from 'node:path';
import { defineConfig } from 'vite';
import { computeManifest } from './scripts/r2-assets.mjs';

// base './' keeps the build embeddable at any path on the host site.
// Production builds bake the R2 URLs of the heavyweight payloads into
// `assetUrl` (src/assets.ts); dev serves the same files from public/.
export default defineConfig(({ command }) => ({
  base: './',
  define: {
    __ARCADE_ASSETS__: JSON.stringify(command === 'build' ? computeManifest() : {}),
  },
  build: {
    target: 'esnext',
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, 'index.html'),
        azeroTest: resolve(import.meta.dirname, 'azero-test.html'),
        goAzeroTest: resolve(import.meta.dirname, 'go-azero-test.html'),
        bench: resolve(import.meta.dirname, 'bench.html'),
      },
    },
  },
  worker: { format: 'es' },
}));
