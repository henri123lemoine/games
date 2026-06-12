import { resolve } from 'node:path';
import { defineConfig } from 'vite';

// base './' keeps the build embeddable at any path on the host site.
export default defineConfig({
  base: './',
  build: {
    target: 'esnext',
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, 'index.html'),
        azeroTest: resolve(import.meta.dirname, 'azero-test.html'),
      },
    },
  },
  worker: { format: 'es' },
});
