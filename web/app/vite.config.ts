import { defineConfig } from 'vite';

// base './' keeps the build embeddable at any path on the host site.
export default defineConfig({
  base: './',
  build: { target: 'esnext' },
  worker: { format: 'es' },
});
