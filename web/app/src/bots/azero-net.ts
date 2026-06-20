// The page-lifetime caches every AlphaZero bot shares: the raw export bytes
// (fetched once, handed to both the GPU evaluator and — as the CPU fallback —
// the wasm engine) and the GPU device + weight upload (one per page, not per
// match, dropped on device loss). The chess/go/snake bots differ in their host
// API and advance loop, but this fetch-once / init-once-per-device pair was
// byte-identical across all three; it lives here once.

/** A `getWeights` closure for a bot's export URL: fetched once and shared, but
 * re-armed on failure so a transient error doesn't pin a rejected promise. */
export function weightsLoader(url: string): () => Promise<ArrayBuffer> {
  let cached: Promise<ArrayBuffer> | null = null;
  return () => {
    cached ??= (async () => {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`weights ${url} missing (HTTP ${resp.status})`);
      return resp.arrayBuffer();
    })();
    cached.catch(() => {
      cached = null;
    });
    return cached;
  };
}

/** A GPU evaluator exposing the device-loss signal the cache re-arms on. */
interface GpuNet {
  readonly lost: Promise<unknown>;
}

/** A `getGpu` closure: one device + weight upload per page. The cache is cleared
 * both on init failure and when the device is lost, so the next match re-inits
 * cleanly. */
export function gpuLoader<G extends GpuNet>(
  init: (weights: ArrayBuffer) => Promise<G>,
  getWeights: () => Promise<ArrayBuffer>,
): () => Promise<G> {
  let cached: Promise<G> | null = null;
  return () => {
    cached ??= (async () => {
      const gpu = await init(await getWeights());
      void gpu.lost.then(() => {
        cached = null;
      });
      return gpu;
    })();
    cached.catch(() => {
      cached = null;
    });
    return cached;
  };
}
