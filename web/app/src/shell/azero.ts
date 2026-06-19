// Shared AlphaZero web policy: where it runs (WebGPU vs the in-wasm CPU
// forward) and the budget the CPU fallback is locked to. The shell uses this
// to label/lock difficulty; the drivers use it to pick a backend.

/** Whether AlphaZero must fall back to the CPU forward — true when the browser
 * exposes no WebGPU. The same net plays either way; the CPU forward is just
 * far slower, so it is pinned to the trivial visit budget. */
export function isCpuFallback(): boolean {
  return !('gpu' in navigator);
}

/** The visit budget the CPU path is locked to: one simulation ≈ the network's
 * raw-policy move (~1–2 forwards), responsive even in wasm. */
export const TRIVIAL_SIMS = 1;
