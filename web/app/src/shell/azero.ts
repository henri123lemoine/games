// Shared AlphaZero web policy: where it runs (WebGPU vs the in-wasm CPU
// forward) and the budget the CPU fallback is locked to. The shell uses this
// to label/lock difficulty; the drivers use it to pick a backend.

/** Whether AlphaZero must fall back to the CPU forward — true when the browser
 * exposes no WebGPU. The same net plays either way; the CPU forward is just
 * far slower, so it is pinned to the trivial visit budget. */
export function isCpuFallback(): boolean {
  return !('gpu' in navigator);
}

/** Trivial play: one simulation ≈ the network's raw-policy move (~1–2
 * forwards), responsive even in wasm. The CPU default. */
export const TRIVIAL_SIMS = 1;

/** Difficulty levels offered without a GPU. The reference forward is ~150×
 * slower than WebGPU, so only the two responsive budgets are available — the
 * rest of the ladder needs a GPU. (Light ≈ a few seconds/move on a typical
 * machine at 9×9; larger boards are slower.) */
export const CPU_LEVELS: [string, string][] = [
  ['Trivial', String(TRIVIAL_SIMS)],
  ['Light', '16'],
];

/** Safety cap on the CPU visit budget, matching the top `CPU_LEVELS` entry. */
export const CPU_MAX_SIMS = 16;
