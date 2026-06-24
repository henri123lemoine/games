// Forward-parity self-check for the go WebGPU evaluator: it compares the GPU
// kernels against the hardware-independent wasm reference forward (the exact
// CPU fallback a no-GPU visitor plays) over the shipped fixtures. The dev test
// page and the live per-visitor check both run through here so the two paths
// can never drift.

import init, { go_reference_forward } from 'web-engine';
import wasmUrl from 'web-engine/web_engine_bg.wasm?url';
import { type GoGpu, softmaxOver } from './azgpu';

export interface Fixture {
  size: number;
  plies: number;
  planes: number[];
  support: number[];
  priors: number[];
  value: number;
}

export interface FixtureCompare {
  /** GPU priors over the legal support, in support order. */
  gpuPriors: number[];
  /** wasm reference priors over the same support. */
  cpuPriors: number[];
  gpuValue: number;
  cpuValue: number;
  /** GPU vs golden-fixture deltas (the dev page's strict reference check). */
  dpVsGolden: number;
  dvVsGolden: number;
  /** GPU vs live wasm reference deltas (the live oracle the alarm watches). */
  dpVsRef: number;
  dvVsRef: number;
}

export interface ConformanceResult {
  pass: boolean;
  maxDp: number;
  maxDv: number;
  count: number;
  worst: { plies: number; size: number } | null;
}

// Looser than the dev page's strict 1e-3 PASS gate: a faithful GPU lands at
// ~1e-4–1e-3 against the reference, so a 3e-3 alarm threshold leaves headroom
// for benign fp rounding while still catching a driver that computes the net
// meaningfully differently. The dev page keeps its strict 1e-3 checks.
export const ALARM_TOL = 3e-3;

let wasmReady: Promise<unknown> | null = null;
function ensureWasm(): Promise<unknown> {
  wasmReady ??= init({ module_or_path: wasmUrl });
  return wasmReady;
}

export async function fetchFixtures(): Promise<Fixture[]> {
  const url = `${import.meta.env.BASE_URL}azero/go-fixtures.json`;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`go fixtures ${url} missing (HTTP ${resp.status})`);
  return resp.json() as Promise<Fixture[]>;
}

/** One fixture's GPU-vs-reference and GPU-vs-golden comparison. The wasm
 * reference must already be initialized (the dev page does its own init; the
 * live check goes through `runGoConformance`). */
export async function compareFixture(
  gpu: GoGpu,
  weights: Uint8Array,
  fx: Fixture,
): Promise<FixtureCompare> {
  const planes = new Float32Array(fx.planes);
  const g = await gpu.forward(planes, 1, fx.size);
  const c = go_reference_forward(weights, planes, 1, fx.size);
  const gpuPriors = softmaxOver(g.logits, fx.support);
  const cpuPriors = softmaxOver(c.logits, fx.support);
  let dpVsGolden = 0;
  let dpVsRef = 0;
  for (let i = 0; i < gpuPriors.length; i++) {
    dpVsGolden = Math.max(dpVsGolden, Math.abs(gpuPriors[i] - fx.priors[i]));
    dpVsRef = Math.max(dpVsRef, Math.abs(gpuPriors[i] - cpuPriors[i]));
  }
  return {
    gpuPriors,
    cpuPriors,
    gpuValue: g.values[0],
    cpuValue: c.values[0],
    dpVsGolden,
    dvVsGolden: Math.abs(g.values[0] - fx.value),
    dpVsRef,
    dvVsRef: Math.abs(g.values[0] - c.values[0]),
  };
}

/** Live forward-parity check: GPU vs the wasm reference over the fixtures,
 * optionally capped to the highest-ply entries (where divergence bites the
 * hardest and the deepest search positions live) to bound visitor-side cost. */
export async function runGoConformance(
  gpu: GoGpu,
  weights: ArrayBuffer,
  opts: { limit?: number } = {},
): Promise<ConformanceResult> {
  await ensureWasm();
  const w = new Uint8Array(weights);
  let fixtures = await fetchFixtures();
  if (opts.limit !== undefined && opts.limit < fixtures.length) {
    fixtures = [...fixtures].sort((a, b) => b.plies - a.plies).slice(0, opts.limit);
  }
  let maxDp = 0;
  let maxDv = 0;
  let worst: { plies: number; size: number } | null = null;
  for (const fx of fixtures) {
    const cmp = await compareFixture(gpu, w, fx);
    if (cmp.dpVsRef > maxDp || cmp.dvVsRef > maxDv)
      worst = { plies: fx.plies, size: fx.size };
    maxDp = Math.max(maxDp, cmp.dpVsRef);
    maxDv = Math.max(maxDv, cmp.dvVsRef);
  }
  return {
    pass: maxDp < ALARM_TOL && maxDv < ALARM_TOL,
    maxDp,
    maxDv,
    count: fixtures.length,
    worst,
  };
}
