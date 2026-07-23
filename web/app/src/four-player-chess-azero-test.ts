// Live parity/throughput harness for the exact WebGPU and wasm-CPU forwards
// used by the four-player chess bot.

import init, { four_player_chess_reference_forward } from 'web-engine';
import wasmUrl from 'web-engine/web_engine_bg.wasm?url';
import {
  AREA,
  FourPlayerAzGpu,
  PLANES,
  VALUE_SEATS,
} from './frontends/four-player-chess/azgpu';

const logEl = document.getElementById('log')!;
logEl.innerHTML = '';
const log = (html: string): void => {
  logEl.innerHTML += `<br>${html}`;
};

function deterministicFeatures(batch: number): Float32Array<ArrayBuffer> {
  const features = new Float32Array(batch * PLANES * AREA);
  let state = 0x4f50454e;
  for (let i = 0; i < features.length; i++) {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    if ((state & 15) === 0) features[i] = ((state >>> 8) & 255) / 127.5 - 1;
  }
  return features;
}

function maxDelta(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) return Number.POSITIVE_INFINITY;
  let max = 0;
  for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] - b[i]));
  return max;
}

(async () => {
  try {
    const base = import.meta.env.BASE_URL;
    const bin = await fetch(`${base}azero/four-player-chess.azweb`).then((response) => {
      if (!response.ok) throw new Error(`model fetch failed: HTTP ${response.status}`);
      return response.arrayBuffer();
    });
    const gpu = await FourPlayerAzGpu.init(bin);
    await init({ module_or_path: wasmUrl });
    log(
      `model: ${gpu.model.blocks}x${gpu.model.C}, ${(bin.byteLength / 1e6).toFixed(1)} MB · ` +
        `${VALUE_SEATS} value seats · WebGPU ready`,
    );

    const batch = 2;
    const features = deterministicFeatures(batch);
    const gpuOut = await gpu.forward(features, batch);
    const cpuOut = four_player_chess_reference_forward(new Uint8Array(bin), features, batch);
    const maxPolicy = maxDelta(gpuOut.logits, cpuOut.logits);
    const maxValue = maxDelta(gpuOut.values, cpuOut.values);
    const pass = maxPolicy < 1e-3 && maxValue < 1e-3;
    log(
      `GPU vs CPU: max |Δpolicy logit| = <span class="brass">${maxPolicy.toExponential(2)}</span>, ` +
        `max |Δseat logit| = <span class="brass">${maxValue.toExponential(2)}</span> → ` +
        (pass
          ? '<span class="ok">PASS — both play paths agree</span>'
          : '<span class="bad">FAIL</span>'),
    );

    await gpu.forward(features, batch);
    const iterations = 20;
    const started = performance.now();
    for (let i = 0; i < iterations; i++) await gpu.forward(features, batch);
    const seconds = (performance.now() - started) / 1000;
    log(
      `throughput: <span class="brass">${Math.round((batch * iterations) / seconds)}</span> ` +
        `evals/s at batch ${batch} (${((seconds / iterations) * 1000).toFixed(1)} ms/forward)`,
    );
  } catch (error) {
    log(`<span class="bad">ERROR: ${error instanceof Error ? error.message : String(error)}</span>`);
    throw error;
  }
})();
