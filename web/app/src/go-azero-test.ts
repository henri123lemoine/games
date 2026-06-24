// Validation harness for the go WebGPU evaluator (the /go-azero-test.html
// page): checks the kernels against nn-infer's reference forward over the
// committed fixtures, compares the WebGPU and in-wasm CPU forwards head-to-head
// (the exact two backends the bot picks between at runtime), then measures
// throughput at an MCTS-ish batch. The per-fixture comparison is the same
// helper the live per-visitor self-check runs, so the dev gate and the field
// alarm can never drift.

import init from 'web-engine';
import wasmUrl from 'web-engine/web_engine_bg.wasm?url';
import { GoGpu, PLANES, policyLen } from './frontends/go/azgpu';
import { compareFixture, fetchFixtures } from './frontends/go/conformance';

const logEl = document.getElementById('log')!;
logEl.innerHTML = '';
const log = (html: string): void => {
  logEl.innerHTML += '<br>' + html;
};

(async () => {
  try {
    const base = import.meta.env.BASE_URL;
    const [bin, fixtures] = await Promise.all([
      fetch(`${base}azero/azero-go.azweb`).then((r) => r.arrayBuffer()),
      fetchFixtures(),
    ]);
    const gpu = await GoGpu.init(bin);
    log(
      `model: ${gpu.model.blocks}x${gpu.model.C}, ${gpu.model.size}×${gpu.model.size}, ` +
        `${(bin.byteLength / 1e6).toFixed(1)} MB · WebGPU ready`,
    );
    const stride = policyLen(gpu.model.size);

    // The reference forward runs in-wasm, so init the engine before comparing.
    await init({ module_or_path: wasmUrl });
    const weights = new Uint8Array(bin);

    let maxDp = 0;
    let maxDv = 0;
    let maxGpuCpuP = 0;
    let maxGpuCpuV = 0;
    for (const fx of fixtures) {
      const cmp = await compareFixture(gpu, weights, fx);
      maxDp = Math.max(maxDp, cmp.dpVsGolden);
      maxDv = Math.max(maxDv, cmp.dvVsGolden);
      maxGpuCpuP = Math.max(maxGpuCpuP, cmp.dpVsRef);
      maxGpuCpuV = Math.max(maxGpuCpuV, cmp.dvVsRef);
      log(`fixture @ply ${fx.plies}  v=${cmp.gpuValue.toFixed(4)} (exp ${fx.value.toFixed(4)})`);
    }
    const pass = maxDp < 1e-3 && maxDv < 1e-3;
    log(
      `max |Δprior| = <span class="brass">${maxDp.toExponential(2)}</span>, ` +
        `max |Δvalue| = <span class="brass">${maxDv.toExponential(2)}</span> → ` +
        (pass
          ? '<span class="ok">PASS — kernels agree with the reference forward</span>'
          : '<span class="bad">FAIL</span>'),
    );

    // Live GPU-vs-CPU calibration: the no-GPU fallback plays this same net
    // through the wasm reference forward, so confirm the two backends a real
    // visitor's browser picks between agree on the same positions.
    const calPass = maxGpuCpuP < 1e-3 && maxGpuCpuV < 1e-3;
    log(
      `GPU vs CPU (live): max |Δprior| = <span class="brass">${maxGpuCpuP.toExponential(2)}</span>, ` +
        `max |Δvalue| = <span class="brass">${maxGpuCpuV.toExponential(2)}</span> → ` +
        (calPass
          ? '<span class="ok">PASS — the CPU fallback matches the GPU bot</span>'
          : '<span class="bad">FAIL</span>'),
    );

    const B = 8;
    const area = gpu.model.size * gpu.model.size;
    const planes = new Float32Array(B * PLANES * area);
    for (let b = 0; b < B; b++) planes.set(fixtures[b % fixtures.length].planes, b * PLANES * area);
    await gpu.forward(planes, B, gpu.model.size);
    const t0 = performance.now();
    const iters = 100;
    for (let i = 0; i < iters; i++) await gpu.forward(planes, B, gpu.model.size);
    const dt = (performance.now() - t0) / 1000;
    log(
      `throughput: <span class="brass">${Math.round((B * iters) / dt)}</span> evals/s ` +
        `at batch ${B} (${((dt / iters) * 1000).toFixed(1)} ms/forward, stride ${stride})`,
    );
  } catch (e) {
    log(`<span class="bad">ERROR: ${e instanceof Error ? e.message : String(e)}</span>`);
    throw e;
  }
})();
