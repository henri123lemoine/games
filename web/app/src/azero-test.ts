// Validation harness for the WebGPU evaluator (the /azero-test.html page):
// checks the kernels against azinfer's reference forward over the committed
// fixtures, then measures throughput at an MCTS-ish batch.

import { AzGpu, PLANES, softmaxOver } from './frontends/chess/azgpu';

interface Fixture {
  fen: string;
  planes: number[];
  support: number[];
  priors: number[];
  value: number;
}

const logEl = document.getElementById('log')!;
logEl.innerHTML = '';
const log = (html: string): void => {
  logEl.innerHTML += '<br>' + html;
};

(async () => {
  try {
    const base = import.meta.env.BASE_URL;
    const [bin, fixtures] = await Promise.all([
      fetch(`${base}azero/azero-chess.azweb`).then((r) => r.arrayBuffer()),
      fetch(`${base}azero/fixtures.json`).then((r) => r.json() as Promise<Fixture[]>),
    ]);
    const gpu = await AzGpu.init(bin);
    log(
      `model: ${gpu.model.blocks}x${gpu.model.C}, ${(bin.byteLength / 1e6).toFixed(1)} MB · WebGPU ready`,
    );

    let maxDp = 0;
    let maxDv = 0;
    for (const fx of fixtures) {
      const { logits, values } = await gpu.forward(new Float32Array(fx.planes), 1);
      const priors = softmaxOver(logits, fx.support);
      fx.priors.forEach((p, i) => {
        maxDp = Math.max(maxDp, Math.abs(p - priors[i]));
      });
      maxDv = Math.max(maxDv, Math.abs(values[0] - fx.value));
      log(
        `fixture ${fx.fen.split(' ')[0].slice(0, 24)}…  v=${values[0].toFixed(4)} (exp ${fx.value.toFixed(4)})`,
      );
    }
    const pass = maxDp < 1e-3 && maxDv < 1e-3;
    log(
      `max |Δprior| = <span class="brass">${maxDp.toExponential(2)}</span>, ` +
        `max |Δvalue| = <span class="brass">${maxDv.toExponential(2)}</span> → ` +
        (pass
          ? '<span class="ok">PASS — kernels agree with the reference forward</span>'
          : '<span class="bad">FAIL</span>'),
    );

    const B = 8;
    const planes = new Float32Array(B * PLANES * 64);
    for (let b = 0; b < B; b++)
      planes.set(fixtures[b % fixtures.length].planes, b * PLANES * 64);
    await gpu.forward(planes, B);
    const t0 = performance.now();
    const iters = 100;
    for (let i = 0; i < iters; i++) await gpu.forward(planes, B);
    const dt = (performance.now() - t0) / 1000;
    log(
      `throughput: <span class="brass">${Math.round((B * iters) / dt)}</span> evals/s ` +
        `at batch ${B} (${((dt / iters) * 1000).toFixed(1)} ms/forward)`,
    );
  } catch (e) {
    log(`<span class="bad">ERROR: ${e instanceof Error ? e.message : String(e)}</span>`);
    throw e;
  }
})();
