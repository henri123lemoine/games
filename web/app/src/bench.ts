// One-page full benchmark matrix for the AlphaZero bots: actual search-move
// timings, GPU (WebGPU) vs CPU (the in-wasm reference forward `play_cpu`), for
// chess and go at 9/13/19, over 1 -> 16384 sims. Each move is the real search
// (the same code paths the site plays through, run on the main thread). A move
// is measured while its projected cost stays under a per-backend budget; beyond
// that the cost is projected from the measured per-forward rate, since the
// search is linear in sims. Open it in a WebGPU browser.

import init, { AzChessBot, AzGoBot } from 'web-engine';
import wasmUrl from 'web-engine/web_engine_bg.wasm?url';
import { AzGpu, POLICY_LEN, softmaxOver as chessSoftmax } from './frontends/chess/azgpu';
import { GoGpu, policyLen, softmaxOver as goSoftmax } from './frontends/go/azgpu';

const LEAVES = 8;
const SIMS = [1, 4, 16, 64, 256, 1024, 4096, 16384];
const BUDGET_GPU = 8000; // ms: measure real until a move would exceed this, then project
const BUDGET_CPU = 6000;

// Forward (network-eval) calls a move makes: the GPU answers a batch of up to 8
// leaves per call; the CPU reference forward does one leaf at a time. Plus the
// root.
const gpuCalls = (sims: number) => Math.max(1, Math.ceil(sims / LEAVES)) + 1;
const cpuCalls = (sims: number) => sims + 1;

const statusEl = document.getElementById('status')!;
const outEl = document.getElementById('out')!;
const setStatus = (s: string) => {
  statusEl.textContent = s;
};

function fmt(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)} s`;
  return `${(ms / 60_000).toFixed(1)} min`;
}

interface Cell {
  ms: number;
  measured: boolean;
}
interface Column {
  label: string;
  backend: 'GPU' | 'CPU';
  cells: Cell[];
  msPerLeaf: number;
}

async function chessGpuMove(gpu: AzGpu, sims: number): Promise<void> {
  const bot = new AzChessBot(sims, LEAVES, 7);
  try {
    let priors = new Float32Array(0);
    let values = new Float32Array(0);
    for (;;) {
      const n = bot.advance(priors, values);
      if (n === 0) break;
      const support = bot.batch_support();
      const offsets = bot.batch_offsets();
      const { logits, values: v } = await gpu.forward(new Float32Array(bot.batch_features()), n);
      const flat: number[] = [];
      for (let i = 0; i < n; i++)
        flat.push(...chessSoftmax(logits, support.subarray(offsets[i], offsets[i + 1]), i * POLICY_LEN));
      priors = Float32Array.from(flat);
      values = v.slice(0, n);
    }
    bot.best();
  } finally {
    bot.free();
  }
}

async function goGpuMove(gpu: GoGpu, sims: number, size: number): Promise<void> {
  const bot = new AzGoBot(sims, LEAVES, 7, size);
  const stride = policyLen(size);
  try {
    let priors = new Float32Array(0);
    let values = new Float32Array(0);
    for (;;) {
      const n = bot.advance(priors, values);
      if (n === 0) break;
      const support = bot.batch_support();
      const offsets = bot.batch_offsets();
      const { logits, values: v } = await gpu.forward(new Float32Array(bot.batch_features()), n, size);
      const flat: number[] = [];
      for (let i = 0; i < n; i++)
        flat.push(...goSoftmax(logits, support.subarray(offsets[i], offsets[i + 1]), i * stride));
      priors = Float32Array.from(flat);
      values = v.slice(0, n);
    }
    bot.best();
  } finally {
    bot.free();
  }
}

function chessCpuMove(weights: Uint8Array, sims: number): void {
  const bot = new AzChessBot(sims, LEAVES, 7);
  bot.load_weights(weights);
  try {
    bot.play_cpu();
  } finally {
    bot.free();
  }
}

function goCpuMove(weights: Uint8Array, sims: number, size: number): void {
  const bot = new AzGoBot(sims, LEAVES, 7, size);
  bot.load_weights(weights);
  try {
    bot.play_cpu();
  } finally {
    bot.free();
  }
}

async function runColumn(
  label: string,
  backend: 'GPU' | 'CPU',
  move: (sims: number) => Promise<void> | void,
): Promise<Column> {
  const budget = backend === 'GPU' ? BUDGET_GPU : BUDGET_CPU;
  const calls = backend === 'GPU' ? gpuCalls : cpuCalls;
  await move(8); // warm up (shader compile / first parse)
  let rate: number | null = null;
  const cells: Cell[] = [];
  for (const sims of SIMS) {
    const c = calls(sims);
    const projected = rate != null ? rate * c : null;
    if (projected != null && projected > budget) {
      cells.push({ ms: projected, measured: false });
      continue;
    }
    setStatus(`${label} · ${backend}  sims=${sims} …`);
    const t0 = performance.now();
    await move(sims);
    const dt = performance.now() - t0;
    rate = dt / c;
    cells.push({ ms: dt, measured: true });
  }
  // Per *leaf*: a GPU call covers up to 8 leaves, a CPU call one. This makes the
  // GPU and CPU rates directly comparable.
  const msPerLeaf = (rate ?? 0) / (backend === 'GPU' ? LEAVES : 1);
  return { label, backend, cells, msPerLeaf };
}

function render(columns: Column[]): void {
  const head =
    `<tr><th class="cfg">config</th><th>ms/leaf</th>` +
    SIMS.map((s) => `<th>${s}</th>`).join('') +
    `</tr>`;
  const rows = columns
    .map((col) => {
      const cells = col.cells
        .map((c) => {
          const cls = !c.measured ? 'proj' : c.ms > 20_000 ? 'slow' : '';
          return `<td class="${cls}">${fmt(c.ms)}</td>`;
        })
        .join('');
      return `<tr><td class="cfg">${col.label} · ${col.backend}</td><td>${col.msPerLeaf.toFixed(2)}</td>${cells}</tr>`;
    })
    .join('');
  outEl.innerHTML =
    `<table><caption>move time by simulations (italic = projected from measured rate)</caption>${head}${rows}</table>` +
    `<p class="muted">Bench runs moves on the main thread, so the GPU column omits the worker round-trip the live app adds (a few ms/batch). CPU is in-wasm and matches the live path.</p>`;
}

(async () => {
  try {
    const base = import.meta.env.BASE_URL;
    setStatus('fetching nets…');
    const [chessBuf, goBuf] = await Promise.all([
      fetch(`${base}azero/azero-chess.azweb`).then((r) => r.arrayBuffer()),
      fetch(`${base}azero/azero-go.azweb`).then((r) => r.arrayBuffer()),
    ]);
    const chessW = new Uint8Array(chessBuf);
    const goW = new Uint8Array(goBuf);
    await init({ module_or_path: wasmUrl });
    setStatus('initialising WebGPU…');
    const azGpu = await AzGpu.init(chessBuf);
    const goGpu = await GoGpu.init(goBuf);

    const plan: { label: string; backend: 'GPU' | 'CPU'; move: (s: number) => Promise<void> | void }[] = [
      { label: 'chess', backend: 'GPU', move: (s) => chessGpuMove(azGpu, s) },
      { label: 'chess', backend: 'CPU', move: (s) => chessCpuMove(chessW, s) },
      { label: 'go 9×9', backend: 'GPU', move: (s) => goGpuMove(goGpu, s, 9) },
      { label: 'go 9×9', backend: 'CPU', move: (s) => goCpuMove(goW, s, 9) },
      { label: 'go 13×13', backend: 'GPU', move: (s) => goGpuMove(goGpu, s, 13) },
      { label: 'go 13×13', backend: 'CPU', move: (s) => goCpuMove(goW, s, 13) },
      { label: 'go 19×19', backend: 'GPU', move: (s) => goGpuMove(goGpu, s, 19) },
      { label: 'go 19×19', backend: 'CPU', move: (s) => goCpuMove(goW, s, 19) },
    ];

    const columns: Column[] = [];
    for (const p of plan) {
      columns.push(await runColumn(p.label, p.backend, p.move));
      render(columns);
    }
    setStatus('done.');
  } catch (e) {
    setStatus('');
    outEl.innerHTML = `<p class="bad">ERROR: ${e instanceof Error ? e.message : String(e)}</p>`;
    throw e;
  }
})();
