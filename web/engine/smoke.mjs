// Node smoke test for the wasm engine (no browser needed):
//   wasm-pack build web/engine --target web --out-dir pkg
//   node web/engine/smoke.mjs
// Exercises the manifest, a spectated match, a human turn, the pair/field
// runners, and the stats bindings.

import { execFileSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import init, * as engine from './pkg/web_engine.js';

/** Payload bytes from the arcade-assets bucket (tools/fetch-asset.sh caches
 * and checksum-verifies; the nets are not in git). */
const fetchAsset = (logical) =>
  readFile(
    execFileSync(new URL('../../tools/fetch-asset.sh', import.meta.url).pathname, [logical], {
      encoding: 'utf8',
    }).trim(),
  );

const wasm = await readFile(new URL('./pkg/web_engine_bg.wasm', import.meta.url));
await init({ module_or_path: wasm });

const assert = (cond, msg) => {
  if (!cond) throw new Error(`smoke failed: ${msg}`);
};

const manifest = JSON.parse(engine.list_games());
assert(manifest.games.length >= 8, `expected >= 8 games, got ${manifest.games.length}`);
assert(
  manifest.compare.some((c) => c.field),
  'expected at least one field-capable compare entry',
);
for (const g of manifest.games) {
  assert(Array.isArray(g.optsSchema) && g.optsSchema.length > 0, `${g.id} has no optsSchema`);
  assert(
    g.optsSchema.every((o) => o.key && o.value !== undefined),
    `${g.id} schema entries need key+value`,
  );
}
console.log('games:', manifest.games.map((g) => g.id).join(','));

let m = engine.create_match('connect4', JSON.stringify({ seat: 'watch', depth: 3, seed: 42 }));
let steps = 0;
while (m.step()) steps++;
assert(m.is_over() && steps >= 7, `connect4 watch ended after ${steps} steps`);
console.log('connect4 watch:', steps, 'moves —', m.result_text());

m = engine.create_match(
  'liars-dice',
  JSON.stringify({ players: 3, dice: 2, rollouts: 100, seed: 7 }),
);
while (m.step());
const labels = JSON.parse(m.legal_labels());
assert(m.to_act() === m.human_seat() && labels.length > 0, 'human to act with legal actions');
const ev = JSON.parse(m.apply_human('0'));
assert(ev.text.startsWith('You:'), `apply_human narration: ${ev.text}`);
console.log('liars-dice human move:', ev.text);

// Liar's Dice history net: proves the trained champion actually loads through
// load_artifact and plays — not just that the crate compiles for wasm.
const ldHistoryWeights = await fetchAsset('artifacts/ld-history-champion.bin');
engine.load_artifact('runs/ld_history/best.bin', new Uint8Array(ldHistoryWeights));
m = engine.create_match(
  'liars-dice',
  JSON.stringify({ players: 3, dice: 2, bot: 'history', seat: 'watch', seed: 13 }),
);
let ldHistorySteps = 0;
while (m.step()) ldHistorySteps++;
assert(m.is_over() && ldHistorySteps > 0, `liars-dice history watch ended after ${ldHistorySteps} steps`);
console.log('liars-dice history watch:', ldHistorySteps, 'moves —', m.result_text());

// Battlesnake: all current moves resolve atomically. Watch BNS to a result,
// then verify the human submits one move while the opponent still acts from
// the same pre-state.
m = engine.create_match('snake', JSON.stringify({ seat: 'watch', bot: 'bns', millis: 2, depth: 3, qdepth: 1, 'tt-bits': 10, seed: 5 }));
let snakeSteps = 0;
for (;;) {
  m.prepare();
  if (!m.step()) break;
  snakeSteps++;
}
assert(m.is_over() && snakeSteps >= 1, `snake watch ended after ${snakeSteps} turns`);
const snakeView = JSON.parse(m.view_data());
assert(snakeView.side === 11 && snakeView.snakes.length === 2, 'snake view has two snakes on an 11-grid');
assert(snakeView.simultaneous === true && !('pending' in snakeView), 'snake view is simultaneous');
assert(Array.isArray(snakeView.food) && Array.isArray(snakeView.hazards), 'snake view carries canonical maps');
assert(['win0', 'win1', 'draw'].includes(snakeView.outcome), `snake outcome: ${snakeView.outcome}`);
assert(
  snakeView.snakes.every((s) => typeof s.health === 'number' && s.health >= 0 && s.health <= 100),
  `snake view carries per-snake health: ${JSON.stringify(snakeView.snakes.map((s) => s.health))}`,
);
console.log('snake watch:', snakeSteps, 'joint turns —', m.result_text());

m = engine.create_match('snake', JSON.stringify({ seat: 0, bot: 'random', seed: 6 }));
while (m.step());
const snakeLabels = JSON.parse(m.legal_labels());
assert(
  m.to_act() === m.human_seat() && snakeLabels.includes('right'),
  'snake human steers with absolute headings',
);
m.prepare();
const snakeMove = JSON.parse(m.apply_human('right'));
assert(snakeMove.text.startsWith('You:'), `snake human move: ${snakeMove.text}`);
console.log('snake human move:', snakeMove.text);

const pairs = JSON.parse(
  engine.play_pairs('connect4', '{}', 'alphabeta:depth=4', 'alphabeta:depth=2', 123, 0, 4),
);
assert(pairs.w + pairs.d + pairs.l === 8, 'pair runner plays 2 games per pair');
const field = JSON.parse(
  engine.play_field('liars-dice', JSON.stringify({ players: 3, dice: 2 }), 'belief', 'random', 9, 0, 6),
);
assert(field.wins + field.losses === 6, 'field runner plays one game per index');
const elo = JSON.parse(engine.elo(pairs.w, pairs.d, pairs.l));
assert(Number.isFinite(elo.elo), 'elo estimate');
const table = JSON.parse(
  engine.fit_elo_table(
    JSON.stringify([
      [
        [0, 0, 0],
        [6, 2, 0],
      ],
      [
        [0, 2, 6],
        [0, 0, 0],
      ],
    ]),
  ),
);
assert(table.length === 2 && table[0] > table[1], 'fit_elo orders the stronger bot first');

// The azero-gpu seam: an externally driven seat, the AzChessBot mirror, and
// the park/resume wire format — evaluated here with uniform priors (the
// browser supplies the WebGPU net; strength is not under test).
m = engine.create_match('chess', JSON.stringify({ bot: 'azero-gpu', seat: 0, seed: 11 }));
assert(m.step() === '', 'no engine-side bot moves in an externally driven match');
const bot = new engine.AzChessBot(64, 8, 11);
const evalUniform = (n) => {
  const offsets = bot.batch_offsets();
  assert(offsets.length === n + 1, 'offsets delimit the batch');
  const support = bot.batch_support();
  assert(offsets[n] === support.length, 'offsets cover the support');
  assert(bot.batch_features().length === n * 18 * 64, 'feature shape');
  const priors = new Float32Array(support.length);
  for (let i = 0; i < n; i++) {
    const k = offsets[i + 1] - offsets[i];
    priors.fill(1 / k, offsets[i], offsets[i + 1]);
  }
  return { priors, values: new Float32Array(n) };
};
const azMove = () => {
  let priors = new Float32Array(0);
  let values = new Float32Array(0);
  for (;;) {
    const n = bot.advance(priors, values);
    if (n === 0) break;
    ({ priors, values } = evalUniform(n));
  }
  return bot.best();
};
let plies = 0;
while (!m.is_over() && plies < 30) {
  const turn = m.to_act();
  const want = JSON.parse(m.legal_labels());
  const input = turn === m.human_seat() ? want[plies % want.length] : azMove();
  if (turn !== m.human_seat()) assert(want.includes(input), `az move ${input} is legal`);
  const mev = JSON.parse(m.apply_human(input));
  bot.push(mev.label);
  const stats = JSON.parse(bot.stats());
  assert(Number.isFinite(stats.value), 'bot stats parse');
  plies++;
}
assert(plies >= 30 || m.is_over(), 'azero-gpu match advanced');
console.log('azero-gpu seam:', plies, 'plies, ok');

// The no-GPU fallback: the same externally driven seat, but the leaves are
// evaluated in-wasm by the reference forward (load_weights + play_cpu) instead
// of WebGPU — the exact path a visitor without a GPU hits. Locked to 1 sim.
const goWeights = await fetchAsset('azero/azero-go.azweb');
const gm = engine.create_match('go', JSON.stringify({ bot: 'azero-gpu', size: 9, seat: 0, seed: 5 }));
assert(gm.step() === '', 'no engine-side bot moves in an externally driven go match');
const goBot = new engine.AzGoBot(1, 8, 5, 9);
goBot.load_weights(new Uint8Array(goWeights));
let goPlies = 0;
while (!gm.is_over() && goPlies < 4) {
  const turn = gm.to_act();
  const want = JSON.parse(gm.legal_labels());
  const input = turn === gm.human_seat() ? (want.find((l) => l !== 'pass') ?? want[0]) : goBot.play_cpu();
  if (turn !== gm.human_seat()) assert(want.includes(input), `cpu move ${input} is legal`);
  const mev = JSON.parse(gm.apply_human(input));
  goBot.push(mev.label);
  goPlies++;
}
assert(goPlies >= 4, 'go CPU fallback advanced');
console.log('azero CPU fallback:', goPlies, 'plies, ok');

console.log('SMOKE OK');
