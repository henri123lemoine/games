// Node smoke test for the wasm engine (no browser needed):
//   wasm-pack build web/engine --target web --out-dir pkg
//   node web/engine/smoke.mjs
// Exercises the manifest, a spectated match, a human turn, the pair/field
// runners, and the stats bindings.

import { readFile } from 'node:fs/promises';
import init, * as engine from './pkg/web_engine.js';

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

console.log('SMOKE OK');
