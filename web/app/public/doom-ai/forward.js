// tch-free forward pass for the trained Doom PPO bot.
//
// TRAIN<->DEPLOY PARITY: the observation() and decodeAction() here are exact
// ports of doomtrain/src/env.rs (observation, TURNS/MOVES, decode_action). The
// per-seat numbers come from the WASM web_player_state() — the SAME LOS-gated
// state the trainer's PlayerState fed into observation(). Do not re-specify any
// input; this is the contract.

const OBS_DIM = 18;
const NUM_ACTIONS = 54;
const GRU_HIDDEN = 128;

const TURNS = [-1300, -700, -300, -120, 0, 120, 300, 700, 1300];
const MOVES = [-40, 0, 50];

// Layout of the 23 floats written by web_player_state() (see doomrl_web.c).
const S = {
  alive: 0, x: 1, y: 2, z: 3, angle_deg: 4, momx: 5, momy: 6,
  health: 7, armor: 8, ready_weapon: 9, ammo0: 10, frags: 11, deaths: 12,
  opponent_visible: 13, opp_bearing_deg: 14, opp_dist: 15,
  opp_rel_vx: 16, opp_rel_vy: 17, opp_health: 18,
  opp_mem_valid: 19, opp_mem_ticks: 20, opp_mem_last_bearing: 21, opp_mem_last_dist: 22,
};
export const PLAYER_STATE_FLOATS = 23;

const DEG2RAD = Math.PI / 180;

// Exact port of doomtrain/src/env.rs `observation`.
export function observation(s) {
  const ang = s[S.angle_deg] * DEG2RAD;
  const oppBear = s[S.opp_bearing_deg] * DEG2RAD;
  const memBear = s[S.opp_mem_last_bearing] * DEG2RAD;
  return new Float32Array([
    s[S.health] / 100.0,
    s[S.armor] / 100.0,
    s[S.ammo0] / 50.0,
    Math.sin(ang),
    Math.cos(ang),
    s[S.momx] / 16.0,
    s[S.momy] / 16.0,
    s[S.opponent_visible],
    Math.sin(oppBear),
    Math.cos(oppBear),
    Math.min(s[S.opp_dist] / 512.0, 8.0),
    s[S.opp_rel_vx] / 16.0,
    s[S.opp_rel_vy] / 16.0,
    s[S.opp_health] / 100.0,
    s[S.opp_mem_valid],
    Math.min(s[S.opp_mem_ticks] / 35.0, 20.0),
    Math.sin(memBear),
    Math.cos(memBear),
  ]);
}

// Exact port of env.rs `decode_action`: idx -> {forward, side, turn, fire, use, weapon}.
export function decodeAction(idx) {
  const fire = idx % 2;
  const m = Math.floor(idx / 2) % MOVES.length;
  const t = Math.floor(Math.floor(idx / 2) / MOVES.length);
  return { forward: MOVES[m], side: 0, turn: TURNS[t], fire, use: 0, weapon: 0 };
}

function relu(a) {
  for (let i = 0; i < a.length; i++) if (a[i] < 0) a[i] = 0;
  return a;
}

// y = W x + b, with W stored row-major [out, in].
function linear(W, b, x, outDim, inDim) {
  const y = new Float32Array(outDim);
  for (let o = 0; o < outDim; o++) {
    let acc = b[o];
    const base = o * inDim;
    for (let i = 0; i < inDim; i++) acc += W[base + i] * x[i];
    y[o] = acc;
  }
  return y;
}

function sigmoid(v) { return 1 / (1 + Math.exp(-v)); }

// Parse the DOOMDFP1 flat file into a name -> {dims, data} map.
export function parseWeights(buf) {
  const dv = new DataView(buf);
  let p = 0;
  const magic = new TextDecoder().decode(new Uint8Array(buf, 0, 8));
  if (magic !== "DOOMDFP1") throw new Error("bad magic: " + magic);
  p = 8;
  const n = dv.getUint32(p, true); p += 4;
  const out = {};
  for (let k = 0; k < n; k++) {
    const nlen = dv.getUint16(p, true); p += 2;
    const name = new TextDecoder().decode(new Uint8Array(buf, p, nlen)); p += nlen;
    const nd = dv.getUint8(p); p += 1;
    const dims = [];
    let numel = 1;
    for (let d = 0; d < nd; d++) { const v = dv.getUint32(p, true); p += 4; dims.push(v); numel *= v; }
    // Read floats via DataView — the byte offset is not guaranteed 4-aligned
    // (variable-length name strings precede it), so a Float32Array view would throw.
    const data = new Float32Array(numel);
    for (let i = 0; i < numel; i++) { data[i] = dv.getFloat32(p, true); p += 4; }
    out[name] = { dims, data };
  }
  return out;
}

// The bot policy: GRU recurrent actor; argmax of the policy logits each tic.
export class DoomBot {
  constructor(weights) {
    const w = (n) => {
      if (!weights[n]) throw new Error("missing tensor " + n);
      return weights[n].data;
    };
    this.obs_fc_w = w("obs_fc.weight"); this.obs_fc_b = w("obs_fc.bias");
    this.gru_ih_w = w("gru.weight_ih_l0"); this.gru_hh_w = w("gru.weight_hh_l0");
    this.gru_ih_b = w("gru.bias_ih_l0"); this.gru_hh_b = w("gru.bias_hh_l0");
    this.pih_w = w("pi_hidden.weight"); this.pih_b = w("pi_hidden.bias");
    this.pi_w = w("pi.weight"); this.pi_b = w("pi.bias");
    this.h = new Float32Array(GRU_HIDDEN); // recurrent hidden state
  }

  reset() { this.h.fill(0); }

  // PyTorch GRU cell (one layer). gates packed [r, z, n] each GRU_HIDDEN.
  gruStep(x) {
    const H = GRU_HIDDEN;
    const gi = linear(this.gru_ih_w, this.gru_ih_b, x, 3 * H, H); // x is already 128-d (obs_fc out)
    const gh = linear(this.gru_hh_w, this.gru_hh_b, this.h, 3 * H, H);
    const hNew = new Float32Array(H);
    for (let j = 0; j < H; j++) {
      const r = sigmoid(gi[j] + gh[j]);
      const z = sigmoid(gi[H + j] + gh[H + j]);
      const n = Math.tanh(gi[2 * H + j] + r * gh[2 * H + j]);
      hNew[j] = (1 - z) * n + z * this.h[j];
    }
    this.h = hNew;
    return hNew;
  }

  // state floats (23) -> action index. Greedy (argmax), as in eval.
  act(stateFloats) {
    const obs = observation(stateFloats);
    const enc = relu(linear(this.obs_fc_w, this.obs_fc_b, obs, GRU_HIDDEN, OBS_DIM));
    const h = this.gruStep(enc);
    const pih = relu(linear(this.pih_w, this.pih_b, h, 256, GRU_HIDDEN));
    const logits = linear(this.pi_w, this.pi_b, pih, NUM_ACTIONS, 256);
    let best = 0;
    for (let i = 1; i < NUM_ACTIONS; i++) if (logits[i] > logits[best]) best = i;
    return best;
  }
}
