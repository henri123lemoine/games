// tch-free forward pass for the trained Doom PPO bot (strategic 1v1).
//
// TRAIN<->DEPLOY PARITY: observation() and decodeAction() here are exact ports of
// doomtrain/src/env.rs (observation, TURNS/MOVES/STRAFE/WEAPONS, decode_action).
// The per-seat numbers come from the WASM web_player_state() — the SAME LOS-gated
// state the trainer's PlayerState fed into observation(). The state-float layout,
// OBS layout and ACTION decode are the authoritative parity contract; see
// doomrl/STRATEGIC_CONTRACT.md. Do not re-specify any input.

const OBS_DIM = 40;
const NUM_ACTIONS = 486;
const GRU_HIDDEN = 128;

const TURNS = [-1300, -700, -300, -120, 0, 120, 300, 700, 1300];
const MOVES = [-40, 0, 50];
const STRAFE = [-40, 0, 40];
const WEAPONS = [0, 3, 5]; // Doom BT_CHANGE slot: keep / shotgun / rocket

const NUM_KEY_ITEMS = 3;
const ARENA_HALF = 1024.0;
const WP_SHOTGUN = 3, WP_CHAINGUN = 4, WP_ROCKET = 5;

// Layout of the 39 floats written by web_player_state() (see doomrl_web.c).
const S = {
  alive: 0, x: 1, y: 2, z: 3, angle_deg: 4, momx: 5, momy: 6,
  health: 7, armor: 8, armortype: 9, ready_weapon: 10,
  ammo_clip: 11, ammo_shell: 12, ammo_cell: 13, ammo_misl: 14,
  frags: 15, deaths: 16,
  opponent_visible: 17, opp_bearing_deg: 18, opp_dist: 19,
  opp_rel_vx: 20, opp_rel_vy: 21, opp_health: 22,
  opp_mem_valid: 23, opp_mem_ticks: 24, opp_mem_last_bearing: 25, opp_mem_last_dist: 26,
  // key items: base 27, 4 floats each [available, respawn_secs, bearing_deg, dist]
  item0: 27,
};
export const PLAYER_STATE_FLOATS = 39;

const DEG2RAD = Math.PI / 180;

// Exact port of doomtrain/src/env.rs `observation`.
export function observation(s) {
  const o = new Float32Array(OBS_DIM);
  const ang = s[S.angle_deg] * DEG2RAD;
  const oppBear = s[S.opp_bearing_deg] * DEG2RAD;
  const memBear = s[S.opp_mem_last_bearing] * DEG2RAD;
  o[0] = s[S.health] / 100.0;
  o[1] = s[S.armor] / 200.0;
  o[2] = s[S.armortype] === 1 ? 1 : 0;
  o[3] = s[S.armortype] === 2 ? 1 : 0;
  o[4] = s[S.ammo_clip] / 200.0;
  o[5] = s[S.ammo_shell] / 50.0;
  o[6] = s[S.ammo_cell] / 300.0;
  o[7] = s[S.ammo_misl] / 50.0;
  o[8] = s[S.ready_weapon] === WP_SHOTGUN ? 1 : 0;
  o[9] = s[S.ready_weapon] === WP_CHAINGUN ? 1 : 0;
  o[10] = s[S.ready_weapon] === WP_ROCKET ? 1 : 0;
  o[11] = Math.sin(ang);
  o[12] = Math.cos(ang);
  o[13] = s[S.x] / ARENA_HALF;
  o[14] = s[S.y] / ARENA_HALF;
  o[15] = s[S.momx] / 16.0;
  o[16] = s[S.momy] / 16.0;
  o[17] = s[S.opponent_visible];
  o[18] = Math.sin(oppBear);
  o[19] = Math.cos(oppBear);
  o[20] = Math.min(s[S.opp_dist] / 512.0, 8.0);
  o[21] = s[S.opp_rel_vx] / 16.0;
  o[22] = s[S.opp_rel_vy] / 16.0;
  o[23] = s[S.opp_health] / 100.0;
  o[24] = s[S.opp_mem_valid];
  o[25] = Math.min(s[S.opp_mem_ticks] / 35.0, 20.0);
  o[26] = Math.sin(memBear);
  o[27] = Math.cos(memBear);
  for (let k = 0; k < NUM_KEY_ITEMS; k++) {
    const si = S.item0 + k * 4;
    const avail = s[si];
    const respawn = s[si + 1];
    const bearing = s[si + 2] * DEG2RAD;
    const dist = s[si + 3];
    const invdist = Math.min(512.0 / Math.max(dist, 1.0), 1.0);
    const base = 28 + k * 4;
    o[base] = avail;
    o[base + 1] = Math.min(Math.max(respawn / 30.0, 0.0), 1.0);
    o[base + 2] = Math.sin(bearing) * invdist;
    o[base + 3] = Math.cos(bearing) * invdist;
  }
  return o;
}

// Exact port of env.rs `decode_action` (mixed-radix, least-significant first).
export function decodeAction(idx) {
  const weaponSel = idx % WEAPONS.length;
  let rest = Math.floor(idx / WEAPONS.length);
  const fire = rest % 2;
  rest = Math.floor(rest / 2);
  const strafeI = rest % STRAFE.length;
  rest = Math.floor(rest / STRAFE.length);
  const forwardI = rest % MOVES.length;
  const turnI = Math.floor(rest / MOVES.length);
  return {
    forward: MOVES[forwardI],
    side: STRAFE[strafeI],
    turn: TURNS[turnI],
    fire,
    use: 0,
    weapon: WEAPONS[weaponSel],
  };
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
    const gi = linear(this.gru_ih_w, this.gru_ih_b, x, 3 * H, H);
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

  // state floats (39) -> action index. Greedy (argmax), as in eval.
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
