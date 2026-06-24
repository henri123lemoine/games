// Doom — 1v1 vs AI. The human drives seat 0 (rendered POV), the trained PPO bot
// drives seat 1 via the tch-free forward (forward.js). One doomrl WASM engine
// steps both seats' ticcmds per tic; JS blits seat 0's framebuffer to canvas.
import { DoomBot, decodeAction, parseWeights, PLAYER_STATE_FLOATS } from "./forward.js";
import DoomRL from "./doomrl.js";

const TICRATE = 35;
const RESX = 640;
const RESY = 400;

const SPEED = 50; // forward/back move units
const TURN = 900; // human turn rate per tic

const els = {
  canvas: document.getElementById("canvas"),
  overlay: document.getElementById("overlay"),
  start: document.getElementById("start"),
  status: document.getElementById("status"),
  hud: document.getElementById("hud"),
};

const keys = Object.create(null);
let running = false;

// Twin-stick touch controls feed the SAME `keys` map the keyboard writes, so the
// rest of the pipeline (humanAction) is untouched. Three independent systems —
// left move stick, right drag-to-turn, and fire/use buttons — each track their
// own pointerId so move + turn + fire register simultaneously and one finger's
// release never clears another's keys.
const STICK_DEADZONE = 16; // px from stick centre before a move key engages
const STICK_RADIUS = 52; // knob travel cap
const TURN_DEADZONE = 4; // px of horizontal drag before a turn key engages
const TURN_IDLE_MS = 90; // release turn keys after the finger stops moving

function setupTouchControls() {
  const isTouch =
    "ontouchstart" in window ||
    navigator.maxTouchPoints > 0 ||
    matchMedia("(pointer: coarse)").matches;
  if (!isTouch) return;
  document.body.classList.add("touch", "menu");

  setupMoveStick();
  setupTurnZone();
  for (const btn of document.querySelectorAll("#touch .btn")) setupActionButton(btn);
}

// Pointer capture keeps a drag bound to its zone even if the finger wanders over
// a sibling button; harmless if it throws (synthetic events, unsupported).
function capture(el, pointerId) {
  try {
    el.setPointerCapture(pointerId);
  } catch {}
}

function clearMoveKeys() {
  keys["KeyW"] = keys["KeyS"] = keys["KeyA"] = keys["KeyD"] = false;
}

function setupMoveStick() {
  const zone = document.getElementById("move-zone");
  const stick = document.getElementById("stick");
  const knob = document.getElementById("knob");
  let pointerId = null;
  let originX = 0;
  let originY = 0;

  const place = (el, x, y) => {
    el.style.left = x + "px";
    el.style.top = y + "px";
  };

  const onDown = (e) => {
    if (pointerId !== null) return;
    e.preventDefault();
    pointerId = e.pointerId;
    capture(zone, pointerId);
    originX = e.clientX;
    originY = e.clientY;
    place(stick, originX, originY);
    knob.style.transform = "translate(0px, 0px)";
    stick.classList.add("active");
  };

  const onMove = (e) => {
    if (e.pointerId !== pointerId) return;
    e.preventDefault();
    let dx = e.clientX - originX;
    let dy = e.clientY - originY;
    const dist = Math.hypot(dx, dy);
    // 8-way thresholding off the drag angle keeps strafing crisp like a D-pad
    // while feeling analog: forward/back from vertical, strafe from horizontal.
    clearMoveKeys();
    if (dist >= STICK_DEADZONE) {
      const ax = Math.abs(dx);
      const ay = Math.abs(dy);
      if (dy < 0 && ay > ax * 0.4) keys["KeyW"] = true;
      if (dy > 0 && ay > ax * 0.4) keys["KeyS"] = true;
      if (dx < 0 && ax > ay * 0.4) keys["KeyA"] = true;
      if (dx > 0 && ax > ay * 0.4) keys["KeyD"] = true;
    }
    const clamp = Math.min(dist, STICK_RADIUS) / (dist || 1);
    knob.style.transform = `translate(${dx * clamp}px, ${dy * clamp}px)`;
  };

  const onUp = (e) => {
    if (e.pointerId !== pointerId) return;
    e.preventDefault();
    pointerId = null;
    clearMoveKeys();
    stick.classList.remove("active");
  };

  zone.addEventListener("pointerdown", onDown);
  zone.addEventListener("pointermove", onMove);
  zone.addEventListener("pointerup", onUp);
  zone.addEventListener("pointercancel", onUp);
}

function clearTurnKeys() {
  keys["ArrowLeft"] = keys["ArrowRight"] = false;
}

function setupTurnZone() {
  const zone = document.getElementById("turn-zone");
  let pointerId = null;
  let lastX = 0;
  let idle = null;

  const stopTurning = () => clearTurnKeys();

  const onDown = (e) => {
    if (pointerId !== null) return;
    e.preventDefault();
    pointerId = e.pointerId;
    capture(zone, pointerId);
    lastX = e.clientX;
    document.body.classList.add("aiming");
  };

  // DOOM auto-aims vertically, so only horizontal drag turns. The key is held
  // for the direction of the most recent horizontal motion and released shortly
  // after the finger stops, giving continuous turning while dragging.
  const onMove = (e) => {
    if (e.pointerId !== pointerId) return;
    e.preventDefault();
    const dx = e.clientX - lastX;
    lastX = e.clientX;
    if (Math.abs(dx) >= TURN_DEADZONE) {
      keys["ArrowRight"] = dx > 0;
      keys["ArrowLeft"] = dx < 0;
      if (idle) clearTimeout(idle);
      idle = setTimeout(stopTurning, TURN_IDLE_MS);
    }
  };

  const onUp = (e) => {
    if (e.pointerId !== pointerId) return;
    e.preventDefault();
    pointerId = null;
    if (idle) clearTimeout(idle);
    clearTurnKeys();
    document.body.classList.remove("aiming");
  };

  zone.addEventListener("pointerdown", onDown);
  zone.addEventListener("pointermove", onMove);
  zone.addEventListener("pointerup", onUp);
  zone.addEventListener("pointercancel", onUp);
}

function setupActionButton(btn) {
  const code = btn.dataset.code;
  let pointerId = null;
  const press = (e) => {
    if (pointerId !== null) return;
    e.preventDefault();
    pointerId = e.pointerId;
    keys[code] = true;
    btn.classList.add("pressed");
  };
  const release = (e) => {
    if (e.pointerId !== pointerId) return;
    e.preventDefault();
    pointerId = null;
    keys[code] = false;
    btn.classList.remove("pressed");
  };
  btn.addEventListener("pointerdown", press);
  btn.addEventListener("pointerup", release);
  btn.addEventListener("pointercancel", release);
  btn.addEventListener("pointerleave", release);
}

function humanAction() {
  // Keyboard -> seat 0 ticcmd. Arrows/WASD move + turn; Ctrl/Space fire; E use.
  let forward = 0,
    side = 0,
    turn = 0,
    fire = 0;
  if (keys["KeyW"] || keys["ArrowUp"]) forward += SPEED;
  if (keys["KeyS"] || keys["ArrowDown"]) forward -= SPEED;
  if (keys["ArrowLeft"]) turn += TURN;
  if (keys["ArrowRight"]) turn -= TURN;
  if (keys["KeyA"]) side -= SPEED;
  if (keys["KeyD"]) side += SPEED;
  if (keys["ControlLeft"] || keys["ControlRight"] || keys["Space"]) fire = 1;
  return { forward, side, turn, fire, use: keys["KeyE"] ? 1 : 0, weapon: 0 };
}

async function boot() {
  els.status.textContent = "loading engine…";
  const Module = await DoomRL();

  // typed C entry points
  const c = {
    init: Module.cwrap("web_init", null, []),
    setAction: Module.cwrap("web_set_action", null, [
      "number", "number", "number", "number", "number", "number", "number",
    ]),
    step: Module.cwrap("web_step", null, []),
    spawnNear: Module.cwrap("web_spawn_near", null, ["number"]),
    reset: Module.cwrap("web_reset", null, []),
    screenbuffer: Module.cwrap("web_screenbuffer", "number", []),
    playerState: Module.cwrap("web_player_state", null, ["number", "number"]),
  };

  els.status.textContent = "loading bot weights…";
  const buf = await (await fetch("./doomppo_best.bin")).arrayBuffer();
  const bot = new DoomBot(parseWeights(buf));

  els.status.textContent = "starting match…";
  c.init();
  c.spawnNear(384);
  bot.reset();

  // scratch buffer in WASM heap for web_player_state output
  const statePtr = Module._malloc(PLAYER_STATE_FLOATS * 4);
  const readState = (seat) => {
    c.playerState(seat, statePtr);
    return Module.HEAPF32.subarray(
      statePtr >> 2,
      (statePtr >> 2) + PLAYER_STATE_FLOATS,
    );
  };

  const ctx = els.canvas.getContext("2d");
  const img = ctx.createImageData(RESX, RESY);

  function blit() {
    const ptr = c.screenbuffer();
    if (!ptr) return;
    const src = Module.HEAPU8.subarray(ptr, ptr + RESX * RESY * 4);
    const dst = img.data;
    // doomgeneric rgba8888 stores bytes [B,G,R,A]; canvas wants [R,G,B,A].
    for (let i = 0; i < RESX * RESY * 4; i += 4) {
      dst[i] = src[i + 2];
      dst[i + 1] = src[i + 1];
      dst[i + 2] = src[i];
      dst[i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  function hud() {
    const me = readState(0);
    const myFrags = me[11];
    const oppState = readState(1);
    const botFrags = oppState[11];
    els.hud.textContent = `YOU ${myFrags}  —  AI ${botFrags}    [hp ${Math.max(0, me[7] | 0)}]`;
  }

  // Validation/demo mode: drive seat 0 with the net too (bot-vs-bot), so the
  // match advances and fights with no human. Enable with ?botboth.
  const params = new URLSearchParams(location.search);
  const botBoth = params.has("botboth");
  const bot0 = botBoth ? new DoomBot(parseWeights(buf)) : null;
  if (bot0) bot0.reset();

  let acc = 0;
  let last = performance.now();
  let ticCount = 0;
  const TIC_MS = 1000 / TICRATE;

  function frame(now) {
    if (!running) return;
    acc += now - last;
    last = now;
    let steps = 0;
    while (acc >= TIC_MS && steps < 4) {
      // seat 0 = human (or the net in ?botboth); seat 1 = bot, each acting from
      // its OWN LOS-gated state.
      if (bot0) {
        const a = decodeAction(bot0.act(readState(0)));
        c.setAction(0, a.forward, a.side, a.turn, a.fire, a.use, a.weapon);
      } else {
        const ha = humanAction();
        c.setAction(0, ha.forward, ha.side, ha.turn, ha.fire, ha.use, ha.weapon);
      }

      const botState = readState(1);
      const ba = decodeAction(bot.act(botState));
      c.setAction(1, ba.forward, ba.side, ba.turn, ba.fire, ba.use, ba.weapon);

      c.step();
      ticCount++;
      // keep the duel close: re-converge if both alive and they drift apart, so
      // an idle wander can't masquerade as the match (and the bot keeps fighting).
      if (ticCount % 350 === 0) {
        const a = readState(0), b = readState(1);
        const dx = a[1] - b[1], dy = a[2] - b[2];
        if (a[0] && b[0] && dx * dx + dy * dy > 700 * 700) c.spawnNear(384);
      }
      acc -= TIC_MS;
      steps++;
    }
    blit();
    hud();
    requestAnimationFrame(frame);
  }

  els.start.disabled = false;
  els.status.textContent = "ready";
  els.start.addEventListener("click", () => {
    els.overlay.style.display = "none";
    els.canvas.focus();
    document.body.classList.remove("menu");
    running = true;
    last = performance.now();
    acc = 0;
    requestAnimationFrame(frame);
  });

  // expose for the validation harness to assert the bot is fighting
  window.__doomAI = { readState, running: () => running };
}

window.addEventListener(
  "keydown",
  (e) => {
    keys[e.code] = true;
    if (running) e.preventDefault();
  },
  { passive: false },
);
window.addEventListener(
  "keyup",
  (e) => {
    keys[e.code] = false;
    if (running) e.preventDefault();
  },
  { passive: false },
);

setupTouchControls();

// expose the synthesized human intent so the validation harness can assert the
// touch wiring without the WASM engine loaded
window.__doomInput = humanAction;

boot().catch((err) => {
  console.error(err);
  els.status.textContent = "error: " + err.message;
});
