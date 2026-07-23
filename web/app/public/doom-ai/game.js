// Strategic Doom FFA. Seat 0 is the rendered human; seats 1-3 can be
// deterministic, route-aware tactical opponents. The future strategic PPO
// policy can replace TacticalBot at this seam once a compatible
// 40-input/486-action net exists.
import DoomRL from "./doomrl.js";
import { PLAYER_STATE_FLOATS, S, TacticalBot } from "./tactical.js";

const TICRATE = 35;
const MAX_PLAYERS = 4;
const MOVE = 50;
const KEY_TURN = 900;

// The one-sided linedef loops that must remain solid. Keep these in the smoke
// telemetry so long bot soaks catch collision regressions, not just rendering
// regressions. Sector-height borders (reactor, stairs, shrines) are omitted
// because they are intentionally traversable.
const SOLID_RECTS = [
  [-712, -568, 240, 620], [-712, -568, -620, -240],
  [568, 712, 240, 620], [568, 712, -620, -240],
  [-435, -225, 210, 390], [225, 435, -390, -210],
  [-1020, -840, 120, 340], [840, 1020, -340, -120],
  [-170, 170, 523, 647], [-170, 170, -647, -523],
];
const ARENA_OUTLINE = [
  [-1120, 768], [1120, 768], [1280, 608], [1280, -608],
  [1120, -768], [-1120, -768], [-1280, -608], [-1280, 608],
];

const canvas = document.getElementById("canvas");
const overlay = document.getElementById("overlay");
const startButton = document.getElementById("start");
const setupBotsButton = document.getElementById("setup-bots");
const setupDifficultyButton = document.getElementById("setup-difficulty");
const pauseOverlay = document.getElementById("pause-overlay");
const pauseContinueButton = document.getElementById("pause-continue");
const pauseHomeButton = document.getElementById("pause-home");

const keys = Object.create(null);
let mouseTurn = 0;
let mouseFire = false;
let running = false;
let pauseArena = null;
let resumeArena = null;
let movePauseSelection = null;
let activatePauseSelection = null;
let moveSetupSelection = null;
let changeSetupValue = null;
let activateSetupSelection = null;

const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));

function pointInPolygon(x, y, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function wallBreachAt(x, y) {
  if (!pointInPolygon(x, y, ARENA_OUTLINE)) return "outer";
  const rect = SOLID_RECTS.findIndex(([left, right, bottom, top]) => (
    x > left && x < right && y > bottom && y < top
  ));
  return rect >= 0 ? `solid-${rect}` : null;
}

function humanAction() {
  let forward = 0;
  let side = 0;
  let turn = 0;
  if (keys.KeyW || keys.ArrowUp) forward += MOVE;
  if (keys.KeyS || keys.ArrowDown) forward -= MOVE;
  if (keys.KeyA) side -= MOVE;
  if (keys.KeyD) side += MOVE;
  if (keys.ArrowLeft) turn += KEY_TURN;
  if (keys.ArrowRight) turn -= KEY_TURN;
  turn -= clamp(Math.round(mouseTurn * 34), -1300, 1300);
  mouseTurn = 0;

  let weapon = 0;
  if (keys.Digit1) weapon = 1;
  else if (keys.Digit3) weapon = 3;
  else if (keys.Digit4) weapon = 4;
  else if (keys.Digit5) weapon = 5;

  return {
    forward,
    side,
    turn,
    fire: mouseFire || keys.Space || keys.ControlLeft || keys.ControlRight ? 1 : 0,
    use: keys.KeyE ? 1 : 0,
    weapon,
  };
}

function sendAction(c, seat, action) {
  c.setAction(
    seat,
    action.forward,
    action.side,
    action.turn,
    action.fire,
    action.use,
    action.weapon,
  );
}

async function boot() {
  const Module = await DoomRL();
  const params = new URLSearchParams(location.search);
  const botBoth = params.has("botboth");
  const difficulties = ["casual", "standard", "relentless"];
  let selectedBots = clamp(Number(params.get("bots")) || 1, 1, 3);
  let selectedDifficulty = difficulties.includes(params.get("difficulty"))
    ? params.get("difficulty")
    : "standard";
  const c = {
    init: Module.cwrap("web_init", null, []),
    setAction: Module.cwrap("web_set_action", null, [
      "number", "number", "number", "number", "number", "number", "number",
    ]),
    step: Module.cwrap("web_step", null, []),
    drawPause: Module.cwrap("web_draw_pause", null, ["number"]),
    drawSetup: Module.cwrap("web_draw_setup", null, ["number", "number", "number"]),
    reset: Module.cwrap("web_reset", null, []),
    setPlayerCount: Module.cwrap("web_set_player_count", "number", ["number"]),
    numPlayers: Module.cwrap("web_num_players", "number", []),
    screenbuffer: Module.cwrap("web_screenbuffer", "number", []),
    screenWidth: Module.cwrap("web_screen_w", "number", []),
    screenHeight: Module.cwrap("web_screen_h", "number", []),
    playerState: Module.cwrap("web_player_state", null, ["number", "number"]),
  };

  c.init();
  const screenWidth = c.screenWidth();
  const screenHeight = c.screenHeight();
  if (screenWidth <= 0 || screenHeight <= 0) {
    throw new Error(`invalid framebuffer dimensions: ${screenWidth}x${screenHeight}`);
  }
  canvas.width = screenWidth;
  canvas.height = screenHeight;
  const statePtrs = Array.from(
    { length: MAX_PLAYERS },
    () => Module._malloc(PLAYER_STATE_FLOATS * 4),
  );
  const readState = (seat) => {
    c.playerState(seat, statePtrs[seat]);
    return new Float32Array(
      Module.HEAPF32.subarray(statePtrs[seat] >> 2, (statePtrs[seat] >> 2) + PLAYER_STATE_FLOATS),
    );
  };

  const timeScale = botBoth ? clamp(Number(params.get("speed")) || 1, 1, 8) : 1;
  let playerCount = c.numPlayers();
  let bots = [];
  let bot0 = null;
  let configuredDifficulty = "";

  function configureMatch() {
    const wantedPlayers = selectedBots + 1;
    const difficulty = selectedDifficulty;
    const changed = wantedPlayers !== playerCount || difficulty !== configuredDifficulty;
    if (!changed) return false;

    if (wantedPlayers !== playerCount) playerCount = c.setPlayerCount(wantedPlayers);
    configuredDifficulty = difficulty;
    bots = Array.from(
      { length: playerCount - 1 },
      (_unused, index) => new TacticalBot(index + 1, configuredDifficulty),
    );
    bot0 = botBoth ? new TacticalBot(0, configuredDifficulty) : null;
    return true;
  }

  const ctx = canvas.getContext("2d", { alpha: false });
  ctx.imageSmoothingEnabled = false;
  const image = ctx.createImageData(screenWidth, screenHeight);
  const frameBytes = screenWidth * screenHeight * 4;
  const metrics = { tics: 0, frames: 0 };
  let maxObservedZ = 0;
  let wallBreaches = 0;
  let wallBreachDetails = [];
  let pauseSelection = 0;
  let setupSelection = 0;

  function blit() {
    const ptr = c.screenbuffer();
    if (!ptr) return;
    const src = Module.HEAPU8.subarray(ptr, ptr + frameBytes);
    const dst = image.data;
    for (let i = 0; i < dst.length; i += 4) {
      dst[i] = src[i + 2];
      dst[i + 1] = src[i + 1];
      dst[i + 2] = src[i];
      dst[i + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
    metrics.frames += 1;
  }

  function updateHud() {
    const states = Array.from({ length: playerCount }, (_unused, seat) => readState(seat));
    maxObservedZ = Math.max(maxObservedZ, ...states.map((state) => state[3]));
    states.forEach((state, seat) => {
      if (state[S.alive] < 0.5) return;
      const wall = wallBreachAt(state[S.x], state[S.y]);
      if (wall === null) return;
      wallBreaches += 1;
      if (wallBreachDetails.length < 16) {
        wallBreachDetails.push({
          tic: metrics.tics,
          seat,
          wall,
          x: Math.round(state[S.x]),
          y: Math.round(state[S.y]),
        });
      }
    });
    // Keep compact, DOM-visible telemetry for the browser smoke test. This is
    // deliberately read-only and contains nothing the tactical bots cannot see.
    canvas.dataset.players = JSON.stringify(states.map((state, seat) => ({
      seat,
      alive: state[S.alive] > 0.5,
      x: Math.round(state[S.x]),
      y: Math.round(state[S.y]),
      z: Math.round(state[3]),
      frags: state[S.frags] | 0,
      deaths: state[S.deaths] | 0,
    })));
    canvas.dataset.tics = String(metrics.tics);
    canvas.dataset.playerCount = String(playerCount);
    canvas.dataset.maxZ = String(Math.round(maxObservedZ));
    canvas.dataset.wallBreaches = String(wallBreaches);
    canvas.dataset.wallBreachDetails = JSON.stringify(wallBreachDetails);
  }

  let nextHudTic = 0;

  function resetMatch() {
    c.reset();
    for (const bot of bots) bot.reset();
    if (bot0) bot0.reset();
    metrics.tics = 0;
    maxObservedZ = 0;
    wallBreaches = 0;
    wallBreachDetails = [];
    nextHudTic = 0;
    mouseTurn = 0;
    updateHud();
    blit();
  }

  let accumulator = 0;
  let last = performance.now();
  const ticMs = 1000 / TICRATE;

  function frame(now) {
    if (!running) return;
    accumulator += (now - last) * timeScale;
    last = now;
    let steps = 0;
    while (accumulator >= ticMs && steps < 32) {
      sendAction(c, 0, bot0 ? bot0.act(readState(0)) : humanAction());
      for (let seat = 1; seat < playerCount; seat += 1) {
        sendAction(c, seat, bots[seat - 1].act(readState(seat)));
      }
      c.step();
      metrics.tics += 1;
      accumulator -= ticMs;
      steps += 1;
    }
    if (steps > 0) {
      blit();
      if (metrics.tics >= nextHudTic) {
        updateHud();
        nextHudTic = metrics.tics + 4;
      }
    }
    requestAnimationFrame(frame);
  }

  function startMatch() {
    if (running) return;
    if (configureMatch()) resetMatch();
    overlay.hidden = true;
    pauseOverlay.hidden = true;
    running = true;
    last = performance.now();
    accumulator = 0;
    if (!botBoth) canvas.requestPointerLock?.();
    requestAnimationFrame(frame);
  }

  function renderSetup() {
    c.drawSetup(selectedBots, difficulties.indexOf(selectedDifficulty), setupSelection);
    blit();
  }

  function showSetup() {
    running = false;
    mouseFire = false;
    for (const key of Object.keys(keys)) keys[key] = false;
    pauseOverlay.hidden = true;
    overlay.hidden = false;
    document.exitPointerLock?.();
    setupSelection = 0;
    renderSetup();
  }

  function setSetupSelection(selection) {
    if (overlay.hidden) return;
    setupSelection = clamp(selection, 0, 2);
    renderSetup();
  }

  function cycleSetupValue(direction) {
    if (overlay.hidden) return;
    if (setupSelection === 0) {
      selectedBots = ((selectedBots - 1 + direction + 3) % 3) + 1;
    } else if (setupSelection === 1) {
      const index = difficulties.indexOf(selectedDifficulty);
      selectedDifficulty = difficulties[(index + direction + difficulties.length) % difficulties.length];
    }
    renderSetup();
  }

  function activateSetup() {
    if (overlay.hidden) return;
    if (setupSelection === 2) startMatch();
    else cycleSetupValue(1);
  }

  function pauseMatch() {
    if (!running) return;
    running = false;
    mouseFire = false;
    for (const key of Object.keys(keys)) keys[key] = false;
    pauseOverlay.hidden = false;
    document.exitPointerLock?.();
    pauseSelection = 0;
    c.drawPause(pauseSelection);
    blit();
  }

  function resumeMatch() {
    if (running || pauseOverlay.hidden) return;
    pauseOverlay.hidden = true;
    running = true;
    last = performance.now();
    accumulator = 0;
    if (!botBoth) canvas.requestPointerLock?.();
    requestAnimationFrame(frame);
  }

  function returnToSetup() {
    resetMatch();
    showSetup();
  }

  function setPauseSelection(selection) {
    if (pauseOverlay.hidden) return;
    pauseSelection = selection ? 1 : 0;
    c.drawPause(pauseSelection);
    blit();
  }

  function activatePause() {
    if (pauseOverlay.hidden) return;
    if (pauseSelection === 0) resumeMatch();
    else returnToSetup();
  }

  configureMatch();
  startButton.addEventListener("click", startMatch);
  setupBotsButton.addEventListener("click", () => {
    setSetupSelection(0);
    cycleSetupValue(1);
  });
  setupBotsButton.addEventListener("pointerenter", () => setSetupSelection(0));
  setupDifficultyButton.addEventListener("click", () => {
    setSetupSelection(1);
    cycleSetupValue(1);
  });
  setupDifficultyButton.addEventListener("pointerenter", () => setSetupSelection(1));
  startButton.addEventListener("pointerenter", () => setSetupSelection(2));
  pauseArena = pauseMatch;
  resumeArena = resumeMatch;
  movePauseSelection = (direction) => setPauseSelection(pauseSelection + direction > 0 ? 1 : 0);
  activatePauseSelection = activatePause;
  moveSetupSelection = (direction) => setSetupSelection(setupSelection + direction);
  changeSetupValue = cycleSetupValue;
  activateSetupSelection = activateSetup;
  pauseContinueButton.addEventListener("click", resumeMatch);
  pauseContinueButton.addEventListener("pointerenter", () => setPauseSelection(0));
  pauseHomeButton.addEventListener("click", returnToSetup);
  pauseHomeButton.addEventListener("pointerenter", () => setPauseSelection(1));
  canvas.addEventListener("click", () => {
    if (running && !botBoth) canvas.requestPointerLock?.();
  });

  resetMatch();
  window.__doomStrategic = {
    readState,
    metrics,
    screen: { width: screenWidth, height: screenHeight },
    playerCount: () => playerCount,
    running: () => running,
    start: startMatch,
    reset: resetMatch,
  };
  if (params.has("autostart")) startMatch();
  else showSetup();
}

const blockedKeys = new Set([
  "KeyW", "KeyA", "KeyS", "KeyD", "KeyE", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
  "Space", "ControlLeft", "ControlRight", "Digit1", "Digit3", "Digit4", "Digit5",
]);

window.addEventListener("keydown", (event) => {
  if (!overlay.hidden && ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Enter", "Space"].includes(event.code)) {
    event.preventDefault();
    if (event.code === "ArrowUp") moveSetupSelection?.(-1);
    else if (event.code === "ArrowDown") moveSetupSelection?.(1);
    else if (event.code === "ArrowLeft") changeSetupValue?.(-1);
    else if (event.code === "ArrowRight") changeSetupValue?.(1);
    else activateSetupSelection?.();
    return;
  }
  if (!pauseOverlay.hidden && ["ArrowUp", "ArrowDown", "Enter", "Space"].includes(event.code)) {
    event.preventDefault();
    if (event.code === "ArrowUp") movePauseSelection?.(-1);
    else if (event.code === "ArrowDown") movePauseSelection?.(1);
    else activatePauseSelection?.();
    return;
  }
  if (event.code === "Escape" && !event.repeat) {
    event.preventDefault();
    if (running) pauseArena?.();
    else if (!pauseOverlay.hidden) resumeArena?.();
    return;
  }
  keys[event.code] = true;
  if (running && blockedKeys.has(event.code)) event.preventDefault();
}, { passive: false });
window.addEventListener("keyup", (event) => {
  keys[event.code] = false;
  if (running && blockedKeys.has(event.code)) event.preventDefault();
}, { passive: false });
window.addEventListener("mousemove", (event) => {
  if (document.pointerLockElement === canvas) mouseTurn += event.movementX;
});
window.addEventListener("mousedown", (event) => {
  if (event.button === 0 && running) mouseFire = true;
});
window.addEventListener("mouseup", (event) => {
  if (event.button === 0) mouseFire = false;
});
window.addEventListener("blur", () => {
  mouseFire = false;
  for (const key of Object.keys(keys)) keys[key] = false;
});
canvas.addEventListener("contextmenu", (event) => event.preventDefault());

boot().catch((error) => {
  console.error(error);
});
