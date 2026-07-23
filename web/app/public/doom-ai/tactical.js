// A deterministic, observation-respecting opponent for the strategic arena.
// It sees the same per-seat state surface exposed to the future PPO policy:
// its own pose/economy, LOS-gated opponent data, and public item timers. The
// only map knowledge is the fixed objective locations and the hall mouths.

export const PLAYER_STATE_FLOATS = 39;

export const S = Object.freeze({
  alive: 0,
  x: 1,
  y: 2,
  angle: 4,
  health: 7,
  armor: 8,
  armorType: 9,
  readyWeapon: 10,
  bullets: 11,
  shells: 12,
  cells: 13,
  rockets: 14,
  frags: 15,
  deaths: 16,
  opponentVisible: 17,
  opponentBearing: 18,
  opponentDistance: 19,
  opponentHealth: 22,
  item0: 27,
});

const READY_SHOTGUN = 2;
const READY_ROCKET = 4;
const SLOT_SHOTGUN = 3;
const SLOT_ROCKET = 5;

const OBJECTIVES = [
  { name: "rocket", x: 0, y: 0, state: S.item0 },
  { name: "armor", x: 980, y: 430, state: S.item0 + 4 },
  { name: "soul", x: -930, y: -500, state: S.item0 + 8 },
];

const DIFFICULTIES = Object.freeze({
  casual: { reaction: 8, jitter: 12, aimGain: 42, move: 35, strafe: 18, fireCycle: 16, fireWindow: 4 },
  standard: { reaction: 5, jitter: 7, aimGain: 52, move: 40, strafe: 24, fireCycle: 11, fireWindow: 4 },
  relentless: { reaction: 1, jitter: 1.5, aimGain: 78, move: 50, strafe: 40, fireCycle: 1, fireWindow: 1 },
});

const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));

function wrap180(degrees) {
  let value = degrees;
  while (value > 180) value -= 360;
  while (value < -180) value += 360;
  return value;
}

function distance(a, b, x, y) {
  return Math.hypot(a - x, b - y);
}

export class TacticalBot {
  constructor(seat = 1, difficulty = "standard") {
    this.seat = seat;
    this.profile = DIFFICULTIES[difficulty] ?? DIFFICULTIES.standard;
    this.reset();
  }

  reset() {
    this.tick = 0;
    this.patrol = this.seat % OBJECTIVES.length;
    this.lastX = null;
    this.lastY = null;
    this.lastMoving = false;
    this.unstickUntil = 0;
    this.unstickDirection = this.seat === 0 ? -1 : 1;
    this.nextCombatDecision = 0;
    this.lastCombatAction = null;
  }

  objective(state) {
    const available = (item) => state[item.state] > 0.5;
    if (state[S.health] < 72 && available(OBJECTIVES[2])) return OBJECTIVES[2];
    if (state[S.armor] < 80 && available(OBJECTIVES[1])) return OBJECTIVES[1];
    if (state[S.rockets] < 2 && available(OBJECTIVES[0])) return OBJECTIVES[0];

    let target = OBJECTIVES[this.patrol];
    if (distance(state[S.x], state[S.y], target.x, target.y) < 90) {
      this.patrol = (this.patrol + 1) % OBJECTIVES.length;
      target = OBJECTIVES[this.patrol];
    }
    return target;
  }

  // Cross-base travel uses the gatehouses' wide center doors. Human players
  // can also flank through the north and south lanes around each gatehouse.
  waypoint(x, y, target) {
    if (x < -700 && target.x > -700) {
      if (distance(x, y, -720, 0) > 72) return { x: -720, y: 0 };
      return { x: Math.min(target.x, 0), y: 0 };
    }
    if (x > 700 && target.x < 700) {
      if (distance(x, y, 720, 0) > 72) return { x: 720, y: 0 };
      return { x: Math.max(target.x, 0), y: 0 };
    }
    if (Math.abs(x) <= 730 && target.x > 730) return { x: 740, y: 0 };
    if (Math.abs(x) <= 730 && target.x < -730) return { x: -740, y: 0 };
    return target;
  }

  navigationAction(state) {
    const target = this.objective(state);
    const waypoint = this.waypoint(state[S.x], state[S.y], target);
    const wanted = Math.atan2(waypoint.y - state[S.y], waypoint.x - state[S.x]) * 180 / Math.PI;
    const bearing = wrap180(wanted - state[S.angle]);
    const near = distance(state[S.x], state[S.y], waypoint.x, waypoint.y);

    if (this.unstickUntil > this.tick) {
      this.lastMoving = true;
      return {
        forward: -40,
        side: 40 * this.unstickDirection,
        turn: 1100 * this.unstickDirection,
        fire: 0,
        use: 0,
        weapon: 0,
      };
    }

    const moving = near > 48 && Math.abs(bearing) < 76;
    this.lastMoving = moving;
    return {
      forward: moving ? this.profile.move : 0,
      side: 0,
      turn: clamp(Math.round(bearing * 72), -1300, 1300),
      fire: 0,
      use: 0,
      weapon: this.preferredWeapon(state, 9999),
    };
  }

  preferredWeapon(state, opponentDistance) {
    if (state[S.rockets] > 0 && opponentDistance > 300 && state[S.readyWeapon] !== READY_ROCKET)
      return SLOT_ROCKET;
    if (state[S.shells] > 0 && opponentDistance <= 520 && state[S.readyWeapon] !== READY_SHOTGUN)
      return SLOT_SHOTGUN;
    return 0;
  }

  combatAction(state) {
    const trueBearing = state[S.opponentBearing];
    const opponentDistance = state[S.opponentDistance];
    const usingRocket = state[S.readyWeapon] === READY_ROCKET && state[S.rockets] > 0;
    const drift = Math.sin(this.tick * 0.17 + this.seat * 1.91) * this.profile.jitter;
    const bearing = trueBearing + drift;
    const aimWindow = usingRocket ? 5 : 10;
    const weave = ((Math.floor(this.tick / 42) + this.seat) & 1) === 0 ? -1 : 1;
    const firingBeat = this.tick % this.profile.fireCycle;

    this.lastMoving = true;
    return {
      forward: Math.abs(bearing) > 58 ? 0 : opponentDistance > 300 ? this.profile.move : opponentDistance < 135 ? -35 : 0,
      side: opponentDistance < 620 ? this.profile.strafe * weave : 0,
      turn: clamp(Math.round(bearing * this.profile.aimGain), -1300, 1300),
      fire: Math.abs(bearing) < aimWindow && firingBeat < this.profile.fireWindow ? 1 : 0,
      use: 0,
      weapon: this.preferredWeapon(state, opponentDistance),
    };
  }

  updateStuckState(state) {
    if (this.tick % 20 !== 0) return;
    if (this.lastX !== null) {
      const moved = distance(state[S.x], state[S.y], this.lastX, this.lastY);
      if (this.lastMoving && moved < 10 && state[S.opponentVisible] < 0.5) {
        this.unstickDirection *= -1;
        this.unstickUntil = this.tick + 34;
      }
    }
    this.lastX = state[S.x];
    this.lastY = state[S.y];
  }

  act(state) {
    this.tick += 1;
    if (state[S.alive] < 0.5) {
      this.lastMoving = false;
      return { forward: 0, side: 0, turn: 0, fire: 0, use: 1, weapon: 0 };
    }
    this.updateStuckState(state);
    if (state[S.opponentVisible] > 0.5) {
      if (this.lastCombatAction === null || this.tick >= this.nextCombatDecision) {
        this.lastCombatAction = this.combatAction(state);
        this.nextCombatDecision = this.tick + this.profile.reaction;
      }
      return this.lastCombatAction;
    }
    this.lastCombatAction = null;
    return this.navigationAction(state);
  }
}
