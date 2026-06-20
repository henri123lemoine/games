//! The slither game, driven in the browser. One [`SlitherGame`] owns a
//! [`slither_rl::World`]: the human steers worm 0 by aiming its head at the
//! cursor, every other worm is driven by the trained net through
//! [`slitherinfer`]'s torch-free forward over each worm's own egocentric
//! observation — the same partial view the policy trained on, so the bots see
//! only what a viewport-limited player would.
//!
//! The page runs the loop (one [`SlitherGame::tick`] per animation frame) and
//! reads back a flat `Float32Array` snapshot to draw. Nothing here renders; the
//! TS frontend owns the canvas.

use slither_rl::Rng;
use slither_rl::env::Action;
use slither_rl::geometry::Vec2;
use slither_rl::obs::view_radius;
use slither_rl::world::{START_LENGTH, WORLD, World, WorldConfig, Worm, WormControl};
use slitherinfer::Model;
use slitherinfer::obs::{ObsMemory, act};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// The human's worm is always seat 0.
const HUMAN: usize = 0;
/// Bots spawn at least this far from the human's center spawn, so a fresh human
/// isn't rammed in the first second — the one concession to spawn fairness. The
/// human itself spawns *small* (`START_LENGTH`), exactly like the bots and like
/// real slither.io: no length head-start, so the trained net is a real opponent.
const SPAWN_CLEARANCE: f32 = 700.0;

/// Build the arena: the human at the center facing a random way (clear of the
/// walls and of an instant bot ram), and `bot_count` bots scattered with a
/// margin from the walls and from the human. Everyone spawns at the same small
/// `START_LENGTH` (real slither.io has no head-start); the only fairness aid is
/// the no-instant-ram spawn clearance.
fn build_world(seed: u64, bot_count: usize, pellet_target: usize) -> World {
    let mut rng = Rng::new(seed);
    let center = Vec2::new(WORLD * 0.5, WORLD * 0.5);
    let mut worms = Vec::with_capacity(bot_count + 1);
    worms.push(Worm::spawn(
        center,
        rng.range(0.0, std::f32::consts::TAU),
        START_LENGTH,
    ));
    let margin = 320.0;
    for _ in 0..bot_count {
        // Rejection-sample a spawn that clears the human's center.
        let mut pos = center;
        for _ in 0..16 {
            let p = Vec2::new(
                rng.range(margin, WORLD - margin),
                rng.range(margin, WORLD - margin),
            );
            if p.dist(center) >= SPAWN_CLEARANCE {
                pos = p;
                break;
            }
        }
        let angle = rng.range(0.0, std::f32::consts::TAU);
        let length = START_LENGTH + rng.range(0.0, 40.0);
        worms.push(Worm::spawn(pos, angle, length));
    }
    World::from_worms(seed, worms, pellet_target)
}

/// A respawned bot comes back at the starting size, like a fresh slither.io
/// snake — its accumulated length stays on the field as the death pellets it
/// already dropped.
const RESPAWN_LENGTH: f32 = START_LENGTH;
/// A respawn position must clear every living worm's body by at least this much
/// (plus the bodies' own radii) so a bot doesn't pop into existence already
/// overlapping a snake and instantly die again.
const RESPAWN_CLEARANCE: f32 = 360.0;

#[wasm_bindgen]
pub struct SlitherGame {
    world: World,
    model: Model,
    /// Per-worm observation scratch (delta channels); index 0 (human) unused.
    mem: Vec<ObsMemory>,
    /// Each bot's action this tick. Every living bot re-decides every tick (to
    /// match the 30 Hz decision rate the net trained and was evaluated at), so
    /// this is rewritten in full each tick rather than cached across ticks. Index
    /// 0 (human) unused.
    actions: Vec<Action>,
    /// Whether each worm actually boosted on the last tick (control boost gated by
    /// `can_boost`) — drives the boost glow in the renderer.
    boosting: Vec<bool>,
    /// Dedicated RNG for picking respawn positions, kept off the world's pellet
    /// RNG so respawns don't perturb the food stream.
    respawn_rng: Rng,
    /// How many bots ran a net forward on the last tick — equals the living-bot
    /// count, since every living bot now decides every tick. Read by the parity
    /// check to assert the deploy decision rate matches training.
    decided_last_tick: usize,
    /// Cached so a fresh game reuses the same population/pellet density.
    cfg: WorldConfig,
    seed: u64,
}

#[wasm_bindgen]
impl SlitherGame {
    /// A fresh arena of `worms` snakes (1 human + bots) at pellet density
    /// `pellets`, with the trained net parsed from its `SLNET1` export bytes.
    #[wasm_bindgen(constructor)]
    pub fn new(
        weights: &[u8],
        worms: usize,
        pellets: usize,
        seed: u32,
    ) -> Result<SlitherGame, JsError> {
        let model = Model::parse(weights).map_err(|e| JsError::new(&e))?;
        let cfg = WorldConfig {
            worms: worms.max(2),
            pellet_target: pellets.max(50),
            ..WorldConfig::default()
        };
        let seed = u64::from(seed);
        let world = build_world(seed, cfg.worms - 1, cfg.pellet_target);
        let n = world.worms.len();
        let mem = (0..n).map(|_| ObsMemory::new()).collect();
        Ok(SlitherGame {
            world,
            model,
            mem,
            actions: vec![Action::default(); n],
            boosting: vec![false; n],
            respawn_rng: Rng::new(seed ^ 0x5151_5151_5151_5151),
            decided_last_tick: 0,
            cfg,
            seed,
        })
    }

    /// Restart with a new seed, same population and pellet density.
    pub fn reset(&mut self, seed: u32) {
        self.seed = u64::from(seed);
        self.world = build_world(self.seed, self.cfg.worms - 1, self.cfg.pellet_target);
        let n = self.world.worms.len();
        self.mem = (0..n).map(|_| ObsMemory::new()).collect();
        self.actions = vec![Action::default(); n];
        self.boosting = vec![false; n];
        self.respawn_rng = Rng::new(self.seed ^ 0x5151_5151_5151_5151);
        self.decided_last_tick = 0;
    }

    /// Advance one fixed `DT` tick. The human worm aims its head at
    /// `human_aim` (absolute world radians) and boosts while `human_boost`;
    /// every living bot worm acts from the net. Dead worms are skipped (the
    /// world ignores their controls).
    pub fn tick(&mut self, human_aim: f32, human_boost: bool) {
        let n = self.world.worms.len();

        // Decide every living bot every tick, the same 30 Hz rate the net trained
        // and was evaluated at. (An earlier round-robin throttle re-decided each
        // bot only every few ticks to save forwards at 8 worms; that stale ~133 ms
        // reaction was a reflex handicap the human could exploit, and the deploy
        // policy no longer matched training. With the trained population of 6 the
        // per-tick forward cost is small enough to run them all.)
        let mut decided = 0;
        for i in (HUMAN + 1)..n {
            if !self.world.worms[i].dead {
                self.actions[i] = act(&self.model, &self.world, i, &mut self.mem[i]);
                decided += 1;
            }
        }
        self.decided_last_tick = decided;

        let controls: Vec<WormControl> = (0..n)
            .map(|i| {
                let w = &self.world.worms[i];
                if w.dead {
                    WormControl::default()
                } else if i == HUMAN {
                    WormControl {
                        aim: human_aim,
                        boost: human_boost,
                    }
                } else {
                    self.actions[i].control(w.angle)
                }
            })
            .collect();

        // Record the *effective* boost (control boost gated by `can_boost`) so the
        // renderer can light up only worms actually spending length.
        for (slot, (w, ctrl)) in self
            .boosting
            .iter_mut()
            .zip(self.world.worms.iter().zip(&controls))
        {
            *slot = !w.dead && ctrl.boost && w.can_boost();
        }

        self.world.step(&controls);
        self.respawn_dead_bots();
    }

    /// Bring every dead *bot* back as a fresh small worm at a safe spot, so the
    /// arena stays populated like slither.io instead of emptying out as bots are
    /// killed. The human (seat 0) is never auto-respawned — its death is the
    /// page's game-over — so a dead human stays dead and `human_dead()` still
    /// fires.
    fn respawn_dead_bots(&mut self) {
        let n = self.world.worms.len();
        for i in (HUMAN + 1)..n {
            if !self.world.worms[i].dead {
                continue;
            }
            let (pos, angle) = self.pick_respawn(i);
            self.world.worms[i] = Worm::spawn(pos, angle, RESPAWN_LENGTH);
            self.mem[i] = ObsMemory::new();
            self.actions[i] = Action::default();
            self.boosting[i] = false;
        }
    }

    /// Rejection-sample a respawn pose for worm `idx` that clears the walls and
    /// every other living worm's body. Falls back to the last sampled point if no
    /// clear spot is found in a bounded number of tries (a packed arena), which
    /// is still better than leaving the bot dead.
    fn pick_respawn(&mut self, idx: usize) -> (Vec2, f32) {
        let margin = 320.0;
        let mut pos = Vec2::new(WORLD * 0.5, WORLD * 0.5);
        for _ in 0..24 {
            let cand = Vec2::new(
                self.respawn_rng.range(margin, WORLD - margin),
                self.respawn_rng.range(margin, WORLD - margin),
            );
            pos = cand;
            if self.spawn_is_clear(cand, idx) {
                break;
            }
        }
        let angle = self.respawn_rng.range(0.0, std::f32::consts::TAU);
        (pos, angle)
    }

    /// True if `at` is far enough from every living worm (other than `idx`) that
    /// a fresh worm spawned there won't be touching a body.
    fn spawn_is_clear(&self, at: Vec2, idx: usize) -> bool {
        for (j, w) in self.world.worms.iter().enumerate() {
            if j == idx || w.dead {
                continue;
            }
            let clear = RESPAWN_CLEARANCE + w.radius();
            let clear2 = clear * clear;
            if w.segments.iter().any(|&s| at.dist2(s) <= clear2) {
                return false;
            }
        }
        true
    }

    /// `true` once the human worm has died — the page's game-over trigger.
    pub fn human_dead(&self) -> bool {
        self.world.worms[HUMAN].dead
    }

    /// The human worm's length (its score) — `0` once dead is still reported as
    /// its last length, so the score doesn't reset on death.
    pub fn human_length(&self) -> f32 {
        self.world.worms[HUMAN].length
    }

    /// The human head's world position `[x, y]`, for the camera to follow.
    pub fn human_head(&self) -> Vec<f32> {
        let h = self.world.worms[HUMAN].head();
        vec![h.x, h.y]
    }

    /// The human worm's current heading (radians) — the camera's "up", and the
    /// fallback aim before the cursor has moved.
    pub fn human_angle(&self) -> f32 {
        self.world.worms[HUMAN].angle
    }

    /// How far the human currently sees, in world units (the egocentric
    /// viewport half-width that grows with length) — lets the camera frame the
    /// same neighborhood the bots judge from.
    pub fn human_view_radius(&self) -> f32 {
        view_radius(self.world.worms[HUMAN].length)
    }

    pub fn world_size(&self) -> f32 {
        WORLD
    }

    pub fn worm_count(&self) -> usize {
        self.world.worms.len()
    }

    /// Living-worm count — the page shows it as "snakes left".
    pub fn alive_count(&self) -> usize {
        self.world.alive_count()
    }

    /// Flat render snapshot of every worm, concatenated. Per worm a fixed-width
    /// 8-float header — `[seat, is_human, dead, boosting, radius, length, angle,
    /// seg_count]` — followed by `seg_count` `x,y` pairs. `seat` is the worm's
    /// stable index so the page can match a worm across snapshots (for
    /// interpolation) even as the population changes. Dead worms report
    /// `seg_count = 0` (their remains are pellets now).
    pub fn worms_blob(&self) -> Vec<f32> {
        let mut out = Vec::new();
        for (i, w) in self.world.worms.iter().enumerate() {
            let segs = if w.dead { &[][..] } else { &w.segments[..] };
            out.push(i as f32);
            out.push(if i == HUMAN { 1.0 } else { 0.0 });
            out.push(if w.dead { 1.0 } else { 0.0 });
            out.push(if self.boosting[i] { 1.0 } else { 0.0 });
            out.push(w.radius());
            out.push(w.length);
            out.push(w.angle);
            out.push(segs.len() as f32);
            for s in segs {
                out.push(s.x);
                out.push(s.y);
            }
        }
        out
    }

    /// Flat pellet snapshot `[x0,y0,value0, x1,y1,value1, …]`. The value lets the
    /// page draw the bigger, brighter death-snake orbs (value ≈ 2) distinctly
    /// from natural pellets (value 1).
    pub fn pellets_blob(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.world.pellets.len() * 3);
        for p in &self.world.pellets {
            out.push(p.pos.x);
            out.push(p.pos.y);
            out.push(p.value);
        }
        out
    }

    /// Leaderboard snapshot: every worm's `[seat, is_human, dead, length]`,
    /// already sorted by length descending. The page slices the top N and finds
    /// the human's rank from it.
    pub fn leaderboard_blob(&self) -> Vec<f32> {
        let mut order: Vec<usize> = (0..self.world.worms.len()).collect();
        order.sort_by(|&a, &b| {
            self.world.worms[b]
                .length
                .partial_cmp(&self.world.worms[a].length)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut out = Vec::with_capacity(order.len() * 4);
        for i in order {
            let w = &self.world.worms[i];
            out.push(i as f32);
            out.push(if i == HUMAN { 1.0 } else { 0.0 });
            out.push(if w.dead { 1.0 } else { 0.0 });
            out.push(w.length);
        }
        out
    }
}

/// Test-only hooks for the headless respawn check (the `respawn_check` example).
/// Gated behind `debug-hooks` so they never reach the shipped wasm.
#[cfg(feature = "debug-hooks")]
impl SlitherGame {
    pub fn debug_kill(&mut self, idx: usize) {
        self.world.worms[idx].dead = true;
    }

    pub fn debug_worm_dead(&self, idx: usize) -> bool {
        self.world.worms[idx].dead
    }

    pub fn debug_worm_length(&self, idx: usize) -> f32 {
        self.world.worms[idx].length
    }

    pub fn debug_pellet_count(&self) -> usize {
        self.world.pellets.len()
    }

    pub fn debug_pellet_target(&self) -> usize {
        self.cfg.pellet_target
    }

    pub fn debug_worm_radius(&self, idx: usize) -> f32 {
        self.world.worms[idx].radius()
    }

    pub fn debug_set_length(&mut self, idx: usize, length: f32) {
        self.world.worms[idx].length = length;
    }

    pub fn debug_decided_last_tick(&self) -> usize {
        self.decided_last_tick
    }

    pub fn debug_living_bots(&self) -> usize {
        self.world.worms[(HUMAN + 1)..]
            .iter()
            .filter(|w| !w.dead)
            .count()
    }
}
