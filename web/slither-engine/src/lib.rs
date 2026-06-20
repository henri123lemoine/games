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

/// How many bot net-forwards run per sim tick. The dominant per-tick cost is the
/// CNN forward (~6 ms native, more in wasm); running every bot every tick is what
/// crippled the framerate. Instead the bots are decided round-robin — a small
/// budget per tick, each bot's action cached until its next turn — so a tick
/// costs a bounded number of forwards regardless of population. At 30 Hz with a
/// budget of 2, ~8 bots each re-decide every ~4 ticks (~7.5 Hz), well inside
/// their reaction needs (a worm turns at most `TURN_RATE*DT` ≈ 8°/tick), and the
/// sim stays real-time even while the browser does other work.
const BOT_BUDGET_PER_TICK: usize = 2;

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

#[wasm_bindgen]
pub struct SlitherGame {
    world: World,
    model: Model,
    /// Per-worm observation scratch (delta channels); index 0 (human) unused.
    mem: Vec<ObsMemory>,
    /// Last action each bot decided, replayed on the ticks between its (throttled)
    /// re-decisions. Index 0 (human) unused. Updated round-robin from `bot_cursor`.
    cached: Vec<Action>,
    /// Next bot index to re-decide this tick; walks the worm range round-robin.
    bot_cursor: usize,
    /// Whether each worm actually boosted on the last tick (control boost gated by
    /// `can_boost`) — drives the boost glow in the renderer.
    boosting: Vec<bool>,
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
            cached: vec![Action::default(); n],
            bot_cursor: HUMAN + 1,
            boosting: vec![false; n],
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
        self.cached = vec![Action::default(); n];
        self.bot_cursor = HUMAN + 1;
        self.boosting = vec![false; n];
    }

    /// Advance one fixed `DT` tick. The human worm aims its head at
    /// `human_aim` (absolute world radians) and boosts while `human_boost`;
    /// every living bot worm acts from the net. Dead worms are skipped (the
    /// world ignores their controls).
    pub fn tick(&mut self, human_aim: f32, human_boost: bool) {
        let n = self.world.worms.len();

        // Re-decide a bounded budget of living bots this tick (round-robin over
        // seats 1..n). A bot's net forward reads `self.world` and writes its own
        // `self.mem[i]`; both are decided here and cached. Every other bot replays
        // its last cached action — that's the throttle that keeps the framerate up
        // without changing the policy (each bot still runs the trained net, just a
        // few Hz instead of every tick).
        if n > 1 {
            let mut decided = 0;
            // Scan at most `n-1` seats so a tick can't spin forever if every bot
            // happens to be dead.
            for _ in 0..(n - 1) {
                if decided >= BOT_BUDGET_PER_TICK {
                    break;
                }
                let i = self.bot_cursor;
                self.bot_cursor += 1;
                if self.bot_cursor >= n {
                    self.bot_cursor = HUMAN + 1;
                }
                if i != HUMAN && !self.world.worms[i].dead {
                    self.cached[i] = act(&self.model, &self.world, i, &mut self.mem[i]);
                    decided += 1;
                }
            }
        }

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
                    self.cached[i].control(w.angle)
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
