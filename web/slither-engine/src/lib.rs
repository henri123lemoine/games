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
/// The human starts a touch longer than a bare worm — enough to be competitive
/// in the opening (not the obvious smallest prey) without being a giant. The
/// bots, scattered, start between `START_LENGTH` and roughly this.
const HUMAN_START_LENGTH: f32 = 60.0;
/// Bots spawn at least this far from the human's center spawn, so a fresh human
/// isn't rammed in the first second.
const SPAWN_CLEARANCE: f32 = 700.0;

/// Build the arena: the human at the center facing a random way (clear of the
/// walls and of an instant bot ram), and `bot_count` bots scattered with a
/// margin from the walls and from the human. Keeps play fair at spawn without
/// changing the trained policy the bots run.
fn build_world(seed: u64, bot_count: usize, pellet_target: usize) -> World {
    let mut rng = Rng::new(seed);
    let center = Vec2::new(WORLD * 0.5, WORLD * 0.5);
    let mut worms = Vec::with_capacity(bot_count + 1);
    worms.push(Worm::spawn(
        center,
        rng.range(0.0, std::f32::consts::TAU),
        HUMAN_START_LENGTH,
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
        let mem = (0..world.worms.len()).map(|_| ObsMemory::new()).collect();
        Ok(SlitherGame {
            world,
            model,
            mem,
            cfg,
            seed,
        })
    }

    /// Restart with a new seed, same population and pellet density.
    pub fn reset(&mut self, seed: u32) {
        self.seed = u64::from(seed);
        self.world = build_world(self.seed, self.cfg.worms - 1, self.cfg.pellet_target);
        self.mem = (0..self.world.worms.len())
            .map(|_| ObsMemory::new())
            .collect();
    }

    /// Advance one fixed `DT` tick. The human worm aims its head at
    /// `human_aim` (absolute world radians) and boosts while `human_boost`;
    /// every living bot worm acts from the net. Dead worms are skipped (the
    /// world ignores their controls).
    pub fn tick(&mut self, human_aim: f32, human_boost: bool) {
        let n = self.world.worms.len();
        // The net forward for each bot reads `self.world` and writes its own
        // `self.mem[i]`, so build the controls by index (the borrow checker
        // can't see those are disjoint through a single iterator). Dead worms
        // get the default control, which the world ignores.
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
                    let angle = w.angle;
                    let a: Action = act(&self.model, &self.world, i, &mut self.mem[i]);
                    a.control(angle)
                }
            })
            .collect();
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

    /// Flat render snapshot of every worm, concatenated. Per worm:
    /// `[is_human, dead, radius, length, seg_count, x0,y0, x1,y1, …]`. The
    /// header is fixed-width (5 floats) so the reader can stride worms; segment
    /// pairs follow. Dead worms report `seg_count = 0` (their remains are
    /// pellets now).
    pub fn worms_blob(&self) -> Vec<f32> {
        let mut out = Vec::new();
        for (i, w) in self.world.worms.iter().enumerate() {
            let segs = if w.dead { &[][..] } else { &w.segments[..] };
            out.push(if i == HUMAN { 1.0 } else { 0.0 });
            out.push(if w.dead { 1.0 } else { 0.0 });
            out.push(w.radius());
            out.push(w.length);
            out.push(segs.len() as f32);
            for s in segs {
                out.push(s.x);
                out.push(s.y);
            }
        }
        out
    }

    /// Flat pellet positions `[x0,y0, x1,y1, …]`; value is folded into the
    /// drawing by the page (death pellets are larger but the world doesn't
    /// distinguish them once dropped, so a single size reads fine).
    pub fn pellets_blob(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.world.pellets.len() * 2);
        for p in &self.world.pellets {
            out.push(p.pos.x);
            out.push(p.pos.y);
        }
        out
    }
}
