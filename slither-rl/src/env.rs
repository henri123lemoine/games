//! Gym-like multi-agent env over the slither [`World`]: `reset(seed) -> Obs[]`
//! and `step(actions) -> StepOut` with per-worm reward and done. One env is one
//! arena holding `n` worms; thousands run in parallel for self-play. Actions are
//! discrete relative-turn buckets plus a separate binary boost, matching the
//! blueprint's two policy heads.
//!
//! Reward stack (per worm, per step):
//!   + KILL_BONUS * victim_length     when an enemy head dies on your body
//!   + length delta                   food + absorbed remains (dense, modest)
//!   - DEATH_PENALTY                   on your own head-collision
//!   - BOOST_COST                      while boosting (boost burns length)
//!   + encircle shaping * anneal       shrinking a nearby smaller foe's escape arc
//!
//! The encircle term is the learnable analog of the heuristic's circle-trap; it
//! is annealed toward zero by the trainer (`set_shaping`) as real kills appear,
//! so the final policy is learned, not scripted. Distance-to-target is kept OUT
//! of dense shaping on purpose (dense distance → naive chasing).

use crate::geometry::Vec2;
use crate::obs::{self, CHANNELS, GRID, Obs, SCALARS, SEMANTIC_CHANNELS, view_radius};
use crate::world::{World, WorldConfig};

/// Discrete relative-turn buckets, symmetric about straight-ahead. 9 buckets ≈
/// ±max-turn in 22.5°-ish steps after scaling, which composes cleanly with the
/// egocentric (heading-aligned) observation.
pub const TURN_BUCKETS: usize = 9;
/// Widest relative heading change a single action can request, in radians. The
/// world still caps the realized turn by its own turn-rate, so picking the
/// extreme bucket just means "turn as hard as you can this way".
const MAX_RELATIVE_TURN: f32 = 1.2;

const KILL_BONUS: f32 = 0.02;
const DEATH_PENALTY: f32 = 5.0;
const LENGTH_DELTA_SCALE: f32 = 0.1;
const BOOST_COST: f32 = 0.01;
const ENCIRCLE_SCALE: f32 = 1.0;
/// A foe must be at least this fraction smaller to be worth encircling (the
/// heuristic's `enCircleThreshold` ≈ 0.56 analog: trap things you can outlast).
const ENCIRCLE_SIZE_RATIO: f32 = 0.85;

#[derive(Clone, Copy, Debug)]
pub struct Action {
    pub turn: u8,
    pub boost: bool,
}

impl Default for Action {
    fn default() -> Self {
        Self {
            turn: (TURN_BUCKETS / 2) as u8,
            boost: false,
        }
    }
}

impl Action {
    /// Signed heading delta this action requests, before the world's turn-rate
    /// cap. Bucket `TURN_BUCKETS/2` is straight ahead.
    fn relative_turn(self) -> f32 {
        let mid = (TURN_BUCKETS / 2) as f32;
        let t = (self.turn as f32 - mid) / mid;
        t * MAX_RELATIVE_TURN
    }
}

#[derive(Clone, Debug)]
pub struct StepOut {
    pub obs: Vec<Obs>,
    pub reward: Vec<f32>,
    pub done: Vec<bool>,
    /// Worms killed *by* each worm this step (an enemy head died on its body).
    /// Separate from reward so a trainer or eval can count kills without
    /// reverse-engineering the reward stack.
    pub kills: Vec<u32>,
    /// True once no learner-controlled worm can still act (all dead). The arena
    /// keeps running for survivors, but the trainer typically resets here.
    pub all_done: bool,
}

pub struct Env {
    world: World,
    prev_semantic: Vec<Vec<f32>>,
    prev_length: Vec<f32>,
    prev_alive: Vec<bool>,
    shaping: f32,
    cfg: WorldConfig,
}

impl Env {
    pub fn new(cfg: WorldConfig) -> Self {
        Self {
            world: World::new(0, cfg),
            prev_semantic: Vec::new(),
            prev_length: Vec::new(),
            prev_alive: Vec::new(),
            shaping: 1.0,
            cfg,
        }
    }

    pub fn num_worms(&self) -> usize {
        self.world.worms.len()
    }

    pub fn world(&self) -> &World {
        &self.world
    }

    /// Encircle-shaping weight in `[0, 1]`; the trainer anneals it toward 0.
    pub fn set_shaping(&mut self, w: f32) {
        self.shaping = w.clamp(0.0, 1.0);
    }

    pub fn reset(&mut self, seed: u64) -> Vec<Obs> {
        self.reset_world(World::new(seed, self.cfg))
    }

    /// Reset onto a caller-built world (curriculum scenarios, controlled tests).
    /// Reinitializes the per-worm delta/length/alive bookkeeping and returns the
    /// first observation for each worm.
    pub fn reset_world(&mut self, world: World) -> Vec<Obs> {
        self.world = world;
        let n = self.world.worms.len();
        let semantic_len = SEMANTIC_CHANNELS * GRID * GRID;
        self.prev_semantic = vec![vec![0.0; semantic_len]; n];
        self.prev_length = self.world.worms.iter().map(|w| w.length).collect();
        self.prev_alive = vec![true; n];
        (0..n)
            .map(|i| {
                let mut o = obs::observe(&self.world, i, &mut self.prev_semantic[i]);
                // No previous frame exists at reset, so velocity is undefined —
                // zero the delta channels rather than report everything as newly
                // appearing (which `current - 0` would).
                for v in o.grid[semantic_len..].iter_mut() {
                    *v = 0.0;
                }
                o
            })
            .collect()
    }

    pub fn step(&mut self, actions: &[Action]) -> StepOut {
        let n = self.world.worms.len();

        // Pre-step encircle potential of each foe an actor is trapping; reward is
        // the *reduction* in that foe's open escape arc this step.
        let pre_arc: Vec<f32> = (0..n).map(|i| self.encircle_openness(i)).collect();

        let controls: Vec<_> = (0..n)
            .map(|i| {
                let w = &self.world.worms[i];
                let a = actions.get(i).copied().unwrap_or_default();
                crate::world::WormControl {
                    aim: w.angle + a.relative_turn(),
                    boost: a.boost,
                }
            })
            .collect();

        self.world.step(&controls);

        let mut reward = vec![0.0f32; n];
        let mut done = vec![false; n];
        let mut kills = vec![0u32; n];

        for i in 0..n {
            let w = &self.world.worms[i];

            let grew = w.length - self.prev_length[i];
            if grew > 0.0 {
                reward[i] += grew * LENGTH_DELTA_SCALE;
            }
            self.prev_length[i] = w.length;

            if actions.get(i).copied().unwrap_or_default().boost && w.can_boost() {
                reward[i] -= BOOST_COST;
            }

            let just_died = w.dead && self.prev_alive[i];
            if just_died {
                reward[i] -= DEATH_PENALTY;
                done[i] = true;
                if let Some(killer) = w.killed_by {
                    reward[killer] += KILL_BONUS * w.length;
                    kills[killer] += 1;
                }
            }
            self.prev_alive[i] = !w.dead;
        }

        if self.shaping > 0.0 {
            let post_arc: Vec<f32> = (0..n).map(|i| self.encircle_openness(i)).collect();
            for i in 0..n {
                if !self.world.worms[i].dead {
                    let shrink = pre_arc[i] - post_arc[i];
                    if shrink > 0.0 {
                        reward[i] += self.shaping * ENCIRCLE_SCALE * shrink;
                    }
                }
            }
        }

        let obs: Vec<Obs> = (0..n)
            .map(|i| obs::observe(&self.world, i, &mut self.prev_semantic[i]))
            .collect();

        let all_done = self.world.alive_count() == 0;
        StepOut {
            obs,
            reward,
            done,
            kills,
            all_done,
        }
    }

    /// The forward escape-arc still open to the *smaller* foe nearest worm `i` —
    /// a number in `[0, 1]` where 1 is "fully open, can flee any forward
    /// direction" and 0 is "boxed in by worm i's body and/or the wall". The
    /// shaping reward pays worm `i` for driving this down. Returns 0 when there
    /// is no encircle-worthy foe (no foe, or none small enough), so closing on a
    /// peer or a bigger snake earns nothing.
    fn encircle_openness(&self, i: usize) -> f32 {
        let me = &self.world.worms[i];
        if me.dead {
            return 0.0;
        }
        let view = view_radius(me.length);
        let mut target: Option<usize> = None;
        let mut best_d2 = view * view;
        for (j, foe) in self.world.worms.iter().enumerate() {
            if j == i || foe.dead {
                continue;
            }
            if foe.length > me.length * ENCIRCLE_SIZE_RATIO {
                continue;
            }
            let d2 = me.head().dist2(foe.head());
            if d2 < best_d2 {
                best_d2 = d2;
                target = Some(j);
            }
        }
        let Some(j) = target else { return 0.0 };

        // Sample the foe's forward semicircle; a direction is "blocked" if a short
        // ray from its head crosses worm i's body or leaves the arena. Openness is
        // the blocked-free fraction.
        let foe = &self.world.worms[j];
        const RAYS: usize = 16;
        let probe = view * 0.5;
        let mut open = 0;
        for k in 0..RAYS {
            let off = (k as f32 / (RAYS - 1) as f32 - 0.5) * std::f32::consts::PI;
            let dir = foe.angle + off;
            let tip = Vec2::new(
                foe.head().x + dir.cos() * probe,
                foe.head().y + dir.sin() * probe,
            );
            let leaves = tip.x <= 0.0
                || tip.x >= crate::world::WORLD
                || tip.y <= 0.0
                || tip.y >= crate::world::WORLD;
            let blocked = leaves || ray_crosses_body(foe.head(), tip, &me.segments, me.radius());
            if !blocked {
                open += 1;
            }
        }
        open as f32 / RAYS as f32
    }
}

/// True if the segment from `from` to `to` passes within `radius` of any point of
/// `body` (a coarse encircle probe, sampled along the ray).
fn ray_crosses_body(from: Vec2, to: Vec2, body: &[Vec2], radius: f32) -> bool {
    const SAMPLES: usize = 8;
    let r2 = (radius * 2.0) * (radius * 2.0);
    for s in 1..=SAMPLES {
        let t = s as f32 / SAMPLES as f32;
        let p = Vec2::new(from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t);
        if body.iter().any(|&b| p.dist2(b) <= r2) {
            return true;
        }
    }
    false
}

/// Static tensor shapes for a learner, so the trainer/net can be sized without a
/// live env. `(channels, grid, grid)` for the conv stack, `scalars` appended.
pub struct Shapes {
    pub grid: (usize, usize, usize),
    pub scalars: usize,
    pub turn_buckets: usize,
}

pub const SHAPES: Shapes = Shapes {
    grid: (CHANNELS, GRID, GRID),
    scalars: SCALARS,
    turn_buckets: TURN_BUCKETS,
};
