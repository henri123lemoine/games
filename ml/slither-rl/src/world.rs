//! Headless slither dynamics: continuous worms (head position + heading, turn-rate
//! limited), fixed-spacing segment chains dragged behind the head, food pellets,
//! boost (burns length, faster), and head-vs-body death where the *other* worm
//! survives. Deterministic given a seed and the per-worm action stream; no
//! rendering, no allocation in the hot path beyond the segment chains.
//!
//! Geometry and constants are cribbed from the j-c-m-style TS clone: `WORLD`,
//! `SEG_SPACING`, `BASE_SPEED`, `TURN_RATE`, boost drain, etc. One simulation
//! "step" advances every worm by a fixed `DT` so a fixed action repeats cleanly
//! and thousands of `World`s vectorize at the same wall-clock rate.

use crate::geometry::{Vec2, turn_toward};
use crate::rng::Rng;

pub const WORLD: f32 = 4200.0;
pub const DT: f32 = 1.0 / 30.0;

pub const START_LENGTH: f32 = 22.0;
const SEG_SPACING: f32 = 4.4;
const BASE_SPEED: f32 = 168.0;
const BOOST_SPEED: f32 = 320.0;
const TURN_RATE: f32 = 4.2;
const BOOST_DRAIN_PER_SEC: f32 = 9.0;
const MIN_BOOST_LENGTH: f32 = START_LENGTH + 8.0;
const EAT_PADDING: f32 = 10.0;

pub const PELLET_VALUE: f32 = 1.0;
const DEATH_PELLET_VALUE: f32 = 2.0;
/// Ambient food regrowth, pellets/second, trickled in by `refill_pellets` rather
/// than topping straight back up to the target — slow enough that a grazed-out
/// region stays depleted, so growth means ranging for food, not circling in place.
const REFILL_PER_SEC: f32 = 30.0;

/// A worm's body radius as a function of its length. This is the single source
/// of truth for both collision (here) and rendering (the browser reads it back
/// per worm), so the snake never collides at a size different from how it draws.
///
/// The growth is deliberately *sublinear* — a cube root of length. Real
/// slither.io width grows gently with mass, not in step with it: a snake ten
/// times longer is only a couple of times wider. The previous law grew the
/// radius linearly with length (then hard-capped), so width ballooned with
/// score — a worm at length 300 was already ~3.4x the starting width and kept
/// climbing, which read as the snake getting absurdly fat per point. Cube-root
/// growth keeps the whole playable range to roughly a 1.5–3x spread: ~8.6 at
/// the START_LENGTH spawn, ~13.6 at length 300, ~21 at length 2000.
fn radius_for(length: f32) -> f32 {
    const BASE: f32 = 5.0;
    const GROWTH: f32 = 3.6;
    BASE + GROWTH * (length.max(0.0) / START_LENGTH).cbrt()
}

#[derive(Clone, Debug)]
pub struct Pellet {
    pub pos: Vec2,
    pub value: f32,
}

/// What a worm is asked to do this step: a target heading delta (already turn-rate
/// limited downstream) and whether to boost. The env translates discrete buckets
/// into this; the heuristic fills it directly.
#[derive(Clone, Copy, Debug, Default)]
pub struct WormControl {
    /// Desired absolute heading this step. The worm turns toward it, capped by
    /// `TURN_RATE * DT`.
    pub aim: f32,
    pub boost: bool,
}

#[derive(Clone, Debug)]
pub struct Worm {
    pub segments: Vec<Vec2>,
    pub angle: f32,
    pub length: f32,
    pub speed: f32,
    pub dead: bool,
    food: f32,
    /// Set on the step this worm dies; lets the env read who killed whom.
    pub killed_by: Option<usize>,
}

impl Worm {
    pub fn spawn(pos: Vec2, angle: f32, length: f32) -> Self {
        let count = length.round().max(START_LENGTH) as usize;
        let mut segments = Vec::with_capacity(count);
        for i in 0..count {
            segments.push(Vec2::new(
                pos.x - angle.cos() * i as f32 * SEG_SPACING,
                pos.y - angle.sin() * i as f32 * SEG_SPACING,
            ));
        }
        Self {
            segments,
            angle,
            length,
            speed: BASE_SPEED,
            dead: false,
            food: 0.0,
            killed_by: None,
        }
    }

    pub fn head(&self) -> Vec2 {
        self.segments[0]
    }

    pub fn radius(&self) -> f32 {
        radius_for(self.length)
    }

    pub fn can_boost(&self) -> bool {
        self.length > MIN_BOOST_LENGTH
    }

    fn grow(&mut self, amount: f32) {
        self.food += amount;
        while self.food >= 1.0 {
            self.food -= 1.0;
            self.length += 1.0;
        }
    }

    /// Advance the head by `dist` along `angle`, then resample the trailing path
    /// to a fixed-spacing chain of `length` points so the body reads as one
    /// smooth ribbon (the TS `step`). Clamps the head to the arena.
    fn advance(&mut self, dist: f32) {
        let head = self.head();
        let nx = (head.x + self.angle.cos() * dist).clamp(0.0, WORLD);
        let ny = (head.y + self.angle.sin() * dist).clamp(0.0, WORLD);
        self.segments.insert(0, Vec2::new(nx, ny));

        let want = (self.length.round() as usize).max(START_LENGTH as usize);
        let mut out: Vec<Vec2> = Vec::with_capacity(want);
        out.push(self.segments[0]);
        let mut prev = self.segments[0];
        let mut acc = 0.0f32;
        let mut i = 1;
        while out.len() < want && i < self.segments.len() {
            let cur = self.segments[i];
            let seg_len = prev.dist(cur);
            if seg_len <= 1e-6 {
                i += 1;
                continue;
            }
            acc += seg_len;
            while acc >= SEG_SPACING && out.len() < want {
                acc -= SEG_SPACING;
                let t = 1.0 - acc / seg_len;
                out.push(Vec2::new(
                    prev.x + (cur.x - prev.x) * t,
                    prev.y + (cur.y - prev.y) * t,
                ));
            }
            prev = cur;
            i += 1;
        }
        while out.len() < want {
            out.push(*out.last().unwrap());
        }
        self.segments = out;
    }
}

#[derive(Clone, Debug)]
pub struct World {
    pub worms: Vec<Worm>,
    pub pellets: Vec<Pellet>,
    pub rng: Rng,
    pub steps: u64,
    pellet_target: usize,
}

/// How many worms and pellets a fresh arena starts with. Small by default so a
/// 1v1-plus-prey curriculum and dense parallelism stay cheap; the env picks the
/// numbers.
#[derive(Clone, Copy, Debug)]
pub struct WorldConfig {
    pub worms: usize,
    pub pellet_target: usize,
    /// Starting length of seat 0 (the learner / teacher seat). The encircle
    /// curriculum begins with this *oversized* against small prey — trapping only
    /// works when you are bigger — and ramps toward symmetric self-play.
    pub seat0_length: f32,
    /// Other worms start at `START_LENGTH + uniform(0, prey_jitter)`, capped so a
    /// curriculum can hold them as small prey while seat 0 is large.
    pub prey_jitter: f32,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            worms: 6,
            pellet_target: 250,
            seat0_length: START_LENGTH,
            prey_jitter: 80.0,
        }
    }
}

impl WorldConfig {
    /// The blueprint's opening curriculum: one oversized predator (seat 0) among
    /// small, similarly-sized prey, so encircling has room to pay off.
    pub fn oversized_vs_prey() -> Self {
        Self {
            seat0_length: 220.0,
            prey_jitter: 24.0,
            ..Self::default()
        }
    }
}

impl World {
    pub fn new(seed: u64, cfg: WorldConfig) -> Self {
        let mut rng = Rng::new(seed);
        let mut worms = Vec::with_capacity(cfg.worms);
        let margin = 300.0;
        for i in 0..cfg.worms {
            let pos = Vec2::new(
                rng.range(margin, WORLD - margin),
                rng.range(margin, WORLD - margin),
            );
            let angle = rng.range(0.0, std::f32::consts::TAU);
            let length = if i == 0 {
                cfg.seat0_length
            } else {
                START_LENGTH + rng.range(0.0, cfg.prey_jitter)
            };
            worms.push(Worm::spawn(pos, angle, length));
        }
        let mut world = Self {
            worms,
            pellets: Vec::with_capacity(cfg.pellet_target),
            rng,
            steps: 0,
            pellet_target: cfg.pellet_target,
        };
        for _ in 0..cfg.pellet_target {
            let p = world.random_pellet();
            world.pellets.push(p);
        }
        world
    }

    /// Build an arena from explicit worms — for curriculum scenarios and tests
    /// that need controlled spawn geometry (e.g. a predator placed next to its
    /// prey). Pellets fill to `pellet_target` as usual.
    pub fn from_worms(seed: u64, worms: Vec<Worm>, pellet_target: usize) -> Self {
        let mut world = Self {
            worms,
            pellets: Vec::with_capacity(pellet_target),
            rng: Rng::new(seed),
            steps: 0,
            pellet_target,
        };
        for _ in 0..pellet_target {
            let p = world.random_pellet();
            world.pellets.push(p);
        }
        world
    }

    fn random_pellet(&mut self) -> Pellet {
        Pellet {
            pos: Vec2::new(
                self.rng.range(20.0, WORLD - 20.0),
                self.rng.range(20.0, WORLD - 20.0),
            ),
            value: PELLET_VALUE,
        }
    }

    /// Advance every living worm one `DT` tick under the given controls (indexed
    /// by worm). Boost burns length and may shed a tail pellet; then eating,
    /// collisions, and pellet refill resolve. Length deltas for reward are read
    /// by the caller before/after via `Worm::length`.
    pub fn step(&mut self, controls: &[WormControl]) {
        for (i, w) in self.worms.iter_mut().enumerate() {
            if w.dead {
                continue;
            }
            let ctrl = controls.get(i).copied().unwrap_or_default();
            w.angle = turn_toward(w.angle, ctrl.aim, TURN_RATE * DT);

            let boosting = ctrl.boost && w.can_boost();
            let target_speed = if boosting { BOOST_SPEED } else { BASE_SPEED };
            w.speed += (target_speed - w.speed) * (DT * 8.0).min(1.0);

            if boosting {
                w.length = (w.length - BOOST_DRAIN_PER_SEC * DT).max(MIN_BOOST_LENGTH);
            }
            w.advance(w.speed * DT);
        }

        self.shed_boost_pellets(controls);
        self.eat_pellets();
        self.resolve_collisions();
        self.refill_pellets();
        self.steps += 1;
    }

    fn shed_boost_pellets(&mut self, controls: &[WormControl]) {
        for i in 0..self.worms.len() {
            let w = &self.worms[i];
            if w.dead {
                continue;
            }
            let boosting = controls.get(i).copied().unwrap_or_default().boost && w.can_boost();
            // Shed pellets at the same rate boost drains length, so boosting
            // conserves mass: BOOST_DRAIN_PER_SEC length/s drained == that many
            // PELLET_VALUE pellets/s dropped. (At the old 14/s shed rate a worm
            // could boost in a circle, eat its own shed pellets, and net mass.)
            let shed_rate = BOOST_DRAIN_PER_SEC / PELLET_VALUE;
            if boosting && self.rng.unit() < DT * shed_rate {
                let tail = *self.worms[i].segments.last().unwrap();
                self.drop_pellet(tail, PELLET_VALUE);
            }
        }
    }

    fn eat_pellets(&mut self) {
        for w in &mut self.worms {
            if w.dead {
                continue;
            }
            let head = w.head();
            let reach = w.radius() + EAT_PADDING;
            let reach2 = reach * reach;
            let mut i = 0;
            while i < self.pellets.len() {
                if head.dist2(self.pellets[i].pos) <= reach2 {
                    w.grow(self.pellets[i].value);
                    let last = self.pellets.len() - 1;
                    self.pellets.swap(i, last);
                    self.pellets.pop();
                } else {
                    i += 1;
                }
            }
        }
    }

    /// A worm dies when its head enters another living worm's body (or the arena
    /// wall, handled as a separate check). The head's owner dies; the body's
    /// owner survives. Deaths this step are computed against the pre-resolution
    /// state so a mutual ram kills both, exactly as in slither.io.
    fn resolve_collisions(&mut self) {
        let n = self.worms.len();
        let mut dying: Vec<(usize, Option<usize>)> = Vec::new();

        for w in 0..n {
            if self.worms[w].dead {
                continue;
            }
            let head = self.worms[w].head();
            let wr = self.worms[w].radius();

            if head.x <= wr || head.x >= WORLD - wr || head.y <= wr || head.y >= WORLD - wr {
                dying.push((w, None));
                continue;
            }

            for o in 0..n {
                if o == w || self.worms[o].dead {
                    continue;
                }
                let hit = wr + self.worms[o].radius();
                if head_hits_body(head, hit, &self.worms[o].segments) {
                    dying.push((w, Some(o)));
                    break;
                }
            }
        }

        for (w, by) in dying {
            self.kill_worm(w, by);
        }
    }

    fn kill_worm(&mut self, idx: usize, by: Option<usize>) {
        if self.worms[idx].dead {
            return;
        }
        self.worms[idx].dead = true;
        self.worms[idx].killed_by = by;

        let segs = self.worms[idx].segments.clone();
        let drop = (segs.len() / 2).max(8);
        let stride = (segs.len() / drop).max(1);
        let mut i = 0;
        while i < segs.len() {
            self.drop_pellet(segs[i], DEATH_PELLET_VALUE);
            i += stride;
        }
    }

    fn drop_pellet(&mut self, at: Vec2, value: f32) {
        let jx = self.rng.range(-6.0, 6.0);
        let jy = self.rng.range(-6.0, 6.0);
        self.pellets.push(Pellet {
            pos: Vec2::new(at.x + jx, at.y + jy),
            value,
        });
    }

    fn refill_pellets(&mut self) {
        // Trickle ambient food back toward the target instead of instantly
        // topping up. An instant top-up makes a grazed-out spot regenerate every
        // step, so a worm grows just by circling in regenerating food; a slow
        // trickle keeps a depleted region depleted, so growth means actively
        // ranging for food (or hunting) — closer to real slither.io.
        if self.pellets.len() < self.pellet_target {
            let deficit = self.pellet_target - self.pellets.len();
            let trickle = (REFILL_PER_SEC * DT).max(1.0) as usize;
            for _ in 0..trickle.min(deficit) {
                let p = self.random_pellet();
                self.pellets.push(p);
            }
        }
        let cap = (self.pellet_target as f32 * 1.6) as usize;
        if self.pellets.len() > cap {
            let keep = (self.pellet_target as f32 * 1.4) as usize;
            let drop = self.pellets.len() - keep;
            self.pellets.drain(0..drop);
        }
    }

    pub fn alive_count(&self) -> usize {
        self.worms.iter().filter(|w| !w.dead).count()
    }
}

/// True if `head` is within `hit_dist` of any body point past the neck. Skipping
/// the first two points keeps a worm from colliding with its own head/neck.
pub fn head_hits_body(head: Vec2, hit_dist: f32, body: &[Vec2]) -> bool {
    let hit2 = hit_dist * hit_dist;
    body.iter().skip(2).any(|&s| head.dist2(s) <= hit2)
}
