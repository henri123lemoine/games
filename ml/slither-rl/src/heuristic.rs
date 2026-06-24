//! The hand-coded predatory worm — the future self-play pool's teacher and the
//! gating opponent. Adapted from the j-c-m bot's two behaviors:
//!
//!   * **avoid-collision**: sweep a sector heatmap of danger (enemy bodies, the
//!     wall) over the forward arc and steer to the biggest open gap, weighted
//!     toward food and away from threats. This keeps it alive far longer than a
//!     random worm.
//!   * **hunt**: when it is much bigger than a nearby foe, it stops wandering and
//!     runs it down by lead-intercept (aim where the prey will be, boost to close,
//!     ram on contact) — the predatory pressure the learner is meant to sharpen
//!     into a true wall-off encircle.
//!
//! The hunt here is a lead-pursuit *approximation* of j-c-m's full circle-method
//! trap (no `followCircleSelf` body-curve planning): it reliably runs a fleeing
//! prey down to contact (see the bench's pursuit-closure test) but converting the
//! final gap into a kill on equal top speed still needs the boost edge or a wall —
//! the genuinely hard part the RL exists to solve.
//!
//! It reads the [`World`] directly and emits an [`Action`] for its worm, so it
//! can sit in the opponent slot of an [`crate::env::Env`] alongside learned
//! policies. Determinism comes from the world plus a seed used only for idle
//! wandering.

use crate::env::{Action, TURN_BUCKETS};
use crate::geometry::{Vec2, angle_diff};
use crate::rng::Rng;
use crate::world::{WORLD_RADIUS, World, world_center};

/// Tunables ported from the j-c-m bot, renamed for clarity. Distances are world
/// units; angles radians.
const DANGER_LOOKAHEAD: f32 = 90.0;
const SECTOR_COUNT: usize = 24;
const SECTOR_SPAN: f32 = std::f32::consts::PI; // forward semicircle
const FOOD_LOOK: f32 = 520.0;
const WALL_MARGIN: f32 = 240.0;
/// Be at least this much bigger than a foe before switching from wandering to
/// hunting it down.
const ENCIRCLE_RATIO: f32 = 1.4;
/// How close a smaller foe must be before the predator commits to hunting it.
/// Wide enough that a prey opening a small gap doesn't shake the pursuit.
const ENCIRCLE_RANGE: f32 = 900.0;
/// Extra reach beyond the touching radius at which the predator switches from
/// the lead-intercept to ramming straight at the prey's head for the kill.
const STRIKE_PADDING: f32 = 40.0;

pub struct Heuristic {
    rng: Rng,
    wander: f32,
    retarget: f32,
    engaged_encircle: std::cell::Cell<bool>,
}

impl Heuristic {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Rng::new(seed),
            wander: 0.0,
            retarget: 0.0,
            engaged_encircle: std::cell::Cell::new(false),
        }
    }

    /// Whether the most recent [`Heuristic::act`] chose the circle-trap branch
    /// (it found a much-smaller foe to encircle) rather than plain avoidance.
    /// Lets evals report how often the predatory behavior fires.
    pub fn engaged_encircle(&self) -> bool {
        self.engaged_encircle.get()
    }

    /// Choose this worm's action for the current world state.
    pub fn act(&mut self, world: &World, idx: usize) -> Action {
        let me = &world.worms[idx];
        if me.dead {
            return Action::default();
        }

        if let Some(aim) = self.encircle_aim(world, idx) {
            self.engaged_encircle.set(true);
            return self.aim_to_action(me.angle, aim, true);
        }
        self.engaged_encircle.set(false);

        let aim = self.safe_aim(world, idx);
        // Boost occasionally when long and not in immediate danger, to cover
        // ground — the trap is set up at base speed.
        let boost = me.can_boost() && me.length > 60.0 && self.rng.unit() < 0.03;
        self.aim_to_action(me.angle, aim, boost)
    }

    /// Lead-intercept pursuit of a much-smaller nearby foe: aim where the foe
    /// will be (its heading projected forward), not where it is, so the predator
    /// cuts the angle instead of trailing in the prey's wake. Closes to a straight
    /// ram inside strike range. `None` when there is no foe worth hunting. This is
    /// the closing half of the encircle behavior; converting a closed-in prey into
    /// a kill on equal top speed also needs a boost edge or a wall to pin against.
    fn encircle_aim(&self, world: &World, idx: usize) -> Option<f32> {
        let me = &world.worms[idx];
        let head = me.head();
        let mut victim: Option<usize> = None;
        let mut best_d2 = ENCIRCLE_RANGE * ENCIRCLE_RANGE;
        for (j, foe) in world.worms.iter().enumerate() {
            if j == idx || foe.dead {
                continue;
            }
            if me.length < foe.length * ENCIRCLE_RATIO {
                continue;
            }
            let d2 = head.dist2(foe.head());
            if d2 < best_d2 {
                best_d2 = d2;
                victim = Some(j);
            }
        }
        let j = victim?;
        let foe = &world.worms[j];
        let dist = head.dist(foe.head()).max(1.0);

        // Strike range: once the body can reach the prey's head, ram straight at
        // it for the kill.
        let strike = me.radius() + foe.radius() + STRIKE_PADDING;
        if dist <= strike {
            return Some((foe.head().y - head.y).atan2(foe.head().x - head.x));
        }

        // Lead the prey: aim where it will be, not where it is. Against a fleeing
        // foe this cuts the angle instead of trailing in its wake — the intercept
        // a straight (aim-at-current-position) chaser never makes. Lead scales with
        // the gap (more time to cover) and with the predator's speed edge.
        let lead = (dist * 0.5).min(260.0);
        let aim_pt = Vec2::new(
            foe.head().x + foe.angle.cos() * lead,
            foe.head().y + foe.angle.sin() * lead,
        );
        Some((aim_pt.y - head.y).atan2(aim_pt.x - head.x))
    }

    /// Sector heatmap: score each forward heading by danger (nearby enemy bodies,
    /// the wall) minus food attraction, and return the safest gap's heading.
    fn safe_aim(&mut self, world: &World, idx: usize) -> f32 {
        let me = &world.worms[idx];
        let head = me.head();
        let r = me.radius();

        let mut danger = [0.0f32; SECTOR_COUNT];
        let look = DANGER_LOOKAHEAD + r + me.speed * 0.18;

        for (j, other) in world.worms.iter().enumerate() {
            if other.dead {
                continue;
            }
            let skip = if j == idx { 2 } else { 0 };
            let hit = r + other.radius();
            for &s in other.segments.iter().skip(skip) {
                let d = head.dist(s);
                if d > look + hit {
                    continue;
                }
                let bearing = (s.y - head.y).atan2(s.x - head.x);
                let close = ((look + hit - d) / (look + hit)).clamp(0.0, 1.0);
                self.add_to_sector(&mut danger, me.angle, bearing, close * close);
            }
        }

        // The wall is danger too: project the nearest wall point.
        for (bearing, dist) in wall_bearings(head) {
            if dist < WALL_MARGIN {
                let close = (WALL_MARGIN - dist) / WALL_MARGIN;
                self.add_to_sector(&mut danger, me.angle, bearing, close * close * 1.5);
            }
        }

        // Food pulls the chosen gap, but never overrides danger.
        let mut food_pull = [0.0f32; SECTOR_COUNT];
        if let Some(food) = nearest_pellet(world, head, FOOD_LOOK) {
            let bearing = (food.y - head.y).atan2(food.x - head.x);
            self.add_to_sector(&mut food_pull, me.angle, bearing, 1.0);
        }

        let mut best = 0;
        let mut best_score = f32::NEG_INFINITY;
        for k in 0..SECTOR_COUNT {
            let score = -danger[k] + 0.25 * food_pull[k];
            if score > best_score {
                best_score = score;
                best = k;
            }
        }

        if best_score <= -0.5 {
            // Boxed in on all forward sectors: keep current heading and hope a
            // gap opens, rather than turning blindly into a body.
            self.retarget = 0.0;
        }

        // Idle wander so it doesn't lock to a single heading in open space.
        self.retarget -= crate::world::DT;
        if danger.iter().all(|&d| d < 0.05) {
            if self.retarget <= 0.0 {
                self.retarget = self.rng.range(0.6, 1.8);
                self.wander = me.angle + self.rng.range(-1.1, 1.1);
            }
            if food_pull.iter().all(|&f| f == 0.0) {
                return self.wander;
            }
        }

        self.sector_heading(me.angle, best)
    }

    fn add_to_sector(&self, sectors: &mut [f32; SECTOR_COUNT], heading: f32, bearing: f32, w: f32) {
        let off = angle_diff(heading, bearing);
        if off.abs() > SECTOR_SPAN {
            return;
        }
        let t = (off + SECTOR_SPAN) / (2.0 * SECTOR_SPAN);
        let k = ((t * SECTOR_COUNT as f32) as usize).min(SECTOR_COUNT - 1);
        sectors[k] += w;
    }

    fn sector_heading(&self, heading: f32, sector: usize) -> f32 {
        let t = (sector as f32 + 0.5) / SECTOR_COUNT as f32;
        heading - SECTOR_SPAN + t * 2.0 * SECTOR_SPAN
    }

    /// Quantize a desired absolute heading into the nearest discrete turn bucket,
    /// so the heuristic and a learned policy share one action space.
    fn aim_to_action(&self, current: f32, aim: f32, boost: bool) -> Action {
        let mid = (TURN_BUCKETS / 2) as i32;
        let max_turn = 1.2f32;
        let d = angle_diff(current, aim).clamp(-max_turn, max_turn);
        let frac = d / max_turn;
        let bucket = (mid as f32 + frac * mid as f32).round() as i32;
        let bucket = bucket.clamp(0, (TURN_BUCKETS - 1) as i32) as u8;
        Action {
            turn: bucket,
            boost,
        }
    }
}

fn nearest_pellet(world: &World, from: Vec2, max_dist: f32) -> Option<Vec2> {
    let mut best: Option<Vec2> = None;
    let mut best_d2 = max_dist * max_dist;
    for p in &world.pellets {
        let d2 = from.dist2(p.pos);
        if d2 < best_d2 {
            best_d2 = d2;
            best = Some(p.pos);
        }
    }
    best
}

/// Bearing to, and distance from, the circular wall relative to `head`.
fn wall_bearings(head: Vec2) -> [(f32, f32); 1] {
    let center = world_center();
    let dx = head.x - center.x;
    let dy = head.y - center.y;
    let dist_from_center = (dx * dx + dy * dy).sqrt();
    [(dy.atan2(dx), WORLD_RADIUS - dist_from_center)]
}
