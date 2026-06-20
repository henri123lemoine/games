//! Egocentric, heading-aligned, viewport-clipped semantic-grid observation —
//! partial-obs by construction. Every worm sees a `GRID x GRID` window centered
//! on its head and rotated so its own heading points to grid +y ("up" / forward).
//! The viewport radius scales with the worm's length, so a longer worm sees more
//! of the arena (as in slither.io's zoom-out), which keeps the *resolution* of
//! nearby threats roughly constant.
//!
//! Channels (per the blueprint):
//!   0 enemy body     — obstacle to avoid / your future encircling wall is theirs
//!   1 enemy head     — threat & target; heading written as a small forward smear
//!   2 food density   — pellets
//!   3 own body       — self-collision + the wall you build when you encircle
//!   4 wall mask      — arena boundary inside the viewport
//!
//! A *delta* set (current − previous) of the same 5 channels is appended, giving
//! velocity/▵ without recurrence (the CS229 trick). Total channels = 10. A small
//! scalar vector (own length, speed fraction, boost-available) rounds it out.

use crate::geometry::Vec2;
use crate::world::{START_LENGTH, WORLD, World};

pub const GRID: usize = 32;
pub const SEMANTIC_CHANNELS: usize = 5;
pub const CHANNELS: usize = SEMANTIC_CHANNELS * 2;
pub const SCALARS: usize = 3;

const CH_ENEMY_BODY: usize = 0;
const CH_ENEMY_HEAD: usize = 1;
const CH_FOOD: usize = 2;
const CH_OWN_BODY: usize = 3;
const CH_WALL: usize = 4;

/// One worm's observation for a step: the stacked grid (channels-major,
/// row-major within a channel) and the scalar vector. The grid is
/// `CHANNELS * GRID * GRID` flattened so a CNN can reshape it directly.
#[derive(Clone, Debug)]
pub struct Obs {
    pub grid: Vec<f32>,
    pub scalars: [f32; SCALARS],
}

impl Obs {
    pub fn zeros() -> Self {
        Self {
            grid: vec![0.0; CHANNELS * GRID * GRID],
            scalars: [0.0; SCALARS],
        }
    }

    fn semantic_index(channel: usize, row: usize, col: usize) -> usize {
        (channel * GRID + row) * GRID + col
    }
}

/// Half-width of the egocentric viewport in world units, given the observer's
/// length. Grows sub-linearly so a giant worm still resolves nearby foes.
pub fn view_radius(length: f32) -> f32 {
    let r = 6.0 + (length / 16.0).min(20.0);
    260.0 + 18.0 * r
}

/// Build worm `idx`'s observation. `prev_semantic`, if present, holds the
/// previous step's 5-channel semantic grid (same layout, length
/// `SEMANTIC_CHANNELS * GRID * GRID`); the delta channels are filled from it and
/// it is updated in place for next step. On the first step pass an all-zero
/// buffer — the delta is then zero, which is correct.
pub fn observe(world: &World, idx: usize, prev_semantic: &mut [f32]) -> Obs {
    let mut obs = Obs::zeros();
    let me = &world.worms[idx];
    if me.dead {
        return obs;
    }

    let head = me.head();
    let radius = view_radius(me.length);
    // Rotate world→ego by -heading so the worm's forward maps to +y. cos/sin of
    // the negated angle, computed once.
    let fwd = me.angle;
    let (s, c) = (fwd.sin(), fwd.cos());
    let cell = 2.0 * radius / GRID as f32;
    // The rotated viewport square fits inside an axis-aligned box of this
    // half-extent; anything beyond it can't land in any cell, so reject it
    // before the (more expensive) rotation. Cuts the per-worm pellet/segment
    // sweep down to the local neighborhood.
    let bound = radius * std::f32::consts::SQRT_2;

    // Map a world point into integer grid cell, or None if outside the viewport.
    let to_cell = |p: Vec2| -> Option<(usize, usize)> {
        let dx = p.x - head.x;
        let dy = p.y - head.y;
        if dx.abs() >= bound || dy.abs() >= bound {
            return None;
        }
        // Heading-aligned frame: ex = right of heading, ey = forward.
        let ex = dx * c + dy * s;
        let ey = -dx * s + dy * c;
        if ex.abs() >= radius || ey.abs() >= radius {
            return None;
        }
        let col = ((ex + radius) / cell) as usize;
        let row = ((ey + radius) / cell) as usize;
        Some((row.min(GRID - 1), col.min(GRID - 1)))
    };

    for (j, other) in world.worms.iter().enumerate() {
        if other.dead {
            continue;
        }
        let own = j == idx;
        let body_channel = if own { CH_OWN_BODY } else { CH_ENEMY_BODY };
        let skip = if own { 1 } else { 0 };
        for &seg in other.segments.iter().skip(skip) {
            if let Some((r, col)) = to_cell(seg) {
                obs.grid[Obs::semantic_index(body_channel, r, col)] = 1.0;
            }
        }
        if !own && let Some((r, col)) = to_cell(other.head()) {
            obs.grid[Obs::semantic_index(CH_ENEMY_HEAD, r, col)] = 1.0;
            // Smear the head's heading one cell forward so the net can read its
            // direction without a separate angle scalar per enemy.
            let ahead = Vec2::new(
                other.head().x + other.angle.cos() * cell,
                other.head().y + other.angle.sin() * cell,
            );
            if let Some((r2, c2)) = to_cell(ahead) {
                obs.grid[Obs::semantic_index(CH_ENEMY_HEAD, r2, c2)] = 0.5;
            }
        }
    }

    for p in &world.pellets {
        if let Some((r, col)) = to_cell(p.pos) {
            let i = Obs::semantic_index(CH_FOOD, r, col);
            obs.grid[i] = (obs.grid[i] + 0.34 * p.value).min(1.0);
        }
    }

    // Wall mask: mark cells whose world position is outside the arena. Only a
    // worm within `bound` of a wall can see one, so skip the sweep otherwise.
    let near_wall =
        head.x < bound || head.x > WORLD - bound || head.y < bound || head.y > WORLD - bound;
    if near_wall {
        for row in 0..GRID {
            for col in 0..GRID {
                let ex = (col as f32 + 0.5) * cell - radius;
                let ey = (row as f32 + 0.5) * cell - radius;
                let wx = head.x + ex * c - ey * s;
                let wy = head.y + ex * s + ey * c;
                if wx <= 0.0 || wx >= WORLD || wy <= 0.0 || wy >= WORLD {
                    obs.grid[Obs::semantic_index(CH_WALL, row, col)] = 1.0;
                }
            }
        }
    }

    let semantic_len = SEMANTIC_CHANNELS * GRID * GRID;
    let (semantic, delta) = obs.grid.split_at_mut(semantic_len);
    for ((d, &cur), prev) in delta
        .iter_mut()
        .zip(semantic.iter())
        .zip(prev_semantic.iter_mut())
    {
        *d = cur - *prev;
        *prev = cur;
    }

    obs.scalars = [
        ((me.length - START_LENGTH) / 200.0).clamp(0.0, 1.0),
        me.speed / 320.0,
        if me.can_boost() { 1.0 } else { 0.0 },
    ];

    obs
}
