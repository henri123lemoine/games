//! 2-D vectors and angle math shared by the dynamics, the observation grid, and
//! the heuristic. Angles are radians; "forward" is `+x` at angle 0, matching the
//! TS clone the constants were cribbed from.

use std::f32::consts::{PI, TAU};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn from_angle(angle: f32, len: f32) -> Self {
        Self::new(angle.cos() * len, angle.sin() * len)
    }

    pub fn dist2(self, o: Vec2) -> f32 {
        let dx = self.x - o.x;
        let dy = self.y - o.y;
        dx * dx + dy * dy
    }

    pub fn dist(self, o: Vec2) -> f32 {
        self.dist2(o).sqrt()
    }
}

/// Shortest signed angular difference `target - current`, wrapped to `(-PI, PI]`.
pub fn angle_diff(current: f32, target: f32) -> f32 {
    let mut d = (target - current) % TAU;
    if d > PI {
        d -= TAU;
    } else if d < -PI {
        d += TAU;
    }
    d
}

/// Rotate `current` toward `target` by at most `max_step` radians.
pub fn turn_toward(current: f32, target: f32, max_step: f32) -> f32 {
    let d = angle_diff(current, target).clamp(-max_step, max_step);
    current + d
}
