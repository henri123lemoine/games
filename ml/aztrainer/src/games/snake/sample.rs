//! Battlesnake replay examples with random dihedral augmentation. Absolute
//! direction labels are transformed with the board, preserving the exact
//! policy meaning under all eight square symmetries.

use std::collections::VecDeque;

use game_core::Rng;

use crate::net::NetConfig;
use crate::train::{Batch, TrainSample};

/// One self-play training example.
pub struct Sample {
    pub planes: Vec<f32>,
    /// Sparse visit distribution over policy indices (the four headings).
    pub policy: Vec<(u16, f32)>,
    /// Game outcome from the perspective of the player to move.
    pub z: f32,
    /// The search's root value at this position (player to move) — mixed into
    /// the value target to de-noise the raw outcome.
    pub q: f32,
}

impl TrainSample for Sample {
    fn expand(
        buf: &VecDeque<Self>,
        n: usize,
        cfg: &NetConfig,
        value_mix: f32,
        rng: &mut Rng,
    ) -> Vec<Batch> {
        let size = cfg.size as usize;
        let cells = size * size;
        let policy = cfg.policy() as usize;
        let plane_len = cfg.planes as usize * cells;
        let mut planes = vec![0.0f32; n * plane_len];
        let mut targets = vec![0.0f32; n * policy];
        let mut value = vec![0.0f32; n];
        for i in 0..n {
            let s = &buf[(rng.unit() * buf.len() as f64) as usize % buf.len()];
            let symmetry = rng.below(8);
            transform_planes(
                &s.planes,
                &mut planes[i * plane_len..(i + 1) * plane_len],
                cfg.planes as usize,
                size,
                symmetry,
            );
            for &(idx, p) in &s.policy {
                let transformed = transform_direction(usize::from(idx), symmetry);
                targets[i * policy + transformed] = p;
            }
            value[i] = (1.0 - value_mix) * s.z + value_mix * s.q;
        }
        vec![Batch {
            n,
            size,
            planes,
            policy: targets,
            value,
            ownership: Vec::new(),
            score: Vec::new(),
        }]
    }
}

fn transform_planes(
    source: &[f32],
    target: &mut [f32],
    planes: usize,
    size: usize,
    symmetry: usize,
) {
    let cells = size * size;
    for plane in 0..planes {
        for y in 0..size {
            for x in 0..size {
                let (tx, ty) = transform_xy(x, y, size, symmetry);
                target[plane * cells + ty * size + tx] = source[plane * cells + y * size + x];
            }
        }
    }
}

fn transform_xy(mut x: usize, y: usize, size: usize, symmetry: usize) -> (usize, usize) {
    let mut y = y;
    if symmetry >= 4 {
        x = size - 1 - x;
    }
    for _ in 0..symmetry % 4 {
        (x, y) = (size - 1 - y, x);
    }
    (x, y)
}

fn transform_direction(direction: usize, symmetry: usize) -> usize {
    let (mut dx, mut dy) = match direction {
        0 => (0, 1),
        1 => (1, 0),
        2 => (0, -1),
        3 => (-1, 0),
        _ => panic!("direction outside 0..4"),
    };
    if symmetry >= 4 {
        dx = -dx;
    }
    for _ in 0..symmetry % 4 {
        (dx, dy) = (-dy, dx);
    }
    match (dx, dy) {
        (0, 1) => 0,
        (1, 0) => 1,
        (0, -1) => 2,
        (-1, 0) => 3,
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_symmetry_is_a_bijection_and_moves_policy_with_cells() {
        for symmetry in 0..8 {
            let mut seen = [false; 121];
            for y in 0..11 {
                for x in 0..11 {
                    let (tx, ty) = transform_xy(x, y, 11, symmetry);
                    assert!(!seen[ty * 11 + tx]);
                    seen[ty * 11 + tx] = true;
                }
            }
            for direction in 0..4 {
                assert!(transform_direction(direction, symmetry) < 4);
            }
        }
    }
}
