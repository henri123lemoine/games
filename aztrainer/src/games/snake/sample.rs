//! Snake's replay example and its minibatch expansion. The board is small
//! (20×20 × ~9 planes) and the buffer is bounded, so planes are stored unpacked
//! — the memory win from bit-packing is not worth the complexity here. Single
//! board size, no augmentation, no auxiliary heads.

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
            planes[i * plane_len..(i + 1) * plane_len].copy_from_slice(&s.planes);
            for &(idx, p) in &s.policy {
                targets[i * policy + usize::from(idx)] = p;
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
