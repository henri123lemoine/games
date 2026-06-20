//! Chess's replay example and its minibatch expansion. The binary feature
//! planes are packed as bitboards; plane 17 (the halfmove clock) is a uniform
//! fill reconstructed from the stored counter. Fixed 8×8 board, full 4672-wide
//! dense policy target, no augmentation, no auxiliary heads.

use std::collections::VecDeque;

use chess::Board;
use chess::encode::{PLANE_COUNT, encode_planes};
use game_core::Rng;

use crate::net::NetConfig;
use crate::train::{Batch, TrainSample};

/// One training example, planes packed as bitboards (plane 17, the halfmove
/// clock, is a uniform fill reconstructed from `halfmove`).
pub struct Sample {
    pub planes: [u64; 17],
    pub halfmove: u8,
    /// Sparse visit distribution over AZ policy indices.
    pub policy: Vec<(u16, f32)>,
    /// Game outcome from the perspective of the player to move.
    pub z: f32,
    /// The search's root value at this position (player to move) — mixed into
    /// the value target to de-noise the raw outcome.
    pub q: f32,
}

/// Bit-packs the binary planes of [`encode_planes`]; plane 17 is uniform
/// `halfmove / 100`, stored as the raw counter.
pub fn compact_planes(b: &Board) -> ([u64; 17], u8) {
    let x = encode_planes(b);
    let mut planes = [0u64; 17];
    for (p, plane) in planes.iter_mut().enumerate() {
        for sq in 0..64 {
            if x[p * 64 + sq] != 0.0 {
                *plane |= 1 << sq;
            }
        }
    }
    (planes, b.halfmove.min(100) as u8)
}

/// Reconstructs the full feature planes from the bit-packed form into `out`.
pub fn expand_planes(planes: &[u64; 17], halfmove: u8, out: &mut [f32]) {
    debug_assert_eq!(out.len(), PLANE_COUNT * 64);
    out.fill(0.0);
    for (p, &bits) in planes.iter().enumerate() {
        let mut b = bits;
        while b != 0 {
            let sq = b.trailing_zeros() as usize;
            out[p * 64 + sq] = 1.0;
            b &= b - 1;
        }
    }
    out[17 * 64..].fill(f32::from(halfmove) / 100.0);
}

impl TrainSample for Sample {
    fn expand(
        buf: &VecDeque<Self>,
        n: usize,
        cfg: &NetConfig,
        value_mix: f32,
        rng: &mut Rng,
    ) -> Vec<Batch> {
        let policy = cfg.policy() as usize;
        let plane_len = PLANE_COUNT * 64;
        let mut planes = vec![0.0f32; n * plane_len];
        let mut targets = vec![0.0f32; n * policy];
        let mut value = vec![0.0f32; n];
        for i in 0..n {
            let s = &buf[(rng.unit() * buf.len() as f64) as usize % buf.len()];
            expand_planes(
                &s.planes,
                s.halfmove,
                &mut planes[i * plane_len..(i + 1) * plane_len],
            );
            for &(idx, p) in &s.policy {
                targets[i * policy + usize::from(idx)] = p;
            }
            value[i] = (1.0 - value_mix) * s.z + value_mix * s.q;
        }
        vec![Batch {
            n,
            size: 8,
            planes,
            policy: targets,
            value,
            ownership: Vec::new(),
            score: Vec::new(),
        }]
    }
}
