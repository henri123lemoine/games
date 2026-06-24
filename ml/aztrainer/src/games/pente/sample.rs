//! Pente's replay example and its minibatch expansion. The first five planes are
//! position-dependent bitsets (own/opponent stones, last placement, vulnerable
//! and capturable pairs); the last three — the two captured-pair-count scalars
//! and the ones plane — are constant fills reconstructed at expansion, so only
//! the bitset planes are packed. Every draw applies a uniformly random dihedral
//! symmetry (Pente's 8-fold board symmetry, an 8× data multiplier). Single board
//! size, no auxiliary heads.

use std::collections::VecDeque;

use game_core::Rng;
use pente::PAIRS_TO_WIN;
use pente::encode::{PLANES, d8, d8_policy};

use crate::net::NetConfig;
use crate::train::{Batch, TrainSample};

/// The five position-dependent bitset planes (own, opp, last, own-vulnerable,
/// opp-capturable). The two pair-count planes and the ones plane are constant
/// fills rebuilt at expansion, so they are not packed.
const PACKED_PLANES: usize = PLANES - 3;

/// One self-play training example.
pub struct Sample {
    pub planes: Box<[u64]>,
    /// Sparse visit distribution over policy indices (board points; never the
    /// pass slot, which Pente has no legal action for).
    pub policy: Vec<(u16, f32)>,
    /// Game outcome from the perspective of the player to move.
    pub z: f32,
    /// The search's root value at this position (player to move) — mixed into
    /// the value target to de-noise the raw outcome.
    pub q: f32,
    /// Captured pairs of the mover and the opponent at this position, broadcast
    /// to the two constant pair-count planes (rebuilt on expand).
    pub own_pairs: u8,
    pub opp_pairs: u8,
    /// Board edge length of this sample.
    pub size: u8,
}

fn words_per_plane(cells: usize) -> usize {
    cells.div_ceil(64)
}

/// Packs the encoder's f32 features into per-plane bitsets for a `size`×`size`
/// board (the five position-dependent planes only). The pair-count and ones
/// planes are constant fills carried separately and rebuilt on expand.
pub fn compact(features: &[f32], size: usize) -> Box<[u64]> {
    let cells = size * size;
    debug_assert_eq!(features.len(), PLANES * cells);
    let wpp = words_per_plane(cells);
    let mut planes = vec![0u64; PACKED_PLANES * wpp];
    for p in 0..PACKED_PLANES {
        for cell in 0..cells {
            if features[p * cells + cell] != 0.0 {
                planes[p * wpp + cell / 64] |= 1 << (cell % 64);
            }
        }
    }
    planes.into_boxed_slice()
}

/// Expands packed planes under symmetry `t` (0..8) into the net's input layout
/// for a `size`×`size` board, rebuilding the constant pair-count planes (mover
/// then opponent, each scaled toward the five-pair win) and the ones plane.
pub fn expand(planes: &[u64], own_pairs: u8, opp_pairs: u8, t: u8, size: usize, out: &mut [f32]) {
    let cells = size * size;
    debug_assert_eq!(out.len(), PLANES * cells);
    let wpp = words_per_plane(cells);
    out.fill(0.0);
    for p in 0..PACKED_PLANES {
        for w in 0..wpp {
            let mut bits = planes[p * wpp + w];
            while bits != 0 {
                let cell = w * 64 + bits.trailing_zeros() as usize;
                out[p * cells + d8(cell, t, size)] = 1.0;
                bits &= bits - 1;
            }
        }
    }
    let denom = f32::from(PAIRS_TO_WIN);
    let own = PACKED_PLANES;
    let opp = PACKED_PLANES + 1;
    let ones = PACKED_PLANES + 2;
    out[own * cells..(own + 1) * cells].fill(f32::from(own_pairs) / denom);
    out[opp * cells..(opp + 1) * cells].fill(f32::from(opp_pairs) / denom);
    out[ones * cells..(ones + 1) * cells].fill(1.0);
}

impl TrainSample for Sample {
    fn expand(
        buf: &VecDeque<Self>,
        batch: usize,
        _cfg: &NetConfig,
        value_mix: f32,
        rng: &mut Rng,
    ) -> Vec<Batch> {
        let n = batch;
        let s0 = &buf[(rng.unit() * buf.len() as f64) as usize % buf.len()];
        let size = s0.size as usize;
        let cells = size * size;
        let policy = size * size + 1;
        let plane_len = PLANES * cells;
        let mut planes = vec![0.0f32; n * plane_len];
        let mut targets = vec![0.0f32; n * policy];
        let mut value = vec![0.0f32; n];
        for i in 0..n {
            let s = &buf[(rng.unit() * buf.len() as f64) as usize % buf.len()];
            let t = rng.below(8) as u8;
            expand(
                &s.planes,
                s.own_pairs,
                s.opp_pairs,
                t,
                size,
                &mut planes[i * plane_len..(i + 1) * plane_len],
            );
            for &(idx, p) in &s.policy {
                let ti = d8_policy(idx, t, size);
                targets[i * policy + usize::from(ti)] = p;
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

#[cfg(test)]
mod tests {
    use super::{compact, expand};
    use game_core::{Game, PolicyValueEncoder};
    use pente::encode::{PenteEncoder, d8};
    use pente::{Pente, PenteAction};

    #[test]
    fn compact_expand_roundtrip_under_identity() {
        for size in [9usize, 13] {
            let g = Pente::new(size);
            let enc = PenteEncoder::new(size);
            let mut s = g.initial_state();
            for _ in 0..8 {
                let actions = g.legal_actions(&s);
                g.apply(&mut s, actions[actions.len() / 2]);
                if g.is_terminal(&s) {
                    break;
                }
            }
            let x = enc.encode_state(&g, &s);
            let planes = compact(&x, size);
            let pairs = s.pairs();
            let own = pairs[s.to_move()];
            let opp = pairs[s.to_move() ^ 1];
            let mut back = vec![0.0f32; x.len()];
            expand(&planes, own, opp, 0, size, &mut back);
            assert_eq!(x, back, "size {size}");
        }
    }

    #[test]
    fn expand_under_symmetry_matches_encoding_of_transformed_board() {
        let size = 9;
        let g = Pente::new(size);
        let enc = PenteEncoder::new(size);
        let coords = ["e5", "c3", "g7", "d4", "f6"];
        let mut s = g.initial_state();
        // First move forced to center; then steer placements onto coords.
        let actions = g.legal_actions(&s);
        g.apply(&mut s, actions[0]);
        for c in &coords {
            let p = g.point(c).unwrap();
            if let Some(&a) = g.legal_actions(&s).iter().find(|a| a.0 == p) {
                g.apply(&mut s, a);
                if g.is_terminal(&s) {
                    break;
                }
            }
        }
        let x = enc.encode_state(&g, &s);
        let planes = compact(&x, size);
        let pairs = s.pairs();
        let own = pairs[s.to_move()];
        let opp = pairs[s.to_move() ^ 1];
        for t in 0..8u8 {
            let mut got = vec![0.0f32; x.len()];
            expand(&planes, own, opp, t, size, &mut got);
            // Every plane point maps to its d8 image of the identity encoding.
            for plane in 0..super::PLANES {
                for p in 0..size * size {
                    assert_eq!(
                        got[plane * size * size + d8(p, t, size)],
                        x[plane * size * size + p],
                        "plane {plane} point {p} symmetry {t}"
                    );
                }
            }
        }
        let _ = PenteAction(0);
    }
}
