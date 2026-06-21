//! Go's replay example and its minibatch expansion. Planes are bit-packed
//! (`PACKED_PLANES` planes × ⌈cells/64⌉ words) so a replay buffer of a million
//! 19×19 samples stays in hundreds of MB rather than tens of GB. Every draw
//! applies a uniformly random dihedral symmetry (go's 8-fold board symmetry, an
//! 8× data multiplier), and the buffer mixes board sizes (KataGo-style), so a
//! minibatch is grouped by size — each group one fixed-shape forward.

use std::collections::VecDeque;

use game_core::Rng;
use go::encode::{KOMI_SCALE, PLANES, d8, d8_policy};

use crate::net::NetConfig;
use crate::train::{Batch, TrainSample};

/// Every plane except the last two is a position-dependent bitset (stones,
/// liberties, illegal points, move history, ladders); the final two — the
/// signed-komi scalar and the ones plane — are constant fills reconstructed at
/// expansion, so only the bitset planes are packed.
const PACKED_PLANES: usize = PLANES - 2;

/// One self-play training example.
pub struct Sample {
    pub planes: Box<[u64]>,
    pub stm_white: bool,
    /// Sparse visit distribution over policy indices.
    pub policy: Vec<(u16, f32)>,
    /// Game outcome from the perspective of the player to move.
    pub z: f32,
    /// The search's root value at this position (player to move) — mixed into
    /// the value target to de-noise the raw outcome.
    pub q: f32,
    /// Final-board ownership, absolute (`+1` Black, `-1` White, `0` neutral),
    /// shared by every position in the game — the auxiliary territory target.
    pub ownership: Box<[i8]>,
    /// Final area-score margin in points from the player-to-move's view — the
    /// auxiliary score target (denser than the win/loss `z`).
    pub score: f32,
    /// This game's komi (points). Reconstructs the komi input plane on expand;
    /// constant within a game.
    pub komi: f32,
    /// Board edge length of this sample — the replay buffer mixes sizes
    /// (KataGo-style), so each example carries its own.
    pub size: u8,
}

fn words_per_plane(cells: usize) -> usize {
    cells.div_ceil(64)
}

/// Packs the encoder's f32 features into per-plane bitsets for a `size`×`size`
/// board. Returns the packed planes and whether the mover is White (the sign of
/// the signed-komi plane: `+` when White is to move).
pub fn compact(features: &[f32], size: usize) -> (Box<[u64]>, bool) {
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
    (
        planes.into_boxed_slice(),
        features[PACKED_PLANES * cells] > 0.0,
    )
}

/// Expands packed planes under symmetry `t` (0..8) into the net's input layout
/// for a `size`×`size` board, rebuilding the constant komi plane (signed by
/// mover) and the ones plane.
pub fn expand(planes: &[u64], stm_white: bool, komi: f32, t: u8, size: usize, out: &mut [f32]) {
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
    let self_komi = if stm_white { komi } else { -komi };
    out[PACKED_PLANES * cells..(PACKED_PLANES + 1) * cells]
        .fill((f64::from(self_komi) / KOMI_SCALE) as f32);
    out[(PACKED_PLANES + 1) * cells..PLANES * cells].fill(1.0);
}

impl TrainSample for Sample {
    fn expand(
        buf: &VecDeque<Self>,
        batch: usize,
        _cfg: &NetConfig,
        value_mix: f32,
        rng: &mut Rng,
    ) -> Vec<Batch> {
        // Group the minibatch by board size so each group is one fixed-shape
        // forward; the trainer sums their losses weighted by batch share.
        let mut groups: std::collections::HashMap<usize, Vec<&Self>> =
            std::collections::HashMap::new();
        for _ in 0..batch {
            let s = &buf[(rng.unit() * buf.len() as f64) as usize % buf.len()];
            groups.entry(s.size as usize).or_default().push(s);
        }
        groups
            .into_iter()
            .map(|(size, group)| {
                let n = group.len();
                let cells = size * size;
                let policy = size * size + 1;
                let plane_len = PLANES * cells;
                let mut planes = vec![0.0f32; n * plane_len];
                let mut targets = vec![0.0f32; n * policy];
                let mut value = vec![0.0f32; n];
                let mut owns = vec![0.0f32; n * cells];
                let mut scores = vec![0.0f32; n];
                for (i, s) in group.iter().enumerate() {
                    let t = rng.below(8) as u8;
                    expand(
                        &s.planes,
                        s.stm_white,
                        s.komi,
                        t,
                        size,
                        &mut planes[i * plane_len..(i + 1) * plane_len],
                    );
                    for &(idx, p) in &s.policy {
                        let ti = d8_policy(idx, t, size);
                        targets[i * policy + usize::from(ti)] = p;
                    }
                    value[i] = (1.0 - value_mix) * s.z + value_mix * s.q;
                    // Ownership from the mover's view (negate when White is to
                    // move; `s.ownership` is absolute Black-positive), under the
                    // same dihedral symmetry as the planes.
                    let sign = if s.stm_white { -1.0 } else { 1.0 };
                    let base = i * cells;
                    for (p, &o) in s.ownership.iter().enumerate() {
                        owns[base + d8(p, t, size)] = sign * f32::from(o);
                    }
                    // Score is already mover-relative and scalar — no transform.
                    scores[i] = s.score;
                }
                Batch {
                    n,
                    size,
                    planes,
                    policy: targets,
                    value,
                    ownership: owns,
                    score: scores,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{compact, expand};
    use game_core::{Game, PolicyValueEncoder};
    use go::encode::GoEncoder;
    use go::{Go, GoAction};

    #[test]
    fn compact_expand_roundtrip_under_identity() {
        // 19 exercises the >128-cell packing (361 bits / plane) that 9 does not.
        for size in [9usize, 19] {
            let g = Go::new(size);
            let enc = GoEncoder::new(size);
            let mut s = g.initial_state();
            for (i, _) in (0..6).enumerate() {
                let p = (i * 37 + 5) % (size * size);
                let placements: Vec<_> = g
                    .legal_actions(&s)
                    .into_iter()
                    .filter(|a| matches!(a, GoAction::Place(q) if (*q as usize) == p))
                    .collect();
                if let Some(&a) = placements.first() {
                    g.apply(&mut s, a);
                }
            }
            let x = enc.encode_state(&g, &s);
            let (planes, stm_white) = compact(&x, size);
            let mut back = vec![0.0f32; x.len()];
            expand(&planes, stm_white, g.komi() as f32, 0, size, &mut back);
            assert_eq!(x, back, "size {size}");
        }
    }

    #[test]
    fn expand_under_symmetry_matches_encoding_of_transformed_board() {
        for (size, coords) in [
            (9usize, vec!["e5", "c3", "g7", "d4", "f6"]),
            (19, vec!["k10", "d4", "q16", "c15", "r5"]),
        ] {
            let g = Go::new(size);
            let enc = GoEncoder::new(size);
            let mut s = g.initial_state();
            for c in &coords {
                g.apply(&mut s, GoAction::Place(g.point(c).unwrap()));
            }
            let (planes, stm_white) = compact(&enc.encode_state(&g, &s), size);
            for t in 0..8u8 {
                let mut ts = g.initial_state();
                for c in &coords {
                    let p = g.point(c).unwrap() as usize;
                    g.apply(&mut ts, GoAction::Place(go::encode::d8(p, t, size) as u16));
                }
                let want = enc.encode_state(&g, &ts);
                let mut got = vec![0.0f32; want.len()];
                expand(&planes, stm_white, g.komi() as f32, t, size, &mut got);
                assert_eq!(got, want, "size {size} symmetry {t}");
            }
        }
    }
}
