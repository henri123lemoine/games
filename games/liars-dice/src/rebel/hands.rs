//! Multiset hand enumeration and a fixed global index space.
//!
//! A hand for `d` dice over `F` faces is the per-face count vector
//! `[c_0, ..., c_{F-1}]` summing to `d`. Hands are enumerated in a fixed
//! canonical order (the combinatorial rank of their count vector, which is
//! lexicographic over the vector); [`index_within`] and [`from_index_within`]
//! are mutual inverses over that order.
//!
//! The global index space concatenates the per-dice-count blocks for
//! `d = 0..=MAX_DICE` over [`MAX_FACES`] faces, giving a single layout of size
//! [`H`] that a network spanning every supported seat dice count can address. A
//! hand from a fewer-face config embeds with zero counts on the unused faces.

pub type Hand = [u8; MAX_FACES];

/// Faces the global index space (and the network) span. Liar's Dice configs use
/// 2..=6 faces; the flagship is 6 faces.
pub const MAX_FACES: usize = 6;

/// Largest per-seat dice count the global index space (and the network) span.
/// The flagship config is 5p5d6f and a seat's dice only ever decrease in play;
/// supporting more dice is a matter of raising this and retraining.
pub const MAX_DICE: usize = 5;

/// Size of the global hand layout: `Σ_{d=0}^{MAX_DICE} C(d + MAX_FACES - 1, d)`.
/// For `MAX_DICE = 5`, `MAX_FACES = 6` this is 462.
pub const H: usize = global_size();

const fn comb(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = if k > n - k { n - k } else { k };
    let mut num = 1;
    let mut den = 1;
    let mut i = 0;
    while i < k {
        num *= n - i;
        den *= i + 1;
        i += 1;
    }
    num / den
}

const fn global_size() -> usize {
    let mut total = 0;
    let mut d = 0;
    while d <= MAX_DICE {
        total += comb(d + MAX_FACES - 1, d);
        d += 1;
    }
    total
}

/// Number of distinct multiset hands for `d` dice over `faces` faces.
pub fn hand_count(d: u8, faces: u8) -> usize {
    comb(d as usize + faces as usize - 1, d as usize)
}

/// Number of non-negative integer vectors of length `len` summing to `sum`.
fn count_vectors(len: usize, sum: usize) -> usize {
    if len == 0 {
        return usize::from(sum == 0);
    }
    comb(sum + len - 1, len - 1)
}

/// Rank of `hand` within the canonical enumeration of `d`-dice `faces`-face
/// hands. Inverse of [`from_index_within`].
pub fn index_within(hand: &Hand, d: u8, faces: u8) -> usize {
    let f = faces as usize;
    let mut rank = 0;
    let mut rem = d as usize;
    for (i, &count) in hand.iter().enumerate().take(f) {
        let after = f - 1 - i;
        let ci = count as usize;
        for v in 0..ci {
            rank += count_vectors(after, rem - v);
        }
        rem -= ci;
    }
    rank
}

/// The `idx`-th hand of `d` dice over `faces` faces. Inverse of [`index_within`].
pub fn from_index_within(mut idx: usize, d: u8, faces: u8) -> Hand {
    let f = faces as usize;
    let mut hand = [0u8; MAX_FACES];
    let mut rem = d as usize;
    for (i, slot) in hand.iter_mut().enumerate().take(f) {
        let after = f - 1 - i;
        for v in 0..=rem {
            let ways = count_vectors(after, rem - v);
            if idx < ways {
                *slot = v as u8;
                rem -= v;
                break;
            }
            idx -= ways;
        }
    }
    hand
}

/// Every `d`-dice `faces`-face hand in canonical order.
pub fn enumerate(d: u8, faces: u8) -> Vec<Hand> {
    (0..hand_count(d, faces))
        .map(|i| from_index_within(i, d, faces))
        .collect()
}

fn block_offset(d: u8) -> usize {
    (0..d).map(|dd| hand_count(dd, MAX_FACES as u8)).sum()
}

/// The slice of the global `H`-vector owned by a seat holding `d` dice.
pub fn global_block(d: u8) -> std::ops::Range<usize> {
    let start = block_offset(d);
    start..start + hand_count(d, MAX_FACES as u8)
}

/// Global index of `hand` (a `d`-dice hand) in the fixed `H`-wide layout.
pub fn global_index(hand: &Hand, d: u8) -> usize {
    block_offset(d) + index_within(hand, d, MAX_FACES as u8)
}

fn factorial(n: u8) -> f64 {
    (1..=u64::from(n)).product::<u64>() as f64
}

fn multinomial_coeff(hand: &Hand, d: u8) -> f64 {
    let mut coeff = factorial(d);
    for &c in hand.iter() {
        coeff /= factorial(c);
    }
    coeff
}

/// Multinomial probability of each `d`-dice `faces`-face hand under fair dice,
/// indexed within and summing to 1.
pub fn prior(d: u8, faces: u8) -> Vec<f64> {
    let p_each = 1.0 / f64::from(faces);
    enumerate(d, faces)
        .iter()
        .map(|hand| multinomial_coeff(hand, d) * p_each.powi(i32::from(d)))
        .collect()
}

/// Distribution over how many of a `d`-dice hand's dice equal `face` (0-based),
/// given a `belief` indexed within the `d`-dice `faces`-face hands. Length
/// `d + 1`; sums to whatever `belief` sums to (1 for a normalized belief).
pub fn face_count_marginal(belief: &[f64], d: u8, faces: u8, face: usize) -> Vec<f64> {
    let mut out = vec![0.0; d as usize + 1];
    for (hand, &b) in enumerate(d, faces).iter().zip(belief) {
        out[hand[face] as usize] += b;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hand_count_matches_binomial() {
        for &(d, f) in &[(0u8, 6u8), (1, 4), (2, 3), (5, 6), (3, 5), (4, 2)] {
            assert_eq!(
                hand_count(d, f),
                comb(d as usize + f as usize - 1, d as usize)
            );
            assert_eq!(enumerate(d, f).len(), hand_count(d, f));
        }
    }

    #[test]
    fn index_round_trips_over_all_hands() {
        for &(d, f) in &[(0u8, 6u8), (1, 4), (1, 5), (2, 3), (3, 6), (5, 6)] {
            let hands = enumerate(d, f);
            for (i, h) in hands.iter().enumerate() {
                assert_eq!(index_within(h, d, f), i);
                assert_eq!(&from_index_within(i, d, f), h);
                assert_eq!(h.iter().map(|&c| c as usize).sum::<usize>(), d as usize);
            }
            let unique: std::collections::HashSet<_> = hands.iter().collect();
            assert_eq!(unique.len(), hands.len());
        }
    }

    fn brute_prior(d: u8, faces: u8) -> Vec<f64> {
        let mut out = vec![0.0; hand_count(d, faces)];
        let total = (faces as usize).pow(d as u32);
        for code in 0..total {
            let mut c = code;
            let mut hand = [0u8; MAX_FACES];
            for _ in 0..d as usize {
                hand[c % faces as usize] += 1;
                c /= faces as usize;
            }
            out[index_within(&hand, d, faces)] += 1.0 / total as f64;
        }
        out
    }

    #[test]
    fn prior_sums_to_one_and_matches_ordered_rolls() {
        for &(d, f) in &[(1u8, 4u8), (1, 5), (2, 3), (3, 6), (2, 6)] {
            let p = prior(d, f);
            let s: f64 = p.iter().sum();
            assert!((s - 1.0).abs() < 1e-12, "prior {d}x{f} sums to {s}");
            let b = brute_prior(d, f);
            assert_eq!(p.len(), b.len());
            for (pi, bi) in p.iter().zip(&b) {
                assert!((pi - bi).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn global_index_is_a_bijection_over_the_blocks() {
        let mut seen = vec![false; H];
        let mut count = 0;
        for d in 0..=MAX_DICE as u8 {
            let block = global_block(d);
            for hand in enumerate(d, MAX_FACES as u8) {
                let g = global_index(&hand, d);
                assert!(
                    block.contains(&g),
                    "global index {g} outside block for d={d}"
                );
                assert!(g < H);
                assert!(!seen[g], "global index {g} collided");
                seen[g] = true;
                count += 1;
            }
        }
        assert_eq!(count, H);
        assert!(seen.iter().all(|&x| x));
        assert_eq!(H, 462);
    }

    fn brute_face_marginal(d: u8, faces: u8, face: usize) -> Vec<f64> {
        let mut out = vec![0.0; d as usize + 1];
        let total = (faces as usize).pow(d as u32);
        for code in 0..total {
            let mut c = code;
            let mut k = 0;
            for _ in 0..d as usize {
                if c % faces as usize == face {
                    k += 1;
                }
                c /= faces as usize;
            }
            out[k] += 1.0 / total as f64;
        }
        out
    }

    #[test]
    fn face_count_marginal_sums_to_one_and_matches_brute_force() {
        for &(d, f) in &[(1u8, 4u8), (2, 3), (3, 6), (5, 6)] {
            for face in 0..f as usize {
                let m = face_count_marginal(&prior(d, f), d, f, face);
                let s: f64 = m.iter().sum();
                assert!((s - 1.0).abs() < 1e-12);
                let b = brute_face_marginal(d, f, face);
                assert_eq!(m.len(), b.len());
                for (mi, bi) in m.iter().zip(&b) {
                    assert!((mi - bi).abs() < 1e-12);
                }
            }
        }
    }
}
