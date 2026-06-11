//! Small, dependency-free hash/mix primitives shared by games and solvers.
//!
//! Games need cheap, well-distributed keys for [`crate::Game::infoset_key`] /
//! [`crate::Game::state_key`]; before this module each crate carried its own
//! copy (and they drifted). These are not cryptographic.

/// SplitMix64 finalizer: a fast, well-distributed 64-bit mix.
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// Folds `v` into a running key. Sequence-sensitive: `combine(combine(0, a), b)`
/// differs from `combine(combine(0, b), a)`.
pub fn combine(key: u64, v: u64) -> u64 {
    splitmix64(key ^ v.wrapping_mul(0x9e3779b97f4a7c15))
}

/// FNV-1a over raw bytes — for hashing rendered/encoded values where a
/// streaming byte hash is the natural fit.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_distributes_low_entropy_inputs() {
        let a = splitmix64(1);
        let b = splitmix64(2);
        assert_ne!(a, b);
        assert!(
            (a ^ b).count_ones() > 16,
            "consecutive seeds should differ widely"
        );
    }

    #[test]
    fn combine_is_order_sensitive() {
        assert_ne!(combine(combine(0, 1), 2), combine(combine(0, 2), 1));
    }
}
