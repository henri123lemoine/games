//! Reproducible sampling helpers: distributions live in [`game_core::rand`];
//! this module adds the seed-mixing the training loop uses.

pub(crate) use game_core::rand::normal;

#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) fn mix(a: u64, b: u64) -> u64 {
    game_core::hash::combine(a, b)
}
