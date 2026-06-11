//! Generic game-playing algorithms, written once against `game_core`'s traits
//! and reusable by every game (the OpenSpiel pattern). What a game must provide
//! is stated per algorithm:
//!
//! | algorithm | requires | for |
//! |-----------|----------|-----|
//! | [`Cfr`] (vanilla CFR+, exact exploitability) | `Game` (2p zero-sum, small) | tiny imperfect-info games, correctness yardsticks |
//! | [`Mccfr`] (external-sampling MCCFR+) | `Game` (2p+) | mid-size imperfect-info games with shallow action ladders |
//! | [`OsMccfr`](os_mccfr::OsMccfr) (outcome-sampling MCCFR) | `Game` (2p+) | deep action ladders external sampling can't afford |
//! | [`AlphaBeta`] | `Game + Eval` (+ optional [`SearchSpec`]) | perfect-information games |
//! | [`Mcts`](mcts::Mcts) (UCT, solver, RAVE) | `Game` (+ optional `Eval`/[`SearchSpec`]) | perfect-information games without a sharp eval |
//! | [`azero`] (PUCT + self-play training) | `Game + PolicyValueEncoder` | perfect-information games, learned eval |
//! | [`Reinforce`](pg::Reinforce) (policy gradient) | `Game + PolicyValueEncoder` | small games, baseline learning curves |
//! | [`QLearner`](qlearn::QLearner) (tabular Q) | `Game` | tiny games, baseline learning curves |
//! | [`Rollout`] | `Game + Determinizer` + a base [`Agent`] | large imperfect-info games |
//!
//! Game-*specific* algorithms (e.g. Twenty-One's round-decomposed solver) live
//! with their game; this crate is only for algorithms that generalize.

pub mod azero;
mod cfr;
mod mccfr;
pub mod mcts;
pub mod os_mccfr;
pub mod pg;
pub mod qlearn;
mod rollout;
pub mod search;
mod tabular;

pub use cfr::Cfr;
pub use mccfr::Mccfr;
pub use rollout::Rollout;
pub use search::AlphaBeta;

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// FxHash-style hasher for already-well-distributed `u64` keys.
#[derive(Default)]
pub(crate) struct FxHasher(u64);
impl Hasher for FxHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}
pub(crate) type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;
