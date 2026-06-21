//! Headless slither.io RL substrate.
//!
//! [`world`] is the deterministic continuous-worm dynamics (head + heading,
//! fixed-spacing chains, food, boost, head-vs-body death). [`obs`] turns one
//! worm's view into an egocentric, viewport-clipped semantic grid. [`env`] wraps
//! the world in a Gym-like `reset`/`step` with discrete actions and the
//! encircle-shaped reward. [`heuristic`] is the hand-coded encircle worm — the
//! self-play pool's teacher and the gating opponent.
//!
//! No learning here yet: this milestone is the substrate the PPO trainer (next
//! milestone) plugs into. Everything is deterministic from a `u64` seed and
//! vectorizes over thousands of independent [`env::Env`]s.

pub mod env;
pub mod geometry;
pub mod heuristic;
pub mod obs;
pub mod rng;
pub mod world;

pub use env::{Action, Env, SHAPES, StepOut};
pub use heuristic::Heuristic;
pub use obs::{CHANNELS, GRID, Obs};
pub use rng::Rng;
pub use world::{World, WorldConfig};

/// A random worm policy: uniform over turn buckets, with an occasional boost.
/// The throwaway baseline the heuristic must clearly beat.
pub fn random_action(rng: &mut Rng) -> Action {
    Action {
        turn: rng.below(env::TURN_BUCKETS) as u8,
        boost: rng.unit() < 0.05,
    }
}
