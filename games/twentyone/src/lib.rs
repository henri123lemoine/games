//! Twenty-One: a 2-player hidden-information card game, with its fast
//! specialized solver and a [`cfr_core::Game`] adapter.
//!
//! - [`env`] — the game engine (rules, observations, controllable chance).
//! - [`solver`] — decomposed external-sampling MCCFR+ over lossless sufficient
//!   information sets, with exact best-response exploitability and save/load.
//!   This is the strong way to solve the real game (see `BAKEOFF.md`).
//! - [`game`] — the engine adapted to the generic [`cfr_core::Game`] trait so
//!   Twenty-One plugs into the same arena/tooling as the other games.

pub mod env;
pub mod game;
pub mod solver;

pub use env::{Action, Env, EnvError, Observation, RoundOutcome, StepResult};
pub use game::{T21State, TwentyOne};
pub use solver::Solver;
