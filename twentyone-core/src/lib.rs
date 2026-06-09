//! Pure Rust implementation of the Twenty-One card game environment.
//!
//! This library provides a fast, allocation-conscious game engine for the Twenty-One
//! card game suitable for reinforcement learning experiments and other applications.
//!
//! # Example
//!
//! ```rust
//! use twentyone_core::{Env, Action};
//!
//! let mut env = Env::new(42);
//! env.start_new_round().unwrap();
//!
//! loop {
//!     let p = env.current_player();
//!     let obs = env.observation(p);
//!     let action = if obs.self_total < 17 { Action::Draw } else { Action::Stand };
//!     let result = env.step(action).unwrap();
//!     if result.round_over { break; }
//! }
//! ```

pub mod env;
pub mod solver;

pub use env::{Action, Env, EnvError, Observation, RoundOutcome, StepResult};
pub use solver::Solver;
