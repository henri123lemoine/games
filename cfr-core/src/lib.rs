//! Generic counterfactual-regret machinery over a [`Game`] trait.
//!
//! A `Game` is a two-player zero-sum extensive game with chance and imperfect
//! information. Implement the trait once per game and you get, for free:
//!
//! * [`Solver`] — external-sampling MCCFR+ training that converges to a Nash
//!   equilibrium, and
//! * [`Solver::exploitability`] — *exact* best-response exploitability
//!   (NashConv), the real measure of how close a strategy is to unbeatable.
//!
//! The design follows the OpenSpiel pattern: the game exposes chance vs. decision
//! nodes, legal actions, an information-set key per acting player, and terminal
//! returns; the algorithms are written once against that interface.

mod arena;
mod mccfr;
mod solver;

pub use arena::{Agent, Rng, play, play_n, playout_from, win_rate, winrate_vs_field};
pub use mccfr::Mccfr;
pub use solver::Solver;

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

/// Whose turn it is at a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Turn {
    /// A chance node — outcomes come from [`Game::chance_outcomes`].
    Chance,
    /// Decision node for the given player index (0 or 1).
    Player(usize),
}

/// A two-player zero-sum extensive game with chance and imperfect information.
///
/// Implementors describe the rules; [`Solver`] supplies the algorithms. Actions
/// at a decision node are identified by their position in [`Game::legal_actions`],
/// which must be deterministic for a given information set so regret/strategy
/// vectors line up across states that share a key.
pub trait Game: Sync {
    /// Game state. Cloning must be cheap-ish — the solver clones to branch.
    type State: Clone;
    /// An action token. `apply` interprets it; the solver only stores positions.
    type Action: Copy + std::fmt::Debug;

    /// Number of players (must be 2 for the zero-sum solver).
    fn num_players(&self) -> usize {
        2
    }

    /// The starting state (typically a chance node that deals/rolls).
    fn initial_state(&self) -> Self::State;

    /// Whose move it is — chance, or a specific player.
    fn turn(&self, state: &Self::State) -> Turn;

    /// Whether `state` is terminal.
    fn is_terminal(&self, state: &Self::State) -> bool;

    /// Utility to `player` at a terminal state, in `[-1, 1]` for a win/loss game.
    fn returns(&self, state: &Self::State, player: usize) -> f64;

    /// Legal actions at a decision node, in a stable order for the information set.
    fn legal_actions(&self, state: &Self::State) -> Vec<Self::Action>;

    /// Chance outcomes and their probabilities at a chance node (sum to 1).
    fn chance_outcomes(&self, state: &Self::State) -> Vec<(Self::Action, f64)>;

    /// Apply an action (decision or chance outcome), mutating the state.
    fn apply(&self, state: &mut Self::State, action: Self::Action);

    /// Information-set key for `player`: identical for states `player` cannot
    /// tell apart, and distinct otherwise. Must encode everything `player`
    /// observes (their private info + public history) and nothing they don't.
    fn infoset_key(&self, state: &Self::State, player: usize) -> u64;

    /// God's-eye canonical key for a state, used to memoize the exact
    /// best-response over the game DAG. Default `None` disables memoization
    /// (correct but slower); override for any game with revisited states.
    fn state_key(&self, _state: &Self::State) -> Option<u64> {
        None
    }
}
