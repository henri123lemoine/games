//! The foundations of the games lab: the [`Game`] trait, the [`Agent`]
//! interface, capability traits, and the arena for running matches.
//!
//! This crate deliberately contains **no algorithms**. Algorithms live in the
//! `solvers` crate, written once against these interfaces (the OpenSpiel
//! pattern); games live in `games/*` and provide only their rules plus whatever
//! *game knowledge* an algorithm needs, declared through capability traits:
//!
//! * [`Game`] — the rules: chance vs. decision nodes, legal actions, terminal
//!   returns, information-set keys. Implementing it is the only requirement.
//! * [`Eval`] — a heuristic state value, which unlocks depth-limited search
//!   (alpha-beta, MCTS-style cutoffs) for perfect-information games.
//! * [`Determinizer`] — sampling of hidden information consistent with one
//!   player's view, which unlocks determinized Monte-Carlo methods for
//!   imperfect-information games.
//! * [`GameUi`] — per-player rendering and action labels/parsing, which gives
//!   every game the same terminal client and, later, the same web service.

mod arena;
pub mod hash;
pub mod json;
pub mod rand;
pub mod stats;
mod ui;

pub use arena::{
    Agent, RandomAgent, Rng, play, play_n, playout_from, win_rate, win_share, winner,
    winrate_vs_field,
};
pub use ui::GameUi;

/// Whose turn it is at a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Turn {
    /// A chance node — outcomes come from [`Game::chance_outcomes`].
    Chance,
    /// Decision node for the given player index.
    Player(usize),
}

/// An N-player zero-sum extensive game with chance and imperfect information.
///
/// Implementors describe the rules; the `solvers` crate supplies the
/// algorithms. Actions at a decision node are identified by their position in
/// [`Game::legal_actions`], which must be deterministic for a given information
/// set so regret/strategy vectors line up across states that share a key.
pub trait Game: Sync {
    /// Game state. Cloning must be cheap-ish — algorithms clone to branch,
    /// and parallel algorithms move states across threads.
    type State: Clone + Send + Sync;
    /// An action token. `apply` interprets it; algorithms only store positions.
    type Action: Copy + std::fmt::Debug + Send + Sync;

    /// Number of players (tabular CFR and exploitability require 2).
    fn num_players(&self) -> usize {
        2
    }

    /// The starting state (typically a chance node that deals/rolls).
    fn initial_state(&self) -> Self::State;

    /// Whose move it is — chance, or a specific player.
    fn turn(&self, state: &Self::State) -> Turn;

    /// Whether `state` is terminal.
    fn is_terminal(&self, state: &Self::State) -> bool;

    /// Utility to `player` at a terminal state, within
    /// `[-max_return, max_return]`.
    fn returns(&self, state: &Self::State, player: usize) -> f64;

    /// Upper bound on `|returns|` over all terminal states (default 1.0 — the
    /// win/loss convention). Algorithms that mix static evaluations with
    /// returns or detect proven wins rely on this bound being tight enough
    /// that no return exceeds it.
    fn max_return(&self) -> f64 {
        1.0
    }

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

    /// God's-eye canonical key for a state, used to memoize over the game DAG.
    /// Default `None` disables memoization (correct but slower); override for
    /// any game with revisited states.
    fn state_key(&self, _state: &Self::State) -> Option<u64> {
        None
    }

    /// A stable 64-bit identity for an action, used by algorithms to index
    /// killer/history/RAVE tables across states. Two actions must collide only
    /// if they are the same move. The default hashes the `Debug` rendering —
    /// correct whenever `Debug` is injective, but it allocates on search's
    /// hottest path; override with a cheap exact encoding for any game played
    /// by alpha-beta or MCTS.
    fn action_id(&self, action: &Self::Action) -> u64 {
        hash::fnv1a(format!("{action:?}").as_bytes())
    }
}

/// Heuristic static evaluation of a (typically non-terminal) state from
/// `player`'s perspective, on the same scale as [`Game::returns`]. Game
/// knowledge supplied by a game crate; consumed by depth-limited search.
pub trait Eval<G: Game>: Sync {
    fn eval(&self, game: &G, state: &G::State, player: usize) -> f64;
}

/// Rewrites `state`'s hidden information into a concrete sample consistent
/// with everything `observer` can see — a *determinization*. Game knowledge
/// (e.g. "a bidder plausibly holds the face they bid"); consumed by
/// determinized Monte-Carlo algorithms. For perfect-information games use
/// [`Identity`].
pub trait Determinizer<G: Game>: Sync {
    fn determinize(&self, game: &G, state: &mut G::State, observer: usize, rng: &mut Rng);
}

/// The trivial determinizer for perfect-information games: nothing is hidden.
pub struct Identity;
impl<G: Game> Determinizer<G> for Identity {
    fn determinize(&self, _game: &G, _state: &mut G::State, _observer: usize, _rng: &mut Rng) {}
}

/// Optional game-supplied guidance for depth-limited search. Defaults are valid
/// for any game (no quiescence, no ordering) — implement to make search strong.
pub trait SearchSpec<G: Game>: Sync {
    /// Actions worth searching past the depth horizon (e.g. chess captures).
    /// With no noisy actions, horizon nodes return the static evaluation.
    fn is_noisy(&self, _game: &G, _state: &G::State, _action: G::Action) -> bool {
        false
    }
    /// Higher = searched first. Good ordering makes alpha-beta prune far more.
    fn order_hint(&self, _game: &G, _state: &G::State, _action: G::Action) -> i64 {
        0
    }
}

/// A [`SearchSpec`] with no guidance — correct for any game.
pub struct NoSpec;
impl<G: Game> SearchSpec<G> for NoSpec {}

/// Game knowledge required by policy-value learning (AlphaZero-style
/// self-play, policy gradient): a flat `f32` encoding of states and a dense
/// index for actions in a fixed policy space. Like every capability trait, a
/// game declares it here and the algorithms in `solvers` consume it.
pub trait PolicyValueEncoder<G: Game>: Sync {
    /// Length of [`PolicyValueEncoder::encode_state`]'s output.
    fn input_len(&self) -> usize;

    /// Size of the fixed action-encoding space (the policy head's width).
    fn policy_len(&self) -> usize;

    /// Flat features of `state`. Must encode the side to move.
    fn encode_state(&self, game: &G, state: &G::State) -> Vec<f32>;

    /// Index of `action` in the policy space, in `0..policy_len()`. Must be
    /// injective over the legal actions of any one state.
    fn action_index(&self, game: &G, state: &G::State, action: G::Action) -> usize;
}
