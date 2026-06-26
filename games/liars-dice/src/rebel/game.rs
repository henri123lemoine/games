//! The integration interface the vector-CFR solver consumes: a Liar's-Dice-like
//! game expressed over public states with per-hand terminal payoffs.

use crate::rebel::pbs::{Belief, PublicState};

/// A bidding action. The reference standard game uses only higher-bid-or-liar
/// (`Raise`/`Call`); the deploy adapter for the real non-standard rules also uses
/// [`Bid::CallExact`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Bid {
    /// Raise to `(qty, face)` with a 0-based `face`.
    Raise { qty: u8, face: u8 },
    /// Call the standing bid a lie (the real rules' `CallLiar`).
    Call,
    /// Call the standing bid exactly right (the real rules' `CallExact`): if the
    /// true count equals the bid no die is lost, otherwise the caller loses one.
    CallExact,
}

/// A game defined over public states, abstracted so the public-tree vector CFR
/// solver can drive it without ever seeing a private hand.
///
/// Contract: [`legal_actions`](RebelGame::legal_actions) MUST return a stable
/// order for a given public state — the solver aligns per-hand regret and
/// strategy vectors and the tree aligns child indices to that order, so the
/// order may not vary between calls on equal states.
pub trait RebelGame {
    fn players(&self) -> usize;
    fn faces(&self) -> u8;

    /// Per-seat dice remaining at the subgame root.
    fn dice_left(&self) -> [u8; 8];

    /// The public state at the subgame root.
    fn root(&self) -> PublicState;

    /// The seat to act at `p` (valid only when `p` is not terminal).
    fn acting(&self, p: &PublicState) -> usize;

    fn is_terminal(&self, p: &PublicState) -> bool;

    /// The legal actions at `p` in a stable order (empty at a terminal state).
    fn legal_actions(&self, p: &PublicState) -> Vec<Bid>;

    fn apply(&self, p: &PublicState, a: Bid) -> PublicState;

    /// Per-hand counterfactual values at a terminal public state `p` for the
    /// `traverser` seat, given the (normalized) beliefs of the other seats.
    ///
    /// Returns a vector over the traverser's hands indexed within its
    /// `dice_left` support. Opponents are assumed to play their *normalized*
    /// belief (no reach-mass weighting): the solver applies opponent-reach
    /// scaling uniformly to every leaf, so this returns the per-hand expected
    /// value under the normalized opponent belief. Payoffs are constant-sum
    /// (zero-sum for two players).
    fn terminal_cfv(&self, p: &PublicState, traverser: usize, belief: &Belief) -> Vec<f64>;
}
