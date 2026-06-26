//! Leaf-value abstraction for the vector CFR solver.
//!
//! A [`LeafValue`] returns the traverser's per-hand value at a leaf given that
//! leaf's per-seat normalized beliefs, with opponents assumed to play their
//! normalized belief; the solver multiplies by opponent reach mass. [`TerminalLeaf`]
//! reads exact terminal payoffs; [`PerfectOracleLeaf`] solves the full-depth
//! subgame rooted at a depth-limited leaf to recover its exact value.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::rebel::cfr::{CfrParams, CfrVariant, Solver};
use crate::rebel::game::{Bid, RebelGame};
use crate::rebel::pbs::{Belief, PublicState};

/// Per-hand value for `traverser` at a leaf, given that leaf's per-seat
/// normalized beliefs.
pub trait LeafValue {
    fn values(
        &self,
        public: &PublicState,
        traverser: usize,
        normalized_belief: &Belief,
    ) -> Vec<f64>;

    /// Per-hand values for `traverser` at each `(public, belief)` query, aligned
    /// to the input order. The default loops over [`LeafValue::values`]; a
    /// net-backed leaf overrides this to amortize a single batched forward pass.
    fn values_batch(
        &self,
        publics: &[PublicState],
        traverser: usize,
        beliefs: &[Belief],
    ) -> Vec<Vec<f64>> {
        publics
            .iter()
            .zip(beliefs)
            .map(|(public, belief)| self.values(public, traverser, belief))
            .collect()
    }
}

/// Values a terminal leaf by the game's exact terminal counterfactual payoffs.
pub struct TerminalLeaf<'a, G: RebelGame> {
    game: &'a G,
}

impl<'a, G: RebelGame> TerminalLeaf<'a, G> {
    pub fn new(game: &'a G) -> Self {
        Self { game }
    }
}

impl<G: RebelGame> LeafValue for TerminalLeaf<'_, G> {
    fn values(&self, public: &PublicState, traverser: usize, belief: &Belief) -> Vec<f64> {
        assert!(
            self.game.is_terminal(public),
            "TerminalLeaf queried on a non-terminal public state"
        );
        self.game.terminal_cfv(public, traverser, belief)
    }
}

/// A game re-rooted at an arbitrary public state, delegating all mechanics to an
/// inner game. Lets a leaf — or the recursive self-play loop — spin up the
/// subgame that begins there.
pub struct RootedGame<'a, G: RebelGame> {
    inner: &'a G,
    root: PublicState,
}

impl<'a, G: RebelGame> RootedGame<'a, G> {
    pub fn new(inner: &'a G, root: PublicState) -> Self {
        Self { inner, root }
    }
}

impl<G: RebelGame> RebelGame for RootedGame<'_, G> {
    fn players(&self) -> usize {
        self.inner.players()
    }
    fn faces(&self) -> u8 {
        self.inner.faces()
    }
    fn dice_left(&self) -> [u8; 8] {
        self.root.dice_left
    }
    fn root(&self) -> PublicState {
        self.root.clone()
    }
    fn acting(&self, p: &PublicState) -> usize {
        self.inner.acting(p)
    }
    fn is_terminal(&self, p: &PublicState) -> bool {
        self.inner.is_terminal(p)
    }
    fn legal_actions(&self, p: &PublicState) -> Vec<Bid> {
        self.inner.legal_actions(p)
    }
    fn apply(&self, p: &PublicState, a: Bid) -> PublicState {
        self.inner.apply(p, a)
    }
    fn terminal_cfv(&self, p: &PublicState, traverser: usize, belief: &Belief) -> Vec<f64> {
        self.inner.terminal_cfv(p, traverser, belief)
    }
}

type OracleKey = (PublicState, usize, Vec<u64>);

/// Values a depth-limited leaf by solving the full-depth subgame rooted there to
/// convergence with the CFR solver, returning the traverser's average-strategy
/// root values. Terminal leaves fall back to exact terminal payoffs.
pub struct PerfectOracleLeaf<'a, G: RebelGame> {
    game: &'a G,
    iters: usize,
    cache: RefCell<HashMap<OracleKey, Vec<f64>>>,
}

impl<'a, G: RebelGame> PerfectOracleLeaf<'a, G> {
    pub fn new(game: &'a G, iters: usize) -> Self {
        Self {
            game,
            iters,
            cache: RefCell::new(HashMap::new()),
        }
    }
}

fn belief_key(public: &PublicState, traverser: usize, belief: &Belief) -> OracleKey {
    let bits = belief
        .per_seat
        .iter()
        .flat_map(|seat| seat.iter().map(|x| x.to_bits()))
        .collect();
    (public.clone(), traverser, bits)
}

impl<G: RebelGame> LeafValue for PerfectOracleLeaf<'_, G> {
    fn values(&self, public: &PublicState, traverser: usize, belief: &Belief) -> Vec<f64> {
        if self.game.is_terminal(public) {
            return self.game.terminal_cfv(public, traverser, belief);
        }
        let key = belief_key(public, traverser, belief);
        if let Some(cached) = self.cache.borrow().get(&key) {
            return cached.clone();
        }

        let subgame = RootedGame::new(self.game, public.clone());
        let params = CfrParams {
            num_iters: self.iters,
            max_depth: u32::MAX,
            variant: CfrVariant::LinearCfr,
            alternating: true,
            cfr_avg: false,
            leaf_refresh_every: 1,
        };
        let terminal = TerminalLeaf::new(&subgame);
        let mut solver = Solver::new(&subgame, params, &terminal, belief.clone());
        solver.multistep();
        let values = solver.root_values_mean(traverser).to_vec();

        self.cache.borrow_mut().insert(key, values.clone());
        values
    }
}
