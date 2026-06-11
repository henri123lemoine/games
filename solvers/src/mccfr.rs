//! External-sampling MCCFR(+) — the scalable counterfactual-regret variant.
//!
//! Where [`crate::Cfr`] walks the whole tree every iteration (exact, but only
//! for tiny games), this samples: on each traversal one player is the *traverser*
//! whose actions are all expanded to update regret, while chance and every other
//! player are sampled to a single action. That makes the per-iteration cost
//! independent of the opponents' branching, so it scales to large games and to
//! any number of players (each player minimizes their own sampled regret).
//!
//! Regrets use the CFR+ floor (clipped at zero), which converges fast in
//! practice. The stored average strategy is what you play.

use game_core::{Game, Rng, Turn};

use crate::FastMap;
use crate::tabular::{normalized_or_uniform, regret_match};

/// External-sampling MCCFR+ trainer for an N-player [`Game`].
pub struct Mccfr<G: Game> {
    game: G,
    regret: FastMap<u64, Vec<f64>>,
    strategy: FastMap<u64, Vec<f64>>,
    rng: Rng,
    iterations: u64,
}

impl<G: Game> Mccfr<G> {
    pub fn new(game: G, seed: u64) -> Self {
        Self {
            game,
            regret: FastMap::default(),
            strategy: FastMap::default(),
            rng: Rng::new(seed),
            iterations: 0,
        }
    }

    pub fn game(&self) -> &G {
        &self.game
    }
    pub fn num_infosets(&self) -> usize {
        self.strategy.len()
    }
    pub fn iterations(&self) -> u64 {
        self.iterations
    }

    /// Run `iters` iterations; each expands one traversal per player as traverser.
    pub fn run(&mut self, iters: u64) {
        let n = self.game.num_players();
        for _ in 0..iters {
            for t in 0..n {
                let state = self.game.initial_state();
                self.traverse(&state, t);
            }
        }
        self.iterations += iters;
    }

    /// External-sampling traversal returning the value to `traverser`.
    fn traverse(&mut self, state: &G::State, traverser: usize) -> f64 {
        if self.game.is_terminal(state) {
            return self.game.returns(state, traverser);
        }
        match self.game.turn(state) {
            Turn::Chance => {
                let outs = self.game.chance_outcomes(state);
                let i = game_core::rand::sample_outcome(&outs, &mut self.rng);
                let mut child = state.clone();
                self.game.apply(&mut child, outs[i].0);
                self.traverse(&child, traverser)
            }
            Turn::Player(p) => {
                let actions = self.game.legal_actions(state);
                let n = actions.len();
                let key = self.game.infoset_key(state, p);
                let sigma = {
                    let r = self.regret.entry(key).or_insert_with(|| vec![0.0; n]);
                    debug_assert_eq!(
                        r.len(),
                        n,
                        "action count changed for infoset {key:#x} — legal_actions must be \
                         stable per information set"
                    );
                    regret_match(r)
                };
                if p == traverser {
                    let mut child_v = vec![0.0; n];
                    let mut v = 0.0;
                    for (i, &a) in actions.iter().enumerate() {
                        let mut child = state.clone();
                        self.game.apply(&mut child, a);
                        child_v[i] = self.traverse(&child, traverser);
                        v += sigma[i] * child_v[i];
                    }
                    let r = self.regret.get_mut(&key).unwrap();
                    for i in 0..n {
                        r[i] = (r[i] + child_v[i] - v).max(0.0);
                    }
                    v
                } else {
                    {
                        let s = self.strategy.entry(key).or_insert_with(|| vec![0.0; n]);
                        for i in 0..n {
                            s[i] += sigma[i];
                        }
                    }
                    let i = self.rng.pick(&sigma);
                    let mut child = state.clone();
                    self.game.apply(&mut child, actions[i]);
                    self.traverse(&child, traverser)
                }
            }
        }
    }

    /// Average-strategy distribution at `state`'s information set for `player`.
    pub fn policy(&self, state: &G::State, player: usize) -> Vec<f64> {
        let n = self.game.legal_actions(state).len();
        let key = self.game.infoset_key(state, player);
        normalized_or_uniform(self.strategy.get(&key), n)
    }

    /// Sample an action index from the average strategy. Usable as an
    /// [`game_core::Agent`] for the arena.
    pub fn sample_action(&self, state: &G::State, player: usize, rng: &mut Rng) -> usize {
        rng.pick(&self.policy(state, player))
    }
}
