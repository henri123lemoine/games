//! External-sampling MCCFR(+) — the scalable counterfactual-regret variant.
//!
//! Where [`crate::Solver`] walks the whole tree every iteration (exact, but only
//! for tiny games), this samples: on each traversal one player is the *traverser*
//! whose actions are all expanded to update regret, while chance and every other
//! player are sampled to a single action. That makes the per-iteration cost
//! independent of the opponents' branching, so it scales to large games and to
//! any number of players (each player minimizes their own sampled regret).
//!
//! Regrets use the CFR+ floor (clipped at zero), which converges fast in
//! practice. The stored average strategy is what you play.

use crate::{FastMap, Game, Turn};

fn regret_match(regret: &[f64]) -> Vec<f64> {
    let sum: f64 = regret.iter().map(|r| r.max(0.0)).sum();
    let n = regret.len();
    if sum > 0.0 {
        regret.iter().map(|r| r.max(0.0) / sum).collect()
    } else {
        vec![1.0 / n as f64; n]
    }
}

/// External-sampling MCCFR+ trainer for an N-player [`Game`].
pub struct Mccfr<G: Game> {
    game: G,
    regret: FastMap<u64, Vec<f64>>,
    strategy: FastMap<u64, Vec<f64>>,
    rng: u64,
    iterations: u64,
}

impl<G: Game> Mccfr<G> {
    pub fn new(game: G, seed: u64) -> Self {
        Self {
            game,
            regret: FastMap::default(),
            strategy: FastMap::default(),
            rng: seed | 1,
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

    fn rand(&mut self) -> f64 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        (x >> 11) as f64 / (1u64 << 53) as f64
    }

    fn sample(&mut self, dist: &[f64]) -> usize {
        let r = self.rand();
        let mut acc = 0.0;
        for (i, p) in dist.iter().enumerate() {
            acc += p;
            if r < acc {
                return i;
            }
        }
        dist.len() - 1
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
                let probs: Vec<f64> = outs.iter().map(|(_, p)| *p).collect();
                let i = self.sample(&probs);
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
                    let i = self.sample(&sigma);
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
        match self.strategy.get(&key) {
            Some(s) => {
                let sum: f64 = s.iter().sum();
                if sum > 0.0 {
                    s.iter().map(|x| x / sum).collect()
                } else {
                    vec![1.0 / n as f64; n]
                }
            }
            None => vec![1.0 / n as f64; n],
        }
    }

    /// Sample an action index from the average strategy. Usable as an
    /// [`crate::Agent`] for the arena.
    pub fn sample_action(&self, state: &G::State, player: usize, r: f64) -> usize {
        let policy = self.policy(state, player);
        let mut acc = 0.0;
        for (i, p) in policy.iter().enumerate() {
            acc += p;
            if r < acc {
                return i;
            }
        }
        policy.len() - 1
    }
}
