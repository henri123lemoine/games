//! Outcome-sampling MCCFR — one sampled trajectory per traversal.
//!
//! Where [`crate::Mccfr`] (external sampling) expands *every* traverser action
//! at each of the traverser's decision nodes, this samples a single action at
//! every node, so a traversal costs O(trajectory length) regardless of
//! branching. That is the difference between impossible and trivial on deep
//! action ladders (e.g. bidding games where every raise opens another full
//! subtree): external sampling's per-traversal cost grows exponentially with
//! ladder depth, outcome sampling's stays linear.
//!
//! The estimator is the outcome-sampling scheme of Lanctot, Waugh, Zinkevich &
//! Bowling, "Monte Carlo Sampling for Regret Minimization in Extensive Games"
//! (NIPS 2009): at the traverser's information sets actions are drawn from an
//! ε-greedy mix of uniform and the regret-matched strategy (ε = 0.6); chance
//! and opponents are sampled on-policy; sampled counterfactual regrets are
//! importance-weighted by the trajectory's sample probability. The average
//! strategy uses the paper's stochastically-weighted scheme: during player i's
//! traversal, i's own infosets accumulate σ(I) weighted by π_i(h)/q(h), the
//! traverser's reach over the sample reach.
//!
//! Regret matching floors at zero — only positive parts of the cumulative
//! regret shape the strategy — but unlike [`crate::Mccfr`] the *stored*
//! regrets are not clipped (plain accumulation, as in Lanctot 2009). CFR+'s
//! storage floor interacts badly with outcome sampling's high-variance
//! importance-weighted increments: flooring turns zero-mean noise into a
//! systematic upward drift whose size varies per action with 1/q, distorting
//! relative regrets. Empirically that stalls Kuhn at NashConv ≈ 0.5 where
//! plain accumulation reaches < 0.03.

use game_core::{Game, Turn};

use crate::FastMap;

/// Uniform-exploration weight in the traverser's sampling policy.
const EPSILON: f64 = 0.6;

fn regret_match(regret: &[f64]) -> Vec<f64> {
    let sum: f64 = regret.iter().map(|r| r.max(0.0)).sum();
    let n = regret.len();
    if sum > 0.0 {
        regret.iter().map(|r| r.max(0.0) / sum).collect()
    } else {
        vec![1.0 / n as f64; n]
    }
}

/// Outcome-sampling MCCFR+ trainer for an N-player [`Game`].
pub struct OsMccfr<G: Game> {
    game: G,
    regret: FastMap<u64, Vec<f64>>,
    strategy: FastMap<u64, Vec<f64>>,
    rng: u64,
    iterations: u64,
}

impl<G: Game> OsMccfr<G> {
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

    /// Run `iters` iterations; each samples one trajectory per player as
    /// traverser (alternating updates, mirroring [`crate::Mccfr::run`]).
    pub fn run(&mut self, iters: u64) {
        let n = self.game.num_players();
        for _ in 0..iters {
            for t in 0..n {
                let state = self.game.initial_state();
                self.traverse(&state, t, 1.0, 1.0, 1.0);
            }
        }
        self.iterations += iters;
    }

    /// One sampled trajectory for `traverser`. `my_reach` is the traverser's
    /// σ-reach π_i(h), `opp_reach` the opponents'+chance σ-reach π₋ᵢ(h), and
    /// `sample_reach` the trajectory sample probability q(h). Returns
    /// `(u, tail)`: the terminal utility to the traverser divided by the full
    /// trajectory sample probability, u_i(z)/q(z), and the traverser's own
    /// σ-probability of the trajectory suffix from this node, π_i(h → z).
    fn traverse(
        &mut self,
        state: &G::State,
        traverser: usize,
        my_reach: f64,
        opp_reach: f64,
        sample_reach: f64,
    ) -> (f64, f64) {
        if self.game.is_terminal(state) {
            return (self.game.returns(state, traverser) / sample_reach, 1.0);
        }
        match self.game.turn(state) {
            Turn::Chance => {
                let outs = self.game.chance_outcomes(state);
                let probs: Vec<f64> = outs.iter().map(|(_, p)| *p).collect();
                let i = self.sample(&probs);
                let mut child = state.clone();
                self.game.apply(&mut child, outs[i].0);
                self.traverse(
                    &child,
                    traverser,
                    my_reach,
                    opp_reach * probs[i],
                    sample_reach * probs[i],
                )
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
                    {
                        let w = my_reach / sample_reach;
                        let s = self.strategy.entry(key).or_insert_with(|| vec![0.0; n]);
                        for i in 0..n {
                            s[i] += w * sigma[i];
                        }
                    }
                    let explore: Vec<f64> = sigma
                        .iter()
                        .map(|&pr| EPSILON / n as f64 + (1.0 - EPSILON) * pr)
                        .collect();
                    let i = self.sample(&explore);
                    let mut child = state.clone();
                    self.game.apply(&mut child, actions[i]);
                    let (u, tail) = self.traverse(
                        &child,
                        traverser,
                        my_reach * sigma[i],
                        opp_reach,
                        sample_reach * explore[i],
                    );
                    // Sampled counterfactual values: ṽ(I,a) = u·π₋ᵢ(h)·π_i(ha→z)
                    // for the sampled action, 0 for the rest, and
                    // ṽ(I) = σ(a|I)·ṽ(I,a); regret increments are ṽ(I,·) − ṽ(I).
                    let w = u * opp_reach * tail;
                    let r = self.regret.get_mut(&key).unwrap();
                    for (j, rj) in r.iter_mut().enumerate() {
                        let inc = if j == i {
                            w * (1.0 - sigma[i])
                        } else {
                            -w * sigma[i]
                        };
                        *rj += inc;
                    }
                    (u, tail * sigma[i])
                } else {
                    let i = self.sample(&sigma);
                    let mut child = state.clone();
                    self.game.apply(&mut child, actions[i]);
                    self.traverse(
                        &child,
                        traverser,
                        my_reach,
                        opp_reach * sigma[i],
                        sample_reach * sigma[i],
                    )
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
