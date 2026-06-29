//! Net-guided truncated rollout — AlphaZero-style search for Liar's Dice.
//!
//! This mirrors [`solvers::Rollout`] move for move (determinize the hidden dice,
//! try every candidate action, common random numbers so each candidate faces the
//! *same* sampled world) but replaces the to-the-end playout with a truncated one:
//! it plays `plies` player-moves with the net's own policy on every seat and then
//! values the leaf with the net's VALUE head instead of `Game::returns`.
//!
//! That truncation is the lever that lets a strong net's search dominate the
//! heuristic full-rollout baseline: short playouts that still see far through the
//! learned value, guided by the learned policy rather than a hand-tuned one. The
//! plain net-guided rollout (`Rollout::new(rollouts, NetAgent::new(net),
//! BidConditioned::default())`) is the simpler sibling — same structure, full
//! playouts, no value head — and needs no new type.
//!
//! Value-head caveat: the Deep CFR / distillation value head is trained on
//! round-*opening* public states (the fitted-value-iteration target — see
//! [`crate::NetValue`]). A leaf that lands at a round boundary (chance pending,
//! hands not yet rolled) is therefore on-distribution and read directly; a leaf
//! mid-bid is off-distribution, an inherent approximation of ply-granular
//! truncation against a round-granular value head. With an undertrained net the
//! value head is weak regardless, so this variant is only as good as the net.

use game_core::{Agent, Determinizer, Game, Rng, Turn};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use solvers::azero::{InferCache, Mlp};

use crate::features::{encode, legal_actions_and_support};
use crate::{BidConditioned, LdState, LiarsDice};

/// Net-guided truncated-rollout agent: `rollouts` determinized worlds per
/// candidate action, each played `plies` net-policy moves then closed by the
/// net's value head. Picks the action with the greatest mean leaf value from the
/// acting player's perspective.
pub struct NetTruncRollout {
    pub rollouts: u32,
    /// Above this many legal actions (the wide opening grid) per-candidate
    /// rollouts would be spread too thin to rank, so fall back to the net policy
    /// — exactly as [`solvers::Rollout`] falls back to its base agent.
    pub cand_cap: usize,
    /// Player-moves played with the net policy before the value head closes the
    /// leaf. Larger values reach further (and land on round boundaries, where the
    /// value head is on-distribution) at more forward-pass cost.
    pub plies: u32,
    net: Mlp,
    cache: InferCache,
    det: BidConditioned,
}

impl NetTruncRollout {
    pub fn new(net: Mlp, rollouts: u32, plies: u32) -> Self {
        let cache = net.infer_cache();
        Self {
            rollouts,
            cand_cap: 8,
            plies,
            net,
            cache,
            det: BidConditioned::default(),
        }
    }

    pub fn from_bytes(data: &[u8], rollouts: u32, plies: u32) -> std::io::Result<Self> {
        Ok(Self::new(Mlp::from_bytes(data)?, rollouts, plies))
    }

    /// Sample a legal action index from the net policy at a player node.
    fn net_action(&self, game: &LiarsDice, s: &LdState, player: usize, rng: &mut Rng) -> usize {
        let (acts, sup) = legal_actions_and_support(game, s);
        if acts.is_empty() {
            return 0;
        }
        let x = encode(game, s, player);
        let (probs, _) = self.net.policy_value_cached(&self.cache, &x, &sup);
        let weights: Vec<f64> = probs.iter().map(|&p| f64::from(p)).collect();
        if weights.iter().sum::<f64>() <= 0.0 {
            return 0;
        }
        rng.pick(&weights)
    }

    /// The leaf's value to `player`: the real return if terminal, otherwise the
    /// net's value head read from `player`'s perspective. At a round boundary
    /// (chance pending) the unrolled opening state is read as-is — the value
    /// head's training distribution — rather than rolling a fresh hand.
    fn leaf_value(&self, game: &LiarsDice, s: &LdState, player: usize) -> f64 {
        if game.is_terminal(s) {
            return game.returns(s, player);
        }
        let x = encode(game, s, player);
        let (_, v) = self.net.policy_value_cached(&self.cache, &x, &[]);
        f64::from(v)
    }

    /// Play up to `plies` net-policy moves on every seat from a determinized
    /// world, resolving chance along the way, then value the leaf.
    fn truncated_value(
        &self,
        game: &LiarsDice,
        mut s: LdState,
        player: usize,
        rng: &mut Rng,
    ) -> f64 {
        let mut moves = 0;
        while moves < self.plies && !game.is_terminal(&s) {
            match game.turn(&s) {
                Turn::Chance => {
                    let a = game.sample_chance_action(&s, rng);
                    game.apply(&mut s, a);
                }
                Turn::Player(p) => {
                    let acts = game.legal_actions(&s);
                    let i = self.net_action(game, &s, p, rng);
                    game.apply(&mut s, acts[i]);
                    moves += 1;
                }
            }
        }
        self.leaf_value(game, &s, player)
    }
}

impl Agent<LiarsDice> for NetTruncRollout {
    fn act(&self, game: &LiarsDice, state: &LdState, player: usize, rng: &mut Rng) -> usize {
        let actions = game.legal_actions(state);
        if actions.len() == 1 {
            return 0;
        }
        if actions.len() > self.cand_cap {
            return self.net_action(game, state, player, rng);
        }
        let seed0 = rng.next_u64();
        let rollouts = self.rollouts;
        let n_chunks = 8u32;
        let tasks: Vec<(usize, u32)> = (0..actions.len())
            .flat_map(|k| (0..n_chunks).map(move |c| (k, c)))
            .collect();
        let run = |&(k, c): &(usize, u32)| {
            let mut sum = 0.0;
            for j in (rollouts * c / n_chunks)..(rollouts * (c + 1) / n_chunks) {
                let mut rng = Rng::new(seed0 ^ (j as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let mut sim = state.clone();
                self.det.determinize(game, &mut sim, player, &mut rng);
                game.apply(&mut sim, actions[k]);
                sum += self.truncated_value(game, sim, player, &mut rng);
            }
            (k, sum)
        };
        #[cfg(feature = "parallel")]
        let results = tasks.par_iter().map(run).collect::<Vec<_>>();
        #[cfg(not(feature = "parallel"))]
        let results = tasks.iter().map(run).collect::<Vec<_>>();
        let mut totals = vec![0.0f64; actions.len()];
        for (k, sum) in results {
            totals[k] += sum;
        }
        let mut best = 0;
        for k in 1..totals.len() {
            if totals[k] > totals[best] {
                best = k;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{feature_len, policy_len};

    #[test]
    fn trunc_rollout_plays_only_legal_actions() {
        let net = Mlp::new(feature_len(), 32, policy_len(), 0xC0FFEE);
        let agent = NetTruncRollout::new(net, 8, 3);
        for &(p, d, f) in &[(2u8, 2u8, 6u8), (3, 3, 6), (5, 5, 6)] {
            let game = LiarsDice::new(p, d, f);
            let mut rng = Rng::new(0x5151 + u64::from(p));
            for _ in 0..6 {
                let mut s = game.initial_state();
                let mut steps = 0;
                while !game.is_terminal(&s) {
                    steps += 1;
                    assert!(steps < 100_000);
                    match game.turn(&s) {
                        Turn::Chance => {
                            let a = game.sample_chance_action(&s, &mut rng);
                            game.apply(&mut s, a);
                        }
                        Turn::Player(pl) => {
                            let acts = game.legal_actions(&s);
                            let i = agent.act(&game, &s, pl, &mut rng);
                            assert!(i < acts.len(), "trunc rollout must pick a legal action");
                            game.apply(&mut s, acts[i]);
                        }
                    }
                }
            }
        }
    }

    /// `NetTruncRollout` must be `Sync` so it can head a parallel field eval and
    /// be shared as the base across rayon workers.
    #[test]
    fn trunc_rollout_is_sync() {
        fn requires_sync<T: Sync>(_: &T) {}
        let net = Mlp::new(feature_len(), 8, policy_len(), 1);
        requires_sync(&NetTruncRollout::new(net, 1, 1));
    }
}
