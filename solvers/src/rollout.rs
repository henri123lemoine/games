//! Determinized Monte-Carlo rollout: at each decision, sample concrete worlds
//! consistent with the player's information (via the game's [`Determinizer`]),
//! play every candidate action to the end with a base policy on all seats, and
//! pick the action with the highest mean return — so margins and draws count,
//! not just outright wins.
//!
//! Two measured design points (see games/liars-dice/examples/ab):
//! * **Common random numbers** — rollout `j` re-seeds from `(seed, j)` only, so
//!   every candidate faces the *same* world `j` and the choice is a paired
//!   comparison, collapsing decision noise.
//! * `(candidate × chunk)` rayon fan-out fills all cores even at 2-4 candidates.

use std::marker::PhantomData;

use game_core::{Agent, Determinizer, Game, Rng, playout_from};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Monte-Carlo lookahead over any [`Game`] with a [`Determinizer`]. The base
/// agent plays every seat during playouts and answers directly at nodes wider
/// than `cand_cap` (a safety valve for huge action menus, where per-candidate
/// rollouts would be spread too thin to rank reliably).
pub struct Rollout<G: Game, A: Agent<G> + Sync, D: Determinizer<G>> {
    pub rollouts: u32,
    pub cand_cap: usize,
    pub base: A,
    pub det: D,
    _g: PhantomData<fn(G)>,
}

impl<G: Game, A: Agent<G> + Sync, D: Determinizer<G>> Rollout<G, A, D> {
    pub fn new(rollouts: u32, base: A, det: D) -> Self {
        Self {
            rollouts,
            cand_cap: 8,
            base,
            det,
            _g: PhantomData,
        }
    }
}

impl<G: Game, A: Agent<G> + Sync, D: Determinizer<G>> Agent<G> for Rollout<G, A, D> {
    fn act(&self, game: &G, state: &G::State, player: usize, rng: &mut Rng) -> usize {
        let actions = game.legal_actions(state);
        if actions.len() == 1 {
            return 0;
        }
        if actions.len() > self.cand_cap {
            return self.base.act(game, state, player, rng);
        }
        let seed0 = rng.next_u64();
        let (rollouts, n) = (self.rollouts, game.num_players());
        let (base, det) = (&self.base, &self.det);
        let n_chunks = 8u32;
        let tasks: Vec<(usize, u32)> = (0..actions.len())
            .flat_map(|k| (0..n_chunks).map(move |c| (k, c)))
            .collect();
        let mut totals = vec![0.0f64; actions.len()];
        let run = |&(k, c): &(usize, u32)| {
            let seats: Vec<&dyn Agent<G>> = (0..n).map(|_| base as &dyn Agent<G>).collect();
            let mut sum = 0.0;
            for j in (rollouts * c / n_chunks)..(rollouts * (c + 1) / n_chunks) {
                let mut rng = Rng::new(seed0 ^ (j as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let mut sim = state.clone();
                det.determinize(game, &mut sim, player, &mut rng);
                game.apply(&mut sim, actions[k]);
                let terminal = playout_from(game, sim, &seats, &mut rng);
                sum += game.returns(&terminal, player);
            }
            (k, sum)
        };
        #[cfg(feature = "parallel")]
        let results = tasks.par_iter().map(run).collect::<Vec<_>>();
        #[cfg(not(feature = "parallel"))]
        let results = tasks.iter().map(run).collect::<Vec<_>>();
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
