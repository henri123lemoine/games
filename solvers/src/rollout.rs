//! Determinized Monte-Carlo rollout: at each decision, sample concrete worlds
//! consistent with the player's information (via the game's [`Determinizer`]),
//! play every candidate action to the end with a base policy on all seats, and
//! pick the action with the highest estimated win rate.
//!
//! Two measured design points (see games/liars-dice/examples/ab):
//! * **Common random numbers** — rollout `j` re-seeds from `(seed, j)` only, so
//!   every candidate faces the *same* world `j` and the choice is a paired
//!   comparison, collapsing decision noise.
//! * `(candidate × chunk)` rayon fan-out fills all cores even at 2-4 candidates.

use std::cell::Cell;
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
    rng: Cell<u64>,
    _g: PhantomData<fn(G)>,
}

impl<G: Game, A: Agent<G> + Sync, D: Determinizer<G>> Rollout<G, A, D> {
    pub fn new(rollouts: u32, base: A, det: D, seed: u64) -> Self {
        Self {
            rollouts,
            cand_cap: 8,
            base,
            det,
            rng: Cell::new(seed | 1),
            _g: PhantomData,
        }
    }
}

impl<G: Game, A: Agent<G> + Sync, D: Determinizer<G>> Agent<G> for Rollout<G, A, D> {
    fn act(&self, game: &G, state: &G::State, player: usize, r: f64) -> usize {
        let actions = game.legal_actions(state);
        if actions.len() == 1 {
            return 0;
        }
        if actions.len() > self.cand_cap {
            return self.base.act(game, state, player, r);
        }
        let seed0 = self.rng.get() ^ r.to_bits();
        self.rng.set(
            self.rng
                .get()
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1),
        );
        let (rollouts, n) = (self.rollouts, game.num_players());
        // Borrow the Sync fields individually so the parallel closures never
        // capture `&self` (the rng Cell makes Self !Sync).
        let (base, det) = (&self.base, &self.det);
        let n_chunks = 8u32;
        let tasks: Vec<(usize, u32)> = (0..actions.len())
            .flat_map(|k| (0..n_chunks).map(move |c| (k, c)))
            .collect();
        let mut wins = vec![0u32; actions.len()];
        let run = |&(k, c): &(usize, u32)| {
            let seats: Vec<&dyn Agent<G>> = (0..n).map(|_| base as &dyn Agent<G>).collect();
            let mut w = 0u32;
            for j in (rollouts * c / n_chunks)..(rollouts * (c + 1) / n_chunks) {
                let mut rng = Rng::new(seed0 ^ (j as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let mut sim = state.clone();
                det.determinize(game, &mut sim, player, &mut rng);
                game.apply(&mut sim, actions[k]);
                if playout_from(game, sim, &seats, &mut rng) == player {
                    w += 1;
                }
            }
            (k, w)
        };
        #[cfg(feature = "parallel")]
        let results = tasks.par_iter().map(run).collect::<Vec<_>>();
        #[cfg(not(feature = "parallel"))]
        let results = tasks.iter().map(run).collect::<Vec<_>>();
        for (k, w) in results {
            wins[k] += w;
        }
        let mut best = 0;
        for k in 1..wins.len() {
            if wins[k] > wins[best] {
                best = k;
            }
        }
        best
    }
}
