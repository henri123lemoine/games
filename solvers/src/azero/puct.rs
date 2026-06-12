//! Synchronous PUCT guided by an [`Mlp`]: the batched [`Search`] driven with
//! one leaf at a time (the sequential special case), priors from the policy
//! head over the legal actions, leaf values from the value head.
//!
//! Two-player zero-sum only, like the search it drives.

/// Re-exported from `game_core`, where the capability trait lives.
pub use game_core::PolicyValueEncoder;
use game_core::{Agent, Game, Rng};

use super::mlp::{InferCache, Mlp};
use super::search::{EvalResult, Gather, PuctConfig, Search, argmax};

pub struct Puct<'a, G: Game, E: PolicyValueEncoder<G>> {
    pub game: &'a G,
    pub enc: &'a E,
    pub net: &'a Mlp,
    pub sims: usize,
    pub c_puct: f32,
    /// First-play urgency: unvisited edges score `node value − fpu`.
    pub fpu: f32,
    pub dirichlet_alpha: f32,
    /// Weight of Dirichlet noise mixed into the root prior; 0 disables it.
    pub root_noise: f32,
    /// Sparse-input fast path for `net`, snapshotted at construction (sound:
    /// the shared borrow keeps the net frozen for this `Puct`'s lifetime).
    cache: InferCache,
}

impl<'a, G: Game, E: PolicyValueEncoder<G>> Puct<'a, G, E> {
    pub fn new(game: &'a G, enc: &'a E, net: &'a Mlp, sims: usize) -> Self {
        Puct {
            game,
            enc,
            net,
            sims,
            c_puct: 1.5,
            fpu: 0.0,
            dirichlet_alpha: 0.3,
            root_noise: 0.0,
            cache: net.infer_cache(),
        }
    }

    /// Runs `sims` simulations from `root` (a non-terminal decision node) and
    /// returns the root visit counts, aligned with `legal_actions(root)`.
    pub fn search(&self, root: &G::State, rng: &mut Rng) -> Vec<u32> {
        debug_assert!(!self.game.is_terminal(root));
        let cfg = PuctConfig {
            sims: self.sims as u32,
            c_puct: self.c_puct,
            fpu: self.fpu,
            dirichlet_alpha: f64::from(self.dirichlet_alpha),
            root_noise: self.root_noise,
            max_leaves: 1,
            cycle_draws: false,
        };
        let mut search = Search::new(None);
        let mut results = Vec::new();
        loop {
            match search.advance(
                self.game,
                self.enc,
                root,
                &cfg,
                rng,
                std::mem::take(&mut results),
                &|_| false,
            ) {
                Gather::Requests(reqs) => {
                    results = reqs
                        .iter()
                        .map(|r| {
                            let support: Vec<usize> =
                                r.support.iter().map(|&s| usize::from(s)).collect();
                            let (priors, value) =
                                self.net
                                    .policy_value_cached(&self.cache, &r.features, &support);
                            EvalResult { priors, value }
                        })
                        .collect();
                }
                Gather::Done => return search.root_visits().to_vec(),
            }
        }
    }
}

/// [`Puct`] as an arena [`Agent`]: deterministic argmax over visit counts,
/// searching with the arena-supplied randomness.
pub struct PuctAgent<'a, G: Game, E: PolicyValueEncoder<G>>(pub Puct<'a, G, E>);

impl<G: Game, E: PolicyValueEncoder<G>> Agent<G> for PuctAgent<'_, G, E> {
    fn act(&self, _game: &G, state: &G::State, _player: usize, rng: &mut Rng) -> usize {
        argmax(&self.0.search(state, rng))
    }
}
