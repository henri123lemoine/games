//! Recursive self-play data generation — the ReBeL `RlRunner::step` loop.
//!
//! Starting from the game root with uniform-prior beliefs, repeatedly: build a
//! depth-limited subgame at the current public belief state, run a uniformly
//! random number of CFR iterations, sample one trajectory down to a leaf (one
//! player exploring uniformly with probability `explore_eps`, beliefs
//! Bayesian-updated along the way), finish the remaining CFR iterations, emit one
//! value example per seat (the subgame-root query mapped to that seat's
//! `root_values_mean`), and recurse into the sampled leaf's public belief state.

use game_core::Rng;

use solvers::rebel_mlp::Sample;

use crate::rebel::cfr::{CfrParams, Solver};
use crate::rebel::game::RebelGame;
use crate::rebel::leaf::RootedGame;
use crate::rebel::pbs::{Belief, PublicState, bayes_update};
use crate::rebel::tree::Tree;
use crate::rebel::value_net::{NetLeaf, PbsNet};

/// Parameters of one self-play episode.
#[derive(Clone, Copy, Debug)]
pub struct SelfPlayParams {
    pub cfr: CfrParams,
    /// Probability the single exploring seat takes a uniform-random action when
    /// sampling the trajectory (off-policy exploration for data coverage).
    pub explore_eps: f64,
}

impl Default for SelfPlayParams {
    fn default() -> Self {
        Self {
            cfr: CfrParams::default(),
            explore_eps: 0.25,
        }
    }
}

/// Hard cap on subgame transitions per episode, guarding against a pathological
/// non-terminating descent. Single-round configs terminate well within this.
const MAX_TRANSITIONS: usize = 4096;

/// Generate the value-net training samples from one recursive self-play episode.
pub fn generate_episode<G: RebelGame>(
    game: &G,
    params: SelfPlayParams,
    net: &PbsNet,
    rng: &mut Rng,
) -> Vec<Sample> {
    let players = game.players();
    let mut state = game.root();
    let mut belief = Belief::uniform_prior(&state);
    let mut samples = Vec::new();

    let mut transitions = 0;
    while !game.is_terminal(&state) && transitions < MAX_TRANSITIONS {
        transitions += 1;

        let rooted = RootedGame::new(game, state.clone());
        let leaf = NetLeaf::new(net, game);
        let mut solver = Solver::new(&rooted, params.cfr, &leaf, belief.clone());

        let num_iters = params.cfr.num_iters;
        let act_it = rng.below(num_iters + 1);
        for it in 0..act_it {
            solver.step(it % players);
        }

        let (next_state, next_belief) = sample_to_leaf(
            solver.tree(),
            solver.last_strategy(),
            &belief,
            players,
            params.explore_eps,
            rng,
        );

        for it in act_it..num_iters {
            solver.step(it % players);
        }

        for traverser in 0..players {
            samples.push(net.to_sample(
                &state,
                traverser,
                &belief,
                solver.root_values_mean(traverser),
            ));
        }

        state = next_state;
        belief = next_belief;
    }

    samples
}

/// Descend a sampled trajectory from the subgame root to a leaf under the current
/// strategy, returning the leaf's public state and the Bayesian-updated belief.
pub(crate) fn sample_to_leaf(
    tree: &Tree,
    strategy: &[Vec<Vec<f64>>],
    root_belief: &Belief,
    players: usize,
    explore_eps: f64,
    rng: &mut Rng,
) -> (PublicState, Belief) {
    let mut belief = root_belief.clone();
    let explorer = rng.below(players);
    let mut node = 0usize;
    loop {
        let n = &tree.nodes[node];
        if n.is_leaf {
            return (n.public.clone(), belief);
        }
        let acting = n.acting;
        let strat = &strategy[node];
        let action = if acting == explorer && rng.unit() < explore_eps {
            rng.below(n.legal.len())
        } else {
            let hand = rng.pick(&belief.per_seat[acting]);
            rng.pick(&strat[hand])
        };
        let per_hand_prob: Vec<f64> = strat.iter().map(|row| row[action]).collect();
        bayes_update(&mut belief.per_seat[acting], &per_hand_prob);
        node = n.children[action];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rebel::hands::{self, global_index, hand_count};
    use crate::rebel::standard::StandardLiarsDice;
    use crate::rebel::value_net::{INPUT_DIM, OUTPUT_DIM};

    #[test]
    fn episode_emits_well_formed_samples() {
        let game = StandardLiarsDice::new(1, 4);
        let net = PbsNet::new(64, 2, 3);
        let params = SelfPlayParams {
            cfr: CfrParams {
                num_iters: 64,
                ..CfrParams::default()
            },
            explore_eps: 0.25,
        };
        let mut rng = Rng::new(11);
        let samples = generate_episode(&game, params, &net, &mut rng);

        assert!(!samples.is_empty());
        assert_eq!(samples.len() % game.players(), 0);

        let trav_hands = hand_count(1, 4);
        for s in &samples {
            assert_eq!(s.input.len(), INPUT_DIM);
            assert_eq!(s.target.len(), OUTPUT_DIM);
            assert_eq!(s.mask.len(), OUTPUT_DIM);
            assert!(s.input.iter().all(|v| v.is_finite()));
            assert!(s.target.iter().all(|v| v.is_finite()));
            let mask_sum: f32 = s.mask.iter().sum();
            assert!((mask_sum - trav_hands as f32).abs() < 1e-6);
            for hand in hands::enumerate(1, 4) {
                let g = global_index(&hand, 1);
                assert_eq!(s.mask[g], 1.0);
                assert!(s.target[g].abs() <= 1.0 + 1e-6);
            }
        }
    }
}
