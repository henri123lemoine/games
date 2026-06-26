//! Public belief state: the common-knowledge public state plus per-seat beliefs
//! over each seat's own hand, with Bayesian propagation of a seat's belief along
//! one of its actions.
//!
//! The joint posterior over hands factorizes into a product of per-seat
//! marginals (each seat's bids depend only on its own hand plus public history),
//! so a [`Belief`] is one normalized marginal per seat.

use crate::rebel::hands;

/// Seats a [`PublicState`] reserves room for.
pub const MAX_SEATS: usize = 8;

/// The common-knowledge state of a Liar's Dice subgame node: everything needed
/// to define the node without any private hand.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PublicState {
    pub players: u8,
    pub faces: u8,
    /// Per-seat dice remaining; 0 marks an eliminated seat.
    pub dice_left: [u8; MAX_SEATS],
    /// Standing bid `(qty, face)`; `None` is a free open / round start.
    pub bid: Option<(u8, u8)>,
    pub turn: usize,
    pub last_bidder: usize,
    pub first_round: bool,
}

impl PublicState {
    /// The seat to act. Meaningful only at a non-terminal node.
    pub fn acting(&self) -> usize {
        self.turn
    }
}

/// Per-seat normalized distribution over that seat's own hands, each indexed
/// within the seat's `dice_left` support (length `hand_count(dice_left, faces)`).
/// An eliminated seat (0 dice) has the degenerate single-element `[1.0]`.
#[derive(Clone, Debug)]
pub struct Belief {
    pub per_seat: Vec<Vec<f64>>,
}

impl Belief {
    /// The round-opening belief: each seat's fair-dice prior over its hands.
    pub fn uniform_prior(public: &PublicState) -> Belief {
        let per_seat = (0..public.players as usize)
            .map(|s| {
                hands::tables(public.dice_left[s], public.faces)
                    .prior
                    .clone()
            })
            .collect();
        Belief { per_seat }
    }

    /// Renormalize every seat's marginal to sum to 1.
    pub fn normalize(&mut self) {
        for seat in self.per_seat.iter_mut() {
            let sum: f64 = seat.iter().sum();
            let sum = sum.max(EPS);
            for p in seat.iter_mut() {
                *p /= sum;
            }
        }
    }
}

/// Public belief state.
#[derive(Clone, Debug)]
pub struct Pbs {
    pub public: PublicState,
    pub belief: Belief,
}

impl Pbs {
    /// A PBS whose belief is the round-opening fair-dice prior.
    pub fn uniform(public: PublicState) -> Pbs {
        let belief = Belief::uniform_prior(&public);
        Pbs { public, belief }
    }
}

const EPS: f64 = 1e-80;

/// In-place Bayesian update of one seat's marginal given the per-hand
/// probability that the seat played the observed action: `P'(h) ∝ P(h)·π(a|h)`,
/// renormalized.
pub fn bayes_update(belief_seat: &mut [f64], action_probs_per_hand: &[f64]) {
    for (p, &pi) in belief_seat.iter_mut().zip(action_probs_per_hand) {
        *p *= pi;
    }
    let sum: f64 = belief_seat.iter().sum();
    let sum = sum.max(EPS);
    for p in belief_seat.iter_mut() {
        *p /= sum;
    }
}

/// The belief after the acting seat plays an action, updating only that seat's
/// marginal via [`bayes_update`] and leaving the others unchanged.
pub fn propagate(pbs: &Pbs, acting_seat: usize, per_hand_action_prob: &[f64]) -> Belief {
    let mut belief = pbs.belief.clone();
    bayes_update(&mut belief.per_seat[acting_seat], per_hand_action_prob);
    belief
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rebel::hands;

    fn small_public() -> PublicState {
        let mut dice_left = [0u8; MAX_SEATS];
        dice_left[0] = 1;
        dice_left[1] = 1;
        PublicState {
            players: 2,
            faces: 3,
            dice_left,
            bid: None,
            turn: 0,
            last_bidder: 1,
            first_round: true,
        }
    }

    #[test]
    fn uniform_prior_is_normalized_per_seat() {
        let public = small_public();
        let belief = Belief::uniform_prior(&public);
        assert_eq!(belief.per_seat.len(), 2);
        for seat in &belief.per_seat {
            assert_eq!(seat.len(), hands::hand_count(1, 3));
            let s: f64 = seat.iter().sum();
            assert!((s - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn eliminated_seat_has_degenerate_belief() {
        let mut public = small_public();
        public.players = 2;
        public.dice_left[1] = 0;
        let belief = Belief::uniform_prior(&public);
        assert_eq!(belief.per_seat[1], vec![1.0]);
    }

    #[test]
    fn bayes_update_matches_brute_force_joint_posterior() {
        // 2 players, 1 die, 3 faces. Seat 0 acts with an arbitrary per-hand
        // action policy; the posterior over seat 0's hand given the observed
        // action is computed two ways: bayes_update and an explicit enumeration
        // of the 2-seat joint marginalized over seat 1.
        let prior0 = hands::prior(1, 3);
        let prior1 = hands::prior(1, 3);
        let action_prob = [0.7, 0.1, 0.4];

        let mut updated = prior0.clone();
        bayes_update(&mut updated, &action_prob);

        let mut joint = vec![0.0; prior0.len()];
        let mut z = 0.0;
        for (h0, j) in joint.iter_mut().enumerate() {
            for &p1 in &prior1 {
                let mass = prior0[h0] * p1 * action_prob[h0];
                *j += mass;
                z += mass;
            }
        }
        for v in joint.iter_mut() {
            *v /= z;
        }

        for (u, b) in updated.iter().zip(&joint) {
            assert!((u - b).abs() < 1e-12, "{u} vs {b}");
        }
    }

    #[test]
    fn propagate_updates_only_the_acting_seat() {
        let pbs = Pbs::uniform(small_public());
        let action_prob = [0.2, 0.5, 0.9];
        let next = propagate(&pbs, 0, &action_prob);
        assert_ne!(next.per_seat[0], pbs.belief.per_seat[0]);
        assert_eq!(next.per_seat[1], pbs.belief.per_seat[1]);
        let s: f64 = next.per_seat[0].iter().sum();
        assert!((s - 1.0).abs() < 1e-12);
    }
}
