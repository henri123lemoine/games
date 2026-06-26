//! Deploy continuation: the ReBeL value net valuing its OWN round openings.
//!
//! [`NetContinuation`] closes the round-end calls of a
//! [`LiarsDiceAdapter`](crate::rebel::adapter::LiarsDiceAdapter): after a round
//! re-rolls every hand, a seat's continuation equity is a scalar, and that scalar
//! is the net's belief-weighted mean per-hand value at the next round's opening
//! public belief state (uniform prior). Because the re-roll erases hand
//! information, that mean is exactly the per-seat scalar the proven per-round
//! decomposition needs, so one net values both mid-round per-hand PBS leaves
//! (through [`NetLeaf`](crate::rebel::value_net::NetLeaf)) and round-opening
//! continuations (through this type) — fitted value iteration over the real
//! multi-round game.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::rebel::pbs::{Belief, MAX_SEATS, PublicState};
use crate::rebel::value_net::PbsNet;
use crate::subgame::{ContinuationValue, DiceShareValue};

/// `(dice_vector, next_opener, seat) -> scalar continuation value`, the per-round
/// memo guarding [`NetContinuation`]'s bounded set of net forwards.
type ContMemo = Mutex<HashMap<(Vec<u8>, usize, usize), f64>>;

/// The value net read as a per-round [`ContinuationValue`].
///
/// Memoized by `(dice_left, next_opener, seat)`: one round queries only a handful
/// of post-round dice vectors, each opened by a live seat, from a few seats'
/// perspectives, so the [`Mutex`]-guarded memo bounds the net forwards and keeps
/// the type `Sync` for rayon data generation.
pub struct NetContinuation<'a> {
    net: &'a PbsNet,
    memo: ContMemo,
}

impl<'a> NetContinuation<'a> {
    pub fn new(net: &'a PbsNet) -> Self {
        Self {
            net,
            memo: Mutex::new(HashMap::new()),
        }
    }

    /// The round-opening public state for a generic continuing round: a free open
    /// (`bid = None`, `first_round = false`) with `opener` to act and the live seat
    /// immediately before it carrying the relative bid-owner position — exactly the
    /// state [`LiarsDiceAdapter::root`](crate::rebel::adapter::LiarsDiceAdapter::root)
    /// reconstructs for a non-first round, so the continuation queries the same
    /// public states the self-play loop emits round-opening targets for.
    fn opening_state(faces: u8, dice_left: &[u8], opener: usize) -> PublicState {
        let players = dice_left.len();
        let mut dl = [0u8; MAX_SEATS];
        dl[..players].copy_from_slice(dice_left);
        let mut prev = (opener + players - 1) % players;
        while dl[prev] == 0 {
            prev = (prev + players - 1) % players;
        }
        PublicState {
            players: players as u8,
            faces,
            dice_left: dl,
            bid: None,
            turn: opener,
            last_bidder: prev,
            first_round: false,
        }
    }

    fn compute(&self, faces: u8, dice_left: &[u8], opener: usize, seat: usize) -> f64 {
        let public = Self::opening_state(faces, dice_left, opener);
        let belief = Belief::uniform_prior(&public);
        self.net
            .evaluate(&public, seat, &belief)
            .iter()
            .zip(&belief.per_seat[seat])
            .map(|(v, p)| v * p)
            .sum()
    }
}

impl ContinuationValue for NetContinuation<'_> {
    fn value(&self, faces: u8, dice_left: &[u8], next_opener: usize, player: usize) -> f64 {
        let n = dice_left.len();
        let loser_share = -1.0 / (n as f64 - 1.0);
        let alive = dice_left.iter().filter(|&&d| d > 0).count();
        if alive <= 1 {
            return if dice_left[player] > 0 {
                1.0
            } else {
                loser_share
            };
        }
        // An eliminated seat has already lost: its game return is fixed at the
        // loser share regardless of how the survivors finish, so it needs no net
        // query (the net is never trained on an empty-hand traverser).
        if dice_left[player] == 0 {
            return loser_share;
        }
        let key = (dice_left.to_vec(), next_opener, player);
        let mut memo = self.memo.lock().expect("net-continuation memo poisoned");
        *memo
            .entry(key)
            .or_insert_with(|| self.compute(faces, dice_left, next_opener, player))
    }
}

/// The deploy self-play continuation under the warmup→FVI schedule: the
/// [`DiceShareValue`] heuristic seeds the bootstrap, then the net values its own
/// round openings once it has learned a signal.
pub enum DeployCont<'a> {
    Heuristic(DiceShareValue),
    Net(NetContinuation<'a>),
}

impl ContinuationValue for DeployCont<'_> {
    fn value(&self, faces: u8, dice_left: &[u8], next_opener: usize, player: usize) -> f64 {
        match self {
            DeployCont::Heuristic(h) => h.value(faces, dice_left, next_opener, player),
            DeployCont::Net(n) => n.value(faces, dice_left, next_opener, player),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rebel::adapter::LiarsDiceAdapter;
    use crate::rebel::cfr::{CfrParams, Solver};
    use crate::rebel::game::RebelGame;
    use crate::rebel::hands::hand_count;
    use crate::rebel::value_net::{NetLeaf, PbsNet};

    fn dice_vec(counts: &[u8]) -> [u8; MAX_SEATS] {
        let mut d = [0u8; MAX_SEATS];
        d[..counts.len()].copy_from_slice(counts);
        d
    }

    #[test]
    fn net_continuation_is_the_belief_weighted_mean() {
        let net = PbsNet::new(32, 2, 5);
        let cont = NetContinuation::new(&net);
        for &(faces, dice, opener, seat) in &[
            (3u8, [2u8, 2u8], 0usize, 0usize),
            (4, [2, 1], 1, 0),
            (3, [1, 2], 0, 1),
            (5, [2, 2], 1, 1),
        ] {
            let public = NetContinuation::opening_state(faces, &dice, opener);
            let belief = Belief::uniform_prior(&public);
            let want: f64 = net
                .evaluate(&public, seat, &belief)
                .iter()
                .zip(&belief.per_seat[seat])
                .map(|(v, p)| v * p)
                .sum();
            let got = cont.value(faces, &dice, opener, seat);
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
    }

    #[test]
    fn eliminated_seat_takes_the_loser_share() {
        let net = PbsNet::new(16, 2, 1);
        let cont = NetContinuation::new(&net);
        // Three seats, the queried seat eliminated while two survive.
        assert!((cont.value(4, &[2, 0, 2], 0, 1) + 0.5).abs() < 1e-12);
        // One survivor: the round is over, exact game returns.
        assert_eq!(cont.value(4, &[0, 3, 0], 1, 1), 1.0);
        assert_eq!(cont.value(4, &[0, 3, 0], 1, 0), -0.5);
    }

    #[test]
    fn a_net_trained_to_mimic_dice_share_matches_dice_share() {
        // Overfit a small net to predict the DiceShareValue at every 2-player
        // round opening of a 3-face config, then confirm NetContinuation recovers
        // it: the round-opening belief-weighted mean tracks the heuristic.
        let faces = 3u8;
        let configs: Vec<([u8; 2], usize, usize)> = (1..=3u8)
            .flat_map(|a| (1..=3u8).map(move |b| (a, b)))
            .flat_map(|(a, b)| (0..2usize).map(move |op| (a, b, op)))
            .flat_map(|(a, b, op)| (0..2usize).map(move |seat| ([a, b], op, seat)))
            .collect();

        let encoder = PbsNet::new(64, 2, 0);
        let samples: Vec<_> = configs
            .iter()
            .map(|&(dice, opener, seat)| {
                let public = NetContinuation::opening_state(faces, &dice, opener);
                let belief = Belief::uniform_prior(&public);
                let v = DiceShareValue.value(faces, &dice, opener, seat);
                let target = vec![v; hand_count(dice[seat], faces)];
                encoder.to_sample(&public, seat, &belief, &target)
            })
            .collect();

        let mut net = PbsNet::new(64, 2, 0);
        net.net_mut().set_lr(3e-3);
        for step in 0..6000 {
            if step == 4000 {
                net.net_mut().set_lr(1e-3);
            }
            net.net_mut().train_step(&samples);
        }

        let cont = NetContinuation::new(&net);
        let mut max_err = 0.0f64;
        for &(dice, opener, seat) in &configs {
            let got = cont.value(faces, &dice, opener, seat);
            let want = DiceShareValue.value(faces, &dice, opener, seat);
            max_err = max_err.max((got - want).abs());
        }
        assert!(
            max_err < 0.05,
            "NetContinuation vs DiceShareValue max err {max_err}"
        );
    }

    #[test]
    fn net_leaf_over_the_adapter_drives_a_valid_strategy() {
        let net = PbsNet::new(64, 2, 7);
        let cont = NetContinuation::new(&net);
        let adapter = LiarsDiceAdapter::new(2, 3, dice_vec(&[2, 2]), 0, false, &cont);
        let leaf = NetLeaf::new(&net, &adapter);
        let params = CfrParams {
            num_iters: 64,
            max_depth: 2,
            ..CfrParams::default()
        };
        let initial = Belief::uniform_prior(&adapter.root());
        let mut solver = Solver::new(&adapter, params, &leaf, initial);
        solver.multistep();
        let avg = solver.average_strategy();
        let tree = solver.tree();
        let mut decision_nodes = 0;
        for (node, policy) in tree.nodes.iter().zip(avg) {
            if node.is_leaf {
                continue;
            }
            decision_nodes += 1;
            assert!(!policy.is_empty());
            for row in policy {
                let sum: f64 = row.iter().sum();
                assert!((sum - 1.0).abs() < 1e-6, "row sums to {sum}");
                assert!(row.iter().all(|&p| (0.0..=1.0).contains(&p)));
            }
        }
        assert!(
            decision_nodes > 0,
            "the subgame has decision nodes to solve"
        );

        // The traverser's root values are well-formed per-hand scalars.
        let root_vals = solver.root_values_mean(0);
        assert_eq!(root_vals.len(), hand_count(adapter.root().dice_left[0], 3));
        assert!(root_vals.iter().all(|v| v.is_finite()));
    }
}
