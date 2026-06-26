//! Test-time ReBeL agent for the real (non-standard, multi-round, N-player)
//! Liar's Dice — the playable bot.
//!
//! At each decision the agent re-derives the live round from its opening and
//! plays the depth-limited block-resolving strategy of the trained value net.
//! The mechanism is the deploy-faithful one: a [`LiarsDiceAdapter`] models the
//! current round, closed at round boundaries by a [`NetContinuation`], and a
//! depth-`max_depth` subgame is solved by vector CFR at the opening, advancing a
//! full block at a time down to the hero's live decision while Bayesian-
//! propagating the public beliefs — exactly the continual re-solving the
//! self-play loop and [`recursive_strategy`](crate::rebel::recursive_strategy)
//! use, but following only the single line that reaches the live node.
//!
//! ## Reconstructing the line
//!
//! The real [`LdState`] retains per-seat *bid counts* ([`LdState::raises_this_round`])
//! and the standing bid, not the ordered bid sequence, so the exact within-round
//! line is not recoverable from public state alone. Because bids are monotone
//! (every raise adds one rank step either `+1` face or `+faces` quantity) the
//! agent reconstructs the most-likely line consistent with the public facts: it
//! descends from the round opening choosing, at each ancestor, the
//! highest-probability raise (under the resolved average strategy, marginalized
//! over that seat's belief) that keeps the live standing bid reachable in the
//! remaining bids. Turn order is fixed by the adapter mechanics, so placing
//! exactly `sum(raises_this_round)` raises always lands on the hero's live node;
//! the belief there is the reach-propagated posterior along that line. If the
//! reconstruction ever fails (no reachable raise — not expected for a consistent
//! state) the agent falls back to solving rooted directly at the live public
//! state with the uniform round-opening prior, which is always legal.

use std::io;
use std::path::Path;

use game_core::{Agent, Game, Rng};

use solvers::rebel_mlp::RebelMlp;

use crate::rebel::adapter::LiarsDiceAdapter;
use crate::rebel::cfr::{CfrParams, SMOOTHING_EPS, Solver, parent_actions, reach_probabilities};
use crate::rebel::deploy::NetContinuation;
use crate::rebel::game::{Bid, RebelGame};
use crate::rebel::hands::{self, MAX_FACES};
use crate::rebel::leaf::RootedGame;
use crate::rebel::pbs::{Belief, MAX_SEATS, PublicState};
use crate::rebel::tree::Node;
use crate::rebel::value_net::{NetLeaf, PbsNet};
use crate::subgame::ContinuationValue;
use crate::{Action, LdState, LiarsDice};

/// Default CFR iterations per subgame solve at test time.
pub const DEFAULT_ITERS: usize = 1024;
/// Default depth limit (public plies) of each resolving block.
pub const DEFAULT_DEPTH: u32 = 2;

/// A test-time ReBeL agent: a trained [`PbsNet`] played by depth-limited
/// block-resolving search over the live round.
pub struct RebelAgent {
    net: PbsNet,
    num_iters: usize,
    max_depth: u32,
}

impl RebelAgent {
    /// The agent over `net` with the default test-time search budget.
    pub fn new(net: PbsNet) -> Self {
        Self {
            net,
            num_iters: DEFAULT_ITERS,
            max_depth: DEFAULT_DEPTH,
        }
    }

    /// The agent over `net` with an explicit search budget.
    pub fn with_config(net: PbsNet, num_iters: usize, max_depth: u32) -> Self {
        Self {
            net,
            num_iters,
            max_depth,
        }
    }

    /// Build the agent from a serialized [`PbsNet`]/[`RebelMlp`] checkpoint's bytes.
    pub fn from_bytes(data: &[u8]) -> io::Result<Self> {
        Ok(Self::new(PbsNet::from_mlp(RebelMlp::from_bytes(data)?)))
    }

    /// Load the agent's value net from a checkpoint file.
    pub fn load(path: &Path) -> io::Result<Self> {
        Ok(Self::new(PbsNet::load(path)?))
    }

    fn eval_params(&self) -> CfrParams {
        CfrParams {
            num_iters: self.num_iters,
            max_depth: self.max_depth,
            ..CfrParams::default()
        }
    }

    /// The agent's mixed strategy at `state` as a distribution over
    /// `game.legal_actions(state)`. Deterministic given the net (the search uses
    /// no randomness); [`Agent::act`] samples from it.
    pub fn action_probs(&self, game: &LiarsDice, state: &LdState, player: usize) -> Vec<f64> {
        let real_actions = game.legal_actions(state);
        let n_real = real_actions.len();
        if n_real <= 1 {
            return vec![1.0; n_real.max(1)];
        }

        let players = game.players as usize;
        let faces = game.faces;
        let mut dice_left = [0u8; MAX_SEATS];
        dice_left[..players].copy_from_slice(&state.dice_left()[..players]);
        let opener = game.round_opener(state) as usize;
        let first_round = state.first_round();

        let cont = NetContinuation::new(&self.net);
        let adapter = LiarsDiceAdapter::new(players, faces, dice_left, opener, first_round, &cont);

        let bids: usize = state.raises_this_round()[..players]
            .iter()
            .map(|&r| r as usize)
            .sum();
        let (cq, cf) = state.current_bid();
        let target_rank = if cq == 0 {
            0
        } else {
            (cq as usize - 1) * faces as usize + (cf as usize - 1)
        };

        let my_dice = state.dice_left()[player];
        let mut my_hand = [0u8; MAX_FACES];
        for f in 1..=faces {
            my_hand[(f - 1) as usize] = state.my_count(player, f);
        }
        let hand_idx = hands::index_within(&my_hand, my_dice, faces);

        let eval = self.eval_params();
        descend_probs(
            &self.net,
            &adapter,
            eval,
            faces,
            bids,
            target_rank,
            hand_idx,
            &real_actions,
        )
        .unwrap_or_else(|| {
            fallback_probs(
                &self.net,
                &adapter,
                eval,
                state,
                faces,
                hand_idx,
                &real_actions,
            )
        })
    }
}

impl Agent<LiarsDice> for RebelAgent {
    fn act(&self, game: &LiarsDice, state: &LdState, player: usize, rng: &mut Rng) -> usize {
        if game.legal_actions(state).len() <= 1 {
            return 0;
        }
        let probs = self.action_probs(game, state, player);
        rng.pick(&probs)
    }
}

/// Rank of a bid `(qty, face)` (0-based face) in the monotone bid order: a
/// quantity raise steps `+faces`, a face raise (or its quantity-carry wrap)
/// steps `+1`.
fn bid_rank(qty: u8, face: u8, faces: u8) -> usize {
    (qty as usize - 1) * faces as usize + face as usize
}

/// Whether the target rank is reachable from `rank` in exactly `rem` raises,
/// each adding `1` (face) or `faces` (quantity) to the rank.
fn reachable(rank: usize, rem: usize, target: usize, faces: u8) -> bool {
    if rank > target {
        return false;
    }
    let gap = target - rank;
    if rem == 0 {
        return gap == 0;
    }
    if gap < rem {
        return false;
    }
    let step = faces as usize - 1;
    let extra = gap - rem;
    extra.is_multiple_of(step) && extra / step <= rem
}

/// Pick the descent action at `node`: the highest-probability raise (under
/// `avg` marginalized over the acting seat's `belief`) that keeps `target_rank`
/// reachable in `plies_after` further raises. `None` if no raise qualifies.
fn choose_descend_action(
    node: &Node,
    avg: &[Vec<f64>],
    belief: &Belief,
    faces: u8,
    target_rank: usize,
    plies_after: usize,
) -> Option<usize> {
    let acting_belief = &belief.per_seat[node.acting];
    let mut best: Option<(usize, f64)> = None;
    for (ai, bid) in node.legal.iter().enumerate() {
        let Bid::Raise { qty, face } = bid else {
            continue;
        };
        let rank = bid_rank(*qty, *face, faces);
        if !reachable(rank, plies_after, target_rank, faces) {
            continue;
        }
        let mass: f64 = avg
            .iter()
            .zip(acting_belief)
            .map(|(row, &p)| p * row[ai])
            .sum();
        if best.is_none_or(|(_, b)| mass > b) {
            best = Some((ai, mass));
        }
    }
    best.map(|(ai, _)| ai)
}

/// Block-resolving descent to the hero's live node, returning its strategy as a
/// distribution over `real_actions`. `None` if the line cannot be reconstructed.
#[allow(clippy::too_many_arguments)]
fn descend_probs<C: ContinuationValue>(
    net: &PbsNet,
    adapter: &LiarsDiceAdapter<C>,
    eval: CfrParams,
    faces: u8,
    bids: usize,
    target_rank: usize,
    hand_idx: usize,
    real_actions: &[Action],
) -> Option<Vec<f64>> {
    let players = adapter.players();
    let mut block_root = adapter.root();
    let mut belief = Belief::uniform_prior(&block_root);
    let mut placed = 0usize;

    loop {
        let rooted = RootedGame::new(adapter, block_root.clone());
        let leaf = NetLeaf::new(net, adapter);
        let mut solver = Solver::new(&rooted, eval, &leaf, belief.clone());
        solver.multistep();
        let sub = solver.tree();
        let avg = solver.average_strategy();

        let remaining = bids - placed;
        if remaining == 0 {
            let node = &sub.nodes[0];
            return Some(strategy_to_real_probs(
                node,
                &avg[0][hand_idx],
                faces,
                real_actions,
            ));
        }

        let steps = remaining.min(2);
        let mut node = 0usize;
        for s in 0..steps {
            let plies_after = remaining - s - 1;
            let nn = &sub.nodes[node];
            let ai =
                choose_descend_action(nn, &avg[node], &belief, faces, target_rank, plies_after)?;
            node = nn.children[ai];
        }

        let pa = parent_actions(sub);
        let per_seat = (0..players)
            .map(|s| {
                let reach = reach_probabilities(sub, &pa, avg, &belief.per_seat[s], s);
                let r = &reach[node];
                let z = r.iter().sum::<f64>().max(SMOOTHING_EPS);
                r.iter().map(|x| x / z).collect()
            })
            .collect();
        block_root = sub.nodes[node].public.clone();
        belief = Belief { per_seat };
        placed += steps;
    }
}

/// Solve rooted directly at the live public state with the uniform round-opening
/// prior — the always-legal fallback when the line cannot be reconstructed.
fn fallback_probs<C: ContinuationValue>(
    net: &PbsNet,
    adapter: &LiarsDiceAdapter<C>,
    eval: CfrParams,
    state: &LdState,
    faces: u8,
    hand_idx: usize,
    real_actions: &[Action],
) -> Vec<f64> {
    let players = adapter.players();
    let mut dice_left = [0u8; MAX_SEATS];
    dice_left[..players].copy_from_slice(&state.dice_left()[..players]);
    let (cq, cf) = state.current_bid();
    let bid = if cq == 0 { None } else { Some((cq, cf - 1)) };
    let live = PublicState {
        players: players as u8,
        faces,
        dice_left,
        bid,
        turn: state.turn(),
        last_bidder: state.last_bidder(),
        first_round: state.first_round(),
    };
    let rooted = RootedGame::new(adapter, live);
    let leaf = NetLeaf::new(net, adapter);
    let belief = Belief::uniform_prior(&rooted.root());
    let mut solver = Solver::new(&rooted, eval, &leaf, belief);
    solver.multistep();
    let sub = solver.tree();
    let avg = solver.average_strategy();
    strategy_to_real_probs(&sub.nodes[0], &avg[0][hand_idx], faces, real_actions)
}

/// Map a live node's per-action strategy `row` onto a distribution over the real
/// game's `real_actions`, translating each adapter [`Bid`] to its [`Action`].
fn strategy_to_real_probs(
    node: &Node,
    row: &[f64],
    faces: u8,
    real_actions: &[Action],
) -> Vec<f64> {
    let standing = node.public.bid;
    let mut probs = vec![0.0; real_actions.len()];
    for (ai, bid) in node.legal.iter().enumerate() {
        let action = bid_to_action(bid, standing, faces);
        if let Some(j) = real_actions.iter().position(|a| *a == action) {
            probs[j] += row[ai];
        }
    }
    if probs.iter().sum::<f64>() <= 1e-12 {
        return vec![1.0 / real_actions.len() as f64; real_actions.len()];
    }
    probs
}

/// Translate an adapter [`Bid`] at a node with the given `standing` bid (0-based
/// face) into the real game's [`Action`].
fn bid_to_action(bid: &Bid, standing: Option<(u8, u8)>, faces: u8) -> Action {
    match bid {
        Bid::Call => Action::CallLiar,
        Bid::CallExact => Action::CallExact,
        Bid::Raise { qty, face } => match standing {
            None => Action::Open(*qty, *face + 1),
            Some((cq, cf)) => {
                if *qty == cq + 1 && *face == cf {
                    Action::RaiseQuantity
                } else if *qty == cq && *face == cf + 1 {
                    Action::RaiseFace
                } else {
                    debug_assert!(
                        *qty == cq + 1 && *face == 0 && cf + 1 == faces,
                        "raise {qty}x{face} is not a legal step off {cq}x{cf}"
                    );
                    Action::RaiseFace
                }
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_core::Turn;

    /// A small untrained net for the legality gate: any net must produce only
    /// legal play.
    fn small_agent(seed: u64) -> RebelAgent {
        RebelAgent::with_config(PbsNet::new(32, 2, seed), 64, 2)
    }

    /// Play a full game with the agent in every seat, asserting every chosen
    /// index is legal and no solve panics. Covers heads-up and a 5-player table.
    fn legal_play(players: u8, dice: u8, faces: u8, games: usize, seed: u64) {
        let agent = small_agent(seed);
        let game = LiarsDice::new(players, dice, faces);
        let mut rng = Rng::new(seed ^ 0x1234);
        for _ in 0..games {
            let mut s = game.initial_state();
            while !game.is_terminal(&s) {
                match game.turn(&s) {
                    Turn::Chance => {
                        let a = game.sample_chance(&s, &mut rng).0;
                        game.apply(&mut s, a);
                    }
                    Turn::Player(p) => {
                        let acts = game.legal_actions(&s);
                        let i = agent.act(&game, &s, p, &mut rng);
                        assert!(i < acts.len(), "illegal index {i} of {}", acts.len());
                        let probs = agent.action_probs(&game, &s, p);
                        assert_eq!(probs.len(), acts.len());
                        let sum: f64 = probs.iter().sum();
                        assert!(probs.iter().all(|x| x.is_finite()) && sum > 0.0);
                        game.apply(&mut s, acts[i]);
                    }
                }
            }
        }
    }

    #[test]
    fn legal_play_two_player() {
        legal_play(2, 2, 3, 8, 1);
    }

    #[test]
    fn legal_play_multiplayer() {
        legal_play(3, 2, 3, 4, 5);
    }

    /// The flagship legality gate: full 5p5d6f games produce only legal actions.
    /// Heavy (the wide opening block is re-solved every move) — run explicitly:
    /// `cargo test -p liars-dice --release -- --ignored legal_play_five_player`.
    #[test]
    #[ignore = "full 5p5d6f games; slow, run explicitly"]
    fn legal_play_five_player_flagship() {
        legal_play(5, 5, 6, 2, 7);
    }

    #[test]
    fn reachable_matches_monotone_bid_arithmetic() {
        // 3 faces: a quantity raise is +3, a face raise +1. From rank 0, reaching
        // rank 4 in 2 raises needs one of each (4 = 3 + 1): reachable.
        assert!(reachable(0, 2, 4, 3));
        // Rank 4 in 1 raise is impossible (max single step is 3).
        assert!(!reachable(0, 1, 4, 3));
        // Exact arrival with no raises left only when already there.
        assert!(reachable(5, 0, 5, 3));
        assert!(!reachable(4, 0, 5, 3));
    }

    #[test]
    fn bid_to_action_translates_every_adapter_move() {
        // Off a standing 2x(face 1) over 3 faces.
        let s = Some((2u8, 1u8));
        assert_eq!(
            bid_to_action(&Bid::Raise { qty: 3, face: 1 }, s, 3),
            Action::RaiseQuantity
        );
        assert_eq!(
            bid_to_action(&Bid::Raise { qty: 2, face: 2 }, s, 3),
            Action::RaiseFace
        );
        // Top-face wrap off 2x(face 2): RaiseFace carries to 3x(face 0).
        assert_eq!(
            bid_to_action(&Bid::Raise { qty: 3, face: 0 }, Some((2, 2)), 3),
            Action::RaiseFace
        );
        assert_eq!(bid_to_action(&Bid::Call, s, 3), Action::CallLiar);
        assert_eq!(bid_to_action(&Bid::CallExact, s, 3), Action::CallExact);
        // A free open maps to Open with the real 1-based face.
        assert_eq!(
            bid_to_action(&Bid::Raise { qty: 3, face: 2 }, None, 3),
            Action::Open(3, 3)
        );
    }
}
