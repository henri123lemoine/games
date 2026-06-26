//! Vector-form (public-tree) CFR over a [`Tree`], ported from the reference
//! ReBeL solver. Per-node, per-hand regrets/strategies are maintained for the
//! acting seat; reach probabilities flow down per seat; counterfactual values
//! flow up. Leaves are valued through a [`LeafValue`] and scaled by opponent
//! reach mass. Alternating Linear CFR is the deployed (and default) setting.

use crate::rebel::game::RebelGame;
use crate::rebel::hands::{self, hand_count};
use crate::rebel::leaf::LeafValue;
use crate::rebel::pbs::{Belief, PublicState};
use crate::rebel::tree::Tree;

/// Reach and regret smoothing floor, matching the reference `1e-80`.
pub(crate) const SMOOTHING_EPS: f64 = 1e-80;

/// Regret-discounting scheme applied each iteration.
#[derive(Clone, Copy, Debug)]
pub enum CfrVariant {
    /// Linear CFR: regrets, negative regrets, and the strategy sum are all
    /// discounted by `t/(t+1)`, weighting iteration `t` by `t`.
    LinearCfr,
    /// Undiscounted vanilla CFR.
    Vanilla,
    /// Discounted CFR with the standard three exponents.
    Dcfr { alpha: f64, beta: f64, gamma: f64 },
}

/// Configuration for a [`Solver`] run.
#[derive(Clone, Copy, Debug)]
pub struct CfrParams {
    pub num_iters: usize,
    pub max_depth: u32,
    pub variant: CfrVariant,
    /// Alternating updates (one traverser per iteration). When false, every seat
    /// is updated each iteration.
    pub alternating: bool,
    /// Take leaf beliefs from the average strategy (CFR-AVG) rather than the
    /// current strategy (CFR-D).
    pub cfr_avg: bool,
    /// Re-query each leaf's value (the expensive net forward) only once every this
    /// many of the traverser's own iterations, reusing the cached per-hand output
    /// in between while still applying that iteration's fresh opponent-reach
    /// scaling. `1` (the default) refreshes every iteration — exactly the
    /// un-cached behavior. Larger values trade a small Nash-distance approximation
    /// for a ~K-fold cut in net forwards; gate before raising. Keep at `1` for
    /// exact eval/exploitability paths.
    pub leaf_refresh_every: usize,
}

impl Default for CfrParams {
    fn default() -> Self {
        Self {
            num_iters: 1024,
            max_depth: 2,
            variant: CfrVariant::LinearCfr,
            alternating: true,
            cfr_avg: false,
            leaf_refresh_every: 1,
        }
    }
}

/// The index of each node within its parent's action/child ordering, or `None`
/// for the root.
pub(crate) fn parent_actions(tree: &Tree) -> Vec<Option<usize>> {
    let mut pa = vec![None; tree.len()];
    for node in &tree.nodes {
        for (action_idx, &child) in node.children.iter().enumerate() {
            pa[child] = Some(action_idx);
        }
    }
    pa
}

/// Per-node reach probabilities for one seat: the product of that seat's action
/// probabilities (under `strategy`) along the path from the root, seeded with
/// `initial`. Unchanged at nodes where another seat acted.
pub(crate) fn reach_probabilities(
    tree: &Tree,
    parent_action: &[Option<usize>],
    strategy: &[Vec<Vec<f64>>],
    initial: &[f64],
    player: usize,
) -> Vec<Vec<f64>> {
    let mut reach = vec![Vec::new(); tree.len()];
    reach[0] = initial.to_vec();
    for node_id in 1..tree.len() {
        let parent = tree.nodes[node_id].parent.expect("non-root has a parent");
        if tree.nodes[parent].acting == player {
            let action_idx = parent_action[node_id].expect("non-root has a parent action");
            reach[node_id] = reach[parent]
                .iter()
                .enumerate()
                .map(|(hand, &r)| r * strategy[parent][hand][action_idx])
                .collect();
        } else {
            reach[node_id] = reach[parent].clone();
        }
    }
    reach
}

/// [`reach_probabilities`] computed in place into `dst` — one preallocated vector
/// per node, each already sized to `player`'s (tree-constant) hand count — instead
/// of allocating a fresh reach table per call. Every node's reach is fully
/// overwritten, so prior contents are irrelevant. Children always follow their
/// parent in the BFS node order, so `split_at_mut(node_id)` cleanly separates the
/// parent (read) from the node being written.
pub(crate) fn fill_reach(
    tree: &Tree,
    parent_action: &[Option<usize>],
    strategy: &[Vec<Vec<f64>>],
    initial: &[f64],
    player: usize,
    dst: &mut [Vec<f64>],
) {
    dst[0].copy_from_slice(initial);
    // `node_id` indexes `dst` for the `split_at_mut` parent/child split, so the
    // range loop is intrinsic here, not a needless index.
    #[allow(clippy::needless_range_loop)]
    for node_id in 1..tree.len() {
        let parent = tree.nodes[node_id].parent.expect("non-root has a parent");
        let (left, right) = dst.split_at_mut(node_id);
        let cur = &mut right[0];
        if tree.nodes[parent].acting == player {
            let action_idx = parent_action[node_id].expect("non-root has a parent action");
            let parent_reach = &left[parent];
            let strat_parent = &strategy[parent];
            for (hand, c) in cur.iter_mut().enumerate() {
                *c = parent_reach[hand] * strat_parent[hand][action_idx];
            }
        } else {
            cur.copy_from_slice(&left[parent]);
        }
    }
}

fn normalize_in_place(v: &mut [f64]) {
    let sum: f64 = v.iter().sum::<f64>().max(SMOOTHING_EPS);
    for x in v.iter_mut() {
        *x /= sum;
    }
}

/// Vector-form CFR solver over a public tree. The tree is owned, so the only
/// borrow the solver carries is the leaf value.
pub struct Solver<'a> {
    params: CfrParams,
    leaf: &'a dyn LeafValue,
    tree: Tree,
    parent_action: Vec<Option<usize>>,
    players: usize,
    faces: u8,
    initial_beliefs: Belief,
    regrets: Vec<Vec<Vec<f64>>>,
    sum_strategies: Vec<Vec<Vec<f64>>>,
    last_strategies: Vec<Vec<Vec<f64>>>,
    average_strategies: Vec<Vec<Vec<f64>>>,
    reach: Vec<Vec<Vec<f64>>>,
    traverser_values: Vec<Vec<f64>>,
    root_values_means: Vec<Vec<f64>>,
    num_steps: Vec<usize>,
    /// Node ids of the tree's leaves, ascending (fixed across iterations).
    leaf_ids: Vec<usize>,
    /// Each leaf's public state, cloned once (fixed across iterations).
    leaf_publics: Vec<PublicState>,
    /// Reusable per-leaf normalized beliefs, overwritten each iteration.
    leaf_beliefs: Vec<Belief>,
    /// Reusable per-leaf opponent-reach scalers, overwritten each iteration.
    leaf_scalers: Vec<f64>,
    /// Cached net (normalized-belief) leaf output, indexed `[traverser][leaf]`,
    /// each inner vector sized to that seat's hand count. Refreshed every
    /// `leaf_refresh_every` of the traverser's iterations and reused (still scaled
    /// by fresh opponent reach) in between. Unused when `leaf_refresh_every <= 1`.
    leaf_value_cache: Vec<Vec<Vec<f64>>>,
    /// Reusable counterfactual-value accumulator (length = max traverser hands).
    val_buf: Vec<f64>,
}

impl<'a> Solver<'a> {
    pub fn new<G: RebelGame>(
        game: &G,
        params: CfrParams,
        leaf: &'a dyn LeafValue,
        initial_beliefs: Belief,
    ) -> Self {
        let tree = Tree::build(game, params.max_depth);
        let parent_action = parent_actions(&tree);
        let players = game.players();
        let faces = game.faces();
        let n = tree.len();

        let mut regrets = vec![Vec::new(); n];
        let mut last_strategies = vec![Vec::new(); n];
        let mut sum_strategies = vec![Vec::new(); n];
        let mut average_strategies = vec![Vec::new(); n];
        for (idx, node) in tree.nodes.iter().enumerate() {
            if node.is_leaf {
                continue;
            }
            let num_hands = hand_count(node.public.dice_left[node.acting], faces);
            let num_actions = node.legal.len();
            let uniform = 1.0 / num_actions as f64;
            regrets[idx] = vec![vec![0.0; num_actions]; num_hands];
            last_strategies[idx] = vec![vec![uniform; num_actions]; num_hands];
            sum_strategies[idx] = vec![vec![0.0; num_actions]; num_hands];
            average_strategies[idx] = vec![vec![uniform; num_actions]; num_hands];
        }

        // Dice (hence per-seat hand counts) are tree-constant: `apply` only changes
        // `turn`/`bid`/`last_bidder`, never `dice_left`. So every node's reach and
        // counterfactual-value vectors keep a fixed length and can be preallocated
        // once here and overwritten in place each iteration.
        let root_dice = tree.root().public.dice_left;
        let seat_hands: Vec<usize> = (0..players)
            .map(|s| hand_count(root_dice[s], faces))
            .collect();
        let max_hands = seat_hands.iter().copied().max().unwrap_or(1);

        let reach = (0..players)
            .map(|s| vec![vec![0.0; seat_hands[s]]; n])
            .collect();
        let traverser_values = vec![vec![0.0; max_hands]; n];
        let val_buf = vec![0.0; max_hands];
        let root_values_means = seat_hands.iter().map(|&h| vec![0.0; h]).collect();
        let num_steps = vec![0; players];

        let mut leaf_ids = Vec::new();
        let mut leaf_publics = Vec::new();
        for (idx, node) in tree.nodes.iter().enumerate() {
            if node.is_leaf {
                leaf_ids.push(idx);
                leaf_publics.push(node.public.clone());
            }
        }
        let leaf_beliefs = leaf_ids
            .iter()
            .map(|_| Belief {
                per_seat: seat_hands.iter().map(|&h| vec![0.0; h]).collect(),
            })
            .collect();
        let leaf_scalers = vec![0.0; leaf_ids.len()];
        let leaf_value_cache = (0..players)
            .map(|s| vec![vec![0.0; seat_hands[s]]; leaf_ids.len()])
            .collect();

        let mut solver = Self {
            params,
            leaf,
            tree,
            parent_action,
            players,
            faces,
            initial_beliefs,
            regrets,
            sum_strategies,
            last_strategies,
            average_strategies,
            reach,
            traverser_values,
            root_values_means,
            num_steps,
            leaf_ids,
            leaf_publics,
            leaf_beliefs,
            leaf_scalers,
            leaf_value_cache,
            val_buf,
        };
        solver.seed_uniform_reach_weighted();
        solver
    }

    /// Seed the strategy sum with the uniform strategy weighted by each acting
    /// seat's reach to the node under uniform play, so iteration 0 contributes a
    /// proper reach-weighted policy.
    fn seed_uniform_reach_weighted(&mut self) {
        for seat in 0..self.players {
            let reach = reach_probabilities(
                &self.tree,
                &self.parent_action,
                &self.last_strategies,
                &self.initial_beliefs.per_seat[seat],
                seat,
            );
            for (idx, node) in self.tree.nodes.iter().enumerate() {
                if node.is_leaf || node.acting != seat {
                    continue;
                }
                let uniform = 1.0 / node.legal.len() as f64;
                for (hand, cells) in self.sum_strategies[idx].iter_mut().enumerate() {
                    let weight = uniform * reach[idx][hand];
                    for cell in cells.iter_mut() {
                        *cell = weight;
                    }
                }
            }
        }
        for idx in 0..self.tree.len() {
            if self.tree.nodes[idx].is_leaf {
                continue;
            }
            for hand in 0..self.sum_strategies[idx].len() {
                self.average_strategies[idx][hand].clone_from(&self.sum_strategies[idx][hand]);
                normalize_in_place(&mut self.average_strategies[idx][hand]);
            }
        }
    }

    fn update_regrets(&mut self, traverser: usize) {
        let strat_for_reach = if self.params.cfr_avg {
            &self.average_strategies
        } else {
            &self.last_strategies
        };
        for seat in 0..self.players {
            fill_reach(
                &self.tree,
                &self.parent_action,
                strat_for_reach,
                &self.initial_beliefs.per_seat[seat],
                seat,
                &mut self.reach[seat],
            );
        }

        // The cached net output is recycled across `leaf_refresh_every` of this
        // traverser's iterations; `num_steps[traverser]` is the count of its
        // completed steps, so step 0 always refreshes (and seeds the cache). The
        // opponent-reach scalers below are cheap and recomputed every iteration.
        let k_refresh = self.params.leaf_refresh_every;
        let refresh = k_refresh <= 1 || self.num_steps[traverser].is_multiple_of(k_refresh);

        for k in 0..self.leaf_ids.len() {
            let idx = self.leaf_ids[k];
            if refresh {
                for seat in 0..self.players {
                    let r = &self.reach[seat][idx];
                    let sum = r.iter().sum::<f64>().max(SMOOTHING_EPS);
                    let dst = &mut self.leaf_beliefs[k].per_seat[seat];
                    for (d, x) in dst.iter_mut().zip(r) {
                        *d = *x / sum;
                    }
                }
            }
            self.leaf_scalers[k] = (0..self.players)
                .filter(|&j| j != traverser)
                .map(|j| self.reach[j][idx].iter().sum::<f64>())
                .product();
        }
        if refresh {
            let raws = self
                .leaf
                .values_batch(&self.leaf_publics, traverser, &self.leaf_beliefs);
            for (cached, raw) in self.leaf_value_cache[traverser].iter_mut().zip(&raws) {
                cached.copy_from_slice(raw);
            }
        }
        for k in 0..self.leaf_ids.len() {
            let idx = self.leaf_ids[k];
            let scaler = self.leaf_scalers[k];
            let cached = &self.leaf_value_cache[traverser][k];
            let dst = &mut self.traverser_values[idx];
            for (d, &v) in dst.iter_mut().zip(cached) {
                *d = v * scaler;
            }
        }

        for idx in (0..self.tree.len()).rev() {
            let node = &self.tree.nodes[idx];
            if node.is_leaf {
                continue;
            }
            let acting = node.acting;
            let trav_hands =
                hands::tables(node.public.dice_left[traverser], self.faces).hand_count();
            self.val_buf[..trav_hands].fill(0.0);
            if acting == traverser {
                for (action_idx, &child) in node.children.iter().enumerate() {
                    for hand in 0..trav_hands {
                        self.regrets[idx][hand][action_idx] += self.traverser_values[child][hand];
                        self.val_buf[hand] += self.traverser_values[child][hand]
                            * self.last_strategies[idx][hand][action_idx];
                    }
                }
                for hand in 0..trav_hands {
                    let v = self.val_buf[hand];
                    for regret in self.regrets[idx][hand].iter_mut() {
                        *regret -= v;
                    }
                }
            } else {
                for &child in &node.children {
                    for hand in 0..trav_hands {
                        self.val_buf[hand] += self.traverser_values[child][hand];
                    }
                }
            }
            self.traverser_values[idx][..trav_hands].copy_from_slice(&self.val_buf[..trav_hands]);
        }
    }

    fn discounts(&self, num_strategies: f64) -> (f64, f64, f64) {
        match self.params.variant {
            CfrVariant::LinearCfr => {
                let w = num_strategies / (num_strategies + 1.0);
                (w, w, w)
            }
            CfrVariant::Vanilla => (1.0, 1.0, 1.0),
            CfrVariant::Dcfr { alpha, beta, gamma } => {
                let pos = num_strategies.powf(alpha) / (num_strategies.powf(alpha) + 1.0);
                let neg = num_strategies.powf(beta) / (num_strategies.powf(beta) + 1.0);
                let strat = (num_strategies / (num_strategies + 1.0)).powf(gamma);
                (pos, neg, strat)
            }
        }
    }

    fn is_linear(&self) -> bool {
        matches!(self.params.variant, CfrVariant::LinearCfr)
    }

    fn acts_here(&self, idx: usize, traverser: usize) -> bool {
        let node = &self.tree.nodes[idx];
        !node.is_leaf && node.acting == traverser
    }

    pub fn step(&mut self, traverser: usize) {
        self.update_regrets(traverser);

        let steps = self.num_steps[traverser];
        let alpha = if self.is_linear() {
            2.0 / (steps as f64 + 2.0)
        } else {
            1.0 / (steps as f64 + 1.0)
        };
        for (mean, &value) in self.root_values_means[traverser]
            .iter_mut()
            .zip(&self.traverser_values[0])
        {
            *mean += (value - *mean) * alpha;
        }

        let num_strategies = (steps + 1) as f64;
        let (pos, neg, strat) = self.discounts(num_strategies);

        for idx in 0..self.tree.len() {
            if !self.acts_here(idx, traverser) {
                continue;
            }
            for hand in 0..self.last_strategies[idx].len() {
                for action in 0..self.last_strategies[idx][hand].len() {
                    self.last_strategies[idx][hand][action] =
                        self.regrets[idx][hand][action].max(SMOOTHING_EPS);
                }
                normalize_in_place(&mut self.last_strategies[idx][hand]);
            }
        }

        // The traverser's reach buffer is dead after `update_regrets` consumed it
        // for leaf beliefs, so reuse it for this iteration's (post-regret-match)
        // reach pass rather than allocating a fresh table.
        fill_reach(
            &self.tree,
            &self.parent_action,
            &self.last_strategies,
            &self.initial_beliefs.per_seat[traverser],
            traverser,
            &mut self.reach[traverser],
        );

        for idx in 0..self.tree.len() {
            if !self.acts_here(idx, traverser) {
                continue;
            }
            for hand in 0..self.regrets[idx].len() {
                let reach_hand = self.reach[traverser][idx][hand];
                for action in 0..self.regrets[idx][hand].len() {
                    let regret = self.regrets[idx][hand][action];
                    self.regrets[idx][hand][action] = regret * if regret > 0.0 { pos } else { neg };
                    self.sum_strategies[idx][hand][action] *= strat;
                    self.sum_strategies[idx][hand][action] +=
                        reach_hand * self.last_strategies[idx][hand][action];
                }
                self.average_strategies[idx][hand].clone_from(&self.sum_strategies[idx][hand]);
                normalize_in_place(&mut self.average_strategies[idx][hand]);
            }
        }

        self.num_steps[traverser] += 1;
    }

    pub fn multistep(&mut self) {
        if self.params.alternating {
            for iter in 0..self.params.num_iters {
                self.step(iter % self.players);
            }
        } else {
            for _ in 0..self.params.num_iters {
                for seat in 0..self.players {
                    self.step(seat);
                }
            }
        }
    }

    pub fn average_strategy(&self) -> &[Vec<Vec<f64>>] {
        &self.average_strategies
    }

    /// The current-iteration (regret-matched) strategy per node/hand/action. The
    /// recursive self-play loop descends a sampled trajectory under this policy.
    pub fn last_strategy(&self) -> &[Vec<Vec<f64>>] {
        &self.last_strategies
    }

    pub fn root_values_mean(&self, seat: usize) -> &[f64] {
        &self.root_values_means[seat]
    }

    pub fn tree(&self) -> &Tree {
        &self.tree
    }
}
