//! Vector-form (public-tree) CFR over a [`Tree`], ported from the reference
//! ReBeL solver. Per-node, per-hand regrets/strategies are maintained for the
//! acting seat; reach probabilities flow down per seat; counterfactual values
//! flow up. Leaves are valued through a [`LeafValue`] and scaled by opponent
//! reach mass. Alternating Linear CFR is the deployed (and default) setting.

use crate::rebel::game::RebelGame;
use crate::rebel::hands::hand_count;
use crate::rebel::leaf::LeafValue;
use crate::rebel::pbs::Belief;
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
}

impl Default for CfrParams {
    fn default() -> Self {
        Self {
            num_iters: 1024,
            max_depth: 2,
            variant: CfrVariant::LinearCfr,
            alternating: true,
            cfr_avg: false,
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

        let reach = vec![vec![Vec::new(); n]; players];
        let traverser_values = vec![Vec::new(); n];
        let root_values_means = (0..players)
            .map(|s| vec![0.0; hand_count(tree.root().public.dice_left[s], faces)])
            .collect();
        let num_steps = vec![0; players];

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
            self.reach[seat] = reach_probabilities(
                &self.tree,
                &self.parent_action,
                strat_for_reach,
                &self.initial_beliefs.per_seat[seat],
                seat,
            );
        }

        let mut leaf_ids = Vec::new();
        let mut publics = Vec::new();
        let mut beliefs = Vec::new();
        let mut scalers = Vec::new();
        for idx in 0..self.tree.len() {
            if !self.tree.nodes[idx].is_leaf {
                continue;
            }
            let per_seat = (0..self.players)
                .map(|seat| {
                    let r = &self.reach[seat][idx];
                    let sum = r.iter().sum::<f64>().max(SMOOTHING_EPS);
                    r.iter().map(|x| x / sum).collect()
                })
                .collect();
            let scaler: f64 = (0..self.players)
                .filter(|&j| j != traverser)
                .map(|j| self.reach[j][idx].iter().sum::<f64>())
                .product();
            leaf_ids.push(idx);
            publics.push(self.tree.nodes[idx].public.clone());
            beliefs.push(Belief { per_seat });
            scalers.push(scaler);
        }
        let raws = self.leaf.values_batch(&publics, traverser, &beliefs);
        for ((&idx, raw), &scaler) in leaf_ids.iter().zip(&raws).zip(&scalers) {
            self.traverser_values[idx] = raw.iter().map(|v| v * scaler).collect();
        }

        for idx in (0..self.tree.len()).rev() {
            let (is_leaf, acting, dice_trav, children) = {
                let node = &self.tree.nodes[idx];
                (
                    node.is_leaf,
                    node.acting,
                    node.public.dice_left[traverser],
                    node.children.clone(),
                )
            };
            if is_leaf {
                continue;
            }
            let trav_hands = hand_count(dice_trav, self.faces);
            let mut val = vec![0.0; trav_hands];
            if acting == traverser {
                for (action_idx, &child) in children.iter().enumerate() {
                    for (hand, v) in val.iter_mut().enumerate() {
                        self.regrets[idx][hand][action_idx] += self.traverser_values[child][hand];
                        *v += self.traverser_values[child][hand]
                            * self.last_strategies[idx][hand][action_idx];
                    }
                }
                for (hand, &v) in val.iter().enumerate() {
                    for regret in self.regrets[idx][hand].iter_mut() {
                        *regret -= v;
                    }
                }
            } else {
                for &child in &children {
                    for (hand, v) in val.iter_mut().enumerate() {
                        *v += self.traverser_values[child][hand];
                    }
                }
            }
            self.traverser_values[idx] = val;
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
        let root_values = self.traverser_values[0].clone();
        for (mean, &value) in self.root_values_means[traverser]
            .iter_mut()
            .zip(&root_values)
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

        let reach_buf = reach_probabilities(
            &self.tree,
            &self.parent_action,
            &self.last_strategies,
            &self.initial_beliefs.per_seat[traverser],
            traverser,
        );

        for (idx, reach_node) in reach_buf.iter().enumerate() {
            if !self.acts_here(idx, traverser) {
                continue;
            }
            for (hand, &reach_hand) in reach_node.iter().enumerate() {
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
