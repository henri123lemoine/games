//! UCT Monte-Carlo tree search for perfect-information games.
//!
//! Requires only [`Game`]; an optional [`Eval`] turns full random playouts into
//! depth-limited ones (truncate after `max_playout_depth` moves, score the leaf
//! with the eval). Each simulation does the classic four phases: UCB1 selection
//! down the stored tree, expansion of one untried action, a uniform-random
//! playout, and backpropagation.
//!
//! Multi-player returns are handled without zero-sum negation: every node
//! accumulates the terminal return of the player who *chose the edge into it*
//! (the parent's mover), which is exactly the quantity UCB1 compares when the
//! parent picks among its children. Chance nodes live in the tree too — descent
//! samples an outcome by its probability and descends into that outcome's
//! child, so deeper statistics never mix different chance branches.
//!
//! The tree is an arena (`Vec` of nodes, child links by index); descent clones
//! the root state once per simulation and applies actions along the path.
//! Single-threaded and deterministic given the seed.

use std::cell::Cell;

use game_core::{Agent, Eval, Game, Rng, Turn};

const UNEXPANDED: usize = usize::MAX;

/// UCT agent over any perfect-information [`Game`]. Picks the most-visited
/// root child after `sims` simulations; `c` is the UCB1 exploration constant
/// (on the `[-1, 1]` returns scale).
pub struct Mcts<G: Game> {
    pub sims: u32,
    pub c: f64,
    eval: Option<Box<dyn Eval<G>>>,
    max_playout_depth: u32,
    rng: Cell<u64>,
}

impl<G: Game> Mcts<G> {
    /// MCTS with full random playouts to terminal states.
    pub fn new(sims: u32, seed: u64) -> Self {
        Self {
            sims,
            c: std::f64::consts::SQRT_2,
            eval: None,
            max_playout_depth: u32::MAX,
            rng: Cell::new(seed | 1),
        }
    }

    /// MCTS with playouts truncated after `max_playout_depth` moves and scored
    /// by `eval` — useful where random playouts are too long or too noisy.
    pub fn with_eval(
        sims: u32,
        eval: impl Eval<G> + 'static,
        max_playout_depth: u32,
        seed: u64,
    ) -> Self {
        Self {
            sims,
            c: std::f64::consts::SQRT_2,
            eval: Some(Box::new(eval)),
            max_playout_depth,
            rng: Cell::new(seed | 1),
        }
    }

    fn next_seed(&self) -> u64 {
        let s = self.rng.get();
        self.rng.set(
            s.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407),
        );
        s
    }

    fn simulate(
        &self,
        game: &G,
        root_state: &G::State,
        nodes: &mut Vec<Node<G::Action>>,
        rng: &mut Rng,
    ) {
        let mut state = root_state.clone();
        let mut path = vec![0usize];
        let mut reached_terminal = true;
        loop {
            let id = *path.last().unwrap();
            if nodes[id].actions.is_empty() {
                break;
            }
            match nodes[id].mover {
                None => {
                    let k = pick_weighted(&nodes[id].probs, rng.unit());
                    game.apply(&mut state, nodes[id].actions[k]);
                    if nodes[id].children[k] == UNEXPANDED {
                        let child = nodes.len();
                        let view = nodes[id].view;
                        nodes.push(Node::for_state(game, &state, view));
                        nodes[id].children[k] = child;
                    }
                    path.push(nodes[id].children[k]);
                }
                Some(p) => {
                    let untried: Vec<usize> = nodes[id]
                        .children
                        .iter()
                        .enumerate()
                        .filter(|&(_, &c)| c == UNEXPANDED)
                        .map(|(i, _)| i)
                        .collect();
                    if !untried.is_empty() {
                        let k = untried[rand_below(untried.len(), rng)];
                        game.apply(&mut state, nodes[id].actions[k]);
                        let child = nodes.len();
                        nodes.push(Node::for_state(game, &state, p));
                        nodes[id].children[k] = child;
                        path.push(child);
                        reached_terminal = self.playout(game, &mut state, rng);
                        break;
                    }
                    let ln_n = f64::ln(nodes[id].visits.max(1) as f64);
                    let mut best = 0;
                    let mut best_score = f64::NEG_INFINITY;
                    for (i, &ch) in nodes[id].children.iter().enumerate() {
                        let n = nodes[ch].visits as f64;
                        let score = nodes[ch].value / n + self.c * (ln_n / n).sqrt();
                        if score > best_score {
                            best_score = score;
                            best = i;
                        }
                    }
                    game.apply(&mut state, nodes[id].actions[best]);
                    path.push(nodes[id].children[best]);
                }
            }
        }
        for &id in &path {
            let view = nodes[id].view;
            let v = if reached_terminal {
                game.returns(&state, view)
            } else {
                let eval = self.eval.as_ref().expect("truncated playout without eval");
                eval.eval(game, &state, view)
            };
            nodes[id].visits += 1;
            nodes[id].value += v;
        }
    }

    /// Uniform-random playout. Returns whether a terminal state was reached
    /// (`false` only when an [`Eval`] cutoff truncated the playout).
    fn playout(&self, game: &G, state: &mut G::State, rng: &mut Rng) -> bool {
        let mut depth = 0u32;
        loop {
            if game.is_terminal(state) {
                return true;
            }
            if self.eval.is_some() && depth >= self.max_playout_depth {
                return false;
            }
            match game.turn(state) {
                Turn::Chance => {
                    let outs = game.chance_outcomes(state);
                    let r = rng.unit();
                    let mut acc = 0.0;
                    let mut chosen = outs[outs.len() - 1].0;
                    for &(a, p) in &outs {
                        acc += p;
                        if r < acc {
                            chosen = a;
                            break;
                        }
                    }
                    game.apply(state, chosen);
                }
                Turn::Player(_) => {
                    let actions = game.legal_actions(state);
                    game.apply(state, actions[rand_below(actions.len(), rng)]);
                }
            }
            depth += 1;
        }
    }
}

impl<G: Game> Agent<G> for Mcts<G> {
    fn act(&self, game: &G, state: &G::State, player: usize, _r: f64) -> usize {
        if game.legal_actions(state).len() <= 1 {
            return 0;
        }
        let mut rng = Rng::new(self.next_seed());
        let mut nodes = Vec::with_capacity(self.sims as usize + 1);
        nodes.push(Node::for_state(game, state, player));
        for _ in 0..self.sims {
            self.simulate(game, state, &mut nodes, &mut rng);
        }
        let mut best = 0;
        let mut best_visits = 0u32;
        for (i, &ch) in nodes[0].children.iter().enumerate() {
            let v = if ch == UNEXPANDED {
                0
            } else {
                nodes[ch].visits
            };
            if v > best_visits {
                best_visits = v;
                best = i;
            }
        }
        best
    }
}

/// Tree node: terminal (`actions` empty), chance (`mover == None`), or
/// decision (`mover == Some(p)`). `children[i]` pairs with `actions[i]`;
/// `view` is the player whose returns `value` accumulates — the mover of the
/// nearest decision ancestor, i.e. whoever steered the search into this node.
struct Node<A> {
    actions: Vec<A>,
    probs: Vec<f64>,
    children: Vec<usize>,
    mover: Option<usize>,
    view: usize,
    visits: u32,
    value: f64,
}

impl<A: Copy> Node<A> {
    fn for_state<G: Game<Action = A>>(game: &G, state: &G::State, view: usize) -> Self {
        let (actions, probs, mover) = if game.is_terminal(state) {
            (Vec::new(), Vec::new(), None)
        } else {
            match game.turn(state) {
                Turn::Chance => {
                    let (a, p) = game.chance_outcomes(state).into_iter().unzip();
                    (a, p, None)
                }
                Turn::Player(p) => (game.legal_actions(state), Vec::new(), Some(p)),
            }
        };
        Node {
            children: vec![UNEXPANDED; actions.len()],
            actions,
            probs,
            mover,
            view,
            visits: 0,
            value: 0.0,
        }
    }
}

fn pick_weighted(probs: &[f64], r: f64) -> usize {
    let mut acc = 0.0;
    for (i, p) in probs.iter().enumerate() {
        acc += p;
        if r < acc {
            return i;
        }
    }
    probs.len() - 1
}

fn rand_below(n: usize, rng: &mut Rng) -> usize {
    ((rng.unit() * n as f64) as usize).min(n - 1)
}
