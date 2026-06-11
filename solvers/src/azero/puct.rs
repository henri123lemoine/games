//! PUCT tree search guided by an [`Mlp`]: priors from the policy head over
//! the legal actions, leaf values from the value head, exact values at
//! terminal nodes. One net evaluation per expanded node (no batching).
//!
//! Two-player zero-sum only: the scalar value head is read as "expected
//! return for the player to move", and negated across players on backup.
//! Chance nodes are sampled once at expansion (fine for deterministic games;
//! a stochastic game would bake one outcome per edge into the tree).

/// Re-exported from `game_core`, where the capability trait lives.
pub use game_core::PolicyValueEncoder;
use game_core::{Agent, Game, Rng, Turn};

use super::mlp::{InferCache, Mlp};
use super::rand::dirichlet;

pub struct Puct<'a, G: Game, E: PolicyValueEncoder<G>> {
    pub game: &'a G,
    pub enc: &'a E,
    pub net: &'a Mlp,
    pub sims: usize,
    pub c_puct: f32,
    pub dirichlet_alpha: f32,
    /// Weight of Dirichlet noise mixed into the root prior; 0 disables it.
    pub root_noise: f32,
    /// Sparse-input fast path for `net`, snapshotted at construction (sound:
    /// the shared borrow keeps the net frozen for this `Puct`'s lifetime).
    cache: InferCache,
}

struct Node<S, A> {
    state: S,
    actions: Vec<A>,
    to_move: usize,
    terminal: bool,
    /// Terminal return to player 0 (terminal nodes only).
    value0: f64,
    /// Net value for `to_move` at creation (non-terminal nodes only).
    value: f64,
    prior: Vec<f32>,
    n: Vec<u32>,
    w: Vec<f64>,
    child: Vec<usize>,
}

impl<'a, G: Game, E: PolicyValueEncoder<G>> Puct<'a, G, E> {
    pub fn new(game: &'a G, enc: &'a E, net: &'a Mlp, sims: usize) -> Self {
        Puct {
            game,
            enc,
            net,
            sims,
            c_puct: 1.5,
            dirichlet_alpha: 0.3,
            root_noise: 0.0,
            cache: net.infer_cache(),
        }
    }

    /// Runs `sims` simulations from `root` (a non-terminal decision node) and
    /// returns the root visit counts, aligned with `legal_actions(root)`.
    pub fn search(&self, root: &G::State, rng: &mut Rng) -> Vec<u32> {
        debug_assert!(!self.game.is_terminal(root));
        let mut root_node = self.expand(root.clone());
        if self.root_noise > 0.0 && root_node.prior.len() > 1 {
            let noise = dirichlet(self.dirichlet_alpha as f64, root_node.prior.len(), rng);
            for (p, n) in root_node.prior.iter_mut().zip(noise) {
                *p = (1.0 - self.root_noise) * *p + self.root_noise * n as f32;
            }
        }
        let mut nodes = vec![root_node];

        for _ in 0..self.sims {
            let mut path: Vec<(usize, usize)> = Vec::new();
            let mut cur = 0usize;
            let leaf = loop {
                if nodes[cur].terminal {
                    break cur;
                }
                let e = select_edge(&nodes[cur], self.c_puct);
                path.push((cur, e));
                let child = nodes[cur].child[e];
                if child != usize::MAX {
                    cur = child;
                    continue;
                }
                let mut s = nodes[cur].state.clone();
                let a = nodes[cur].actions[e];
                self.game.apply(&mut s, a);
                sample_chance(self.game, &mut s, rng);
                let node = self.expand(s);
                let idx = nodes.len();
                nodes.push(node);
                nodes[cur].child[e] = idx;
                break idx;
            };

            let (terminal, value0, leaf_player, leaf_value) = {
                let n = &nodes[leaf];
                (n.terminal, n.value0, n.to_move, n.value)
            };
            for &(ni, ei) in &path {
                let p = nodes[ni].to_move;
                let val = if terminal {
                    if p == 0 { value0 } else { -value0 }
                } else if p == leaf_player {
                    leaf_value
                } else {
                    -leaf_value
                };
                nodes[ni].n[ei] += 1;
                nodes[ni].w[ei] += val;
            }
        }

        nodes[0].n.clone()
    }

    fn expand(&self, s: G::State) -> Node<G::State, G::Action> {
        if self.game.is_terminal(&s) {
            return Node {
                value0: self.game.returns(&s, 0),
                terminal: true,
                to_move: usize::MAX,
                value: 0.0,
                state: s,
                actions: Vec::new(),
                prior: Vec::new(),
                n: Vec::new(),
                w: Vec::new(),
                child: Vec::new(),
            };
        }
        let Turn::Player(to_move) = self.game.turn(&s) else {
            unreachable!("chance nodes are sampled before expansion");
        };
        let actions = self.game.legal_actions(&s);
        let x = self.enc.encode_state(self.game, &s);
        let support: Vec<usize> = actions
            .iter()
            .map(|&a| self.enc.action_index(self.game, &s, a))
            .collect();
        let (prior, v) = self.net.policy_value_cached(&self.cache, &x, &support);
        let k = actions.len();
        Node {
            state: s,
            actions,
            to_move,
            terminal: false,
            value0: 0.0,
            value: v as f64,
            prior,
            n: vec![0; k],
            w: vec![0.0; k],
            child: vec![usize::MAX; k],
        }
    }
}

fn select_edge<S, A>(node: &Node<S, A>, c_puct: f32) -> usize {
    let total: u32 = node.n.iter().sum();
    let sqrt_total = f64::from(total + 1).sqrt();
    let mut best = 0;
    let mut best_score = f64::NEG_INFINITY;
    for (i, ((&pr, &n), &w)) in node.prior.iter().zip(&node.n).zip(&node.w).enumerate() {
        let q = if n > 0 { w / f64::from(n) } else { 0.0 };
        let u = f64::from(c_puct) * f64::from(pr) * sqrt_total / (1.0 + f64::from(n));
        if q + u > best_score {
            best_score = q + u;
            best = i;
        }
    }
    best
}

/// Advances `s` through any chance nodes by sampling outcomes.
pub(crate) fn sample_chance<G: Game>(game: &G, s: &mut G::State, rng: &mut Rng) {
    while !game.is_terminal(s) && matches!(game.turn(s), Turn::Chance) {
        let outs = game.chance_outcomes(s);
        let i = game_core::rand::sample_outcome(&outs, rng);
        game.apply(s, outs[i].0);
    }
}

pub(crate) fn argmax(visits: &[u32]) -> usize {
    visits
        .iter()
        .enumerate()
        .max_by_key(|&(_, &n)| n)
        .map_or(0, |(i, _)| i)
}

/// [`Puct`] as an arena [`Agent`]: deterministic argmax over visit counts,
/// searching with the arena-supplied randomness.
pub struct PuctAgent<'a, G: Game, E: PolicyValueEncoder<G>>(pub Puct<'a, G, E>);

impl<G: Game, E: PolicyValueEncoder<G>> Agent<G> for PuctAgent<'_, G, E> {
    fn act(&self, _game: &G, state: &G::State, _player: usize, rng: &mut Rng) -> usize {
        argmax(&self.0.search(state, rng))
    }
}
