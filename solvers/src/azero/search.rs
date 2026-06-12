//! PUCT search built for batched evaluation: `advance` gathers up to
//! `max_leaves` leaves per call (diversified by virtual loss), parks them,
//! and resumes when the caller returns the net's results. Batching happens
//! both within a search (multiple leaves) and across many concurrent
//! searches — the caller owns the evaluator, which may be a GPU batch, a
//! CPU net, or a WebGPU bridge on the other side of a wasm boundary.
//!
//! Generic over [`Game`] + [`PolicyValueEncoder`]. Two-player zero-sum only:
//! the scalar value head is read as "expected return for the player to
//! move", and backups compare each node's player against the leaf's (so
//! non-alternating turn orders are handled). Chance nodes are sampled once
//! at expansion and baked into the tree.
//!
//! Two behaviors that started life chess-side are config, not code:
//!
//! * **Cycle awareness** (`cycle_draws`). A state whose [`Game::repetition_key`]
//!   already occurred in the game (the caller's `seen`) or earlier on the
//!   current descent path backs up a draw immediately — without it,
//!   self-play in repetition games shuffles into threefold draws the tree
//!   cannot see. Cycle draws are transient (path-dependent), never stored
//!   as terminal nodes.
//! * **First-play urgency** (`fpu`). Unvisited edges score
//!   `node value − fpu` rather than 0, so search deepens promising lines
//!   instead of spraying one visit everywhere.

use game_core::rand::dirichlet;
use game_core::{Game, PolicyValueEncoder, Rng, Turn};

#[derive(Clone, Copy)]
pub struct PuctConfig {
    pub sims: u32,
    pub c_puct: f32,
    pub fpu: f32,
    pub dirichlet_alpha: f64,
    /// Weight of Dirichlet noise mixed into the root prior; 0 disables.
    pub root_noise: f32,
    /// Leaves gathered per `advance` call (virtual-loss parallelism).
    pub max_leaves: u32,
    /// Back up a draw when a descent revisits a state (by
    /// [`Game::repetition_key`]) seen earlier in the game or on the current
    /// path. Game knowledge:
    /// enable where repetition means a draw (chess), leave off elsewhere.
    pub cycle_draws: bool,
}

impl Default for PuctConfig {
    fn default() -> Self {
        PuctConfig {
            sims: 320,
            c_puct: 1.6,
            fpu: 0.25,
            dirichlet_alpha: 0.3,
            root_noise: 0.25,
            max_leaves: 8,
            cycle_draws: false,
        }
    }
}

/// One evaluation request: the encoder's features plus the legal policy
/// indices ([`PolicyValueEncoder::action_index`], which must fit in `u16`).
pub struct EvalRequest {
    pub features: Vec<f32>,
    pub support: Vec<u16>,
}

/// Priors over `support` (softmax restricted to the legal subset) and the
/// value, both from the side to move's perspective.
pub struct EvalResult {
    pub priors: Vec<f32>,
    pub value: f32,
}

pub struct Node<G: Game> {
    state: G::State,
    pub actions: Vec<G::Action>,
    to_move: usize,
    pub prior: Vec<f32>,
    pub n: Vec<u32>,
    /// Total action value per edge, from this node's player's perspective.
    pub w: Vec<f64>,
    child: Vec<i32>,
    /// Net value at this node, for the player to move (non-terminal nodes).
    value: f32,
    /// Exact return to player 0 (terminal nodes).
    value0: f64,
    terminal: bool,
}

impl<G: Game> Node<G> {
    fn visits(&self) -> u32 {
        self.n.iter().sum()
    }
}

pub struct Tree<G: Game> {
    pub nodes: Vec<Node<G>>,
    pub root: usize,
}

struct Pending<G: Game> {
    path: Vec<(usize, usize)>,
    state: G::State,
    actions: Vec<G::Action>,
    to_move: usize,
}

/// What `Search::advance` came back with.
pub enum Gather {
    /// Leaves need the net; resume by passing the results back, aligned.
    Requests(Vec<EvalRequest>),
    /// The root has its visit budget; pick a move from `root_visits`.
    Done,
}

/// The leaf value being backed up a path.
#[derive(Clone, Copy)]
enum Leaf {
    /// Net evaluation, from `player`'s perspective.
    Net { player: usize, value: f32 },
    /// Exact return to player 0 (terminal or cycle draw).
    Exact(f64),
}

pub struct Search<G: Game> {
    tree: Tree<G>,
    pending: Vec<Pending<G>>,
    noised: bool,
}

impl<G: Game> Search<G> {
    /// Starts a search, optionally seeded with a reused subtree.
    pub fn new(reuse: Option<Tree<G>>) -> Search<G> {
        Search {
            tree: reuse.unwrap_or(Tree {
                nodes: Vec::new(),
                root: 0,
            }),
            pending: Vec::new(),
            noised: false,
        }
    }

    /// Resolves `results` (aligned with the previous `Gather::Requests`),
    /// then gathers the next batch of leaves or finishes. `seen` answers
    /// "did this repetition key already occur in the game?" (only consulted
    /// when `cycle_draws` is on; pass `&|_| false` otherwise).
    #[allow(clippy::too_many_arguments)]
    pub fn advance<E: PolicyValueEncoder<G>>(
        &mut self,
        game: &G,
        enc: &E,
        root: &G::State,
        cfg: &PuctConfig,
        rng: &mut Rng,
        results: Vec<EvalResult>,
        seen: &dyn Fn(u64) -> bool,
    ) -> Gather {
        debug_assert_eq!(results.len(), self.pending.len(), "results align");
        for (pending, res) in std::mem::take(&mut self.pending).into_iter().zip(results) {
            self.resolve(pending, res);
        }

        // Fresh tree: the root itself needs one evaluation first.
        if self.tree.nodes.is_empty() {
            let actions = game.legal_actions(root);
            assert!(
                !actions.is_empty(),
                "search started from a terminal position"
            );
            let Turn::Player(to_move) = game.turn(root) else {
                panic!("search started from a chance node");
            };
            let req = eval_request(game, enc, root, &actions);
            self.pending.push(Pending {
                path: Vec::new(),
                state: root.clone(),
                actions,
                to_move,
            });
            return Gather::Requests(vec![req]);
        }
        // A reused subtree can be rooted at a terminal node (the extracted
        // move ended the game). There is nothing to search — and descending
        // would back up empty paths forever without ever filling the visit
        // budget.
        if self.tree.nodes[self.tree.root].terminal {
            return Gather::Done;
        }
        if !self.noised && cfg.root_noise > 0.0 {
            add_dirichlet(&mut self.tree.nodes[self.tree.root], cfg, rng);
            self.noised = true;
        }

        let mut requests = Vec::new();
        while self.tree.nodes[self.tree.root].visits() < cfg.sims
            && (requests.len() as u32) < cfg.max_leaves
        {
            if let Some(pending) = self.descend(game, cfg, rng, seen) {
                requests.push(eval_request(game, enc, &pending.state, &pending.actions));
                self.pending.push(pending);
            }
        }
        if requests.is_empty() {
            debug_assert!(self.pending.is_empty());
            Gather::Done
        } else {
            Gather::Requests(requests)
        }
    }

    /// One descent. Terminal and cycle leaves back up immediately and
    /// return `None`; a leaf needing the net gets virtual loss applied and
    /// returns the pending record.
    fn descend(
        &mut self,
        game: &G,
        cfg: &PuctConfig,
        rng: &mut Rng,
        seen: &dyn Fn(u64) -> bool,
    ) -> Option<Pending<G>> {
        let mut cur = self.tree.root;
        let mut path: Vec<(usize, usize)> = Vec::new();
        let mut path_keys: Vec<u64> = Vec::new();
        loop {
            let node = &self.tree.nodes[cur];
            if node.terminal {
                let v = node.value0;
                self.backup(&path, Leaf::Exact(v));
                return None;
            }
            let e = select_edge(node, (cfg.c_puct, cfg.fpu));
            path.push((cur, e));

            let child = self.tree.nodes[cur].child[e];
            if child >= 0 {
                if cfg.cycle_draws
                    && let Some(key) = game.repetition_key(&self.tree.nodes[child as usize].state)
                {
                    if seen(key) || path_keys.contains(&key) {
                        self.backup(&path, Leaf::Exact(0.0));
                        return None;
                    }
                    path_keys.push(key);
                }
                cur = child as usize;
                continue;
            }

            let mut s = self.tree.nodes[cur].state.clone();
            game.apply(&mut s, self.tree.nodes[cur].actions[e]);
            // Resolve chance before the cycle check and expansion, so both
            // see the concrete successor; outcomes are baked into the tree.
            let to_move = loop {
                if game.is_terminal(&s) {
                    break None;
                }
                match game.turn(&s) {
                    Turn::Chance => {
                        let outs = game.chance_outcomes(&s);
                        let i = game_core::rand::sample_outcome(&outs, rng);
                        game.apply(&mut s, outs[i].0);
                    }
                    Turn::Player(p) => break Some(p),
                }
            };
            if cfg.cycle_draws
                && let Some(key) = game.repetition_key(&s)
            {
                if seen(key) || path_keys.contains(&key) {
                    self.backup(&path, Leaf::Exact(0.0));
                    return None;
                }
                path_keys.push(key);
            }
            let Some(to_move) = to_move else {
                let value0 = game.returns(&s, 0);
                let idx = self.tree.nodes.len();
                self.tree.nodes.push(terminal_node(s, value0));
                self.tree.nodes[cur].child[e] = idx as i32;
                self.backup(&path, Leaf::Exact(value0));
                return None;
            };
            let actions = game.legal_actions(&s);
            // Park: apply virtual loss so sibling descents diversify.
            for &(ni, ei) in &path {
                let n = &mut self.tree.nodes[ni];
                n.n[ei] += 1;
                n.w[ei] -= 1.0;
            }
            return Some(Pending {
                path,
                state: s,
                actions,
                to_move,
            });
        }
    }

    fn resolve(&mut self, mut pending: Pending<G>, res: EvalResult) {
        let path = std::mem::take(&mut pending.path);
        // Undo virtual loss.
        for &(ni, ei) in &path {
            let n = &mut self.tree.nodes[ni];
            n.n[ei] -= 1;
            n.w[ei] += 1.0;
        }
        let leaf = Leaf::Net {
            player: pending.to_move,
            value: res.value,
        };
        let &(parent, edge) = match path.last() {
            Some(last) => last,
            None => {
                // Root evaluation of a fresh tree.
                self.tree.nodes.push(expanded_node(pending, res));
                self.tree.root = self.tree.nodes.len() - 1;
                return;
            }
        };
        if self.tree.nodes[parent].child[edge] < 0 {
            let idx = self.tree.nodes.len();
            self.tree.nodes.push(expanded_node(pending, res));
            self.tree.nodes[parent].child[edge] = idx as i32;
        }
        self.backup(&path, leaf);
    }

    /// Backs `leaf` up the path: each node accumulates the value from its
    /// own player's perspective.
    fn backup(&mut self, path: &[(usize, usize)], leaf: Leaf) {
        for &(ni, ei) in path {
            let node = &mut self.tree.nodes[ni];
            let v = match leaf {
                Leaf::Net { player, value } => {
                    if node.to_move == player {
                        f64::from(value)
                    } else {
                        -f64::from(value)
                    }
                }
                Leaf::Exact(value0) => {
                    if node.to_move == 0 {
                        value0
                    } else {
                        -value0
                    }
                }
            };
            node.n[ei] += 1;
            node.w[ei] += v;
        }
    }

    /// Visit counts over the root's actions, aligned with `root_actions`.
    pub fn root_visits(&self) -> &[u32] {
        &self.tree.nodes[self.tree.root].n
    }

    pub fn root_actions(&self) -> &[G::Action] {
        &self.tree.nodes[self.tree.root].actions
    }

    /// Visit-weighted mean value of the root position (player to move):
    /// the search's estimate of the position itself, for value targets.
    pub fn root_value(&self) -> f64 {
        let root = &self.tree.nodes[self.tree.root];
        let n: u32 = root.n.iter().sum();
        let w: f64 = root.w.iter().sum();
        if n > 0 { w / f64::from(n) } else { 0.0 }
    }

    /// Mean value of the most-visited root edge (player to move).
    pub fn root_q(&self) -> f64 {
        let root = &self.tree.nodes[self.tree.root];
        let mut best = (0u32, 0.0f64);
        for (&n, &w) in root.n.iter().zip(&root.w) {
            if n > best.0 {
                best = (n, w);
            }
        }
        if best.0 > 0 {
            best.1 / f64::from(best.0)
        } else {
            0.0
        }
    }

    /// Extracts the subtree under the root's edge `e` for reuse after that
    /// action is played. Returns `None` if the child was never expanded.
    pub fn extract_child(self, e: usize) -> Option<Tree<G>> {
        debug_assert!(self.pending.is_empty(), "extract with leaves in flight");
        let child = self.tree.nodes[self.tree.root].child[e];
        if child < 0 {
            return None;
        }
        let mut map = vec![-1i32; self.tree.nodes.len()];
        let mut old_of_new = vec![child as usize];
        map[child as usize] = 0;
        let mut i = 0;
        while i < old_of_new.len() {
            let old = old_of_new[i];
            i += 1;
            for &c in &self.tree.nodes[old].child {
                if c >= 0 && map[c as usize] < 0 {
                    map[c as usize] = old_of_new.len() as i32;
                    old_of_new.push(c as usize);
                }
            }
        }
        let mut old_nodes: Vec<Option<Node<G>>> = self.tree.nodes.into_iter().map(Some).collect();
        let nodes = old_of_new
            .into_iter()
            .map(|old| {
                let mut n = old_nodes[old].take().expect("node moved once");
                for c in &mut n.child {
                    if *c >= 0 {
                        *c = map[*c as usize];
                    }
                }
                n
            })
            .collect();
        Some(Tree { nodes, root: 0 })
    }
}

fn terminal_node<G: Game>(state: G::State, value0: f64) -> Node<G> {
    Node {
        state,
        actions: Vec::new(),
        to_move: usize::MAX,
        prior: Vec::new(),
        n: Vec::new(),
        w: Vec::new(),
        child: Vec::new(),
        value: 0.0,
        value0,
        terminal: true,
    }
}

fn expanded_node<G: Game>(pending: Pending<G>, res: EvalResult) -> Node<G> {
    let k = pending.actions.len();
    Node {
        state: pending.state,
        actions: pending.actions,
        to_move: pending.to_move,
        prior: res.priors,
        n: vec![0; k],
        w: vec![0.0; k],
        child: vec![-1; k],
        value: res.value,
        value0: 0.0,
        terminal: false,
    }
}

fn eval_request<G: Game, E: PolicyValueEncoder<G>>(
    game: &G,
    enc: &E,
    state: &G::State,
    actions: &[G::Action],
) -> EvalRequest {
    let support = actions
        .iter()
        .map(|&a| {
            let idx = enc.action_index(game, state, a);
            debug_assert!(idx <= usize::from(u16::MAX), "policy index fits u16");
            idx as u16
        })
        .collect();
    EvalRequest {
        features: enc.encode_state(game, state),
        support,
    }
}

fn select_edge<G: Game>(node: &Node<G>, (c_puct, fpu): (f32, f32)) -> usize {
    let total = node.visits();
    let sqrt_total = f64::from(total + 1).sqrt();
    let fpu_q = f64::from(node.value) - f64::from(fpu);
    let mut best = 0;
    let mut best_score = f64::NEG_INFINITY;
    for i in 0..node.actions.len() {
        let q = if node.n[i] > 0 {
            node.w[i] / f64::from(node.n[i])
        } else {
            fpu_q
        };
        let u = f64::from(c_puct) * f64::from(node.prior[i]) * sqrt_total
            / (1.0 + f64::from(node.n[i]));
        if q + u > best_score {
            best_score = q + u;
            best = i;
        }
    }
    best
}

fn add_dirichlet<G: Game>(node: &mut Node<G>, cfg: &PuctConfig, rng: &mut Rng) {
    if node.prior.len() < 2 {
        return;
    }
    let noise = dirichlet(cfg.dirichlet_alpha, node.prior.len(), rng);
    for (p, n) in node.prior.iter_mut().zip(noise) {
        *p = (1.0 - cfg.root_noise) * *p + cfg.root_noise * n as f32;
    }
}

pub fn argmax(visits: &[u32]) -> usize {
    visits
        .iter()
        .enumerate()
        .max_by_key(|&(_, &n)| n)
        .map_or(0, |(i, _)| i)
}
