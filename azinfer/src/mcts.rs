//! PUCT search built for batched evaluation: `advance` gathers up to
//! `max_leaves` leaves per call (diversified by virtual loss), parks them,
//! and resumes when the caller returns the net's results. Batching happens
//! both within a game (multiple leaves) and across many concurrent games.
//!
//! Two chess-specific behaviors live here deliberately:
//!
//! * **Repetition awareness.** A position that already occurred in the game
//!   (`history`) or earlier on the current descent path backs up a draw
//!   immediately — without it, self-play shuffles into threefold draws the
//!   tree cannot see. Repetition draws are transient (path-dependent), never
//!   stored as terminal nodes.
//! * **First-play urgency.** Unvisited edges score `node value − fpu`
//!   rather than 0, so search deepens promising lines instead of spraying
//!   one visit everywhere.
//!
//! Values are always from the perspective of the player to move at each
//! node.

use std::collections::HashMap;

use chess::encode::{az_move_index, encode_planes};
use chess::{Board, Move, legal_moves};
use game_core::Rng;

use crate::{EvalRequest, EvalResult};

#[derive(Clone, Copy)]
pub struct MctsConfig {
    pub sims: u32,
    pub c_puct: f32,
    pub fpu: f32,
    pub dirichlet_alpha: f64,
    /// Weight of Dirichlet noise mixed into the root prior; 0 disables.
    pub root_noise: f32,
    /// Leaves gathered per `advance` call (virtual-loss parallelism).
    pub max_leaves: u32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig {
            sims: 320,
            c_puct: 1.6,
            fpu: 0.25,
            dirichlet_alpha: 0.3,
            root_noise: 0.25,
            max_leaves: 8,
        }
    }
}

pub struct Node {
    pub moves: Vec<Move>,
    pub prior: Vec<f32>,
    pub n: Vec<u32>,
    /// Total action value per edge, from this node's player's perspective.
    pub w: Vec<f64>,
    pub child: Vec<i32>,
    /// Net value at this node (player to move), or the exact return for
    /// terminal nodes.
    pub value: f32,
    pub terminal: bool,
}

impl Node {
    fn visits(&self) -> u32 {
        self.n.iter().sum()
    }
}

pub struct Tree {
    pub nodes: Vec<Node>,
    pub root: usize,
}

struct Pending {
    path: Vec<(usize, usize)>,
    board: Board,
    moves: Vec<Move>,
}

/// What `Search::advance` came back with.
pub enum Gather {
    /// Leaves need the net; resume by passing the results back, aligned.
    Requests(Vec<EvalRequest>),
    /// The root has its visit budget; pick a move from `root_visits`.
    Done,
}

pub struct Search {
    pub(crate) tree: Tree,
    pending: Vec<Pending>,
    noised: bool,
}

impl Search {
    /// Starts a search, optionally seeded with a reused subtree.
    pub fn new(reuse: Option<Tree>) -> Search {
        Search {
            tree: reuse.unwrap_or(Tree {
                nodes: Vec::new(),
                root: 0,
            }),
            pending: Vec::new(),
            noised: false,
        }
    }

    #[cfg(test)]
    fn from_tree(tree: Tree) -> Search {
        Search {
            tree,
            pending: Vec::new(),
            noised: false,
        }
    }

    /// Resolves `results` (aligned with the previous `Gather::Requests`),
    /// then gathers the next batch of leaves or finishes.
    pub fn advance(
        &mut self,
        board: &Board,
        history: &HashMap<u64, u8>,
        cfg: &MctsConfig,
        rng: &mut Rng,
        results: Vec<EvalResult>,
    ) -> Gather {
        debug_assert_eq!(results.len(), self.pending.len(), "results align");
        for (pending, res) in std::mem::take(&mut self.pending).into_iter().zip(results) {
            self.resolve(pending, res);
        }

        // Fresh tree: the root itself needs one evaluation first.
        if self.tree.nodes.is_empty() {
            let moves = legal_moves(board);
            assert!(!moves.is_empty(), "search started from a terminal position");
            let req = eval_request(board, &moves);
            self.pending.push(Pending {
                path: Vec::new(),
                board: board.clone(),
                moves,
            });
            return Gather::Requests(vec![req]);
        }
        if !self.noised && cfg.root_noise > 0.0 {
            add_dirichlet(&mut self.tree.nodes[self.tree.root], cfg, rng);
            self.noised = true;
        }

        let mut requests = Vec::new();
        while self.tree.nodes[self.tree.root].visits() < cfg.sims
            && (requests.len() as u32) < cfg.max_leaves
        {
            if let Some(pending) = self.descend(board, history, (cfg.c_puct, cfg.fpu)) {
                requests.push(eval_request(&pending.board, &pending.moves));
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

    /// One descent. Terminal and repetition leaves back up immediately and
    /// return `None`; a leaf needing the net gets virtual loss applied and
    /// returns the pending record.
    fn descend(
        &mut self,
        board: &Board,
        history: &HashMap<u64, u8>,
        scalars: (f32, f32),
    ) -> Option<Pending> {
        let mut cur = self.tree.root;
        let mut b = board.clone();
        let mut path: Vec<(usize, usize)> = Vec::new();
        let mut path_keys: Vec<u64> = Vec::new();
        loop {
            let node = &self.tree.nodes[cur];
            if node.terminal {
                let v = node.value;
                self.backup(&path, v);
                return None;
            }
            let e = select_edge(node, scalars);
            path.push((cur, e));
            b.apply(self.tree.nodes[cur].moves[e]);
            let key = b.key();
            let seen_in_game = history.get(&key).copied().unwrap_or(0) > 0;
            let seen_on_path = path_keys.contains(&key);
            if seen_in_game || seen_on_path {
                self.backup(&path, 0.0);
                return None;
            }
            path_keys.push(key);

            let child = self.tree.nodes[cur].child[e];
            if child >= 0 {
                cur = child as usize;
                continue;
            }
            let moves = legal_moves(&b);
            if moves.is_empty() || b.halfmove >= 100 || b.insufficient_material() {
                let v = if moves.is_empty() && b.in_check(b.stm) {
                    -1.0
                } else {
                    0.0
                };
                let idx = self.tree.nodes.len();
                self.tree.nodes.push(terminal_node(v));
                self.tree.nodes[cur].child[e] = idx as i32;
                self.backup(&path, v);
                return None;
            }
            // Park: apply virtual loss so sibling descents diversify.
            for &(ni, ei) in &path {
                let n = &mut self.tree.nodes[ni];
                n.n[ei] += 1;
                n.w[ei] -= 1.0;
            }
            return Some(Pending {
                path,
                board: b,
                moves,
            });
        }
    }

    fn resolve(&mut self, pending: Pending, res: EvalResult) {
        // Undo virtual loss.
        for &(ni, ei) in &pending.path {
            let n = &mut self.tree.nodes[ni];
            n.n[ei] -= 1;
            n.w[ei] += 1.0;
        }
        let &(parent, edge) = match pending.path.last() {
            Some(last) => last,
            None => {
                // Root evaluation of a fresh tree.
                let k = pending.moves.len();
                self.tree.nodes.push(Node {
                    moves: pending.moves,
                    prior: res.priors,
                    n: vec![0; k],
                    w: vec![0.0; k],
                    child: vec![-1; k],
                    value: res.value,
                    terminal: false,
                });
                self.tree.root = self.tree.nodes.len() - 1;
                return;
            }
        };
        if self.tree.nodes[parent].child[edge] < 0 {
            let k = pending.moves.len();
            self.tree.nodes.push(Node {
                moves: pending.moves,
                prior: res.priors,
                n: vec![0; k],
                w: vec![0.0; k],
                child: vec![-1; k],
                value: res.value,
                terminal: false,
            });
            self.tree.nodes[parent].child[edge] = (self.tree.nodes.len() - 1) as i32;
        }
        self.backup(&pending.path, res.value);
    }

    /// Backs `leaf_value` (perspective of the player to move below the last
    /// path edge) up the path, flipping sign per ply.
    fn backup(&mut self, path: &[(usize, usize)], leaf_value: f32) {
        let depth = path.len();
        for (i, &(node, edge)) in path.iter().enumerate() {
            let plies_from_leaf = depth - i;
            let v = if plies_from_leaf % 2 == 1 {
                -f64::from(leaf_value)
            } else {
                f64::from(leaf_value)
            };
            let n = &mut self.tree.nodes[node];
            n.n[edge] += 1;
            n.w[edge] += v;
        }
    }

    /// Visit counts over the root's moves, aligned with `root_moves`.
    pub fn root_visits(&self) -> &[u32] {
        &self.tree.nodes[self.tree.root].n
    }

    pub fn root_moves(&self) -> &[Move] {
        &self.tree.nodes[self.tree.root].moves
    }

    /// Mean value of the most-visited edge, from the root player's view.
    /// Visit-weighted mean value of the root position (player to move):
    /// the search's estimate of the position itself, for value targets.
    pub fn root_value(&self) -> f64 {
        let root = &self.tree.nodes[self.tree.root];
        let n: u32 = root.n.iter().sum();
        let w: f64 = root.w.iter().sum();
        if n > 0 { w / f64::from(n) } else { 0.0 }
    }

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
    /// move is played. Returns `None` if the child was never expanded.
    pub fn extract_child(self, e: usize) -> Option<Tree> {
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
        let mut old_nodes: Vec<Option<Node>> = self.tree.nodes.into_iter().map(Some).collect();
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

fn terminal_node(v: f32) -> Node {
    Node {
        moves: Vec::new(),
        prior: Vec::new(),
        n: Vec::new(),
        w: Vec::new(),
        child: Vec::new(),
        value: v,
        terminal: true,
    }
}

fn eval_request(b: &Board, moves: &[Move]) -> EvalRequest {
    let support = moves
        .iter()
        .map(|&m| az_move_index(m, b.stm) as u16)
        .collect();
    EvalRequest {
        planes: encode_planes(b),
        support,
    }
}

fn select_edge(node: &Node, (c_puct, fpu): (f32, f32)) -> usize {
    let total = node.visits();
    let sqrt_total = f64::from(total + 1).sqrt();
    let fpu_q = f64::from(node.value) - f64::from(fpu);
    let mut best = 0;
    let mut best_score = f64::NEG_INFINITY;
    for i in 0..node.moves.len() {
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

fn add_dirichlet(node: &mut Node, cfg: &MctsConfig, rng: &mut Rng) {
    if node.prior.len() < 2 {
        return;
    }
    let noise = dirichlet(cfg.dirichlet_alpha, node.prior.len(), rng);
    for (p, n) in node.prior.iter_mut().zip(noise) {
        *p = (1.0 - cfg.root_noise) * *p + cfg.root_noise * n as f32;
    }
}

/// Gamma(alpha) samples normalized to a Dirichlet draw.
fn dirichlet(alpha: f64, k: usize, rng: &mut Rng) -> Vec<f64> {
    let mut g: Vec<f64> = (0..k).map(|_| gamma(alpha, rng)).collect();
    let sum: f64 = g.iter().sum();
    if sum <= 0.0 {
        return vec![1.0 / k as f64; k];
    }
    for v in &mut g {
        *v /= sum;
    }
    g
}

fn gamma(alpha: f64, rng: &mut Rng) -> f64 {
    if alpha < 1.0 {
        let u: f64 = rng.unit().max(1e-12);
        return gamma(alpha + 1.0, rng) * u.powf(1.0 / alpha);
    }
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = normal(rng);
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u: f64 = rng.unit().max(1e-12);
        if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
            return d * v;
        }
    }
}

fn normal(rng: &mut Rng) -> f64 {
    let u1: f64 = rng.unit().max(1e-12);
    let u2: f64 = rng.unit();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::argmax;

    fn drive_with_uniform_net(board: &Board, cfg: &MctsConfig, rng: &mut Rng) -> Search {
        let mut search = Search::new(None);
        let history = HashMap::new();
        let mut results: Vec<EvalResult> = Vec::new();
        loop {
            match search.advance(board, &history, cfg, rng, std::mem::take(&mut results)) {
                Gather::Requests(reqs) => {
                    results = reqs
                        .iter()
                        .map(|r| EvalResult {
                            priors: vec![1.0 / r.support.len() as f32; r.support.len()],
                            value: 0.0,
                        })
                        .collect();
                }
                Gather::Done => return search,
            }
        }
    }

    #[test]
    fn finds_back_rank_mate_with_uniform_net() {
        let b = Board::from_fen("6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1").unwrap();
        let cfg = MctsConfig {
            sims: 256,
            root_noise: 0.0,
            ..MctsConfig::default()
        };
        let mut rng = Rng::new(7);
        let search = drive_with_uniform_net(&b, &cfg, &mut rng);
        let best = search.root_moves()[argmax(search.root_visits())];
        assert_eq!(best, "e1e8".parse().unwrap());
    }

    #[test]
    fn repeated_position_backs_up_draw() {
        // The only winning try for White is Re8#; if the game history says
        // the position after a rook shuffle already occurred, search must
        // treat that branch as a draw, not as fresh territory.
        let b = Board::from_fen("6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1").unwrap();
        let cfg = MctsConfig {
            sims: 128,
            root_noise: 0.0,
            ..MctsConfig::default()
        };
        let mut rng = Rng::new(3);
        // Mark every possible successor as already seen except the mate.
        let mut history = HashMap::new();
        for m in legal_moves(&b) {
            if m != "e1e8".parse().unwrap() {
                let mut nb = b.clone();
                nb.apply(m);
                history.insert(nb.key(), 1);
            }
        }
        let mut search = Search::new(None);
        let mut results: Vec<EvalResult> = Vec::new();
        loop {
            match search.advance(&b, &history, &cfg, &mut rng, std::mem::take(&mut results)) {
                Gather::Requests(reqs) => {
                    results = reqs
                        .iter()
                        .map(|r| EvalResult {
                            priors: vec![1.0 / r.support.len() as f32; r.support.len()],
                            value: 0.0,
                        })
                        .collect();
                }
                Gather::Done => break,
            }
        }
        let best = search.root_moves()[argmax(search.root_visits())];
        assert_eq!(best, "e1e8".parse().unwrap());
    }

    #[test]
    fn extract_child_preserves_subtree() {
        let b = Board::start();
        let cfg = MctsConfig {
            sims: 128,
            root_noise: 0.0,
            ..MctsConfig::default()
        };
        let mut rng = Rng::new(11);
        let search = drive_with_uniform_net(&b, &cfg, &mut rng);
        let choice = argmax(search.root_visits());
        let child_visits = search.root_visits()[choice];
        assert!(child_visits > 1, "best move got visits");
        let tree = search.extract_child(choice).expect("best child expanded");
        let reused = Search::from_tree(tree);
        let total: u32 = reused.root_visits().iter().sum();
        assert_eq!(
            total + 1,
            child_visits,
            "extracted subtree keeps every visit except the expansion eval"
        );
    }
}
