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
//!
//! An optional [`TerminalProver`] (`advance`'s `prover`) adds a KataGo-style
//! MCTS-solver (Winands et al.): terminal and proven leaves back up exact
//! verdicts, a node is proven once its children force one, proven subtrees are
//! never re-explored, and a proven root ends the search. Strictly opt-in — with
//! no prover, every proof path is inert and the search is byte-for-byte the
//! prover-free one.

use game_core::rand::dirichlet;
use game_core::{Game, PolicyValueEncoder, Proof, Rng, TerminalProver, Turn};

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
    /// KataGo forced playouts: at the root, every child that has been visited
    /// at least once is forced up to `sqrt(k · prior · total_root_visits)`
    /// visits, widening exploration of plausible moves. 0 disables it (the
    /// default everywhere except Go self-play). Pair with policy-target
    /// pruning at the call site, which subtracts these forced visits back out
    /// of the recorded policy target.
    pub forced_playouts_k: f32,
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
            forced_playouts_k: 0.0,
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
    /// Exact return to player 0 for terminal nodes and for proven nodes (set
    /// when `proven` becomes `Some`); 0 otherwise.
    value0: f64,
    terminal: bool,
    /// A proven game-theoretic verdict for this node, from the perspective of
    /// the player to move here (the MCTS-solver). Terminal nodes are proven
    /// from their exact result; interior nodes become proven when their
    /// children's proofs force it. `None` until/unless proven; always `None`
    /// when no [`TerminalProver`] drives the search, so behavior is unchanged.
    proven: Option<Proof>,
    /// For a proven interior node, the edge index whose child witnesses the
    /// proof: a child proving the win to play (proven Win), or — when lost or
    /// drawn — a child realizing the best achievable outcome, for move
    /// extraction. Unused for terminal nodes.
    proof_edge: usize,
}

impl<G: Game> Node<G> {
    fn visits(&self) -> u32 {
        self.n.iter().sum()
    }
}

/// Exact return to player 0 of a node proven `proof` for the given `to_move`
/// player. `max` is [`Game::max_return`]; a win for the mover is `+max` from
/// their seat, mapped to player 0's view.
fn proven_value0(proof: Proof, to_move: usize, max: f64) -> f64 {
    let from_mover = match proof {
        Proof::Win => max,
        Proof::Loss => -max,
        Proof::Draw => 0.0,
    };
    if to_move == 0 {
        from_mover
    } else {
        -from_mover
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
    /// Whether the MCTS-solver is live for this search: set once `advance` is
    /// called with a `Some` prover. When false, every proof
    /// code path is inert and the search is byte-for-byte the prover-free one,
    /// even on a subtree reused from a search that did run the solver.
    solver_active: bool,
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
            solver_active: false,
        }
    }

    /// Resolves `results` (aligned with the previous `Gather::Requests`),
    /// then gathers the next batch of leaves or finishes. `seen` answers
    /// "did this repetition key already occur in the game?" (only consulted
    /// when `cycle_draws` is on; pass `&|_| false` otherwise).
    ///
    /// With an optional MCTS-solver: when `prover` is `Some`, every freshly
    /// expanded non-terminal leaf is offered to it, and a proven verdict is
    /// treated exactly like a terminal one — backed up as an exact value and
    /// propagated up the proof tree (a parent becomes proven once its children
    /// force it). With `prover = None` this drives the search with no terminal
    /// solver: byte-for-byte identical to passing a [`game_core::NoProver`] —
    /// no proof status is ever set and nothing differs.
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
        prover: Option<&dyn TerminalProver<G>>,
    ) -> Gather {
        debug_assert_eq!(results.len(), self.pending.len(), "results align");
        self.solver_active |= prover.is_some();
        for (pending, res) in std::mem::take(&mut self.pending).into_iter().zip(results) {
            self.resolve(pending, res, game, prover);
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
        // budget. A proven root is the same: the verdict is exact, so there is
        // nothing left to search and the caller can stop (solver only).
        if self.tree.nodes[self.tree.root].terminal
            || (self.solver_active && self.tree.nodes[self.tree.root].proven.is_some())
        {
            return Gather::Done;
        }
        if !self.noised && cfg.root_noise > 0.0 {
            add_dirichlet(&mut self.tree.nodes[self.tree.root], cfg, rng);
            self.noised = true;
        }

        let mut requests = Vec::new();
        while self.tree.nodes[self.tree.root].visits() < cfg.sims
            && !(self.solver_active && self.tree.nodes[self.tree.root].proven.is_some())
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
            // A terminal node — or, under the solver, any proven node — is an
            // exact leaf: back up its value and stop. Never descend into a
            // proven subtree (its verdict is already settled).
            if node.terminal || (self.solver_active && node.proven.is_some()) {
                let v = node.value0;
                self.backup(&path, Leaf::Exact(v));
                return None;
            }
            let forced_k = if cur == self.tree.root {
                cfg.forced_playouts_k
            } else {
                0.0
            };
            let e = self.select_edge_for(cur, (cfg.c_puct, cfg.fpu), forced_k);
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
                // A newly discovered terminal is the solver's entry point:
                // propagate its proof up the descent path.
                if self.solver_active {
                    self.solver_backup(&path, game.max_return());
                }
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

    fn resolve(
        &mut self,
        mut pending: Pending<G>,
        res: EvalResult,
        game: &G,
        prover: Option<&dyn TerminalProver<G>>,
    ) {
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
        // Ask the prover about this freshly expanded non-terminal leaf, while
        // its state is still in hand. `None`-prover searches skip this and
        // never set any proof. A directly proven Win carries its witnessing
        // move, whose edge index in the node's own actions pins `proof_edge` so
        // `best_proven_action` is correct for directly proven roots/leaves too.
        let proof = prover.and_then(|p| {
            p.prove(game, &pending.state).map(|(pr, win)| {
                let edge = win.and_then(|a| pending.actions.iter().position(|&x| x == a));
                (pr, pending.to_move, edge)
            })
        });
        let &(parent, edge) = match path.last() {
            Some(last) => last,
            None => {
                // Root evaluation of a fresh tree.
                let to_move = pending.to_move;
                self.tree.nodes.push(expanded_node(pending, res));
                self.tree.root = self.tree.nodes.len() - 1;
                if let Some((pr, _, win_edge)) = proof {
                    let root = self.tree.root;
                    self.mark_proven(root, pr, to_move, game.max_return());
                    if let Some(e) = win_edge {
                        self.tree.nodes[root].proof_edge = e;
                    }
                }
                return;
            }
        };
        let newly_created = self.tree.nodes[parent].child[edge] < 0;
        if newly_created {
            let idx = self.tree.nodes.len();
            self.tree.nodes.push(expanded_node(pending, res));
            self.tree.nodes[parent].child[edge] = idx as i32;
        }
        self.backup(&path, leaf);
        // Prove and propagate only on first creation: a sibling resolve in the
        // same batch may have already created (and proven) this child.
        if newly_created && let Some((pr, to_move, win_edge)) = proof {
            let child = self.tree.nodes[parent].child[edge] as usize;
            self.mark_proven(child, pr, to_move, game.max_return());
            if let Some(e) = win_edge {
                self.tree.nodes[child].proof_edge = e;
            }
            self.solver_backup(&path, game.max_return());
        }
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

    /// Records a proven verdict on `node` (mover `to_move`) and pins its exact
    /// player-0 value so the solver and ordinary backups agree on it.
    fn mark_proven(&mut self, node: usize, proof: Proof, to_move: usize, max: f64) {
        let n = &mut self.tree.nodes[node];
        n.proven = Some(proof);
        n.value0 = proven_value0(proof, to_move, max);
    }

    /// Propagates proofs up `path` (leaf-most parent first): a node becomes
    /// proven once its children force the verdict (Winands et al.). Stops at the
    /// first node whose status does not change — nothing above it can flip
    /// either. The proven node's exact `value0` is pinned for ordinary backups.
    fn solver_backup(&mut self, path: &[(usize, usize)], max: f64) {
        for &(ni, _) in path.iter().rev() {
            if self.tree.nodes[ni].proven.is_some() {
                continue;
            }
            let Some((proof, edge)) = self.recompute_proof(ni) else {
                return;
            };
            let to_move = self.tree.nodes[ni].to_move;
            self.mark_proven(ni, proof, to_move, max);
            self.tree.nodes[ni].proof_edge = edge;
        }
    }

    /// The MCTS-solver verdict for an interior node from its children's proofs,
    /// with a witnessing edge for move extraction, or `None` if not yet forced.
    ///
    /// Mover M (`node.to_move`): **Win** if any child is a forced win for M
    /// (that child is a loss for *its* mover, the opponent). Otherwise, only if
    /// every child edge is expanded and proven: **Loss** if all lose for M,
    /// else **Draw** (no win, ≥1 drawing child, the rest losing or drawn).
    fn recompute_proof(&self, node: usize) -> Option<(Proof, usize)> {
        let n = &self.tree.nodes[node];
        let to_move = n.to_move;
        let mut all_proven = true;
        let mut draw_edge: Option<usize> = None;
        let mut loss_edge = 0;
        for (e, &c) in n.child.iter().enumerate() {
            if c < 0 {
                all_proven = false;
                continue;
            }
            let child = &self.tree.nodes[c as usize];
            let Some(_) = child.proven else {
                all_proven = false;
                continue;
            };
            // Child value from M's seat: +max is a win for M.
            let for_m = if to_move == 0 {
                child.value0
            } else {
                -child.value0
            };
            if for_m > 0.0 {
                return Some((Proof::Win, e));
            } else if for_m == 0.0 {
                draw_edge.get_or_insert(e);
            } else {
                loss_edge = e;
            }
        }
        if !all_proven {
            return None;
        }
        // No winning move and every child settled: drawn if a draw is on offer,
        // otherwise all moves lose.
        match draw_edge {
            Some(e) => Some((Proof::Draw, e)),
            None => Some((Proof::Loss, loss_edge)),
        }
    }

    /// Selection at `node`. Without the solver this is plain [`select_edge`].
    /// With it, a child proven a win for *us* (a loss for the child's mover) is
    /// taken at once; a child proven a win for the opponent scores `Q = -1` so
    /// it is shunned; unproven children keep their PUCT score.
    fn select_edge_for(&self, node: usize, puct: (f32, f32), forced_k: f32) -> usize {
        let n = &self.tree.nodes[node];
        if !self.solver_active {
            return select_edge(n, puct, forced_k);
        }
        let to_move = n.to_move;
        // First pass: a proven win for us is taken outright (same edge the
        // override path would have, since a win short-circuits); otherwise note
        // whether any child is proven at all. With none proven (the common case
        // even under an active solver) this is plain PUCT — no override vector.
        let mut any_proven = false;
        for (e, &c) in n.child.iter().enumerate() {
            if c < 0 {
                continue;
            }
            let child = &self.tree.nodes[c as usize];
            if child.proven.is_none() {
                continue;
            }
            let for_m = if to_move == 0 {
                child.value0
            } else {
                -child.value0
            };
            if for_m > 0.0 {
                return e; // proven win for us — take it immediately
            }
            any_proven = true;
        }
        if !any_proven {
            return select_edge(n, puct, forced_k);
        }
        // ≥1 child proven (loss/draw for us): shun it via a Q override.
        let mut overrides: Vec<Option<f64>> = vec![None; n.child.len()];
        for (e, &c) in n.child.iter().enumerate() {
            if c < 0 {
                continue;
            }
            let child = &self.tree.nodes[c as usize];
            if child.proven.is_none() {
                continue;
            }
            let for_m = if to_move == 0 {
                child.value0
            } else {
                -child.value0
            };
            overrides[e] = Some(if for_m < 0.0 { -1.0 } else { 0.0 });
        }
        select_edge_solver(n, puct, forced_k, &overrides)
    }

    /// Whether the root node exists yet — false on a fresh search before the
    /// first `advance` allocates it. Callers that may read the root mid-search
    /// (an anytime/best-so-far read) must check this first.
    pub fn has_root(&self) -> bool {
        !self.tree.nodes.is_empty()
    }

    /// Visit counts over the root's actions, aligned with `root_actions`.
    pub fn root_visits(&self) -> &[u32] {
        &self.tree.nodes[self.tree.root].n
    }

    pub fn root_actions(&self) -> &[G::Action] {
        &self.tree.nodes[self.tree.root].actions
    }

    /// Net priors over the root's actions, aligned with `root_actions` /
    /// `root_visits`. Lets the caller reconstruct each child's forced-playout
    /// count for policy-target pruning.
    pub fn root_priors(&self) -> &[f32] {
        &self.tree.nodes[self.tree.root].prior
    }

    /// The solver's verdict at the root, if it has been proven, from the root
    /// player's perspective. `None` whenever no solver ran or the root is still
    /// unsettled — fall back to [`Search::root_visits`] + [`argmax`] then.
    pub fn root_proof(&self) -> Option<Proof> {
        if !self.solver_active {
            return None;
        }
        self.tree.nodes[self.tree.root].proven
    }

    /// Index (into `root_actions`) of the root's proof-witnessing move once the
    /// root is proven: the move that forces the win, or — when lost or drawn —
    /// one realizing the proven outcome. `None` if the root is not proven.
    /// Play this in preference to the visit argmax when it is `Some`.
    pub fn best_proven_action(&self) -> Option<usize> {
        if !self.solver_active {
            return None;
        }
        let root = &self.tree.nodes[self.tree.root];
        root.proven.map(|_| root.proof_edge)
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
    // A terminal node is proven by definition. Its proof tag is recorded from
    // player 0's seat (the win/loss/draw `value0` encodes); the only consumers
    // of a *child's* proof — solver backup and selection — read its `value0`
    // relative to the parent's mover, so the tag's seat does not matter here.
    let proven = Some(if value0 > 0.0 {
        Proof::Win
    } else if value0 < 0.0 {
        Proof::Loss
    } else {
        Proof::Draw
    });
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
        proven,
        proof_edge: 0,
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
        proven: None,
        proof_edge: 0,
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

/// Selection bonus that lifts a forced child above every non-forced one;
/// far larger than any PUCT score (q+u is O(1)), so forced children win and
/// their PUCT scores only break ties among themselves.
const FORCED_PLAYOUT_BONUS: f64 = 1e9;

fn select_edge<G: Game>(node: &Node<G>, (c_puct, fpu): (f32, f32), forced_k: f32) -> usize {
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
        let mut score = q + u;
        // Forced playouts (root only, forced_k > 0): a visited child below its
        // forced floor jumps the queue; the PUCT score breaks ties among forced
        // children so the search still prioritizes the better ones.
        if forced_k > 0.0 && node.n[i] >= 1 {
            let n_forced =
                (f64::from(forced_k) * f64::from(node.prior[i]) * f64::from(total)).sqrt();
            if f64::from(node.n[i]) < n_forced {
                score += FORCED_PLAYOUT_BONUS;
            }
        }
        if score > best_score {
            best_score = score;
            best = i;
        }
    }
    best
}

/// [`select_edge`] with proof overrides: `q_override[i] = Some(q)` forces edge
/// `i`'s action value (a proven-loss child to `-1`, a proven-draw child to `0`),
/// otherwise PUCT's usual Q applies. Identical to [`select_edge`] when every
/// override is `None`, so this is only ever reached under an active solver.
fn select_edge_solver<G: Game>(
    node: &Node<G>,
    (c_puct, fpu): (f32, f32),
    forced_k: f32,
    q_override: &[Option<f64>],
) -> usize {
    let total = node.visits();
    let sqrt_total = f64::from(total + 1).sqrt();
    let fpu_q = f64::from(node.value) - f64::from(fpu);
    let mut best = 0;
    let mut best_score = f64::NEG_INFINITY;
    for (i, &override_q) in q_override.iter().enumerate() {
        let q = match override_q {
            Some(forced) => forced,
            None if node.n[i] > 0 => node.w[i] / f64::from(node.n[i]),
            None => fpu_q,
        };
        let u = f64::from(c_puct) * f64::from(node.prior[i]) * sqrt_total
            / (1.0 + f64::from(node.n[i]));
        let mut score = q + u;
        if forced_k > 0.0 && node.n[i] >= 1 {
            let n_forced =
                (f64::from(forced_k) * f64::from(node.prior[i]) * f64::from(total)).sqrt();
            if f64::from(node.n[i]) < n_forced {
                score += FORCED_PLAYOUT_BONUS;
            }
        }
        if score > best_score {
            best_score = score;
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
