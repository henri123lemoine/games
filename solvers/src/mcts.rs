//! UCT Monte-Carlo tree search for perfect-information games, with optional
//! upgrades that stay out of the way until enabled.
//!
//! Requires only [`Game`]; an optional [`Eval`] turns full random playouts into
//! depth-limited ones (truncate after `max_playout_depth` moves, score the leaf
//! with the eval). Each simulation does the classic four phases: selection down
//! the stored tree, expansion of one untried action, a uniform-random playout,
//! and backpropagation.
//!
//! Multi-player returns are handled without zero-sum negation: every node
//! accumulates the terminal return of the player who *chose the edge into it*
//! (the parent's mover), which is exactly the quantity UCB1 compares when the
//! parent picks among its children. Chance nodes live in the tree too — descent
//! samples an outcome from [`Game::chance_outcomes`] and descends into that
//! outcome's child, so deeper statistics never mix different chance branches
//! (2048-style player-move-then-random-tile games work out of the box).
//!
//! The variants:
//!
//! * **MCTS-Solver** (Winands et al., [`Mcts::solver`], on by default): exact
//!   terminal values are backed up as proofs. A decision node is proven once
//!   one child is a proven win for its mover or all children are proven; a
//!   chance node once all outcomes are proven (probability-weighted). Proven
//!   lines stop descending and back up exact values, a proven root stops the
//!   search, and the final move never picks a proven loss — forced lines are
//!   played perfectly.
//! * **Priors** ([`Mcts::with_spec`]): [`SearchSpec::order_hint`]s, normalized
//!   per node into a prior distribution, switch selection to PUCT-style
//!   `Q + c·P·√N/(1+n)`, concentrating simulations on hinted moves.
//! * **RAVE/AMAF** ([`Mcts::rave`], off by default): all-moves-as-first
//!   statistics blended into UCB1 with the MC-RAVE schedule
//!   `β = ñ/(n + ñ + n·ñ/k)`. Actions are matched across nodes by
//!   [`Game::action_id`].
//! * **Transposition merging** ([`Mcts::transpositions`], off by default):
//!   children that share [`Game::state_key`] *and* perspective are merged, so
//!   statistics pool across move orders. The tree becomes a DAG; descent is
//!   capped at 4096 plies as a guard against key cycles.
//!
//! The tree is an arena (`Vec` of nodes, child links by index); descent clones
//! the root state once per simulation and applies actions along the path.
//! Single-threaded and deterministic given the arena's seed (randomness comes
//! from the rng the [`Agent`] contract supplies).

use game_core::{Agent, Eval, Game, Rng, SearchSpec, Turn};

use crate::FastMap;

const UNEXPANDED: usize = usize::MAX;
const MAX_PATH: usize = 4096;

/// UCT agent over any perfect-information [`Game`]. Picks the most-visited
/// root child (proven wins preferred, proven losses avoided) after `sims`
/// simulations; `c` is the exploration constant (on the `[-1, 1]` returns
/// scale).
pub struct Mcts<G: Game> {
    pub sims: u32,
    pub c: f64,
    /// Back up proven wins/losses from terminal positions (MCTS-Solver).
    pub solver: bool,
    /// Blend AMAF statistics into selection (RAVE). Off by default.
    pub rave: bool,
    /// RAVE equivalence parameter `k`: AMAF and direct estimates carry equal
    /// weight when a child has `k` visits.
    pub rave_k: f64,
    /// Merge children that share [`Game::state_key`]. Off by default.
    pub transpositions: bool,
    eval: Option<Box<dyn Eval<G>>>,
    spec: Option<Box<dyn SearchSpec<G>>>,
    max_playout_depth: u32,
}

enum End {
    Proven(usize),
    Terminal,
    Truncated,
    Aborted,
}

impl<G: Game> Mcts<G> {
    fn base(sims: u32) -> Self {
        Self {
            sims,
            c: std::f64::consts::SQRT_2,
            solver: true,
            rave: false,
            rave_k: 300.0,
            transpositions: false,
            eval: None,
            spec: None,
            max_playout_depth: u32::MAX,
        }
    }

    /// MCTS with full random playouts to terminal states.
    pub fn new(sims: u32) -> Self {
        Self::base(sims)
    }

    /// MCTS with playouts truncated after `max_playout_depth` moves and scored
    /// by `eval` — useful where random playouts are too long or too noisy.
    pub fn with_eval(sims: u32, eval: impl Eval<G> + 'static, max_playout_depth: u32) -> Self {
        let mut m = Self::base(sims);
        m.eval = Some(Box::new(eval));
        m.max_playout_depth = max_playout_depth;
        m
    }

    /// MCTS whose selection is biased by `spec`'s [`SearchSpec::order_hint`]s,
    /// normalized into per-node priors and applied PUCT-style.
    pub fn with_spec(sims: u32, spec: impl SearchSpec<G> + 'static) -> Self {
        let mut m = Self::base(sims);
        m.spec = Some(Box::new(spec));
        m
    }

    /// Per-root-action visit counts after a fresh search, aligned with
    /// [`Game::legal_actions`]. Diagnostic companion to [`Agent::act`].
    pub fn root_visits(
        &self,
        game: &G,
        state: &G::State,
        player: usize,
        rng: &mut Rng,
    ) -> Vec<u32> {
        let nodes = self.run(game, state, player, rng);
        nodes[0]
            .children
            .iter()
            .map(|&ch| {
                if ch == UNEXPANDED {
                    0
                } else {
                    nodes[ch].visits
                }
            })
            .collect()
    }

    fn run(
        &self,
        game: &G,
        state: &G::State,
        player: usize,
        rng: &mut Rng,
    ) -> Vec<Node<G::Action>> {
        let mut nodes = Vec::with_capacity((self.sims as usize).min(1 << 20) + 1);
        let mut tt = FastMap::default();
        nodes.push(self.make_node(game, state, player));
        if self.transpositions
            && let Some(sk) = game.state_key(state)
        {
            tt.insert(merge_key(sk, player), 0);
        }
        for _ in 0..self.sims {
            if nodes[0].proven.is_some() {
                break;
            }
            self.simulate(game, state, &mut nodes, &mut tt, rng);
        }
        nodes
    }

    fn simulate(
        &self,
        game: &G,
        root_state: &G::State,
        nodes: &mut Vec<Node<G::Action>>,
        tt: &mut FastMap<u64, usize>,
        rng: &mut Rng,
    ) {
        let mut state = root_state.clone();
        let mut path = vec![0usize];
        let mut moves: Vec<(usize, usize, u64)> = Vec::new();
        let end = loop {
            let id = *path.last().unwrap();
            if nodes[id].proven.is_some() {
                break End::Proven(id);
            }
            if nodes[id].actions.is_empty() {
                break End::Terminal;
            }
            if path.len() > MAX_PATH {
                break End::Aborted;
            }
            match nodes[id].mover {
                None => {
                    let k = rng.pick(&nodes[id].probs);
                    game.apply(&mut state, nodes[id].actions[k]);
                    if nodes[id].children[k] == UNEXPANDED {
                        let view = nodes[id].view;
                        let cid = self.fresh_or_merged(game, &state, view, nodes, tt);
                        nodes[id].children[k] = cid;
                    }
                    path.push(nodes[id].children[k]);
                }
                Some(p) => {
                    let (k, expanding) = if nodes[id].priors.is_empty() {
                        let untried: Vec<usize> = nodes[id]
                            .children
                            .iter()
                            .enumerate()
                            .filter(|&(_, &c)| c == UNEXPANDED)
                            .map(|(i, _)| i)
                            .collect();
                        if untried.is_empty() {
                            (self.select_ucb(game, nodes, id, p), false)
                        } else {
                            (untried[rng.below(untried.len())], true)
                        }
                    } else {
                        let k = self.select_puct(game, nodes, id, p);
                        (k, nodes[id].children[k] == UNEXPANDED)
                    };
                    if self.rave {
                        moves.push((path.len() - 1, p, game.action_id(&nodes[id].actions[k])));
                    }
                    game.apply(&mut state, nodes[id].actions[k]);
                    if expanding {
                        let cid = self.fresh_or_merged(game, &state, p, nodes, tt);
                        nodes[id].children[k] = cid;
                        path.push(cid);
                        if nodes[cid].visits == 0 && nodes[cid].proven.is_none() {
                            let pos = path.len() - 1;
                            break if self.playout(game, &mut state, rng, &mut moves, pos) {
                                End::Terminal
                            } else {
                                End::Truncated
                            };
                        }
                    } else {
                        path.push(nodes[id].children[k]);
                    }
                }
            }
        };
        let np = game.num_players();
        let values: Vec<f64> = match end {
            End::Proven(id) => nodes[id].proven.as_ref().unwrap().to_vec(),
            End::Terminal => (0..np).map(|p| game.returns(&state, p)).collect(),
            End::Truncated => {
                let eval = self.eval.as_ref().expect("truncated playout without eval");
                (0..np).map(|p| eval.eval(game, &state, p)).collect()
            }
            End::Aborted => vec![0.0; np],
        };
        // Walk the path deepest-first, growing the set of (mover, action)
        // pairs played at-or-below the current node, so each AMAF update is a
        // set lookup instead of a trajectory scan (`moves` is sorted by path
        // position: descent pushes in order, the playout appends at the tail).
        let mut played: std::collections::HashSet<(usize, u64)> = std::collections::HashSet::new();
        let mut moves_idx = moves.len();
        for i in (0..path.len()).rev() {
            let id = path[i];
            nodes[id].visits += 1;
            nodes[id].value += values[nodes[id].view];
            if self.rave {
                while moves_idx > 0 && moves[moves_idx - 1].0 >= i {
                    moves_idx -= 1;
                    let (_, mv, k) = moves[moves_idx];
                    played.insert((mv, k));
                }
                update_amaf(&mut nodes[id], &played, &values);
            }
            if self.solver {
                try_prove(nodes, id, game.max_return());
            }
        }
    }

    /// UCB1 over fully-expanded children: mean (optionally RAVE-blended) plus
    /// exploration. Proven children use their exact value as the mean (still
    /// explored, so a proven draw is not starved of the visits the final
    /// most-visited pick compares); a proven win short-circuits.
    fn select_ucb(&self, game: &G, nodes: &[Node<G::Action>], id: usize, p: usize) -> usize {
        let node = &nodes[id];
        let ln_n = f64::ln(node.visits.max(1) as f64);
        let mut best = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (i, &ch) in node.children.iter().enumerate() {
            let child = &nodes[ch];
            let n = child.visits as f64;
            let score = if let Some(v) = &child.proven {
                if v[p] >= game.max_return() {
                    f64::INFINITY
                } else {
                    v[p] + self.c * (ln_n / n.max(1.0)).sqrt()
                }
            } else {
                let mut q = child.value / n;
                if !node.amaf.is_empty() {
                    let (an, av) = node.amaf[i];
                    if an > 0 {
                        let an = an as f64;
                        let beta = an / (n + an + n * an / self.rave_k);
                        q = (1.0 - beta) * q + beta * av / an;
                    }
                }
                q + self.c * (ln_n / n).sqrt()
            };
            if score > best_score {
                best_score = score;
                best = i;
            }
        }
        best
    }

    /// PUCT over all children (unexpanded ones included at `Q = 0`), driven by
    /// the node's normalized order-hint priors.
    fn select_puct(&self, game: &G, nodes: &[Node<G::Action>], id: usize, p: usize) -> usize {
        let node = &nodes[id];
        let sqrt_n = (node.visits.max(1) as f64).sqrt();
        let mut best = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (i, &ch) in node.children.iter().enumerate() {
            let score = if ch == UNEXPANDED {
                self.c * node.priors[i] * sqrt_n
            } else if let Some(v) = &nodes[ch].proven {
                if v[p] >= game.max_return() {
                    f64::INFINITY
                } else {
                    let n = nodes[ch].visits as f64;
                    v[p] + self.c * node.priors[i] * sqrt_n / (1.0 + n)
                }
            } else {
                let n = nodes[ch].visits as f64;
                nodes[ch].value / n.max(1.0) + self.c * node.priors[i] * sqrt_n / (1.0 + n)
            };
            if score > best_score {
                best_score = score;
                best = i;
            }
        }
        best
    }

    fn fresh_or_merged(
        &self,
        game: &G,
        state: &G::State,
        view: usize,
        nodes: &mut Vec<Node<G::Action>>,
        tt: &mut FastMap<u64, usize>,
    ) -> usize {
        if self.transpositions
            && let Some(sk) = game.state_key(state)
        {
            let key = merge_key(sk, view);
            if let Some(&nid) = tt.get(&key) {
                return nid;
            }
            let nid = nodes.len();
            nodes.push(self.make_node(game, state, view));
            tt.insert(key, nid);
            return nid;
        }
        let nid = nodes.len();
        nodes.push(self.make_node(game, state, view));
        nid
    }

    fn make_node(&self, game: &G, state: &G::State, view: usize) -> Node<G::Action> {
        let terminal = game.is_terminal(state);
        let (actions, probs, mover) = if terminal {
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
        let priors = match (self.spec.as_deref(), mover) {
            (Some(spec), Some(_)) => normalized_hints(
                &actions
                    .iter()
                    .map(|&a| spec.order_hint(game, state, a))
                    .collect::<Vec<_>>(),
            ),
            _ => Vec::new(),
        };
        let (amaf, amaf_keys) = if self.rave && mover.is_some() {
            (
                vec![(0u32, 0.0); actions.len()],
                actions.iter().map(|a| game.action_id(a)).collect(),
            )
        } else {
            (Vec::new(), Vec::new())
        };
        let proven = (self.solver && terminal).then(|| {
            (0..game.num_players())
                .map(|p| game.returns(state, p))
                .collect()
        });
        Node {
            children: vec![UNEXPANDED; actions.len()],
            actions,
            probs,
            priors,
            amaf,
            amaf_keys,
            mover,
            view,
            visits: 0,
            value: 0.0,
            proven,
        }
    }

    /// Uniform-random playout. Returns whether a terminal state was reached
    /// (`false` only when an [`Eval`] cutoff truncated the playout).
    fn playout(
        &self,
        game: &G,
        state: &mut G::State,
        rng: &mut Rng,
        moves: &mut Vec<(usize, usize, u64)>,
        pos: usize,
    ) -> bool {
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
                Turn::Player(p) => {
                    let actions = game.legal_actions(state);
                    let a = actions[rng.below(actions.len())];
                    if self.rave {
                        moves.push((pos, p, game.action_id(&a)));
                    }
                    game.apply(state, a);
                }
            }
            depth += 1;
        }
    }
}

impl<G: Game> Agent<G> for Mcts<G> {
    fn act(&self, game: &G, state: &G::State, player: usize, rng: &mut Rng) -> usize {
        if game.legal_actions(state).len() <= 1 {
            return 0;
        }
        let nodes = self.run(game, state, player, rng);
        let mut best = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (i, &ch) in nodes[0].children.iter().enumerate() {
            let score = if ch == UNEXPANDED {
                0.0
            } else if let Some(v) = &nodes[ch].proven {
                v[player] * 1e12 + nodes[ch].visits as f64
            } else {
                nodes[ch].visits as f64
            };
            if score > best_score {
                best_score = score;
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
/// `proven`, when set, is the exact returns vector for all players.
struct Node<A> {
    actions: Vec<A>,
    probs: Vec<f64>,
    priors: Vec<f64>,
    children: Vec<usize>,
    amaf: Vec<(u32, f64)>,
    amaf_keys: Vec<u64>,
    mover: Option<usize>,
    view: usize,
    visits: u32,
    value: f64,
    proven: Option<Box<[f64]>>,
}

/// Winands-style proof backup. A decision node is proven by one child that is
/// a proven win for its mover (a return at [`Game::max_return`] — nothing
/// better exists), or by all children proven (take the mover's best); a chance
/// node by all outcomes proven (probability-weighted expectation, exact).
fn try_prove<A: Copy>(nodes: &mut [Node<A>], id: usize, max_return: f64) {
    if nodes[id].proven.is_some() || nodes[id].actions.is_empty() {
        return;
    }
    let proven = match nodes[id].mover {
        Some(p) => {
            let mut best: Option<Box<[f64]>> = None;
            let mut all = true;
            for ci in 0..nodes[id].children.len() {
                let ch = nodes[id].children[ci];
                let pv = if ch == UNEXPANDED {
                    None
                } else {
                    nodes[ch].proven.clone()
                };
                match pv {
                    Some(v) if v[p] >= max_return => {
                        nodes[id].proven = Some(v);
                        return;
                    }
                    Some(v) => {
                        if best.as_ref().is_none_or(|b| v[p] > b[p]) {
                            best = Some(v);
                        }
                    }
                    None => all = false,
                }
            }
            if all { best } else { None }
        }
        None => {
            let mut acc: Option<Vec<f64>> = None;
            for ci in 0..nodes[id].children.len() {
                let ch = nodes[id].children[ci];
                if ch == UNEXPANDED {
                    return;
                }
                let Some(v) = nodes[ch].proven.clone() else {
                    return;
                };
                let w = nodes[id].probs[ci];
                let sum = acc.get_or_insert_with(|| vec![0.0; v.len()]);
                for (s, x) in sum.iter_mut().zip(v.iter()) {
                    *s += w * x;
                }
            }
            acc.map(Vec::into_boxed_slice)
        }
    };
    if proven.is_some() {
        nodes[id].proven = proven;
    }
}

/// AMAF update: credit every of the node's actions that its mover played at or
/// after this node during the simulation (`played` holds those pairs).
fn update_amaf<A>(
    node: &mut Node<A>,
    played: &std::collections::HashSet<(usize, u64)>,
    values: &[f64],
) {
    let Some(p) = node.mover else { return };
    for j in 0..node.amaf_keys.len() {
        if played.contains(&(p, node.amaf_keys[j])) {
            node.amaf[j].0 += 1;
            node.amaf[j].1 += values[p];
        }
    }
}

/// Shift-to-positive normalization of order hints into a prior distribution:
/// `(h - min + 1) / Σ`. Equal hints give the uniform prior.
fn normalized_hints(hints: &[i64]) -> Vec<f64> {
    let Some(&m) = hints.iter().min() else {
        return Vec::new();
    };
    let w: Vec<f64> = hints.iter().map(|&h| (h as f64 - m as f64) + 1.0).collect();
    let s: f64 = w.iter().sum();
    w.into_iter().map(|x| x / s).collect()
}

fn merge_key(state_key: u64, view: usize) -> u64 {
    state_key ^ (view as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}
