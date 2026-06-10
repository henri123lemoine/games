//! Depth-limited adversarial search for two-player, perfect-information,
//! zero-sum games: iterative-deepening negamax with alpha-beta pruning and
//! optional quiescence.
//!
//! The game supplies its knowledge through capability traits: [`game_core::Eval`]
//! (a static value) makes search possible at all, and an optional [`SearchSpec`]
//! sharpens it (move ordering, and which actions are "noisy" enough that the
//! horizon should be extended over them — captures/promotions in chess).
//!
//! Chance nodes are not supported (perfect information only); games need not
//! strictly alternate turns — the sign flip follows whose turn it actually is.

use std::marker::PhantomData;

use game_core::{Agent, Eval, Game, SearchSpec, Turn};

const INF: f64 = f64::INFINITY;
/// Terminal utilities are scaled by (MATE - ply) so faster wins score higher.
const MATE: f64 = 1.0e9;

/// Iterative-deepening negamax with alpha-beta over any `Game + Eval`.
/// Deterministic: the arena's tie-break `r` is ignored.
pub struct AlphaBeta<G: Game, E: Eval<G>, S: SearchSpec<G>> {
    pub depth: u32,
    pub eval: E,
    pub spec: S,
    _g: PhantomData<fn(G)>,
}

impl<G: Game, E: Eval<G>, S: SearchSpec<G>> AlphaBeta<G, E, S> {
    pub fn new(depth: u32, eval: E, spec: S) -> Self {
        assert!(depth >= 1, "search depth must be at least 1");
        Self {
            depth,
            eval,
            spec,
            _g: PhantomData,
        }
    }

    fn mover(&self, game: &G, state: &G::State) -> usize {
        match game.turn(state) {
            Turn::Player(p) => p,
            Turn::Chance => unreachable!("AlphaBeta does not support chance nodes"),
        }
    }

    /// Child value converted to the parent mover's perspective.
    #[allow(clippy::too_many_arguments)]
    fn child_value(
        &self,
        game: &G,
        child: &G::State,
        parent_mover: usize,
        depth: u32,
        alpha: f64,
        beta: f64,
        ply: u32,
    ) -> f64 {
        if game.is_terminal(child) {
            return game.returns(child, parent_mover) * (MATE - ply as f64);
        }
        let child_mover = self.mover(game, child);
        if child_mover == parent_mover {
            self.negamax(game, child, depth, alpha, beta, ply)
        } else {
            -self.negamax(game, child, depth, -beta, -alpha, ply)
        }
    }

    fn ordered_actions(&self, game: &G, state: &G::State) -> Vec<G::Action> {
        let mut actions = game.legal_actions(state);
        actions.sort_by_key(|&a| -self.spec.order_hint(game, state, a));
        actions
    }

    /// Value of `state` (non-terminal) for its mover.
    fn negamax(
        &self,
        game: &G,
        state: &G::State,
        depth: u32,
        mut alpha: f64,
        beta: f64,
        ply: u32,
    ) -> f64 {
        let mover = self.mover(game, state);
        if depth == 0 {
            return self.quiesce(game, state, alpha, beta, ply);
        }
        let mut best = -INF;
        for a in self.ordered_actions(game, state) {
            let mut child = state.clone();
            game.apply(&mut child, a);
            let score = self.child_value(game, &child, mover, depth - 1, alpha, beta, ply + 1);
            if score > best {
                best = score;
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                break;
            }
        }
        best
    }

    /// Search only "noisy" actions past the horizon, standing pat on the eval.
    fn quiesce(&self, game: &G, state: &G::State, mut alpha: f64, beta: f64, ply: u32) -> f64 {
        let mover = self.mover(game, state);
        let stand = self.eval.eval(game, state, mover);
        if stand >= beta {
            return stand;
        }
        if stand > alpha {
            alpha = stand;
        }
        let mut best = stand;
        for a in self.ordered_actions(game, state) {
            if !self.spec.is_noisy(game, state, a) {
                continue;
            }
            let mut child = state.clone();
            game.apply(&mut child, a);
            let score = if game.is_terminal(&child) {
                game.returns(&child, mover) * (MATE - ply as f64)
            } else if self.mover(game, &child) == mover {
                self.quiesce(game, &child, alpha, beta, ply + 1)
            } else {
                -self.quiesce(game, &child, -beta, -alpha, ply + 1)
            };
            if score > best {
                best = score;
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                break;
            }
        }
        best
    }

    /// Index (into `legal_actions`) of the best move by iterative deepening.
    pub fn best_action(&self, game: &G, state: &G::State) -> usize {
        let mover = self.mover(game, state);
        let actions = game.legal_actions(state);
        assert!(!actions.is_empty(), "best_action on a terminal state");
        // Search a permutation of original indices (so the return value needs
        // no mapping); rotate the incumbent to the front each deepening round
        // so it seeds alpha.
        let mut order: Vec<usize> = (0..actions.len()).collect();
        order.sort_by_key(|&i| -self.spec.order_hint(game, state, actions[i]));
        let mut best = order[0];
        for depth in 1..=self.depth {
            if let Some(pos) = order.iter().position(|&i| i == best) {
                order[..=pos].rotate_right(1);
            }
            let mut alpha = -INF;
            let mut best_this = order[0];
            for &i in &order {
                let mut child = state.clone();
                game.apply(&mut child, actions[i]);
                let score = self.child_value(game, &child, mover, depth - 1, alpha, INF, 1);
                if score > alpha {
                    alpha = score;
                    best_this = i;
                }
            }
            best = best_this;
        }
        best
    }
}

impl<G: Game, E: Eval<G>, S: SearchSpec<G>> Agent<G> for AlphaBeta<G, E, S> {
    fn act(&self, game: &G, state: &G::State, _player: usize, _r: f64) -> usize {
        self.best_action(game, state)
    }
}
