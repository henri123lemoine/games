//! Depth-limited adversarial search for two-player, perfect-information,
//! zero-sum games: iterative-deepening negamax with alpha-beta pruning and
//! optional quiescence.
//!
//! The game supplies its knowledge through capability traits: [`game_core::Eval`]
//! (a static value) makes search possible at all, and an optional [`SearchSpec`]
//! sharpens it (move ordering, and which actions are "noisy" enough that the
//! horizon should be extended over them — captures/promotions in chess).
//!
//! On top of the base search, three classic enhancements are available, each
//! individually toggleable (all default on):
//!
//! * **Transposition table** ([`AlphaBeta::use_tt`]) — keyed by
//!   [`Game::state_key`] (skipped for games that return `None`), storing the
//!   remaining depth, a bound flag, the score, and the best-move index, which
//!   is searched first on revisits.
//! * **Killer moves + history heuristic** ([`AlphaBeta::use_killers`]) — two
//!   killer slots per ply and a cutoff-frequency history table, layered over
//!   [`SearchSpec::order_hint`] as a tie-break for quiet (non-noisy) moves.
//!   Actions carry no `Eq`/`Hash` bound, so moves are identified by a hash of
//!   their `Debug` rendering.
//! * **Aspiration windows** ([`AlphaBeta::use_aspiration`]) — root searches
//!   after the first two deepening rounds open with a narrow window around the
//!   previous score (sized from the score's drift between rounds, so it adapts
//!   to each game's eval scale) and re-search with a full window on failure.
//!
//! Null-move pruning is deliberately **not** implemented: it requires handing
//! the opponent a "pass", and the [`Game`] trait has no generic null/pass
//! action — a game-agnostic version cannot exist. Games whose rules include a
//! pass already get the equivalent effect through the non-alternating-turn
//! handling below.
//!
//! Chance nodes are not supported (perfect information only); games need not
//! strictly alternate turns — the sign flip follows whose turn it actually is.

use std::fmt::{self, Write as _};
use std::hash::Hasher;
use std::marker::PhantomData;
use std::sync::Mutex;
use std::time::Duration;

use web_time::Instant;

use game_core::{Agent, Eval, Game, SearchSpec, Turn};

use crate::{FastMap, FxHasher};

const INF: f64 = f64::INFINITY;
/// Terminal utilities are scaled by (MATE - ply) so faster wins score higher.
const MATE: f64 = 1.0e9;
/// Scores beyond this are mate-like and stored ply-adjusted in the TT.
const MATE_THRESHOLD: f64 = MATE * 0.5;
/// Depth cap for time-budgeted iterative deepening.
const TIME_MAX_DEPTH: u32 = 64;
/// The TT is cleared at the start of a search once it exceeds this many entries.
const TT_CLEAR_LEN: usize = 1 << 21;
/// Deadline is polled once per this many nodes.
const TIME_CHECK_MASK: u64 = 0x3FF;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy)]
struct TtEntry {
    depth: u32,
    flag: Bound,
    score: f64,
    best: u32,
}

/// Search-internal mutable state, behind a `Mutex` so the engine stays usable
/// through `&self` (the [`Agent`] contract) and shareable across threads.
struct Tables {
    tt: FastMap<u64, TtEntry>,
    killers: Vec<[Option<u64>; 2]>,
    history: FastMap<u64, i64>,
    nodes: u64,
    aborted: bool,
    deadline: Option<Instant>,
}

impl Tables {
    fn new() -> Self {
        Self {
            tt: FastMap::default(),
            killers: Vec::new(),
            history: FastMap::default(),
            nodes: 0,
            aborted: false,
            deadline: None,
        }
    }

    fn begin_search(&mut self, budget: Option<Duration>) {
        self.nodes = 0;
        self.aborted = false;
        self.deadline = budget.map(|b| Instant::now() + b);
        self.killers.clear();
        self.history.values_mut().for_each(|v| *v /= 2);
        if self.tt.len() > TT_CLEAR_LEN {
            self.tt.clear();
        }
    }

    fn tick(&mut self) {
        self.nodes += 1;
        if let Some(dl) = self.deadline
            && self.nodes & TIME_CHECK_MASK == 0
            && Instant::now() >= dl
        {
            self.aborted = true;
        }
    }

    fn note_cutoff(&mut self, ply: u32, sig: u64, depth: u32) {
        let ply = ply as usize;
        if self.killers.len() <= ply {
            self.killers.resize(ply + 1, [None; 2]);
        }
        let slots = &mut self.killers[ply];
        if slots[0] != Some(sig) {
            slots[1] = slots[0];
            slots[0] = Some(sig);
        }
        let h = self.history.entry(sig).or_insert(0);
        *h = h.saturating_add((depth as i64) * (depth as i64));
    }
}

struct SigWriter(FxHasher);
impl fmt::Write for SigWriter {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.0.write(s.as_bytes());
        Ok(())
    }
}

/// Identity hash of an action via its `Debug` rendering (actions carry no
/// `Eq`/`Hash` bound, but `Debug` is required and derived renderings are
/// faithful for the move types games use).
fn action_sig<A: fmt::Debug>(a: &A) -> u64 {
    let mut w = SigWriter(FxHasher::default());
    let _ = write!(w, "{a:?}");
    w.0.finish()
}

fn to_tt_score(score: f64, ply: u32) -> f64 {
    if score > MATE_THRESHOLD {
        score + ply as f64
    } else if score < -MATE_THRESHOLD {
        score - ply as f64
    } else {
        score
    }
}

fn from_tt_score(score: f64, ply: u32) -> f64 {
    if score > MATE_THRESHOLD {
        score - ply as f64
    } else if score < -MATE_THRESHOLD {
        score + ply as f64
    } else {
        score
    }
}

/// Iterative-deepening negamax with alpha-beta over any `Game + Eval`.
/// Deterministic: the arena's tie-break `r` is ignored.
pub struct AlphaBeta<G: Game, E: Eval<G>, S: SearchSpec<G>> {
    pub depth: u32,
    pub eval: E,
    pub spec: S,
    /// Transposition table (needs [`Game::state_key`]; no-op without it).
    pub use_tt: bool,
    /// Killer moves (2/ply) + history heuristic for quiet-move ordering.
    pub use_killers: bool,
    /// Narrow root windows across iterative-deepening rounds.
    pub use_aspiration: bool,
    /// Soft time budget; when set, iterative deepening runs until the deadline
    /// (up to an internal depth cap) instead of to the fixed `depth`.
    pub time_budget: Option<Duration>,
    tables: Mutex<Tables>,
    _g: PhantomData<fn(G)>,
}

impl<G: Game, E: Eval<G>, S: SearchSpec<G>> AlphaBeta<G, E, S> {
    pub fn new(depth: u32, eval: E, spec: S) -> Self {
        assert!(depth >= 1, "search depth must be at least 1");
        Self {
            depth,
            eval,
            spec,
            use_tt: true,
            use_killers: true,
            use_aspiration: true,
            time_budget: None,
            tables: Mutex::new(Tables::new()),
            _g: PhantomData,
        }
    }

    /// Switch to a soft time budget: iterative deepening continues until
    /// `millis` have elapsed (the running round is then abandoned and the last
    /// completed round's move returned), making time-fair comparisons possible.
    pub fn with_time(mut self, millis: u64) -> Self {
        self.time_budget = Some(Duration::from_millis(millis));
        self
    }

    /// Nodes visited by the most recent [`Self::best_action`] call.
    pub fn node_count(&self) -> u64 {
        self.tables.lock().expect("search tables poisoned").nodes
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
        tables: &mut Tables,
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
            self.negamax(game, child, tables, depth, alpha, beta, ply)
        } else {
            -self.negamax(game, child, tables, depth, -beta, -alpha, ply)
        }
    }

    fn ordered_actions(&self, game: &G, state: &G::State) -> Vec<G::Action> {
        let mut actions = game.legal_actions(state);
        actions.sort_by_key(|&a| -self.spec.order_hint(game, state, a));
        actions
    }

    /// Search order over `actions` as indices: TT best move first, then by
    /// `order_hint` descending with killers/history breaking ties among quiet
    /// moves.
    fn order_indices(
        &self,
        game: &G,
        state: &G::State,
        actions: &[G::Action],
        tables: &Tables,
        ply: u32,
        tt_best: Option<u32>,
    ) -> Vec<usize> {
        let killers = if self.use_killers {
            tables
                .killers
                .get(ply as usize)
                .copied()
                .unwrap_or([None; 2])
        } else {
            [None; 2]
        };
        let mut keyed: Vec<(i64, i64, usize)> = actions
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let hint = self.spec.order_hint(game, state, *a);
                let mut tiebreak = 0i64;
                if self.use_killers && !self.spec.is_noisy(game, state, *a) {
                    let sig = action_sig(a);
                    tiebreak = if killers[0] == Some(sig) {
                        i64::MAX / 2
                    } else if killers[1] == Some(sig) {
                        i64::MAX / 4
                    } else {
                        tables.history.get(&sig).copied().unwrap_or(0)
                    };
                }
                (hint, tiebreak, i)
            })
            .collect();
        keyed.sort_by(|a, b| (b.0, b.1).cmp(&(a.0, a.1)));
        let mut order: Vec<usize> = keyed.into_iter().map(|(_, _, i)| i).collect();
        if let Some(b) = tt_best
            && let Some(pos) = order.iter().position(|&i| i == b as usize)
        {
            order[..=pos].rotate_right(1);
        }
        order
    }

    /// Value of `state` (non-terminal) for its mover.
    #[allow(clippy::too_many_arguments)]
    fn negamax(
        &self,
        game: &G,
        state: &G::State,
        tables: &mut Tables,
        depth: u32,
        mut alpha: f64,
        mut beta: f64,
        ply: u32,
    ) -> f64 {
        tables.tick();
        if tables.aborted {
            return 0.0;
        }
        let mover = self.mover(game, state);
        if depth == 0 {
            return self.quiesce(game, state, tables, alpha, beta, ply);
        }

        let skey = if self.use_tt {
            game.state_key(state)
        } else {
            None
        };
        let mut tt_best = None;
        if let Some(k) = skey
            && let Some(e) = tables.tt.get(&k)
        {
            tt_best = Some(e.best);
            if e.depth >= depth {
                let score = from_tt_score(e.score, ply);
                match e.flag {
                    Bound::Exact => return score,
                    Bound::Lower => alpha = alpha.max(score),
                    Bound::Upper => beta = beta.min(score),
                }
                if alpha >= beta {
                    return score;
                }
            }
        }
        let alpha_in = alpha;

        let actions = game.legal_actions(state);
        let order = self.order_indices(game, state, &actions, tables, ply, tt_best);
        let mut best = -INF;
        let mut best_i = order[0];
        for &i in &order {
            let a = actions[i];
            let mut child = state.clone();
            game.apply(&mut child, a);
            let score =
                self.child_value(game, &child, tables, mover, depth - 1, alpha, beta, ply + 1);
            if tables.aborted {
                return 0.0;
            }
            if score > best {
                best = score;
                best_i = i;
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                if self.use_killers && !self.spec.is_noisy(game, state, a) {
                    tables.note_cutoff(ply, action_sig(&a), depth);
                }
                break;
            }
        }

        if let Some(k) = skey {
            let flag = if best <= alpha_in {
                Bound::Upper
            } else if best >= beta {
                Bound::Lower
            } else {
                Bound::Exact
            };
            tables.tt.insert(
                k,
                TtEntry {
                    depth,
                    flag,
                    score: to_tt_score(best, ply),
                    best: best_i as u32,
                },
            );
        }
        best
    }

    /// Search only "noisy" actions past the horizon, standing pat on the eval.
    fn quiesce(
        &self,
        game: &G,
        state: &G::State,
        tables: &mut Tables,
        mut alpha: f64,
        beta: f64,
        ply: u32,
    ) -> f64 {
        tables.tick();
        if tables.aborted {
            return 0.0;
        }
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
                self.quiesce(game, &child, tables, alpha, beta, ply + 1)
            } else {
                -self.quiesce(game, &child, tables, -beta, -alpha, ply + 1)
            };
            if tables.aborted {
                return 0.0;
            }
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

    /// One root pass at `depth` within `(alpha0, beta0)`. Returns the fail-soft
    /// best score and the best original action index.
    #[allow(clippy::too_many_arguments)]
    fn root_round(
        &self,
        game: &G,
        state: &G::State,
        actions: &[G::Action],
        order: &[usize],
        tables: &mut Tables,
        depth: u32,
        alpha0: f64,
        beta0: f64,
    ) -> (f64, usize) {
        let mover = self.mover(game, state);
        let mut alpha = alpha0;
        let mut best = -INF;
        let mut best_i = order[0];
        for &i in order {
            let mut child = state.clone();
            game.apply(&mut child, actions[i]);
            let score = self.child_value(game, &child, tables, mover, depth - 1, alpha, beta0, 1);
            if tables.aborted {
                break;
            }
            if score > best {
                best = score;
                best_i = i;
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta0 {
                break;
            }
        }
        (best, best_i)
    }

    /// Index (into `legal_actions`) of the best move by iterative deepening.
    pub fn best_action(&self, game: &G, state: &G::State) -> usize {
        let actions = game.legal_actions(state);
        assert!(!actions.is_empty(), "best_action on a terminal state");
        let mut tables = self.tables.lock().expect("search tables poisoned");
        tables.begin_search(self.time_budget);
        // Search a permutation of original indices (so the return value needs
        // no mapping); rotate the incumbent to the front each deepening round
        // so it seeds alpha.
        let mut order: Vec<usize> = (0..actions.len()).collect();
        order.sort_by_key(|&i| -self.spec.order_hint(game, state, actions[i]));
        let mut best = order[0];
        let max_depth = if self.time_budget.is_some() {
            TIME_MAX_DEPTH
        } else {
            self.depth
        };
        let mut prev_score: Option<f64> = None;
        let mut window: Option<f64> = None;
        for depth in 1..=max_depth {
            if let Some(dl) = tables.deadline
                && Instant::now() >= dl
            {
                break;
            }
            if let Some(pos) = order.iter().position(|&i| i == best) {
                order[..=pos].rotate_right(1);
            }
            let mut round = None;
            if self.use_aspiration
                && let (Some(p), Some(w)) = (prev_score, window)
                && w > 0.0
                && w.is_finite()
                && p.abs() < MATE_THRESHOLD
            {
                let (v, i) = self.root_round(
                    game,
                    state,
                    &actions,
                    &order,
                    &mut tables,
                    depth,
                    p - w,
                    p + w,
                );
                if tables.aborted {
                    break;
                }
                if v > p - w && v < p + w {
                    round = Some((v, i));
                }
            }
            let (score, best_this) = match round {
                Some(r) => r,
                None => {
                    let r = self.root_round(
                        game,
                        state,
                        &actions,
                        &order,
                        &mut tables,
                        depth,
                        -INF,
                        INF,
                    );
                    if tables.aborted {
                        break;
                    }
                    r
                }
            };
            if let Some(p) = prev_score {
                window = Some(((score - p).abs() * 2.0).max(score.abs() * 0.125));
            }
            prev_score = Some(score);
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
