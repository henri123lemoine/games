//! Test-time search primitives (§5): belief-determinized, depth-`D` batched
//! rollouts under a caller-provided move-net policy, returning the per-rollout
//! leaf value and the root action each rollout was seeded with.
//!
//! This is **not** MCTS. The search itself — n_sample beliefs, the per-action
//! world budget, the magnetic-mirror-descent closed form, and the final sample —
//! lives in the Python trainer (`stratego_trainer/search.py`), which owns the
//! MLX move net. This module supplies the three Rust primitives that search
//! needs and that must run against the verified rules engine:
//!
//! 1. [`RootInfo`] — reads, from the real root board, which opponent pieces are
//!    hidden, their row-major POV rank, whether each has moved, and the per-type
//!    hidden counts. The Rust equivalent of the reference env's
//!    `unknown_piece_{position_onehot,has_moved,counts}` tensors.
//! 2. [`assign_hidden`] — the `assign_opponent_hidden_pieces` equivalent: writes
//!    a sampled type onto each hidden opponent piece of a cloned root,
//!    determinizing it into a fully-known board for rollout.
//! 3. [`RolloutBatch`] — a batch of `num_envs` rollout worlds, each a
//!    determinized root with one seeded root action applied, advanced `depth`
//!    plies under the move net (both players sample it) via the same
//!    collect/commit seam the self-play [`Simulator`](crate::sim::Simulator)
//!    uses. [`RolloutBatch::finish`] returns each world's λ-return leaf value
//!    (terminal reward if the game ended, else the value-head bootstrap at the
//!    leaf — `td_lambda = 1.0`, §5) in the search player's POV.
//!
//! ## Rollout / leaf-value contract (matches `core/search.py::estimate_q_values`)
//! The reference loops `depth / 2` (search-player, opponent) pairs, applying
//! `depth` plies, and bootstraps the leaf with the move net's **value at the
//! search player's position reached after the last opponent ply** (`td_lambda =
//! 1.0`: terminal reward if the game ended during the rollout, else the leaf
//! value). We reproduce that exactly: ply 0 is the seeded root action (search
//! player), applied at construction; the batch then drives `depth - 1` net-moved
//! plies followed by one final value-only forward on the search-player leaf.

use game_core::rand::pick_weighted;
use game_core::{Game, Rng};
use rayon::prelude::*;

use crate::action::Action;
use crate::board::{Board, Color, PieceType};
use crate::encode::{EncoderConfig, encode_tokens};
use crate::evaluator::{Decision, Evaluation, Phase};
use crate::game::{Move, State, Stratego};

/// The hidden-opponent inventory of a root position, in the search (acting)
/// player's POV: enough to sample consistent type assignments and to determinize
/// rollout worlds. Mirrors the reference `unknown_piece_*` tensors but as plain
/// Rust over the absolute board.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootInfo {
    /// The search player (whose turn it is at the root).
    pub to_play: usize,
    /// Absolute board cell of each hidden opponent piece, ordered by **row-major
    /// POV rank** (`pov_cell = to_play == 1 ? 99 - cell : cell`, ascending) — the
    /// ordering the reference `AssignOpponentHiddenPieces` (`++j` over POV cells)
    /// and the belief sampler key on.
    pub hidden_cells: Vec<usize>,
    /// Whether each hidden opponent piece (parallel to `hidden_cells`) has moved.
    pub hidden_has_moved: Vec<bool>,
    /// Per-type counts (`[0, 12)`) of the opponent's hidden pieces — the
    /// remaining-supply budget the count/movability mask enforces.
    pub hidden_counts: [u8; 12],
}

impl RootInfo {
    /// Reads the hidden-opponent inventory from `board` for the player about to
    /// act. Hidden opponent pieces are enumerated in ascending POV-cell order so
    /// the `i`-th entry is the reference's `rank == i + 1` piece.
    pub fn from_board(board: &Board, to_play: usize) -> RootInfo {
        let opp = Color::of_player(1 - to_play);
        let mut entries: Vec<(usize, usize, bool)> = Vec::new(); // (pov_cell, abs_cell, has_moved)
        for cell in 0..100usize {
            let p = board.pieces[cell];
            if p.color == opp && !p.visible {
                let pov_cell = if to_play == 1 { 99 - cell } else { cell };
                entries.push((pov_cell, cell, p.has_moved));
            }
        }
        entries.sort_by_key(|e| e.0);

        let hidden_cells = entries.iter().map(|e| e.1).collect();
        let hidden_has_moved = entries.iter().map(|e| e.2).collect();
        let hidden_counts = board.num_hidden[1 - to_play];
        RootInfo {
            to_play,
            hidden_cells,
            hidden_has_moved,
            hidden_counts,
        }
    }

    /// Number of hidden opponent pieces.
    pub fn n_hidden(&self) -> usize {
        self.hidden_cells.len()
    }
}

/// Determinizes a clone of `root` by assigning `assignment[i]` (a [`PieceType`]
/// value in `[0, 12)`) to the `i`-th hidden opponent piece, in the same
/// row-major POV rank order [`RootInfo`] enumerates. The reference
/// `assign_opponent_hidden_pieces`: the piece stays hidden (`visible == false`)
/// — only its true type is now known to the simulator — so all downstream rules
/// (reveal-on-attack, hidden counts) stay consistent.
///
/// `assignment.len()` must equal `info.n_hidden()`. Each assigned type's running
/// use must not exceed the hidden count for that type (the sampler guarantees
/// this); this function trusts the assignment and only rewrites the board.
pub fn assign_hidden(root: &Board, info: &RootInfo, assignment: &[u8]) -> Board {
    debug_assert_eq!(assignment.len(), info.n_hidden());
    let mut board = root.clone();
    let opp = 1 - info.to_play;
    for (i, &cell) in info.hidden_cells.iter().enumerate() {
        let new_type = PieceType::from_u8(assignment[i]);
        let piece = &mut board.pieces[cell];
        let piece_id = piece.piece_id;
        piece.kind = new_type;
        // The cemetery / death-reason channels read the dead piece's type from
        // the initial arrangement (`zero_types`); keep it consistent with the
        // now-known live type so a later death reports the assigned rank.
        board.zero_types[opp][piece_id as usize] = assignment[i];
    }
    board
}

/// The combinatorial / marginalized-uniform belief marginal at each hidden
/// opponent piece (parallel to [`RootInfo::hidden_cells`]): the per-type
/// probability the analytic posterior assigns, gathered from the encoder's
/// opponent-posterior planes (channels 12..=23, the `their_*_prob` channels) at
/// each hidden piece's POV cell, padded to the 14-wide `N_PIECE_TYPE`. The
/// MARGINALIZED_UNIFORM belief samples autoregressively from these marginals
/// (with the count/movability mask applied per draw, in Python).
pub fn marginal_posterior(board: &Board, info: &RootInfo) -> Vec<[f32; 14]> {
    let infostate = crate::encode::encode_infostate(
        board,
        info.to_play,
        &crate::encode::EncoderConfig::default(),
    );
    let mut out = Vec::with_capacity(info.n_hidden());
    for &cell in &info.hidden_cells {
        // The encoder's `prob_types` writes the opponent posterior at POV cell
        // `99 - cell` for to_play 1, `cell` for to_play 0 (reflect = to_play==1).
        let pov_cell = if info.to_play == 1 { 99 - cell } else { cell };
        let mut marg = [0.0f32; 14];
        // Channels 12..=23 are the 12 opponent-posterior type planes (shift 1200).
        for (t, slot) in marg.iter_mut().enumerate().take(12) {
            *slot = infostate[(12 + t) * 100 + pov_cell];
        }
        out.push(marg);
    }
    out
}

/// One rollout world: a live game advanced under the move net, carrying the
/// bookkeeping the λ-return leaf value needs.
#[derive(Debug, Clone)]
struct World {
    state: State,
    rng: Rng,
    /// The 1800-space root action this world was seeded with (for the per-action
    /// scatter back in the search policy).
    root_action: u16,
    /// Whether this world has reached a terminal position during the rollout.
    done: bool,
    /// Player-POV terminal reward once the world terminated; 0 while live.
    terminal_reward: f32,
    /// The leaf value-head bootstrap (search-player POV), captured at the
    /// value-only leaf forward for a world that never terminated; 0 otherwise.
    leaf_value: f32,
}

/// A batch of belief-determinized rollout worlds advanced in lockstep under a
/// caller-provided move-net policy. Drives `depth` plies via the same
/// collect/commit seam as self-play: [`collect`](RolloutBatch::collect) encodes
/// every live world for one batched net forward, [`commit`](RolloutBatch::commit)
/// samples + applies each world's move (or, on the final leaf forward, only
/// records the bootstrap value).
pub struct RolloutBatch {
    worlds: Vec<World>,
    cfg: EncoderConfig,
    /// The search player (constant across the batch — every world shares the
    /// real root's to_play).
    search_player: usize,
    depth: usize,
    /// Number of net forwards already committed. The first `depth - 1` apply a
    /// move; the `depth`-th is the value-only leaf forward.
    forwards: usize,
}

impl RolloutBatch {
    /// Builds `roots.len()` rollout worlds. Each `(board, root_action, seed)`
    /// triple is a determinized root (already type-assigned via
    /// [`assign_hidden`]); the root action is applied immediately (the search
    /// player's ply 0), so the batch is ready for the opponent's reply on the
    /// first [`collect`](RolloutBatch::collect).
    ///
    /// `depth` must be even and ≥ 2: the seeded root ply plus `depth - 1`
    /// net-moved plies, then one value-only leaf forward (`depth` forwards
    /// total), matching the reference's `depth / 2` (player, opponent) pairs.
    pub fn new(
        roots: &[(Board, u16, u64)],
        search_player: usize,
        depth: usize,
        cfg: EncoderConfig,
    ) -> RolloutBatch {
        assert!(
            depth >= 2 && depth.is_multiple_of(2),
            "depth must be even and ≥ 2"
        );
        let game = Stratego;
        let worlds: Vec<World> = roots
            .par_iter()
            .map(|(board, root_action, seed)| {
                let mut state = State::Play {
                    board: Box::new(board.clone()),
                    to_play: search_player,
                    flag_captured: None,
                };
                game.apply(&mut state, Move::Step(Action(*root_action)));
                let mut world = World {
                    state,
                    rng: Rng::new(*seed),
                    root_action: *root_action,
                    done: false,
                    terminal_reward: 0.0,
                    leaf_value: 0.0,
                };
                if game.is_terminal(&world.state) {
                    world.done = true;
                    world.terminal_reward = game.returns(&world.state, search_player) as f32;
                }
                world
            })
            .collect();

        RolloutBatch {
            worlds,
            cfg,
            search_player,
            depth,
            forwards: 0,
        }
    }

    /// Number of rollout worlds.
    pub fn num_envs(&self) -> usize {
        self.worlds.len()
    }

    /// Whether every forward (the `depth - 1` move forwards plus the leaf
    /// forward) has been committed.
    pub fn is_done(&self) -> bool {
        self.forwards >= self.depth
    }

    /// Whether the *next* [`collect`](RolloutBatch::collect) / [`commit`] is the
    /// value-only leaf forward (the search player's leaf position, where the
    /// value bootstraps but no move is applied).
    pub fn is_leaf_forward(&self) -> bool {
        self.forwards + 1 == self.depth
    }

    /// Encodes every still-live world's move decision for one batched net
    /// forward. Terminal worlds yield an inert placeholder (empty legal set) the
    /// commit pass skips; they stay in the batch so row `i` always maps to world
    /// `i`. Pair with [`commit`](RolloutBatch::commit).
    pub fn collect(&self) -> Vec<RolloutDecision> {
        let cfg = &self.cfg;
        self.worlds
            .par_iter()
            .map(|w| match (&w.state, w.done) {
                (State::Play { board, to_play, .. }, false) => {
                    let mask = crate::rules::legal_mask(board, *to_play);
                    let legal: Vec<u16> = (0..mask.len())
                        .filter(|&i| mask[i])
                        .map(|i| i as u16)
                        .collect();
                    let live = !legal.is_empty();
                    let obs = if live {
                        encode_tokens(board, *to_play, cfg)
                    } else {
                        Vec::new()
                    };
                    RolloutDecision {
                        obs,
                        legal,
                        phase: Phase::Move,
                        player: *to_play,
                        live,
                    }
                }
                _ => RolloutDecision {
                    obs: Vec::new(),
                    legal: Vec::new(),
                    phase: Phase::Move,
                    player: self.search_player,
                    live: false,
                },
            })
            .collect()
    }

    /// Applies one net forward to every live world. `evals` is parallel to the
    /// last [`collect`](RolloutBatch::collect). On the first `depth - 1`
    /// forwards, each live world softmax-samples a legal move (its own RNG) and
    /// advances; a world that terminates latches its terminal reward. On the
    /// final (leaf) forward, the search-player value is latched as the leaf
    /// bootstrap and no move is applied.
    pub fn commit(&mut self, decisions: &[RolloutDecision], evals: &[Evaluation]) {
        assert_eq!(decisions.len(), self.worlds.len());
        assert_eq!(evals.len(), self.worlds.len());
        let game = Stratego;
        let search_player = self.search_player;
        let leaf = self.is_leaf_forward();

        self.worlds
            .par_iter_mut()
            .zip(decisions.par_iter())
            .zip(evals.par_iter())
            .for_each(|((world, decision), evaluation)| {
                if world.done || !decision.live {
                    return;
                }
                if leaf {
                    // The leaf forward is the search player's position; bootstrap
                    // with its value and stop (no move applied).
                    debug_assert_eq!(decision.player, search_player);
                    world.leaf_value = evaluation.value;
                    return;
                }

                let log_probs = log_softmax(&evaluation.logits);
                let weights: Vec<f64> = log_probs.iter().map(|&lp| f64::from(lp).exp()).collect();
                let chosen = pick_weighted(weights.iter().copied(), &mut world.rng);
                let action = decision.legal[chosen];

                game.apply(&mut world.state, Move::Step(Action(action)));

                if game.is_terminal(&world.state) {
                    world.done = true;
                    world.terminal_reward = game.returns(&world.state, search_player) as f32;
                }
            });

        self.forwards += 1;
    }

    /// The per-world λ-return leaf value (search-player POV) and the root action
    /// each world was seeded with, parallel along the batch. A world that
    /// terminated returns its terminal reward; a world that ran to depth returns
    /// the value-head bootstrap captured at the leaf forward.
    pub fn finish(&self) -> Vec<(u16, f32)> {
        self.worlds
            .iter()
            .map(|w| {
                let leaf = if w.done {
                    w.terminal_reward
                } else {
                    w.leaf_value
                };
                (w.root_action, leaf)
            })
            .collect()
    }
}

/// An owned move decision for one rollout world — the obs + legal set the move
/// net scores, plus the acting player and a `live` flag the commit pass uses to
/// skip terminal worlds. Mirrors [`Decision`](crate::evaluator::Decision) but
/// owns its buffers so the batch can be built in parallel.
#[derive(Debug, Clone)]
pub struct RolloutDecision {
    pub obs: Vec<f32>,
    pub legal: Vec<u16>,
    pub phase: Phase,
    pub player: usize,
    pub live: bool,
}

impl RolloutDecision {
    /// A borrowed [`Decision`] view for an in-process [`Evaluator`] call (tests).
    pub fn as_decision(&self) -> Decision<'_> {
        Decision {
            phase: self.phase,
            obs: &self.obs,
            legal: &self.legal,
            player: self.player,
        }
    }
}

/// Numerically-stable log-softmax over the legal-option logits (the same softmax
/// the self-play sampler uses, so the rollout policy matches the data policy).
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    let log_sum = max + sum_exp.ln();
    logits.iter().map(|&l| l - log_sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::{Evaluator, UniformEvaluator};

    fn deterministic_root(seed: u64) -> (Board, usize) {
        let mut rng = Rng::new(seed);
        match Stratego::random_play_state(&mut rng) {
            State::Play { board, to_play, .. } => (*board, to_play),
            _ => unreachable!(),
        }
    }

    fn legal_actions(board: &Board, to_play: usize) -> Vec<u16> {
        let mask = crate::rules::legal_mask(board, to_play);
        (0..mask.len())
            .filter(|&i| mask[i])
            .map(|i| i as u16)
            .collect()
    }

    fn true_type_assignment(board: &Board, info: &RootInfo) -> Vec<u8> {
        info.hidden_cells
            .iter()
            .map(|&c| board.pieces[c].kind as u8)
            .collect()
    }

    #[test]
    fn root_info_enumerates_hidden_in_pov_rank_order() {
        let (board, to_play) = deterministic_root(20240624);
        let info = RootInfo::from_board(&board, to_play);
        let opp = Color::of_player(1 - to_play);
        for &cell in &info.hidden_cells {
            let p = board.pieces[cell];
            assert_eq!(p.color, opp);
            assert!(!p.visible);
        }
        let povs: Vec<usize> = info
            .hidden_cells
            .iter()
            .map(|&c| if to_play == 1 { 99 - c } else { c })
            .collect();
        assert!(povs.windows(2).all(|w| w[0] < w[1]), "pov ranks ascending");
        assert_eq!(info.hidden_counts, board.num_hidden[1 - to_play]);
        let total: u32 = info.hidden_counts.iter().map(|&x| x as u32).sum();
        assert_eq!(total as usize, info.n_hidden());
        assert_eq!(info.hidden_has_moved.len(), info.n_hidden());
    }

    #[test]
    fn assign_hidden_keeps_pieces_hidden_and_consistent() {
        let (board, to_play) = deterministic_root(20240624);
        let info = RootInfo::from_board(&board, to_play);
        let opp = 1 - to_play;
        let assignment = true_type_assignment(&board, &info);
        let assigned = assign_hidden(&board, &info, &assignment);
        for &cell in &info.hidden_cells {
            assert!(!assigned.pieces[cell].visible, "stays hidden");
        }
        assert_eq!(assigned.num_hidden[opp], board.num_hidden[opp]);
    }

    /// The analytic marginal gathers the encoder's opponent-posterior planes at
    /// the right POV cell: each hidden piece's marginal is a valid distribution
    /// over the 12 ranks, and a moved piece carries zero flag/bomb probability.
    #[test]
    fn marginal_posterior_is_a_valid_constrained_distribution() {
        let (board, to_play) = deterministic_root(20240624);
        let info = RootInfo::from_board(&board, to_play);
        let marg = marginal_posterior(&board, &info);
        assert_eq!(marg.len(), info.n_hidden());
        assert!(info.n_hidden() > 0, "the seed should leave hidden pieces");
        for (i, row) in marg.iter().enumerate() {
            let sum: f32 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "hidden piece {i} marginal sums to {sum}, not 1"
            );
            if info.hidden_has_moved[i] {
                // A moved piece cannot be the flag (10) or a bomb (11).
                assert!(
                    row[10] == 0.0 && row[11] == 0.0,
                    "moved piece {i} has flag/bomb mass"
                );
            }
        }
    }

    /// A `UniformEvaluator`-driven rollout never panics; leaf values are valid.
    #[test]
    fn rollout_runs_and_finishes() {
        let (board, to_play) = deterministic_root(20240624);
        let info = RootInfo::from_board(&board, to_play);
        let assigned = assign_hidden(&board, &info, &true_type_assignment(&board, &info));
        let legal = legal_actions(&assigned, to_play);
        assert!(!legal.is_empty());
        let roots: Vec<(Board, u16, u64)> = legal
            .iter()
            .take(8)
            .enumerate()
            .map(|(i, &a)| (assigned.clone(), a, 1000 + i as u64))
            .collect();

        let cfg = EncoderConfig::default();
        let mut batch = RolloutBatch::new(&roots, to_play, 6, cfg);
        let eval = UniformEvaluator;
        while !batch.is_done() {
            let decisions = batch.collect();
            let views: Vec<Decision> = decisions.iter().map(|d| d.as_decision()).collect();
            let evals = eval.evaluate_batch(&views);
            batch.commit(&decisions, &evals);
        }
        let leaves = batch.finish();
        assert_eq!(leaves.len(), roots.len());
        for (i, (root_action, leaf)) in leaves.iter().enumerate() {
            assert_eq!(*root_action, legal[i]);
            assert!((-1.0..=1.0).contains(leaf), "leaf {leaf} out of range");
        }
    }

    /// A deterministic value evaluator: uniform logits (uniform sampling) and a
    /// fixed scalar value, so a rollout's bootstrap is reproducible and an
    /// independent λ-return can be computed over the identical trajectory.
    struct FixedValue(f32);
    impl Evaluator for FixedValue {
        fn evaluate_batch(&self, batch: &[Decision]) -> Vec<Evaluation> {
            batch
                .iter()
                .map(|d| Evaluation {
                    logits: vec![0.0; d.legal.len()],
                    value: self.0,
                })
                .collect()
        }
    }

    /// The Q-estimator oracle (`test_search.py`): a rollout's `td_lambda = 1.0`
    /// leaf value must equal an **independent** λ-return over the *same* rollout
    /// trajectory, to 1e-6 — proving the rollout/value plumbing is correct.
    ///
    /// We re-run each world's exact decision sequence (same RNG, same uniform
    /// policy, same fixed value) outside [`RolloutBatch`], then compute the λ=1
    /// return at the root step with the reference `segmented_discounted_cumsum`
    /// math (`core/buffer.py::process_data`, the same recurrence
    /// [`ReplayBuffer::process_data`] is itself verified against): for the search
    /// player's own value/reward sequence, `delta = target - value` with
    /// `target` the next search-player value two plies later (or the terminal
    /// reward), `returns = segmented_discounted_cumsum(delta, 1.0 * ~term) +
    /// value`, read at the root step.
    #[test]
    fn rollout_leaf_matches_independent_lambda_return() {
        for &depth in &[2usize, 4, 6, 10] {
            for &bootstrap in &[0.0f32, 0.4, -0.7] {
                check_oracle(depth, bootstrap);
            }
        }
    }

    fn check_oracle(depth: usize, bootstrap: f32) {
        let (board, to_play) = deterministic_root(20240624);
        let info = RootInfo::from_board(&board, to_play);
        let assigned = assign_hidden(&board, &info, &true_type_assignment(&board, &info));
        let legal = legal_actions(&assigned, to_play);
        let roots: Vec<(Board, u16, u64)> = legal
            .iter()
            .take(16)
            .enumerate()
            .map(|(i, &a)| (assigned.clone(), a, 777 + i as u64))
            .collect();

        let cfg = EncoderConfig::default();
        let eval = FixedValue(bootstrap);

        // Run the batched rollout.
        let mut batch = RolloutBatch::new(&roots, to_play, depth, cfg);
        while !batch.is_done() {
            let decisions = batch.collect();
            let views: Vec<Decision> = decisions.iter().map(|d| d.as_decision()).collect();
            let evals = eval.evaluate_batch(&views);
            batch.commit(&decisions, &evals);
        }
        let leaves = batch.finish();

        // Independently recompute each world's λ=1 root return by replaying the
        // identical decision sequence (same RNG stream, same uniform policy).
        for (i, &(root_action, leaf)) in leaves.iter().enumerate() {
            let independent = independent_root_return(
                &roots[i].0,
                to_play,
                root_action,
                roots[i].2,
                depth,
                bootstrap,
                &cfg,
            );
            assert!(
                (independent - leaf).abs() < 1e-6,
                "depth {depth} boot {bootstrap} world {i}: leaf {leaf} != λ-return {independent}"
            );
        }
    }

    /// Replays one rollout world outside [`RolloutBatch`] (identical RNG, uniform
    /// policy, fixed value `bootstrap`) and returns the λ=1 return at the root
    /// step, via the reference `segmented_discounted_cumsum` recurrence.
    ///
    /// We walk the rollout in **pairs**, mirroring `estimate_q_values`'s `depth /
    /// 2` (search-player, opponent) loop: each pair contributes one search-player
    /// step whose `value = bootstrap` and whose λ=1 `target` is the terminal
    /// reward (search POV) if the game ended during the pair, the leaf value on
    /// the final pair, else the next pair's search value. `returns =
    /// segmented_discounted_cumsum(target - value, 1.0 * ~term) + value`, read at
    /// the root step.
    #[allow(clippy::too_many_arguments)]
    fn independent_root_return(
        root_board: &Board,
        to_play: usize,
        root_action: u16,
        seed: u64,
        depth: usize,
        bootstrap: f32,
        _cfg: &EncoderConfig,
    ) -> f32 {
        let game = Stratego;
        let mut state = State::Play {
            board: Box::new(root_board.clone()),
            to_play,
            flag_captured: None,
        };
        let mut rng = Rng::new(seed);

        // One (value, target, terminal) record per (search, opponent) pair.
        let mut value: Vec<f32> = Vec::new();
        let mut target: Vec<f32> = Vec::new();
        let mut term: Vec<bool> = Vec::new();

        // Apply a uniform-sampled move for the current player; return the
        // post-move terminal flag.
        let apply_uniform = |state: &mut State, rng: &mut Rng| -> bool {
            let (board, cur) = match &*state {
                State::Play { board, to_play, .. } => (board.clone(), *to_play),
                _ => unreachable!(),
            };
            let legal = legal_actions(&board, cur);
            if legal.is_empty() {
                return game.is_terminal(state);
            }
            let weights: Vec<f64> = vec![1.0 / legal.len() as f64; legal.len()];
            let chosen = pick_weighted(weights.iter().copied(), rng);
            game.apply(state, Move::Step(Action(legal[chosen])));
            game.is_terminal(state)
        };

        let pairs = depth / 2;
        let mut terminated = false;
        for d in 0..pairs {
            // Search player's move: the seeded root action on the first pair.
            let search_term = if d == 0 {
                game.apply(&mut state, Move::Step(Action(root_action)));
                game.is_terminal(&state)
            } else {
                apply_uniform(&mut state, &mut rng)
            };

            // Opponent's move (only if the search move did not end the game).
            let pair_term = if search_term {
                true
            } else {
                apply_uniform(&mut state, &mut rng)
            };

            let reward = if pair_term {
                game.returns(&state, to_play) as f32
            } else {
                0.0
            };

            value.push(bootstrap);
            term.push(pair_term);
            // λ=1 target: terminal reward if ended; on the final pair, the leaf
            // bootstrap; else resolved to the next pair's value afterwards.
            let is_last = d == pairs - 1;
            target.push(if pair_term {
                reward
            } else if is_last {
                bootstrap // leaf value
            } else {
                f32::NAN // resolved below
            });

            if pair_term {
                terminated = true;
                break;
            }
        }

        // Resolve non-terminal, non-leaf targets to the next pair's value.
        let n = value.len();
        for s in 0..n {
            if target[s].is_nan() {
                target[s] = value[s + 1];
            }
        }
        let _ = terminated;

        let delta: Vec<f32> = (0..n).map(|s| target[s] - value[s]).collect();
        let td_disc: Vec<f32> = (0..n).map(|s| if term[s] { 0.0 } else { 1.0 }).collect();
        let mut y = vec![0.0f32; n];
        if n > 0 {
            y[n - 1] = delta[n - 1];
            for t in (0..n - 1).rev() {
                y[t] = delta[t] + td_disc[t] * y[t + 1];
            }
        }
        y[0] + value[0]
    }
}
