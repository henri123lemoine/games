//! Shapeshifter-style adversarial search for canonical Battlesnake.
//!
//! The root uses Best Node Search (BNS) with null-window alpha-beta probes.
//! A max node chooses the hero move; a min node chooses a simultaneous joint
//! response without ever exposing one player's current move to another. For
//! multiplayer games the response set can use Shapeshifter's move-combination
//! simplification (MCS), Best Reply Search Plus (BRS+), or the full Cartesian
//! product. Evaluation is a phase-interpolated linear model over bitboard area
//! control, checkerboard capacity, tails, food, health, length, and mobility.

use arrayvec::ArrayVec;
use std::cell::RefCell;
use std::time::Duration;
use web_time::Instant;

use game_core::hash::combine;
use game_core::{Rng, SimultaneousAgent, SimultaneousGame};

use super::{
    Battlesnake, BoardState, CELLS, ChanceAction, Direction, Mode, Phase, SIDE, bit, bits,
    board_mask, next_cell, xy,
};

pub type Score = i16;

const LOSS: Score = -30_000;
const WIN: Score = 30_000;
const DRAW: Score = -10_000;
const INF: Score = 31_000;
const FEATURE_COUNT: usize = 12;
const DEFAULT_TT_BITS: u8 = 19;
const MAX_JOINT_REPLIES: usize = 64;
const MAX_BODY_LEN: usize = 128;
const BOTTOM_EDGE: u128 = (1u128 << SIDE) - 1;
const TOP_EDGE: u128 = BOTTOM_EDGE << (SIDE * (SIDE - 1));
const LEFT_EDGE: u128 = edge_mask(0);
const RIGHT_EDGE: u128 = edge_mask(SIDE - 1);
const CHECKERBOARD: u128 = checkerboard_mask();

const fn edge_mask(x: usize) -> u128 {
    let mut mask = 0u128;
    let mut y = 0;
    while y < SIDE {
        mask |= 1u128 << (y * SIDE + x);
        y += 1;
    }
    mask
}

const fn checkerboard_mask() -> u128 {
    let mut mask = 0u128;
    let mut cell = 0;
    while cell < CELLS {
        if ((cell % SIDE) + (cell / SIDE)) & 1 == 0 {
            mask |= 1u128 << cell;
        }
        cell += 1;
    }
    mask
}

pub const FEATURE_NAMES: [&str; FEATURE_COUNT] = [
    "health",
    "lowest_enemy_health",
    "length_difference",
    "being_longer",
    "controlled_food",
    "checkerboard_area",
    "close_area",
    "food_proximity",
    "controlled_tails",
    "safe_mobility",
    "non_hazard_area",
    "length_parity",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpponentModel {
    /// Exact Cartesian product. This is cheap and preferred for duels.
    Full,
    /// At most four joint replies, pairing ordered moves so every individual
    /// enemy action occurs in at least one reply.
    MoveCombination,
    /// Hold all enemies to their ordered default and vary one enemy at a time.
    BestReplyPlus,
}

#[derive(Clone, Copy, Debug)]
pub struct EvaluationWeights {
    pub early: [Score; FEATURE_COUNT],
    pub late: [Score; FEATURE_COUNT],
    pub late_game_turn: u16,
}

impl EvaluationWeights {
    /// Standard-board weights seeded from the public Shapeshifter evaluator,
    /// extended with mobility and hazard-aware territory.
    pub const fn shapeshifter_standard() -> Self {
        Self {
            early: [1, -2, 0, 9, 0, 1, 0, 7, 6, 2, 1, 0],
            late: [0, 0, 1, 0, 3, 7, 2, 0, 20, 5, 7, 2],
            late_game_turn: 632,
        }
    }
}

impl Default for EvaluationWeights {
    fn default() -> Self {
        Self::shapeshifter_standard()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SearchConfig {
    pub time_limit: Duration,
    pub max_depth: u8,
    pub quiescence_depth: u8,
    pub opponent_model: OpponentModel,
    pub tt_bits: u8,
    pub weights: EvaluationWeights,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            // Leave headroom inside Battlesnake's common 500 ms request limit.
            time_limit: Duration::from_millis(440),
            max_depth: u8::MAX,
            quiescence_depth: 3,
            opponent_model: OpponentModel::MoveCombination,
            tt_bits: DEFAULT_TT_BITS,
            weights: EvaluationWeights::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SearchStats {
    pub depth: u8,
    pub score: Score,
    pub nodes: u64,
    pub tt_hits: u64,
    pub bns_probes: u64,
    pub elapsed: Duration,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchResult {
    pub action: Direction,
    pub stats: SearchStats,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy, Debug)]
struct Entry {
    key: u64,
    value: Score,
    depth: u8,
    bound: Bound,
    best: u8,
}

struct TranspositionTable {
    entries: Vec<Option<Entry>>,
    mask: usize,
}

impl TranspositionTable {
    fn new(bits: u8) -> Self {
        assert!((10..=24).contains(&bits), "tt_bits must be in 10..=24");
        let len = 1usize << bits;
        Self {
            entries: vec![None; len],
            mask: len - 1,
        }
    }

    #[inline]
    fn get(&self, key: u64) -> Option<Entry> {
        self.entries[key as usize & self.mask].filter(|entry| entry.key == key)
    }

    #[inline]
    fn put(&mut self, entry: Entry) {
        let slot = &mut self.entries[entry.key as usize & self.mask];
        if slot.is_none_or(|old| old.key != entry.key || old.depth <= entry.depth) {
            *slot = Some(entry);
        }
    }
}

/// Reusable search state. Keeping the transposition table and history between
/// turns is materially stronger than constructing a fresh search per request.
pub struct Searcher<const N: usize> {
    config: SearchConfig,
    tt: TranspositionTable,
    history: [[[u32; 4]; CELLS]; N],
    previous_best: [Direction; N],
    stats: SearchStats,
    started: Instant,
    deadline: Instant,
    timed_out: bool,
}

impl<const N: usize> Searcher<N> {
    pub fn new(mut config: SearchConfig) -> Self {
        assert!(N <= 4, "Battlesnake search supports at most four players");
        if N == 2 && config.opponent_model == OpponentModel::MoveCombination {
            config.opponent_model = OpponentModel::Full;
        }
        Self {
            config,
            tt: TranspositionTable::new(config.tt_bits),
            history: [[[0; 4]; CELLS]; N],
            previous_best: [Direction::Up; N],
            stats: SearchStats::default(),
            started: Instant::now(),
            deadline: Instant::now(),
            timed_out: false,
        }
    }

    pub fn config(&self) -> SearchConfig {
        self.config
    }

    pub fn search(
        &mut self,
        game: &Battlesnake<N>,
        state: &BoardState<N>,
        hero: usize,
    ) -> SearchResult {
        assert!(hero < N);
        assert_eq!(state.phase, Phase::Moves, "search starts at a joint move");
        self.started = Instant::now();
        self.deadline = self.started + self.config.time_limit;
        self.timed_out = false;
        self.stats = SearchStats::default();

        let mut root_moves = self.ordered_moves(game, state, hero, Some(self.previous_best[hero]));
        let fallback = root_moves[0];
        if root_moves.len() == 1 {
            return SearchResult {
                action: fallback,
                stats: SearchStats {
                    elapsed: self.started.elapsed(),
                    ..SearchStats::default()
                },
            };
        }

        let mut completed_move = fallback;
        let mut completed_score = LOSS;
        let mut last_guess = 0;
        for depth in 1..=self.config.max_depth {
            let Some((best, score)) =
                self.bns_depth(game, state, hero, depth, &root_moves, last_guess)
            else {
                break;
            };
            completed_move = best;
            completed_score = score;
            last_guess = score;
            self.stats.depth = depth;
            self.stats.score = score;
            self.previous_best[hero] = best;
            root_moves = self.ordered_moves(game, state, hero, Some(best));
            if score >= WIN - 1_000 || score <= LOSS + 1_000 {
                break;
            }
        }
        self.stats.elapsed = self.started.elapsed();
        SearchResult {
            action: completed_move,
            stats: SearchStats {
                score: completed_score,
                ..self.stats
            },
        }
    }

    fn bns_depth(
        &mut self,
        game: &Battlesnake<N>,
        state: &BoardState<N>,
        hero: usize,
        depth: u8,
        root_moves: &[Direction],
        previous_guess: Score,
    ) -> Option<(Direction, Score)> {
        let mut candidates: ArrayVec<Direction, 4> = root_moves.iter().copied().collect();
        let mut alpha = LOSS;
        let mut beta = WIN;
        let mut guess = previous_guess.clamp(alpha + 1, beta - 1);
        while i32::from(beta) - i32::from(alpha) >= 2 && candidates.len() > 1 {
            if self.expired() {
                return None;
            }
            let test = next_bns_guess(guess, alpha, beta);
            self.stats.bns_probes += 1;
            let mut better = ArrayVec::<Direction, 4>::new();
            for &action in &candidates {
                let score = self.min_node(
                    game,
                    state,
                    hero,
                    action,
                    depth,
                    self.config.quiescence_depth,
                    test - 1,
                    test,
                    0,
                )?;
                if score >= test {
                    better.push(action);
                }
            }
            if better.is_empty() {
                beta = test;
            } else {
                alpha = test;
                candidates = better;
            }
            guess = test;
        }

        // BNS deliberately stops after identifying the best action. Computing
        // its exact minimax value would throw away the principal speed benefit
        // over MTD(f), and Shapeshifter likewise carries the final null-window
        // threshold into the next iteration.
        Some((candidates[0], guess))
    }

    #[inline]
    fn expired(&mut self) -> bool {
        if self.timed_out {
            return true;
        }
        if self.stats.nodes & 1023 == 0 && Instant::now() >= self.deadline {
            self.timed_out = true;
        }
        self.timed_out
    }
}

/// Arena adapter retaining search state across moves.
pub struct SearchAgent<const N: usize> {
    searcher: RefCell<Searcher<N>>,
    last: RefCell<Option<SearchResult>>,
}

impl<const N: usize> SearchAgent<N> {
    pub fn new(config: SearchConfig) -> Self {
        Self {
            searcher: RefCell::new(Searcher::new(config)),
            last: RefCell::new(None),
        }
    }

    pub fn last_result(&self) -> Option<SearchResult> {
        *self.last.borrow()
    }
}

impl<const N: usize> SimultaneousAgent<Battlesnake<N>> for SearchAgent<N> {
    fn act(
        &self,
        game: &Battlesnake<N>,
        state: &BoardState<N>,
        player: usize,
        _rng: &mut Rng,
    ) -> usize {
        let result = self.searcher.borrow_mut().search(game, state, player);
        *self.last.borrow_mut() = Some(result);
        result.action as usize
    }
}

#[inline]
fn next_bns_guess(previous: Score, alpha: Score, beta: Score) -> Score {
    if previous > alpha && previous < beta {
        return previous;
    }
    let midpoint = ((i32::from(alpha) + i32::from(beta)) / 2) as Score;
    midpoint.clamp(alpha + 1, beta - 1)
}

impl<const N: usize> Searcher<N> {
    #[allow(clippy::too_many_arguments)]
    fn min_node(
        &mut self,
        game: &Battlesnake<N>,
        state: &BoardState<N>,
        hero: usize,
        hero_action: Direction,
        depth: u8,
        qdepth: u8,
        mut alpha: Score,
        mut beta: Score,
        ply: u16,
    ) -> Option<Score> {
        if self.expired() {
            return None;
        }
        let original_alpha = alpha;
        let original_beta = beta;
        let key = combine(
            combine(
                combine(state_key(game, state, hero), 0x6d69_6e00),
                u64::from(qdepth),
            ),
            hero_action as u64,
        );
        let mut tt_reply = None;
        if let Some(entry) = self.tt.get(key) {
            self.stats.tt_hits += 1;
            tt_reply = Some(unpack_joint(entry.best));
            if entry.depth >= depth {
                match entry.bound {
                    Bound::Exact => return Some(entry.value),
                    Bound::Lower => alpha = alpha.max(entry.value),
                    Bound::Upper => beta = beta.min(entry.value),
                }
                if alpha >= beta {
                    return Some(entry.value);
                }
            }
        }

        let replies = self.opponent_replies(game, state, hero, tt_reply);
        let mut best = INF;
        let mut best_joint = replies[0];
        for mut joint in replies {
            joint[hero] = hero_action;
            let mut child = *state;
            game.resolve(&mut child, &joint);
            settle_modal_food(game, &mut child);
            self.stats.nodes += 1;

            let score = if !child.snakes[hero].is_alive() || child.alive_count() <= 1 {
                terminal_score(&child, hero, ply + 1)
            } else if depth <= 1 {
                if qdepth > 0 && !is_stable(game, &child, hero) {
                    self.max_node(
                        game,
                        &child,
                        hero,
                        1,
                        qdepth - 1,
                        alpha,
                        beta.min(best),
                        ply + 1,
                    )?
                } else {
                    self.evaluate(game, &child, hero)
                }
            } else {
                self.max_node(
                    game,
                    &child,
                    hero,
                    depth - 1,
                    qdepth,
                    alpha,
                    beta.min(best),
                    ply + 1,
                )?
            };
            if score < best {
                best = score;
                best_joint = joint;
                beta = beta.min(score);
            }
            if best <= alpha {
                break;
            }
        }

        self.tt.put(Entry {
            key,
            value: best,
            depth,
            bound: classify_bound(best, original_alpha, original_beta),
            best: pack_joint(&best_joint),
        });
        Some(best)
    }

    #[allow(clippy::too_many_arguments)]
    fn max_node(
        &mut self,
        game: &Battlesnake<N>,
        state: &BoardState<N>,
        hero: usize,
        depth: u8,
        qdepth: u8,
        mut alpha: Score,
        mut beta: Score,
        ply: u16,
    ) -> Option<Score> {
        if self.expired() {
            return None;
        }
        let original_alpha = alpha;
        let original_beta = beta;
        let key = combine(
            combine(state_key(game, state, hero), 0x6d61_7800),
            u64::from(qdepth),
        );
        let mut tt_move = None;
        if let Some(entry) = self.tt.get(key) {
            self.stats.tt_hits += 1;
            tt_move = direction(entry.best);
            if entry.depth >= depth {
                match entry.bound {
                    Bound::Exact => return Some(entry.value),
                    Bound::Lower => alpha = alpha.max(entry.value),
                    Bound::Upper => beta = beta.min(entry.value),
                }
                if alpha >= beta {
                    return Some(entry.value);
                }
            }
        }

        let moves = self.ordered_moves(game, state, hero, tt_move);
        let search_depth =
            if moves.len() == 1 && self.opponent_replies(game, state, hero, None).len() == 1 {
                // Forced corridors cost no branching. Preserve nominal depth so
                // iterative deepening reaches the next actual choice instead of
                // spending a ply on a transition nobody can avoid.
                depth.saturating_add(1)
            } else {
                depth
            };
        let mut best = -INF;
        let mut best_move = moves[0];
        for action in moves {
            let score = self.min_node(
                game,
                state,
                hero,
                action,
                search_depth,
                qdepth,
                alpha,
                beta,
                ply,
            )?;
            if score > best {
                best = score;
                best_move = action;
                alpha = alpha.max(score);
            }
            if best >= beta {
                let head = state.snakes[hero].head() as usize;
                self.history[hero][head][action as usize] = self.history[hero][head]
                    [action as usize]
                    .saturating_add(u32::from(depth).pow(2));
                break;
            }
        }

        self.tt.put(Entry {
            key,
            value: best,
            depth,
            bound: classify_bound(best, original_alpha, original_beta),
            best: best_move as u8,
        });
        Some(best)
    }

    fn ordered_moves(
        &self,
        game: &Battlesnake<N>,
        state: &BoardState<N>,
        player: usize,
        preferred: Option<Direction>,
    ) -> ArrayVec<Direction, 4> {
        let snake = state.snakes[player];
        let obstacles = body_obstacles(state);
        let scores = Direction::ALL
            .map(|action| move_order_score_with_obstacles(game, state, player, action, obstacles));
        let mut moves: ArrayVec<Direction, 4> = Direction::ALL
            .into_iter()
            .filter(|&action| scores[action as usize] > -5_000)
            .collect();
        if moves.is_empty() {
            // Every action is immediately fatal, but they are not equivalent:
            // one can still force every opponent to die on the same joint
            // move and turn a loss into a draw. Keep all four for exact
            // simultaneous resolution in this uncommon case.
            moves.extend(Direction::ALL);
        }
        moves.sort_by_key(|&action| {
            let preference = i64::from(preferred == Some(action)) * 1_000_000;
            let history = i64::from(self.history[player][snake.head() as usize][action as usize]);
            let tactical = i64::from(scores[action as usize]);
            std::cmp::Reverse(preference + history + tactical)
        });
        moves
    }

    fn opponent_replies(
        &self,
        game: &Battlesnake<N>,
        state: &BoardState<N>,
        hero: usize,
        preferred: Option<[Direction; N]>,
    ) -> ArrayVec<[Direction; N], MAX_JOINT_REPLIES> {
        let per_player: [ArrayVec<Direction, 4>; N] = std::array::from_fn(|player| {
            if player == hero || !state.snakes[player].is_alive() {
                ArrayVec::from_iter([state.snakes[player].heading()])
            } else {
                self.ordered_moves(game, state, player, None)
            }
        });
        let mut replies = match self.config.opponent_model {
            OpponentModel::Full => full_replies(&per_player, state, hero),
            OpponentModel::MoveCombination => mcs_replies(&per_player, state, hero),
            OpponentModel::BestReplyPlus => brs_plus_replies(&per_player, state, hero),
        };
        if let Some(mut preferred) = preferred {
            // A TT reply is a concrete joint action, not an index into a list
            // whose history ordering can change between probes. For the
            // reduced multiplayer models, retaining a previously dangerous
            // legal combination as one extra reply is conservative and makes
            // the ordering useful across iterative-deepening passes.
            preferred[hero] = state.snakes[hero].heading();
            let legal = (0..N).all(|player| {
                player == hero
                    || !state.snakes[player].is_alive()
                    || per_player[player].contains(&preferred[player])
            });
            if legal {
                if let Some(index) = replies.iter().position(|joint| *joint == preferred) {
                    replies.swap(0, index);
                } else if !replies.is_full() {
                    replies.insert(0, preferred);
                }
            }
        }
        replies
    }
}

#[inline]
fn classify_bound(value: Score, alpha: Score, beta: Score) -> Bound {
    if value <= alpha {
        Bound::Upper
    } else if value >= beta {
        Bound::Lower
    } else {
        Bound::Exact
    }
}

#[inline]
fn direction(value: u8) -> Option<Direction> {
    Direction::ALL.get(value as usize).copied()
}

#[inline]
fn pack_joint<const N: usize>(joint: &[Direction; N]) -> u8 {
    joint
        .iter()
        .enumerate()
        .fold(0, |packed, (player, action)| {
            packed | ((*action as u8) << (player * 2))
        })
}

#[inline]
fn unpack_joint<const N: usize>(packed: u8) -> [Direction; N] {
    std::array::from_fn(|player| Direction::ALL[((packed >> (player * 2)) & 0b11) as usize])
}

fn state_key<const N: usize>(game: &Battlesnake<N>, state: &BoardState<N>, hero: usize) -> u64 {
    combine(
        game.state_key(state).expect("Battlesnake has a state key"),
        hero as u64,
    )
}

fn settle_modal_food<const N: usize>(game: &Battlesnake<N>, state: &mut BoardState<N>) {
    loop {
        match state.phase {
            Phase::Moves => break,
            Phase::FoodRoll => game.apply_chance(state, ChanceAction::NoFood),
            Phase::RequiredFood(_) => {
                let available = state.available_food_cells();
                if available == 0 {
                    game.apply_chance(state, ChanceAction::NoFood);
                    continue;
                }
                // Minimum food is guaranteed, but its square is unknowable to
                // the live snake. A stable hash sample avoids treating it as
                // either adversarial or hero-favouring while keeping one path.
                let hash = game.state_key(state).expect("state key");
                let index = hash as usize % available.count_ones() as usize;
                let cell = bits(available).nth(index).expect("available food");
                game.apply_chance(state, ChanceAction::PlaceFood(cell));
            }
        }
    }
}

fn full_replies<const N: usize>(
    moves: &[ArrayVec<Direction, 4>; N],
    state: &BoardState<N>,
    hero: usize,
) -> ArrayVec<[Direction; N], MAX_JOINT_REPLIES> {
    let mut replies = ArrayVec::new();
    replies.push(std::array::from_fn(|p| state.snakes[p].heading()));
    for player in 0..N {
        if player == hero || !state.snakes[player].is_alive() {
            continue;
        }
        let prior = replies;
        replies = ArrayVec::new();
        for base in prior {
            for &action in &moves[player] {
                let mut joint = base;
                joint[player] = action;
                replies.push(joint);
            }
        }
    }
    replies
}

fn mcs_replies<const N: usize>(
    moves: &[ArrayVec<Direction, 4>; N],
    state: &BoardState<N>,
    hero: usize,
) -> ArrayVec<[Direction; N], MAX_JOINT_REPLIES> {
    let count = (0..N)
        .filter(|&player| player != hero && state.snakes[player].is_alive())
        .map(|player| moves[player].len())
        .max()
        .unwrap_or(1);
    (0..count)
        .map(|index| {
            std::array::from_fn(|player| {
                if player == hero || !state.snakes[player].is_alive() {
                    state.snakes[player].heading()
                } else {
                    moves[player][index.min(moves[player].len() - 1)]
                }
            })
        })
        .collect()
}

fn brs_plus_replies<const N: usize>(
    moves: &[ArrayVec<Direction, 4>; N],
    state: &BoardState<N>,
    hero: usize,
) -> ArrayVec<[Direction; N], MAX_JOINT_REPLIES> {
    let default = std::array::from_fn(|player| {
        if player == hero || !state.snakes[player].is_alive() {
            state.snakes[player].heading()
        } else {
            moves[player][0]
        }
    });
    let mut replies = ArrayVec::new();
    replies.push(default);
    for player in 0..N {
        if player == hero || !state.snakes[player].is_alive() {
            continue;
        }
        for &action in moves[player].iter().skip(1) {
            let mut joint = default;
            joint[player] = action;
            replies.push(joint);
        }
    }
    replies
}

fn terminal_score<const N: usize>(state: &BoardState<N>, hero: usize, ply: u16) -> Score {
    let distance = ply.min(999) as Score;
    if !state.snakes[hero].is_alive() {
        if state.alive_count() == 0 {
            DRAW + distance
        } else {
            LOSS + distance
        }
    } else if state.alive_count() == 1 {
        WIN - distance
    } else {
        unreachable!("terminal score requested for a live multiplayer state")
    }
}

#[derive(Clone, Copy, Debug)]
struct AreaControl {
    hero: u128,
    enemies: u128,
    close_hero: u128,
    close_enemies: u128,
    food_distance: Score,
}

impl<const N: usize> Searcher<N> {
    fn evaluate(&self, game: &Battlesnake<N>, state: &BoardState<N>, hero: usize) -> Score {
        if !state.snakes[hero].is_alive() || state.alive_count() <= 1 {
            return terminal_score(state, hero, 0);
        }
        let features = evaluation_features(game, state, hero);
        if let Some(solved) = separated_endgame(game, state, hero, &features.1) {
            return solved;
        }
        let progress = state.turn.min(self.config.weights.late_game_turn) as i32;
        let end = i32::from(self.config.weights.late_game_turn.max(1));
        let mut early = 0i32;
        let mut late = 0i32;
        for (index, feature) in features.0.into_iter().enumerate() {
            early += i32::from(self.config.weights.early[index]) * feature;
            late += i32::from(self.config.weights.late[index]) * feature;
        }
        let blended = (early * (end - progress) + late * progress) / end;
        blended.clamp(i32::from(LOSS + 1_000), i32::from(WIN - 1_000)) as Score
    }
}

/// Feature vector used by the handcrafted evaluator. Exposing this makes the
/// later weight ablation/tuning stage reproducible without duplicating search
/// internals in the trainer.
pub fn features<const N: usize>(
    game: &Battlesnake<N>,
    state: &BoardState<N>,
    hero: usize,
) -> [i32; FEATURE_COUNT] {
    evaluation_features(game, state, hero).0
}

fn evaluation_features<const N: usize>(
    game: &Battlesnake<N>,
    state: &BoardState<N>,
    hero: usize,
) -> ([i32; FEATURE_COUNT], AreaControl) {
    let area = area_control(game, state, hero, 5);
    let me = state.snakes[hero];
    let mut largest_enemy_len = 0;
    let mut lowest_enemy_health = 100;
    let mut enemy_mobility = 0;
    let mut enemies = 0;
    for player in 0..N {
        if player == hero || !state.snakes[player].is_alive() {
            continue;
        }
        enemies += 1;
        largest_enemy_len = largest_enemy_len.max(state.snakes[player].len() as i32);
        lowest_enemy_health = lowest_enemy_health.min(i32::from(state.snakes[player].health()));
        enemy_mobility += safe_mobility(game, state, player);
    }
    let length_diff = me.len() as i32 - largest_enemy_len;
    let being_longer = if length_diff == 0 {
        0
    } else {
        // Shapeshifter scales the raw length gap by board width before a
        // base-1.5 logarithm, then scales once more. The steep first step is
        // intentional: becoming just one cell longer changes head-to-head
        // legality everywhere the two influence fronts meet.
        let scaled = length_diff.unsigned_abs() as f64 * SIDE as f64 + 1.0;
        (scaled.log(1.5) * SIDE as f64) as i32 * length_diff.signum()
    };
    let my_food = (area.hero & state.food).count_ones() as i32;
    let enemy_food = (area.enemies & state.food).count_ones() as i32;
    let my_tails = controlled_tails(state, area.hero);
    let enemy_tails = controlled_tails(state, area.enemies);
    let my_area = checkerboard_capacity(area.hero);
    let enemy_area = checkerboard_capacity(area.enemies);
    let my_close = area.close_hero.count_ones() as i32;
    let enemy_close = area.close_enemies.count_ones() as i32;
    let hazard_free_my = (area.hero & !state.hazards).count_ones() as i32;
    let hazard_free_enemy = (area.enemies & !state.hazards).count_ones() as i32;
    let my_mobility = safe_mobility(game, state, hero);

    (
        [
            i32::from(me.health()),
            lowest_enemy_health,
            length_diff * SIDE as i32,
            being_longer,
            my_food - enemy_food,
            my_area - enemy_area,
            my_close - enemy_close,
            SIDE as i32 - i32::from(area.food_distance),
            my_tails - enemy_tails,
            my_mobility * enemies.max(1) - enemy_mobility,
            hazard_free_my - hazard_free_enemy,
            (me.len() & 1) as i32,
        ],
        area,
    )
}

fn area_control<const N: usize>(
    game: &Battlesnake<N>,
    state: &BoardState<N>,
    hero: usize,
    close_distance: usize,
) -> AreaControl {
    let mut hero_area = bit(state.snakes[hero].head());
    let mut enemy_area = 0;
    let mut longest_enemy = 0;
    for player in 0..N {
        if player != hero && state.snakes[player].is_alive() {
            enemy_area |= bit(state.snakes[player].head());
            longest_enemy = longest_enemy.max(state.snakes[player].len());
        }
    }
    // Release one old body segment per ply. This is the useful optimistic
    // approximation for territory search: it follows the canonical
    // tail-before-growth update, while deliberately ignoring future eating.
    // Counts preserve doubled tails and the stacked opening correctly.
    let mut bodies = [[0u8; MAX_BODY_LEN]; N];
    let mut body_lens = [0usize; N];
    let mut occupied_counts = [0u8; CELLS];
    let mut blocked = 0u128;
    for (player, snake) in state.snakes.iter().enumerate() {
        if !snake.is_alive() {
            continue;
        }
        for (index, cell) in snake.cells().enumerate() {
            bodies[player][index] = cell;
            body_lens[player] += 1;
            occupied_counts[cell as usize] = occupied_counts[cell as usize].saturating_add(1);
            blocked |= bit(cell);
        }
    }
    let mut hero_front = hero_area;
    let mut enemy_front = enemy_area;
    let mut close_hero = hero_area;
    let mut close_enemies = enemy_area;
    let mut food_distance = None;
    let hero_longer = state.snakes[hero].len().cmp(&longest_enemy);

    for distance in 1..=CELLS {
        for player in 0..N {
            if distance > body_lens[player] {
                continue;
            }
            let released = bodies[player][body_lens[player] - distance] as usize;
            occupied_counts[released] -= 1;
            if occupied_counts[released] == 0 {
                blocked &= !bit(released as u8);
            }
        }
        let unclaimed = board_mask() & !blocked & !(hero_area | enemy_area);
        let mut next_hero = neighbors(hero_front, game.rules.mode) & unclaimed;
        let mut next_enemies = neighbors(enemy_front, game.rules.mode) & unclaimed;
        let contested = next_hero & next_enemies;
        match hero_longer {
            std::cmp::Ordering::Greater if N == 2 => next_enemies &= !contested,
            std::cmp::Ordering::Less if N == 2 => next_hero &= !contested,
            _ => {
                next_hero &= !contested;
                next_enemies &= !contested;
            }
        }
        if next_hero == 0 && next_enemies == 0 {
            break;
        }
        hero_area |= next_hero;
        enemy_area |= next_enemies;
        hero_front = next_hero;
        enemy_front = next_enemies;
        if food_distance.is_none() && next_hero & state.food != 0 {
            food_distance = Some(distance as Score);
        }
        if distance == close_distance {
            close_hero = hero_area;
            close_enemies = enemy_area;
        }
    }
    if close_distance >= CELLS || close_hero == bit(state.snakes[hero].head()) {
        close_hero = hero_area;
        close_enemies = enemy_area;
    }
    AreaControl {
        hero: hero_area,
        enemies: enemy_area,
        close_hero,
        close_enemies,
        food_distance: food_distance.unwrap_or(SIDE as Score),
    }
}

#[inline]
fn neighbors(board: u128, mode: Mode) -> u128 {
    let mut expanded = ((board & !RIGHT_EDGE) << 1)
        | ((board & !LEFT_EDGE) >> 1)
        | (board << SIDE)
        | (board >> SIDE);
    if mode.wrapped() {
        expanded |= ((board & RIGHT_EDGE) >> (SIDE - 1))
            | ((board & LEFT_EDGE) << (SIDE - 1))
            | ((board & BOTTOM_EDGE) << (SIDE * (SIDE - 1)))
            | ((board & TOP_EDGE) >> (SIDE * (SIDE - 1)));
    }
    expanded & board_mask()
}

fn body_obstacles<const N: usize>(state: &BoardState<N>) -> u128 {
    let mut occupied = 0;
    for snake in state.snakes {
        if !snake.is_alive() {
            continue;
        }
        // One final segment is removed by every standard move. If the tail is
        // doubled, the preceding copy remains and therefore stays occupied.
        for cell in snake.cells().take(snake.len().saturating_sub(1)) {
            occupied |= bit(cell);
        }
    }
    occupied
}

fn controlled_tails<const N: usize>(state: &BoardState<N>, area: u128) -> i32 {
    state
        .snakes
        .iter()
        .filter(|snake| snake.is_alive() && area & bit(snake.tail()) != 0)
        .count() as i32
}

fn checkerboard_capacity(area: u128) -> i32 {
    let first = (area & CHECKERBOARD).count_ones() as i32;
    let second = (area & !CHECKERBOARD).count_ones() as i32;
    2 * first.min(second) + i32::from(first != second)
}

fn safe_mobility<const N: usize>(
    game: &Battlesnake<N>,
    state: &BoardState<N>,
    player: usize,
) -> i32 {
    let obstacles = body_obstacles(state);
    Direction::ALL
        .into_iter()
        .filter(|&action| {
            move_order_score_with_obstacles(game, state, player, action, obstacles) > -5_000
        })
        .count() as i32
}

fn move_order_score_with_obstacles<const N: usize>(
    game: &Battlesnake<N>,
    state: &BoardState<N>,
    player: usize,
    action: Direction,
    obstacles: u128,
) -> i32 {
    let snake = state.snakes[player];
    let Some(destination) = next_cell(snake.head(), action, game.rules.mode.wrapped()) else {
        return -20_000;
    };
    if obstacles & bit(destination) != 0 {
        return -10_000;
    }
    if state.hazards & bit(destination) != 0
        && state.food & bit(destination) == 0
        && snake.health() <= game.rules.hazard_damage + 1
    {
        return -8_000;
    }
    let mut score = 0;
    let exits = neighbors(bit(destination), game.rules.mode) & !obstacles;
    score += exits.count_ones() as i32 * 200;
    if state.food & bit(destination) != 0 {
        score += 400 + (100 - i32::from(snake.health())) * 10;
    }
    let (x, y) = xy(destination);
    let center = (SIDE / 2) as i32;
    score -= ((i32::from(x) - center).abs() + (i32::from(y) - center).abs()) * 4;
    score
}

fn is_stable<const N: usize>(game: &Battlesnake<N>, state: &BoardState<N>, hero: usize) -> bool {
    if safe_mobility(game, state, hero) <= 1 {
        return false;
    }
    let my_head = state.snakes[hero].head();
    let (mx, my) = xy(my_head);
    for player in 0..N {
        if player == hero || !state.snakes[player].is_alive() {
            continue;
        }
        let (x, y) = xy(state.snakes[player].head());
        let dx = (i16::from(mx) - i16::from(x)).unsigned_abs() as usize;
        let dy = (i16::from(my) - i16::from(y)).unsigned_abs() as usize;
        if dx + dy < 3 {
            return false;
        }
    }
    neighbors(bit(my_head), game.rules.mode) & state.food == 0
}

fn separated_endgame<const N: usize>(
    game: &Battlesnake<N>,
    state: &BoardState<N>,
    hero: usize,
    area: &AreaControl,
) -> Option<Score> {
    if N != 2 || state.turn < 50 || neighbors(area.hero, game.rules.mode) & area.enemies != 0 {
        return None;
    }
    let enemy = (0..N).find(|&player| player != hero && state.snakes[player].is_alive())?;
    let my_capacity = checkerboard_capacity(area.hero);
    let enemy_capacity = checkerboard_capacity(area.enemies);
    let my_food = area.hero & state.food != 0;
    let enemy_food = area.enemies & state.food != 0;
    let my_life = if my_food {
        my_capacity
    } else {
        my_capacity.min(i32::from(state.snakes[hero].health()))
    };
    let enemy_life = if enemy_food {
        enemy_capacity
    } else {
        enemy_capacity.min(i32::from(state.snakes[enemy].health()))
    };
    // Only return a near-terminal bound with a conservative two-turn margin;
    // close estimates stay in the ordinary linear evaluator.
    if my_life + 2 < enemy_life {
        Some(LOSS + 1_000 + my_life.clamp(0, 999) as Score)
    } else if enemy_life + 2 < my_life {
        Some(WIN - 1_000 - enemy_life.clamp(0, 999) as Score)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::battlesnake::{BattleSnake, Rules, cell};

    fn snake(cells: &[(u8, u8)], health: i16, heading: Direction) -> BattleSnake {
        let body: Vec<_> = cells.iter().map(|&(x, y)| cell(x, y)).collect();
        BattleSnake::from_cells(&body, health, heading)
    }

    fn fast_config(model: OpponentModel) -> SearchConfig {
        SearchConfig {
            time_limit: Duration::from_secs(2),
            max_depth: 2,
            quiescence_depth: 1,
            opponent_model: model,
            tt_bits: 10,
            weights: EvaluationWeights::default(),
        }
    }

    #[test]
    fn packed_joint_action_round_trips_all_players() {
        let joint = [
            Direction::Left,
            Direction::Down,
            Direction::Right,
            Direction::Up,
        ];
        assert_eq!(unpack_joint::<4>(pack_joint(&joint)), joint);
    }

    #[test]
    fn all_fatal_moves_are_retained_to_search_for_a_joint_draw() {
        let game = Battlesnake::<2>::new(Rules {
            food_spawn_chance: 0,
            minimum_food: 0,
            ..Rules::default()
        });
        let state = BoardState::from_parts(
            [
                snake(&[(0, 0)], 100, Direction::Up),
                snake(&[(1, 0), (0, 1), (1, 1)], 100, Direction::Left),
            ],
            0,
            0,
            0,
        );
        let searcher = Searcher::new(fast_config(OpponentModel::Full));

        assert_eq!(searcher.ordered_moves(&game, &state, 0, None).len(), 4);
    }

    #[test]
    fn move_combination_covers_every_enemy_action_once() {
        let game = Battlesnake::<4>::new(Rules::default());
        let state = game.initial_state();
        let per_player = std::array::from_fn(|_| Direction::ALL.into_iter().collect());
        let replies = mcs_replies(&per_player, &state, 0);
        assert_eq!(replies.len(), 4);
        for player in 1..4 {
            for action in Direction::ALL {
                assert!(replies.iter().any(|joint| joint[player] == action));
            }
        }
    }

    #[test]
    fn best_reply_plus_varies_only_one_enemy_from_default() {
        let game = Battlesnake::<4>::new(Rules::default());
        let state = game.initial_state();
        let per_player = std::array::from_fn(|_| Direction::ALL.into_iter().collect());
        let replies = brs_plus_replies(&per_player, &state, 0);
        assert_eq!(replies.len(), 10);
        for joint in replies.iter().skip(1) {
            assert_eq!(
                (1..4)
                    .filter(|&player| joint[player] != Direction::Up)
                    .count(),
                1
            );
        }
    }

    #[test]
    fn search_rejects_a_head_to_head_draw_when_survival_exists() {
        let game = Battlesnake::<2>::new(Rules {
            food_spawn_chance: 0,
            minimum_food: 0,
            ..Rules::default()
        });
        let state = BoardState::from_parts(
            [
                snake(&[(1, 1), (1, 0), (0, 0)], 100, Direction::Up),
                snake(&[(3, 1), (3, 0), (4, 0)], 100, Direction::Up),
            ],
            0,
            0,
            0,
        );
        let result = Searcher::new(fast_config(OpponentModel::Full)).search(&game, &state, 0);
        assert_ne!(result.action, Direction::Right);
        assert_ne!(result.action, Direction::Down);
        assert!(result.stats.nodes > 0);
    }

    #[test]
    fn search_finds_a_forced_winning_head_to_head() {
        let game = Battlesnake::<2>::new(Rules {
            food_spawn_chance: 0,
            minimum_food: 0,
            ..Rules::default()
        });
        let state = BoardState::from_parts(
            [
                snake(&[(8, 0), (8, 1), (8, 2), (7, 2)], 100, Direction::Down),
                snake(&[(10, 0), (10, 1), (9, 1)], 100, Direction::Down),
            ],
            0,
            0,
            20,
        );
        let result = Searcher::new(fast_config(OpponentModel::Full)).search(&game, &state, 0);
        assert_eq!(result.action, Direction::Right);
        assert!(
            result.stats.score > 0,
            "BNS reports its final separating threshold, not an exact mate score: {result:?}"
        );
    }

    #[test]
    fn features_are_antisymmetric_on_a_symmetric_duel() {
        let game = Battlesnake::<2>::new(Rules::default());
        let state = BoardState::from_parts(
            [
                snake(&[(2, 5), (1, 5), (0, 5)], 80, Direction::Right),
                snake(&[(8, 5), (9, 5), (10, 5)], 80, Direction::Left),
            ],
            bit(cell(5, 5)),
            0,
            12,
        );
        let left = features(&game, &state, 0);
        let right = features(&game, &state, 1);
        for index in [2, 3, 4, 5, 6, 8, 9, 10] {
            assert_eq!(left[index], -right[index], "{}", FEATURE_NAMES[index]);
        }
        assert_eq!(left[0], right[0]);
        assert_eq!(left[1], right[1]);
        assert_eq!(left[7], right[7]);
        assert_eq!(left[11], right[11]);
    }
}
