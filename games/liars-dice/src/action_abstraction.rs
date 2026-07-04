//! Shared action abstraction for large Liar's Dice decision nodes.
//!
//! The live game's mid-round action set is already tiny: at most two relative
//! raises plus Call Liar and Call Exact. The expensive node is a free opening,
//! where 5p5d6f exposes up to 150 `Open(q, face)` actions. This module keeps the
//! full mid-round set and narrows only free openings to a small, belief-grounded
//! candidate set that search/CFR contenders can share. MCCFR can also train on a
//! depth-capped view of the same abstracted game, making the cap explicit in the
//! caller instead of silently pretending a full-horizon solve finished.

use std::cmp::Ordering;

use game_core::{Agent, Determinizer, Game, Rng, Turn, playout_from};
use solvers::Mccfr;
use solvers::mcts::Mcts;
use solvers::qlearn::{QConfig, QLearner};

use crate::rebel::principled_open_cap;
use crate::{Action, BidConditioned, LdState, LiarsDice, ProbabilisticAgent};

/// Why a candidate survived the abstraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidateReason {
    /// Challenge actions: Call Liar / Call Exact.
    Call,
    /// The lowest legal continuation of the bid ladder.
    MinRaise,
    /// A bid on a face the acting player actually holds.
    OwnFace,
    /// A quantity chosen from `P(bid true | own dice)` thresholds.
    PosteriorQuantile,
}

/// One legal action selected by the abstraction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CandidateAction {
    /// Index into `game.legal_actions(state)` / `game.action_at(state, index)`.
    pub index: usize,
    pub action: Action,
    pub reason: CandidateReason,
    /// `P(action's bid is true | own dice)` for opening bids; `NaN` for calls.
    pub truth_prob: f64,
}

/// Tunable knobs for [`candidate_actions`].
#[derive(Clone, Debug)]
pub struct ActionAbstractionConfig {
    /// Maximum candidates returned at a wide free-open node.
    pub max_candidates: usize,
    /// Opening bid truth-probability thresholds. Lower thresholds create more
    /// pressure/polarized bids; higher thresholds stay plausible.
    pub opening_truth_thresholds: [f64; 4],
    /// Include `Open(1, face)` for each face before capping.
    pub include_min_open_each_face: bool,
    /// Optional maximum opening quantity. `None` uses ReBeL's principled cap.
    pub open_qty_cap: Option<u8>,
}

impl Default for ActionAbstractionConfig {
    fn default() -> Self {
        Self {
            max_candidates: 24,
            opening_truth_thresholds: [0.75, 0.50, 0.25, 0.10],
            include_min_open_each_face: true,
            open_qty_cap: None,
        }
    }
}

/// Determinized rollout over [`candidate_actions`].
///
/// This is the Liar's-Dice-specific counterpart to `solvers::Rollout`: it keeps
/// common random worlds across candidates, but it can search a narrowed opening
/// set instead of falling back to the base policy at wide free-open nodes.
pub struct AbstractedRolloutAgent {
    pub rollouts: u32,
    pub base: ProbabilisticAgent,
    pub determinizer: BidConditioned,
    pub abstraction: ActionAbstractionConfig,
}

impl AbstractedRolloutAgent {
    pub fn new(rollouts: u32) -> Self {
        Self {
            rollouts,
            base: ProbabilisticAgent::default_agent(),
            determinizer: BidConditioned::default(),
            abstraction: ActionAbstractionConfig::default(),
        }
    }

    pub fn with_config(
        rollouts: u32,
        base: ProbabilisticAgent,
        determinizer: BidConditioned,
        abstraction: ActionAbstractionConfig,
    ) -> Self {
        Self {
            rollouts,
            base,
            determinizer,
            abstraction,
        }
    }
}

/// Imperfect-information MCTS by determinization.
///
/// Each move samples `worlds` hidden-dice assignments consistent with the
/// acting player's information, runs the generic perfect-information [`Mcts`]
/// for `sims_per_world` simulations in each sampled world, and aggregates root
/// visit counts over the shared abstracted action set.
pub struct DeterminizedMctsAgent {
    pub worlds: u32,
    pub sims_per_world: u32,
    pub determinizer: BidConditioned,
    pub abstraction: ActionAbstractionConfig,
}

impl DeterminizedMctsAgent {
    pub fn new(worlds: u32, sims_per_world: u32) -> Self {
        Self {
            worlds,
            sims_per_world,
            determinizer: BidConditioned::default(),
            abstraction: ActionAbstractionConfig::default(),
        }
    }

    pub fn with_config(
        worlds: u32,
        sims_per_world: u32,
        determinizer: BidConditioned,
        abstraction: ActionAbstractionConfig,
    ) -> Self {
        Self {
            worlds,
            sims_per_world,
            determinizer,
            abstraction,
        }
    }
}

/// A Liar's Dice [`Game`] view whose decision nodes are restricted by
/// [`candidate_actions`].
///
/// The state and action tokens are the live game's `LdState`/`Action`, so
/// policies trained here can be translated back into ordinary Liar's Dice
/// legal-action indices without an adapter action type.
#[derive(Clone)]
pub struct AbstractedLiarsDice {
    pub game: LiarsDice,
    pub abstraction: ActionAbstractionConfig,
}

impl AbstractedLiarsDice {
    pub fn new(game: LiarsDice) -> Self {
        Self {
            game,
            abstraction: ActionAbstractionConfig::default(),
        }
    }

    pub fn with_config(game: LiarsDice, abstraction: ActionAbstractionConfig) -> Self {
        Self { game, abstraction }
    }

    fn same_rules(&self, game: &LiarsDice) -> bool {
        self.game.players == game.players
            && self.game.dice == game.dice
            && self.game.faces == game.faces
    }
}

impl Game for AbstractedLiarsDice {
    type State = LdState;
    type Action = Action;

    fn num_players(&self) -> usize {
        self.game.num_players()
    }

    fn initial_state(&self) -> Self::State {
        self.game.initial_state()
    }

    fn turn(&self, state: &Self::State) -> game_core::Turn {
        self.game.turn(state)
    }

    fn is_terminal(&self, state: &Self::State) -> bool {
        self.game.is_terminal(state)
    }

    fn returns(&self, state: &Self::State, player: usize) -> f64 {
        self.game.returns(state, player)
    }

    fn max_return(&self) -> f64 {
        self.game.max_return()
    }

    fn legal_actions(&self, state: &Self::State) -> Vec<Self::Action> {
        match self.game.turn(state) {
            game_core::Turn::Player(player) => {
                candidate_actions(&self.game, state, player, &self.abstraction)
                    .into_iter()
                    .map(|c| c.action)
                    .collect()
            }
            game_core::Turn::Chance => self.game.legal_actions(state),
        }
    }

    fn num_actions(&self, state: &Self::State) -> usize {
        match self.game.turn(state) {
            game_core::Turn::Player(player) => {
                candidate_actions(&self.game, state, player, &self.abstraction).len()
            }
            game_core::Turn::Chance => self.game.num_actions(state),
        }
    }

    fn action_at(&self, state: &Self::State, i: usize) -> Self::Action {
        match self.game.turn(state) {
            game_core::Turn::Player(player) => {
                candidate_actions(&self.game, state, player, &self.abstraction)[i].action
            }
            game_core::Turn::Chance => self.game.action_at(state, i),
        }
    }

    fn chance_outcomes(&self, state: &Self::State) -> Vec<(Self::Action, f64)> {
        self.game.chance_outcomes(state)
    }

    fn sample_chance(&self, state: &Self::State, rng: &mut Rng) -> (Self::Action, f64) {
        self.game.sample_chance(state, rng)
    }

    fn sample_chance_action(&self, state: &Self::State, rng: &mut Rng) -> Self::Action {
        self.game.sample_chance_action(state, rng)
    }

    fn apply(&self, state: &mut Self::State, action: Self::Action) {
        self.game.apply(state, action);
    }

    fn infoset_key(&self, state: &Self::State, player: usize) -> u64 {
        self.game.infoset_key(state, player)
    }

    fn state_key(&self, state: &Self::State) -> Option<u64> {
        self.game.state_key(state)
    }

    fn action_id(&self, action: &Self::Action) -> u64 {
        self.game.action_id(action)
    }
}

#[derive(Clone)]
struct MccfrTrainingState {
    inner: LdState,
    decision_plies: u16,
}

#[derive(Clone)]
struct MccfrTrainingGame {
    inner: AbstractedLiarsDice,
    max_decision_plies: Option<u16>,
}

impl MccfrTrainingGame {
    fn new(
        game: LiarsDice,
        abstraction: ActionAbstractionConfig,
        max_decision_plies: Option<u16>,
    ) -> Self {
        Self {
            inner: AbstractedLiarsDice::with_config(game, abstraction),
            max_decision_plies,
        }
    }

    fn live_state(&self, state: &LdState) -> MccfrTrainingState {
        MccfrTrainingState {
            inner: state.clone(),
            decision_plies: 0,
        }
    }

    fn same_rules(&self, game: &LiarsDice) -> bool {
        self.inner.same_rules(game)
    }

    fn capped(&self, state: &MccfrTrainingState) -> bool {
        self.max_decision_plies
            .is_some_and(|limit| state.decision_plies >= limit)
            && !self.inner.game.is_terminal(&state.inner)
    }
}

impl Game for MccfrTrainingGame {
    type State = MccfrTrainingState;
    type Action = Action;

    fn num_players(&self) -> usize {
        self.inner.num_players()
    }

    fn initial_state(&self) -> Self::State {
        MccfrTrainingState {
            inner: self.inner.initial_state(),
            decision_plies: 0,
        }
    }

    fn turn(&self, state: &Self::State) -> Turn {
        self.inner.turn(&state.inner)
    }

    fn is_terminal(&self, state: &Self::State) -> bool {
        self.capped(state) || self.inner.is_terminal(&state.inner)
    }

    fn returns(&self, state: &Self::State, player: usize) -> f64 {
        if self.inner.is_terminal(&state.inner) {
            self.inner.returns(&state.inner, player)
        } else {
            dice_count_return(&state.inner, player)
        }
    }

    fn max_return(&self) -> f64 {
        self.inner.max_return()
    }

    fn legal_actions(&self, state: &Self::State) -> Vec<Self::Action> {
        self.inner.legal_actions(&state.inner)
    }

    fn num_actions(&self, state: &Self::State) -> usize {
        self.inner.num_actions(&state.inner)
    }

    fn action_at(&self, state: &Self::State, i: usize) -> Self::Action {
        self.inner.action_at(&state.inner, i)
    }

    fn chance_outcomes(&self, state: &Self::State) -> Vec<(Self::Action, f64)> {
        self.inner.chance_outcomes(&state.inner)
    }

    fn sample_chance(&self, state: &Self::State, rng: &mut Rng) -> (Self::Action, f64) {
        self.inner.sample_chance(&state.inner, rng)
    }

    fn sample_chance_action(&self, state: &Self::State, rng: &mut Rng) -> Self::Action {
        self.inner.sample_chance_action(&state.inner, rng)
    }

    fn apply(&self, state: &mut Self::State, action: Self::Action) {
        let was_player = matches!(self.inner.turn(&state.inner), Turn::Player(_));
        self.inner.apply(&mut state.inner, action);
        if was_player {
            state.decision_plies = state.decision_plies.saturating_add(1);
        }
    }

    fn infoset_key(&self, state: &Self::State, player: usize) -> u64 {
        self.inner.infoset_key(&state.inner, player)
    }

    fn state_key(&self, state: &Self::State) -> Option<u64> {
        self.inner.state_key(&state.inner)
    }

    fn action_id(&self, action: &Self::Action) -> u64 {
        self.inner.action_id(action)
    }
}

fn dice_count_return(state: &LdState, player: usize) -> f64 {
    let live: Vec<usize> = state
        .dice_left()
        .iter()
        .enumerate()
        .filter_map(|(p, &dice)| (dice > 0).then_some(p))
        .collect();
    if live.is_empty() || !live.contains(&player) {
        return -1.0;
    }
    let max_dice = live
        .iter()
        .map(|&p| state.dice_left()[p])
        .max()
        .unwrap_or(0);
    let leaders: Vec<usize> = live
        .iter()
        .copied()
        .filter(|&p| state.dice_left()[p] == max_dice)
        .collect();
    if leaders.len() == live.len() {
        0.0
    } else if leaders.contains(&player) {
        1.0 / leaders.len() as f64
    } else {
        -1.0 / (live.len() - leaders.len()) as f64
    }
}

/// Average-strategy play from MCCFR+ trained on [`AbstractedLiarsDice`].
pub struct AbstractedMccfrAgent {
    solver: Mccfr<MccfrTrainingGame>,
}

impl AbstractedMccfrAgent {
    pub fn train(game: LiarsDice, iterations: u64, seed: u64) -> Self {
        Self::train_with_config(game, iterations, seed, ActionAbstractionConfig::default())
    }

    pub fn train_with_config(
        game: LiarsDice,
        iterations: u64,
        seed: u64,
        abstraction: ActionAbstractionConfig,
    ) -> Self {
        Self::train_with_config_and_max_decision_plies(game, iterations, seed, abstraction, None)
    }

    pub fn train_with_config_and_max_decision_plies(
        game: LiarsDice,
        iterations: u64,
        seed: u64,
        abstraction: ActionAbstractionConfig,
        max_decision_plies: Option<u16>,
    ) -> Self {
        let mut solver = Mccfr::new(
            MccfrTrainingGame::new(game, abstraction, max_decision_plies),
            seed,
        );
        solver.run(iterations);
        Self { solver }
    }

    pub fn iterations(&self) -> u64 {
        self.solver.iterations()
    }

    pub fn num_infosets(&self) -> usize {
        self.solver.num_infosets()
    }
}

impl Agent<LiarsDice> for AbstractedMccfrAgent {
    fn act(&self, game: &LiarsDice, state: &LdState, player: usize, rng: &mut Rng) -> usize {
        if game.num_actions(state) == 1 {
            return 0;
        }
        debug_assert!(
            self.solver.game().same_rules(game),
            "MCCFR agent was trained for a different Liar's Dice config"
        );
        let training_state = self.solver.game().live_state(state);
        let action_index = self.solver.sample_action(&training_state, player, rng);
        let action = self.solver.game().action_at(&training_state, action_index);
        game.legal_actions(state)
            .into_iter()
            .position(|a| a == action)
            .unwrap_or(0)
    }
}

/// Greedy play from tabular Q-learning trained on [`AbstractedLiarsDice`].
///
/// This is intentionally a control contender: it uses the same abstracted legal
/// action set as MCCFR/search, but learns purely from terminal self-play returns.
pub struct AbstractedQAgent {
    game: AbstractedLiarsDice,
    learner: QLearner<AbstractedLiarsDice>,
}

impl AbstractedQAgent {
    pub fn train(game: LiarsDice, episodes: u64, seed: u64) -> Self {
        Self::train_with_config(
            game,
            episodes,
            seed,
            QConfig::default(),
            ActionAbstractionConfig::default(),
        )
    }

    pub fn train_with_config(
        game: LiarsDice,
        episodes: u64,
        seed: u64,
        q_config: QConfig,
        abstraction: ActionAbstractionConfig,
    ) -> Self {
        let game = AbstractedLiarsDice::with_config(game, abstraction);
        let mut learner = QLearner::new(game.clone(), q_config, seed);
        learner.train_episodes(episodes);
        Self { game, learner }
    }

    pub fn episodes_trained(&self) -> u64 {
        self.learner.episodes_trained()
    }

    pub fn table_size(&self) -> usize {
        self.learner.table_size()
    }
}

impl Agent<LiarsDice> for AbstractedQAgent {
    fn act(&self, game: &LiarsDice, state: &LdState, player: usize, rng: &mut Rng) -> usize {
        if game.num_actions(state) == 1 {
            return 0;
        }
        debug_assert!(
            self.game.same_rules(game),
            "Q-learning agent was trained for a different Liar's Dice config"
        );
        let action_index = self.learner.greedy().act(&self.game, state, player, rng);
        let action = self.game.action_at(state, action_index);
        game.legal_actions(state)
            .into_iter()
            .position(|a| a == action)
            .unwrap_or(0)
    }
}

impl Agent<LiarsDice> for DeterminizedMctsAgent {
    fn act(&self, game: &LiarsDice, state: &LdState, player: usize, rng: &mut Rng) -> usize {
        if game.num_actions(state) == 1 {
            return 0;
        }
        let candidates = candidate_actions(game, state, player, &self.abstraction);
        if candidates.is_empty() {
            return 0;
        }
        if candidates.len() == 1 {
            return candidates[0].index;
        }

        let seed0 = rng.next_u64();
        let mut totals = vec![0u64; candidates.len()];
        for w in 0..self.worlds.max(1) {
            let mut world_rng =
                Rng::new(seed0 ^ (u64::from(w) + 1).wrapping_mul(0xD1B5_4A32_D192_ED03));
            let mut world = state.clone();
            self.determinizer
                .determinize(game, &mut world, player, &mut world_rng);
            let restricted = RootRestrictedLiarsDice {
                game,
                root_candidates: candidates.clone(),
            };
            let root = RootRestrictedState::Root(world);
            let mcts = Mcts::new(self.sims_per_world.max(1));
            for (slot, visits) in mcts
                .root_visits(&restricted, &root, player, &mut world_rng)
                .into_iter()
                .enumerate()
            {
                totals[slot] += u64::from(visits);
            }
        }

        let mut best = 0;
        for i in 1..totals.len() {
            if totals[i] > totals[best]
                || (totals[i] == totals[best] && candidates[i].index < candidates[best].index)
            {
                best = i;
            }
        }
        candidates[best].index
    }
}

struct RootRestrictedLiarsDice<'a> {
    game: &'a LiarsDice,
    root_candidates: Vec<CandidateAction>,
}

#[derive(Clone)]
enum RootRestrictedState {
    Root(LdState),
    Inner(LdState),
}

impl RootRestrictedState {
    fn inner(&self) -> &LdState {
        match self {
            RootRestrictedState::Root(s) | RootRestrictedState::Inner(s) => s,
        }
    }
}

impl Game for RootRestrictedLiarsDice<'_> {
    type State = RootRestrictedState;
    type Action = Action;

    fn num_players(&self) -> usize {
        self.game.num_players()
    }

    fn initial_state(&self) -> Self::State {
        RootRestrictedState::Root(self.game.initial_state())
    }

    fn turn(&self, state: &Self::State) -> game_core::Turn {
        self.game.turn(state.inner())
    }

    fn is_terminal(&self, state: &Self::State) -> bool {
        self.game.is_terminal(state.inner())
    }

    fn returns(&self, state: &Self::State, player: usize) -> f64 {
        self.game.returns(state.inner(), player)
    }

    fn max_return(&self) -> f64 {
        self.game.max_return()
    }

    fn legal_actions(&self, state: &Self::State) -> Vec<Self::Action> {
        match state {
            RootRestrictedState::Root(_) => self.root_candidates.iter().map(|c| c.action).collect(),
            RootRestrictedState::Inner(s) => self.game.legal_actions(s),
        }
    }

    fn num_actions(&self, state: &Self::State) -> usize {
        match state {
            RootRestrictedState::Root(_) => self.root_candidates.len(),
            RootRestrictedState::Inner(s) => self.game.num_actions(s),
        }
    }

    fn action_at(&self, state: &Self::State, i: usize) -> Self::Action {
        match state {
            RootRestrictedState::Root(_) => self.root_candidates[i].action,
            RootRestrictedState::Inner(s) => self.game.action_at(s, i),
        }
    }

    fn chance_outcomes(&self, state: &Self::State) -> Vec<(Self::Action, f64)> {
        self.game.chance_outcomes(state.inner())
    }

    fn sample_chance(&self, state: &Self::State, rng: &mut Rng) -> (Self::Action, f64) {
        self.game.sample_chance(state.inner(), rng)
    }

    fn sample_chance_action(&self, state: &Self::State, rng: &mut Rng) -> Self::Action {
        self.game.sample_chance_action(state.inner(), rng)
    }

    fn apply(&self, state: &mut Self::State, action: Self::Action) {
        match state {
            RootRestrictedState::Root(s) => {
                let mut next = s.clone();
                self.game.apply(&mut next, action);
                *state = RootRestrictedState::Inner(next);
            }
            RootRestrictedState::Inner(s) => self.game.apply(s, action),
        }
    }

    fn infoset_key(&self, state: &Self::State, player: usize) -> u64 {
        self.game.infoset_key(state.inner(), player)
    }

    fn action_id(&self, action: &Self::Action) -> u64 {
        self.game.action_id(action)
    }
}

impl Agent<LiarsDice> for AbstractedRolloutAgent {
    fn act(&self, game: &LiarsDice, state: &LdState, player: usize, rng: &mut Rng) -> usize {
        if game.num_actions(state) == 1 {
            return 0;
        }
        let candidates = candidate_actions(game, state, player, &self.abstraction);
        if candidates.is_empty() {
            return self.base.act(game, state, player, rng);
        }
        if candidates.len() == 1 {
            return candidates[0].index;
        }

        let seed0 = rng.next_u64();
        let seats: Vec<&dyn Agent<LiarsDice>> = (0..game.num_players())
            .map(|_| &self.base as &dyn Agent<LiarsDice>)
            .collect();
        let mut totals = vec![0.0f64; candidates.len()];
        for j in 0..self.rollouts.max(1) {
            let mut world_rng =
                Rng::new(seed0 ^ (u64::from(j) + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut world = state.clone();
            self.determinizer
                .determinize(game, &mut world, player, &mut world_rng);
            let rollout_rng = world_rng.clone();
            for (slot, candidate) in candidates.iter().enumerate() {
                let mut sim = world.clone();
                let mut sim_rng = rollout_rng.clone();
                game.apply(&mut sim, candidate.action);
                let terminal = playout_from(game, sim, &seats, &mut sim_rng);
                totals[slot] += game.returns(&terminal, player);
            }
        }

        let mut best = 0;
        for i in 1..totals.len() {
            if totals[i] > totals[best] {
                best = i;
            }
        }
        candidates[best].index
    }
}

/// Legal action candidates for `player` at `state`, in stable legal-action order.
pub fn candidate_actions(
    game: &LiarsDice,
    state: &LdState,
    player: usize,
    cfg: &ActionAbstractionConfig,
) -> Vec<CandidateAction> {
    if state.current_bid().0 != 0 {
        return game
            .legal_actions(state)
            .into_iter()
            .enumerate()
            .map(|(index, action)| CandidateAction {
                index,
                action,
                reason: match action {
                    Action::CallLiar | Action::CallExact => CandidateReason::Call,
                    Action::RaiseQuantity | Action::RaiseFace => CandidateReason::MinRaise,
                    Action::Open(_, _) | Action::Roll(_) => unreachable!(),
                },
                truth_prob: f64::NAN,
            })
            .collect();
    }

    let total = total_dice(game, state);
    let qty_cap = cfg
        .open_qty_cap
        .unwrap_or_else(|| principled_open_cap(total, game.faces))
        .clamp(1, total);
    let mut out = Vec::new();
    for face in 1..=game.faces {
        if cfg.include_min_open_each_face {
            add_open_candidate(
                game,
                state,
                player,
                &mut out,
                1,
                face,
                CandidateReason::MinRaise,
            );
        }
        let own = state.my_count(player, face);
        if own > 0 {
            add_open_candidate(
                game,
                state,
                player,
                &mut out,
                own.min(qty_cap),
                face,
                CandidateReason::OwnFace,
            );
        }
        for &threshold in &cfg.opening_truth_thresholds {
            let qty = largest_qty_at_truth(game, state, player, face, qty_cap, threshold);
            add_open_candidate(
                game,
                state,
                player,
                &mut out,
                qty,
                face,
                CandidateReason::PosteriorQuantile,
            );
        }
    }

    out.sort_by_key(|c| c.index);
    out.dedup_by_key(|c| c.index);
    let max_candidates = cfg.max_candidates.max(1);
    if out.len() > max_candidates {
        out.sort_by(|a, b| {
            score_candidate(game, state, player, b)
                .partial_cmp(&score_candidate(game, state, player, a))
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.index.cmp(&b.index))
        });
        out.truncate(max_candidates);
        out.sort_by_key(|c| c.index);
    }
    out
}

/// `P(Binomial(n, p) >= k)`, with fractional/negative needs handled by callers
/// that round into an integer threshold.
pub fn binom_sf(n: u32, p: f64, k: i64) -> f64 {
    if k <= 0 {
        return 1.0;
    }
    if k as u32 > n {
        return 0.0;
    }
    let mut term = (1.0 - p).powi(n as i32);
    let mut cdf_below = 0.0;
    for i in 0..k as u32 {
        cdf_below += term;
        term *= p * (n - i) as f64 / ((i + 1) as f64 * (1.0 - p));
    }
    (1.0 - cdf_below).clamp(0.0, 1.0)
}

/// `P(Open(q, face) is true | player sees their own dice)`.
pub fn bid_truth_prob(game: &LiarsDice, state: &LdState, player: usize, q: u8, face: u8) -> f64 {
    let total = total_dice(game, state);
    let my_dice = state.dice_left()[player];
    let unknown = (total - my_dice) as u32;
    let need = q as i64 - state.my_count(player, face) as i64;
    binom_sf(unknown, 1.0 / game.faces as f64, need)
}

fn total_dice(game: &LiarsDice, state: &LdState) -> u8 {
    state.dice_left()[..game.players as usize].iter().sum()
}

fn largest_qty_at_truth(
    game: &LiarsDice,
    state: &LdState,
    player: usize,
    face: u8,
    qty_cap: u8,
    threshold: f64,
) -> u8 {
    let mut best = 1;
    for qty in 1..=qty_cap {
        if bid_truth_prob(game, state, player, qty, face) >= threshold {
            best = qty;
        }
    }
    best
}

fn add_open_candidate(
    game: &LiarsDice,
    state: &LdState,
    player: usize,
    out: &mut Vec<CandidateAction>,
    qty: u8,
    face: u8,
    reason: CandidateReason,
) {
    let index = usize::from(qty - 1) * usize::from(game.faces) + usize::from(face - 1);
    let action = game.action_at(state, index);
    debug_assert_eq!(action, Action::Open(qty, face));
    out.push(CandidateAction {
        index,
        action,
        reason,
        truth_prob: bid_truth_prob(game, state, player, qty, face),
    });
}

fn score_candidate(
    game: &LiarsDice,
    state: &LdState,
    player: usize,
    candidate: &CandidateAction,
) -> f64 {
    let reason = match candidate.reason {
        CandidateReason::MinRaise => 4.0,
        CandidateReason::OwnFace => 3.0,
        CandidateReason::PosteriorQuantile => 2.0,
        CandidateReason::Call => 5.0,
    };
    let own_face_bonus = match candidate.action {
        Action::Open(_, face) => f64::from(state.my_count(player, face)) * 0.05,
        _ => 0.0,
    };
    let pressure_bonus = match candidate.action {
        Action::Open(qty, _) => f64::from(qty) / f64::from(total_dice(game, state)).max(1.0),
        _ => 0.0,
    };
    reason + candidate.truth_prob.clamp(0.0, 1.0) + own_face_bonus + pressure_bonus * 0.02
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_core::{Rng, Turn};

    fn roll_to_player(game: &LiarsDice, state: &mut LdState, rng: &mut Rng) {
        while let Turn::Chance = game.turn(state) {
            let action = game.sample_chance_action(state, rng);
            game.apply(state, action);
        }
    }

    fn free_open_state(players: u8, dice: u8, faces: u8) -> (LiarsDice, LdState) {
        let game = LiarsDice::new(players, dice, faces);
        let mut rng = Rng::new(0xA8A8);
        let mut state = game.initial_state();
        roll_to_player(&game, &mut state, &mut rng);
        let call_liar = game
            .legal_actions(&state)
            .into_iter()
            .find(|a| *a == Action::CallLiar)
            .unwrap();
        game.apply(&mut state, call_liar);
        assert!(!game.is_terminal(&state));
        roll_to_player(&game, &mut state, &mut rng);
        assert_eq!(state.current_bid().0, 0);
        (game, state)
    }

    #[test]
    fn mid_round_abstraction_keeps_every_raise_and_call() {
        let game = LiarsDice::new(5, 5, 6);
        let mut state = game.initial_state();
        roll_to_player(&game, &mut state, &mut Rng::new(1));
        let legal = game.legal_actions(&state);
        let candidates = candidate_actions(&game, &state, state.turn(), &Default::default());
        assert_eq!(candidates.len(), legal.len());
        for (i, candidate) in candidates.iter().enumerate() {
            assert_eq!(candidate.index, i);
            assert_eq!(candidate.action, legal[i]);
        }
    }

    #[test]
    fn opening_abstraction_caps_wide_free_open_nodes() {
        let (game, state) = free_open_state(5, 5, 6);
        let cfg = ActionAbstractionConfig::default();
        let candidates = candidate_actions(&game, &state, state.turn(), &cfg);
        assert!(game.num_actions(&state) > cfg.max_candidates);
        assert!(candidates.len() <= cfg.max_candidates);
        assert!(candidates.windows(2).all(|w| w[0].index < w[1].index));
        for candidate in &candidates {
            assert_eq!(game.action_at(&state, candidate.index), candidate.action);
            assert!(matches!(candidate.action, Action::Open(_, _)));
            assert!(candidate.truth_prob.is_finite());
        }
    }

    #[test]
    fn opening_abstraction_includes_every_min_face_when_room_allows() {
        let (game, state) = free_open_state(5, 5, 6);
        let cfg = ActionAbstractionConfig {
            max_candidates: 64,
            ..Default::default()
        };
        let candidates = candidate_actions(&game, &state, state.turn(), &cfg);
        for face in 1..=game.faces {
            assert!(
                candidates.iter().any(|c| c.action == Action::Open(1, face)),
                "missing minimum open on face {face}"
            );
        }
    }

    #[test]
    fn abstracted_rollout_returns_a_legal_opening_index() {
        let (game, state) = free_open_state(5, 5, 6);
        let agent = AbstractedRolloutAgent::new(2);
        let mut rng = Rng::new(0xABCD);
        let index = agent.act(&game, &state, state.turn(), &mut rng);
        assert!(index < game.num_actions(&state));
    }

    #[test]
    fn determinized_mcts_returns_a_legal_opening_index() {
        let (game, state) = free_open_state(5, 5, 6);
        let agent = DeterminizedMctsAgent::new(2, 4);
        let mut rng = Rng::new(0xBEEF);
        let index = agent.act(&game, &state, state.turn(), &mut rng);
        assert!(index < game.num_actions(&state));
    }

    #[test]
    fn abstracted_game_caps_free_open_actions() {
        let (game, state) = free_open_state(5, 5, 6);
        let abstracted = AbstractedLiarsDice::new(game.clone());
        assert!(game.num_actions(&state) > abstracted.num_actions(&state));
        assert!(
            abstracted.num_actions(&state) <= ActionAbstractionConfig::default().max_candidates
        );
        for action in abstracted.legal_actions(&state) {
            assert!(game.legal_actions(&state).contains(&action));
        }
    }

    #[test]
    fn abstracted_mccfr_returns_a_legal_index() {
        let game = LiarsDice::new(2, 1, 3);
        let mut state = game.initial_state();
        roll_to_player(&game, &mut state, &mut Rng::new(0xD1CE));
        let agent = AbstractedMccfrAgent::train(game.clone(), 2, 0xC0F5);
        let mut rng = Rng::new(0xA11CE);
        let index = agent.act(&game, &state, state.turn(), &mut rng);
        assert!(index < game.num_actions(&state));
    }

    #[test]
    fn depth_capped_mccfr_returns_a_legal_index_on_full_size_game() {
        let game = LiarsDice::new(5, 5, 6);
        let mut state = game.initial_state();
        roll_to_player(&game, &mut state, &mut Rng::new(0x5EED));
        let agent = AbstractedMccfrAgent::train_with_config_and_max_decision_plies(
            game.clone(),
            1,
            0xC0F5,
            ActionAbstractionConfig::default(),
            Some(2),
        );
        let mut rng = Rng::new(0xA11CE);
        let index = agent.act(&game, &state, state.turn(), &mut rng);
        assert!(index < game.num_actions(&state));
    }

    #[test]
    fn abstracted_q_returns_a_legal_index() {
        let game = LiarsDice::new(2, 1, 3);
        let mut state = game.initial_state();
        roll_to_player(&game, &mut state, &mut Rng::new(0xD1CE));
        let agent = AbstractedQAgent::train(game.clone(), 4, 0xA11CE);
        let mut rng = Rng::new(0xC0DE);
        let index = agent.act(&game, &state, state.turn(), &mut rng);
        assert!(index < game.num_actions(&state));
    }

    #[test]
    fn truth_probability_matches_simple_certainty_cases() {
        let (game, state) = free_open_state(3, 2, 6);
        let player = state.turn();
        let face = state.hand(player)[0];
        let other_face = if face < game.faces { face + 1 } else { 1 };
        assert_eq!(bid_truth_prob(&game, &state, player, 1, face), 1.0);
        assert_eq!(
            bid_truth_prob(&game, &state, player, game.players * game.dice, other_face),
            0.0
        );
    }
}
