use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use twentyone_core::{Action as CoreAction, Env as CoreEnv, Observation as CoreObservation, RoundOutcome as CoreRoundOutcome, Solver as CoreSolver, StepResult as CoreStepResult};

/// Action available to the current player: draw a card or stand.
#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Draw a card from the deck
    Draw,
    /// Stand with current total
    Stand,
}

impl From<Action> for CoreAction {
    fn from(action: Action) -> Self {
        match action {
            Action::Draw => CoreAction::Draw,
            Action::Stand => CoreAction::Stand,
        }
    }
}

impl From<CoreAction> for Action {
    fn from(action: CoreAction) -> Self {
        match action {
            CoreAction::Draw => Action::Draw,
            CoreAction::Stand => Action::Stand,
        }
    }
}

/// Partial observation available to an agent controlling one player.
#[pyclass]
#[derive(Debug, Clone)]
pub struct Observation {
    /// Player's current total (visible + hidden + drawn cards)
    #[pyo3(get)]
    pub self_total: u8,
    /// Opponent's face-up card value
    #[pyo3(get)]
    pub opp_face_up: u8,
    /// Player's face-up card value
    #[pyo3(get)]
    pub self_face_up: u8,
    /// Player's face-down card value (hidden from opponent)
    #[pyo3(get)]
    pub self_face_down: u8,
    /// Whether the player stood in their last action
    #[pyo3(get)]
    pub self_stood: bool,
    /// Whether the opponent stood in their last action
    #[pyo3(get)]
    pub opp_stood: bool,
    /// Number of cards remaining in the deck
    #[pyo3(get)]
    pub deck_count: u8,
    /// Current round number
    #[pyo3(get)]
    pub round: u8,
    /// Player's hearts remaining
    #[pyo3(get)]
    pub self_hearts: u8,
    /// Opponent's hearts remaining
    #[pyo3(get)]
    pub opp_hearts: u8,
}

impl From<CoreObservation> for Observation {
    fn from(obs: CoreObservation) -> Self {
        Self {
            self_total: obs.self_total,
            opp_face_up: obs.opp_face_up,
            self_face_up: obs.self_face_up,
            self_face_down: obs.self_face_down,
            self_stood: obs.self_stood,
            opp_stood: obs.opp_stood,
            deck_count: obs.deck_count,
            round: obs.round,
            self_hearts: obs.self_hearts,
            opp_hearts: obs.opp_hearts,
        }
    }
}

#[pymethods]
impl Observation {
    fn __repr__(&self) -> String {
        format!(
            "Observation(self_total={}, opp_face_up={}, self_face_up={}, self_face_down={}, round={})",
            self.self_total, self.opp_face_up, self.self_face_up, self.self_face_down, self.round
        )
    }

    /// Convert observation to a dictionary for compatibility
    fn to_dict(&self, py: Python) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("self_total", self.self_total)?;
        dict.set_item("opp_face_up", self.opp_face_up)?;
        dict.set_item("self_face_up", self.self_face_up)?;
        dict.set_item("self_face_down", self.self_face_down)?;
        dict.set_item("self_stood", self.self_stood)?;
        dict.set_item("opp_stood", self.opp_stood)?;
        dict.set_item("deck_count", self.deck_count)?;
        dict.set_item("round", self.round)?;
        dict.set_item("self_hearts", self.self_hearts)?;
        dict.set_item("opp_hearts", self.opp_hearts)?;
        Ok(dict.into())
    }
}

/// Result of a completed round.
#[pyclass]
#[derive(Debug, Clone)]
pub struct RoundOutcome {
    /// Winner of the round (0 or 1), or None for tie
    #[pyo3(get)]
    pub winner: Option<usize>,
    /// Damage dealt to the loser
    #[pyo3(get)]
    pub damage: u8,
}

impl From<CoreRoundOutcome> for RoundOutcome {
    fn from(outcome: CoreRoundOutcome) -> Self {
        Self {
            winner: outcome.winner,
            damage: outcome.damage,
        }
    }
}

#[pymethods]
impl RoundOutcome {
    fn __repr__(&self) -> String {
        match self.winner {
            Some(w) => format!("RoundOutcome(winner={}, damage={})", w, self.damage),
            None => format!("RoundOutcome(tie, damage={})", self.damage),
        }
    }
}

/// Result of a step in the game.
#[pyclass]
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Whether the round ended
    #[pyo3(get)]
    pub round_over: bool,
    /// Whether the game ended
    #[pyo3(get)]
    pub game_over: bool,
    /// Round outcome if the round ended
    #[pyo3(get)]
    pub outcome: Option<RoundOutcome>,
}

impl From<CoreStepResult> for StepResult {
    fn from(result: CoreStepResult) -> Self {
        Self {
            round_over: result.round_over,
            game_over: result.game_over,
            outcome: result.outcome.map(|o| o.into()),
        }
    }
}

#[pymethods]
impl StepResult {
    fn __repr__(&self) -> String {
        format!(
            "StepResult(round_over={}, game_over={}, outcome={:?})",
            self.round_over, self.game_over, self.outcome
        )
    }
}

/// Twenty-One game environment for two players.
/// 
/// This is a fast Rust implementation of the Twenty-One card game environment
/// suitable for reinforcement learning experiments.
#[pyclass]
pub struct Env {
    inner: CoreEnv,
}

#[pymethods]
impl Env {
    /// Create a new environment with a random seed.
    /// 
    /// Args:
    ///     seed: Random seed for reproducible games
    /// 
    /// Returns:
    ///     New game environment
    #[new]
    fn new(seed: u64) -> Self {
        Self {
            inner: CoreEnv::new(seed),
        }
    }

    /// Create a new environment with predetermined deck orders.
    /// Intended for deterministic testing.
    /// 
    /// Args:
    ///     preset_decks: List of deck orders (each is a list of 11 cards)
    /// 
    /// Returns:
    ///     New game environment with preset decks
    #[classmethod]
    fn with_preset_decks(_cls: &Bound<'_, PyType>, preset_decks: Vec<Vec<u8>>) -> PyResult<Self> {
        let converted_decks: Result<Vec<[u8; 11]>, _> = preset_decks
            .into_iter()
            .map(|deck| {
                if deck.len() != 11 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Each deck must have exactly 11 cards"
                    ));
                }
                let mut array = [0u8; 11];
                array.copy_from_slice(&deck);
                Ok(array)
            })
            .collect();
        
        match converted_decks {
            Ok(decks) => Ok(Self {
                inner: CoreEnv::new_with_preset_decks(decks),
            }),
            Err(e) => Err(e),
        }
    }

    /// Start a new round: deals cards to both players.
    /// 
    /// Raises:
    ///     RuntimeError: If game is over or round already in progress
    fn start_new_round(&mut self) -> PyResult<()> {
        self.inner.start_new_round().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })
    }

    /// Get observation for a player.
    /// 
    /// Args:
    ///     player: Player index (0 or 1)
    /// 
    /// Returns:
    ///     Observation for the specified player
    fn observation(&self, player: usize) -> Observation {
        self.inner.observation(player).into()
    }

    /// Take an action for the current player.
    /// 
    /// Args:
    ///     action: Action to take (Action.Draw or Action.Stand)
    /// 
    /// Returns:
    ///     StepResult containing game state and outcome
    /// 
    /// Raises:
    ///     RuntimeError: If game is over, no active round, or invalid action
    fn step(&mut self, action: Action) -> PyResult<StepResult> {
        self.inner.step(action.into()).map(|r| r.into()).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })
    }

    /// Current round number (starts at 1).
    fn round(&self) -> u8 {
        self.inner.round()
    }

    /// Hearts remaining for a player.
    /// 
    /// Args:
    ///     player: Player index (0 or 1)
    /// 
    /// Returns:
    ///     Hearts remaining for the player
    fn hearts(&self, player: usize) -> u8 {
        self.inner.hearts(player)
    }

    /// Index of the current player (0 or 1).
    fn current_player(&self) -> usize {
        self.inner.current_player()
    }

    /// Public up cards for a player in the current round.
    /// 
    /// Args:
    ///     player: Player index (0 or 1)
    /// 
    /// Returns:
    ///     List of face-up cards, or None if no active round
    fn public_up_cards(&self, player: usize) -> Option<Vec<u8>> {
        self.inner.public_up_cards(player).map(|(cards, len)| {
            cards[..len as usize].to_vec()
        })
    }

    /// The last round's down cards revealed at round end.
    /// 
    /// Returns:
    ///     List of two down cards [player0_down, player1_down], or None
    fn last_reveal(&self) -> Option<Vec<u8>> {
        self.inner.last_reveal().map(|cards| cards.to_vec())
    }

    /// Return an independent deep copy of the environment, including RNG state.
    ///
    /// Enables tree branching for search and CFR: snapshot a state, then explore
    /// each action on separate copies without mutating the original.
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: Bound<'_, PyDict>) -> Self {
        self.clone()
    }

    /// Lossless sufficient-statistic information-set key for `player`, used to
    /// index an equilibrium strategy table.
    fn sufficient_key(&self, player: usize) -> u64 {
        self.inner.sufficient_key(player)
    }

    fn __repr__(&self) -> String {
        format!(
            "Env(round={}, hearts=[{}, {}])",
            self.round(),
            self.hearts(0),
            self.hearts(1)
        )
    }
}

/// CFR+ solver that computes a near-Nash equilibrium strategy for Twenty-One.
///
/// Train with `solve`, measure quality with `exploitability`, and play with
/// `draw_probability`. Strategies persist via `save`/`load`.
#[pyclass]
pub struct Solver {
    inner: CoreSolver,
}

#[pymethods]
impl Solver {
    /// Create a solver for the full 6-heart game.
    #[new]
    fn new(seed: u64) -> Self {
        Self {
            inner: CoreSolver::new(seed),
        }
    }

    /// Create a solver for a variant with `start_hearts` hearts per player.
    /// Smaller values define a shorter, exactly-solvable game for validation.
    #[classmethod]
    fn with_hearts(_cls: &Bound<'_, PyType>, seed: u64, start_hearts: u8) -> Self {
        Self {
            inner: CoreSolver::with_hearts(seed, start_hearts),
        }
    }

    /// Run `iters_per_subgame` CFR+ iterations on every round subgame.
    fn solve(&mut self, iters_per_subgame: u64) {
        self.inner.solve(iters_per_subgame);
    }

    /// Exact/sampled exploitability: returns (br0, br1, nashconv). With
    /// `deal_samples == 0` the opening deal of each round is enumerated exactly;
    /// otherwise it is Monte-Carlo sampled with the given `seed`.
    fn exploitability(&self, deal_samples: u32, seed: u64) -> (f64, f64, f64) {
        self.inner.exploitability(deal_samples, seed)
    }

    /// Probability of drawing under the equilibrium average strategy for the
    /// current player in `env`. Returns 0.0 when the deck is empty (forced stand).
    fn draw_probability(&self, env: &Env, player: usize) -> f64 {
        if env.inner.deck_mask() == 0 {
            return 0.0;
        }
        self.inner.average_draw_prob(env.inner.sufficient_key(player))
    }

    fn iterations(&self) -> u64 {
        self.inner.iterations()
    }

    fn num_infosets(&self) -> usize {
        self.inner.num_infosets()
    }

    fn save(&self, path: &str) -> PyResult<()> {
        self.inner
            .save(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
    }

    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        CoreSolver::load(path)
            .map(|inner| Self { inner })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "Solver(iterations={}, infosets={})",
            self.inner.iterations(),
            self.inner.num_infosets()
        )
    }
}

/// Twenty-One card game environment with Rust backend.
/// 
/// This module provides a fast implementation of the Twenty-One card game
/// suitable for reinforcement learning experiments and game simulations.
#[pymodule]
fn _twentyone(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Env>()?;
    m.add_class::<Action>()?;
    m.add_class::<Observation>()?;
    m.add_class::<RoundOutcome>()?;
    m.add_class::<StepResult>()?;
    m.add_class::<Solver>()?;
    Ok(())
}