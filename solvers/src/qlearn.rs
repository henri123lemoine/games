//! Tabular Q-learning / SARSA over [`Game::infoset_key`].
//!
//! Model-free control for games small enough to enumerate the infosets a
//! behavior policy visits: act epsilon-greedily, sample chance nodes, and back
//! up the only rewards the [`Game`] trait defines — terminal returns. Each
//! player learns its own table, so one loop covers both single-player MDPs
//! (chance plays the environment) and two-player zero-sum alternating
//! self-play (each seat learns against the other's evolving policy).
//!
//! Updates happen between *consecutive decisions of the same player*: when
//! player `p` reaches its next decision, its previous `(infoset, action)` pair
//! moves toward `gamma * bootstrap`, and at the terminal state toward `p`'s
//! return. The bootstrap is `max Q` at the new infoset for Q-learning, or the
//! Q-value of the epsilon-greedy action actually taken for SARSA
//! ([`QConfig::sarsa`]).

use game_core::{Agent, Game, Rng, Turn};

use crate::FastMap;

/// Hyperparameters for [`QLearner`]. `alpha` and `epsilon` decay linearly from
/// `*_start` to `*_end` over the first [`QConfig::decay_episodes`] training
/// episodes and hold at `*_end` afterwards.
#[derive(Debug, Clone, Copy)]
pub struct QConfig {
    /// Discount per decision of the acting player; a terminal return reached
    /// directly by an action is credited undiscounted.
    pub gamma: f64,
    pub alpha_start: f64,
    pub alpha_end: f64,
    pub epsilon_start: f64,
    pub epsilon_end: f64,
    pub decay_episodes: u64,
    /// `false`: Q-learning (off-policy, bootstrap on `max Q`).
    /// `true`: SARSA (on-policy, bootstrap on the action actually taken).
    pub sarsa: bool,
}

impl Default for QConfig {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            alpha_start: 0.5,
            alpha_end: 0.01,
            epsilon_start: 1.0,
            epsilon_end: 0.05,
            decay_episodes: 100_000,
            sarsa: false,
        }
    }
}

/// Tabular epsilon-greedy Q-learning / SARSA with one Q-table per player.
/// Train with [`QLearner::train_episodes`], then play via the greedy [`Agent`]
/// from [`QLearner::greedy`].
pub struct QLearner<G: Game> {
    game: G,
    cfg: QConfig,
    tables: Vec<FastMap<u64, Vec<f64>>>,
    rng: Rng,
    episodes: u64,
}

impl<G: Game> QLearner<G> {
    pub fn new(game: G, cfg: QConfig, seed: u64) -> Self {
        let players = game.num_players();
        Self {
            game,
            cfg,
            tables: (0..players).map(|_| FastMap::default()).collect(),
            rng: Rng::new(seed),
            episodes: 0,
        }
    }

    /// Run `n` training episodes of epsilon-greedy (self-)play.
    pub fn train_episodes(&mut self, n: u64) {
        for _ in 0..n {
            self.episode();
            self.episodes += 1;
        }
    }

    /// The greedy (argmax-Q) policy over the learned tables. Unseen infosets
    /// fall back to a uniform-random action via the match-provided `r`.
    pub fn greedy(&self) -> GreedyQ<'_, G> {
        GreedyQ { learner: self }
    }

    /// Learned Q-values for `player` at `infoset`, in [`Game::legal_actions`]
    /// order, or `None` if that infoset was never visited.
    pub fn q_values(&self, player: usize, infoset: u64) -> Option<&[f64]> {
        self.tables[player].get(&infoset).map(Vec::as_slice)
    }

    /// Total number of `(player, infoset)` rows across all per-player tables.
    pub fn table_size(&self) -> usize {
        self.tables.iter().map(|t| t.len()).sum()
    }

    pub fn episodes_trained(&self) -> u64 {
        self.episodes
    }

    fn schedule(&self, start: f64, end: f64) -> f64 {
        let t = (self.episodes as f64 / self.cfg.decay_episodes.max(1) as f64).min(1.0);
        start + (end - start) * t
    }

    fn episode(&mut self) {
        let alpha = self.schedule(self.cfg.alpha_start, self.cfg.alpha_end);
        let eps = self.schedule(self.cfg.epsilon_start, self.cfg.epsilon_end);
        let mut state = self.game.initial_state();
        let mut pending: Vec<Option<(u64, usize)>> = vec![None; self.tables.len()];
        loop {
            if self.game.is_terminal(&state) {
                for (p, slot) in pending.iter_mut().enumerate() {
                    if let Some((key, a)) = slot.take() {
                        let target = self.game.returns(&state, p);
                        let q = &mut self.tables[p].get_mut(&key).expect("visited row")[a];
                        *q += alpha * (target - *q);
                    }
                }
                return;
            }
            match self.game.turn(&state) {
                Turn::Chance => {
                    game_core::rand::step_chance(&self.game, &mut state, &mut self.rng);
                }
                Turn::Player(p) => {
                    let key = self.game.infoset_key(&state, p);
                    let actions = self.game.legal_actions(&state);
                    let n = actions.len();
                    let (a, bootstrap) = {
                        let row = self.tables[p].entry(key).or_insert_with(|| vec![0.0; n]);
                        let a = if self.rng.unit() < eps {
                            self.rng.below(n)
                        } else {
                            argmax_tiebreak(row, &mut self.rng)
                        };
                        let bootstrap = if self.cfg.sarsa {
                            row[a]
                        } else {
                            row.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                        };
                        (a, bootstrap)
                    };
                    if let Some((pkey, pa)) = pending[p].take() {
                        let q = &mut self.tables[p].get_mut(&pkey).expect("visited row")[pa];
                        *q += alpha * (self.cfg.gamma * bootstrap - *q);
                    }
                    pending[p] = Some((key, a));
                    self.game.apply(&mut state, actions[a]);
                }
            }
        }
    }
}

/// Greedy play from a trained [`QLearner`]; see [`QLearner::greedy`].
pub struct GreedyQ<'a, G: Game> {
    learner: &'a QLearner<G>,
}

impl<G: Game> Agent<G> for GreedyQ<'_, G> {
    fn act(&self, game: &G, state: &G::State, player: usize, rng: &mut Rng) -> usize {
        let key = game.infoset_key(state, player);
        match self.learner.tables[player].get(&key) {
            Some(row) => argmax_first(row),
            None => rng.below(game.legal_actions(state).len()),
        }
    }
}

/// Argmax with uniform tie-breaking (reservoir sampling over the maxima), so
/// untrained all-zero rows explore instead of always picking action 0.
fn argmax_tiebreak(row: &[f64], rng: &mut Rng) -> usize {
    let mut best = 0;
    let mut best_v = row[0];
    let mut ties = 1.0;
    for (i, &v) in row.iter().enumerate().skip(1) {
        if v > best_v {
            best = i;
            best_v = v;
            ties = 1.0;
        } else if v == best_v {
            ties += 1.0;
            if rng.unit() * ties < 1.0 {
                best = i;
            }
        }
    }
    best
}

fn argmax_first(row: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate().skip(1) {
        if v > row[best] {
            best = i;
        }
    }
    best
}
