//! Temporal-difference value learning, TD-Gammon style: online TD(λ)
//! self-play for two-player zero-sum perfect-information games, with a linear
//! sigmoid value v(s, p) = σ(w · φ(s, p) + b) over game-supplied features.
//!
//! The game's knowledge is a [`StateFeatures`] extractor; everything else is
//! generic. Self-play is ε-greedy over 1-ply *afterstates*: the mover picks
//! the action whose successor has the best learned value (terminal successors
//! use the true return), and eligibility traces propagate each new prediction
//! — and finally the actual outcome — back along the trajectory.
//!
//! Two prediction streams, one per player perspective, share the weight
//! vector, so a single run learns both seats even when φ(s, 0) is not the
//! mirror of φ(s, 1). Chance nodes are sampled during self-play (the
//! TD-Gammon case), though search consumers remain perfect-information.
//!
//! The learned [`TdEval`] is both a [`game_core::Eval`] — plug it into
//! [`crate::AlphaBeta`] or [`crate::mcts::Mcts`] — and, via [`TdAgent`], a
//! greedy arena [`Agent`].

use std::io;
use std::path::Path;

use game_core::{Agent, Eval, Game, Rng, Turn};

/// Game knowledge for TD learning: a fixed-length feature vector describing a
/// state from one player's perspective, so one weight vector serves both
/// seats of a zero-sum game.
#[allow(clippy::len_without_is_empty)]
pub trait StateFeatures<G: Game>: Sync {
    /// Number of features; must be constant for the game.
    fn len(&self) -> usize;

    /// Feature vector for `state` from `player`'s perspective, of length
    /// [`StateFeatures::len`].
    fn features(&self, game: &G, state: &G::State, player: usize) -> Vec<f32>;
}

/// TD(λ) hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct TdConfig {
    /// Learning rate.
    pub alpha: f32,
    /// Eligibility-trace decay in `[0, 1]`; 0 is one-step TD(0).
    pub lambda: f32,
    /// Probability of a uniform-random exploratory move during self-play.
    pub epsilon: f64,
}

impl Default for TdConfig {
    fn default() -> Self {
        Self {
            alpha: 0.02,
            lambda: 0.7,
            epsilon: 0.1,
        }
    }
}

const MAGIC: &[u8; 8] = b"td-lin-w";
const VERSION: u32 = 1;

/// A learned value function: the estimated probability that `player` wins,
/// σ(w · φ(s, player) + b). As an [`Eval`] it is rescaled to the
/// [`Game::returns`] scale `[-1, 1]`; [`TdAgent`] wraps it as a greedy agent.
#[derive(Clone)]
pub struct TdEval<F> {
    feats: F,
    weights: Vec<f32>,
}

impl<F> TdEval<F> {
    /// Pair a feature extractor with a weight vector of length
    /// `feats.len() + 1` — the trailing weight is the bias.
    pub fn new(feats: F, weights: Vec<f32>) -> Self {
        Self { feats, weights }
    }

    pub fn feats(&self) -> &F {
        &self.feats
    }

    /// Feature weights followed by the bias.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Win probability of `player` at `state`, in `(0, 1)`.
    pub fn value<G: Game>(&self, game: &G, state: &G::State, player: usize) -> f64
    where
        F: StateFeatures<G>,
    {
        let phi = self.feats.features(game, state, player);
        debug_assert_eq!(phi.len() + 1, self.weights.len());
        sigmoid(dot_bias(&self.weights, &phi))
    }

    /// Versioned binary checkpoint, written via temp file + atomic rename.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        if let Some(dir) = path.parent()
            && !dir.as_os_str().is_empty()
        {
            std::fs::create_dir_all(dir)?;
        }
        let mut buf = Vec::with_capacity(16 + self.weights.len() * 4);
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&(self.weights.len() as u32).to_le_bytes());
        for w in &self.weights {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("weights");
        let tmp = path.with_file_name(format!("{name}.tmp"));
        std::fs::write(&tmp, &buf)?;
        std::fs::rename(&tmp, path)
    }

    /// Load weights saved by [`TdEval::save`]; `feats` must be the extractor
    /// the weights were trained with.
    pub fn load(feats: F, path: &Path) -> io::Result<Self> {
        let data = std::fs::read(path)?;
        let bad = |m: &str| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: {m}", path.display()),
            )
        };
        if data.len() < 16 {
            return Err(bad("truncated header"));
        }
        if &data[..8] != MAGIC {
            return Err(bad("not a TD weight checkpoint"));
        }
        let u32_at = |i: usize| u32::from_le_bytes(data[i..i + 4].try_into().unwrap());
        if u32_at(8) != VERSION {
            return Err(bad("unsupported checkpoint version"));
        }
        let n = u32_at(12) as usize;
        let body = &data[16..];
        if body.len() != n * 4 {
            return Err(bad("weight count does not match header"));
        }
        let weights = body
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok(Self { feats, weights })
    }
}

impl<G: Game, F: StateFeatures<G>> Eval<G> for TdEval<F> {
    fn eval(&self, game: &G, state: &G::State, player: usize) -> f64 {
        2.0 * self.value(game, state, player) - 1.0
    }
}

/// Greedy 1-ply afterstate policy over a learned [`TdEval`]: picks the action
/// whose successor state has the highest value for the mover (terminal
/// successors use the true return). Deterministic — the arena's `r` is ignored.
pub struct TdAgent<F>(pub TdEval<F>);

impl<G: Game, F: StateFeatures<G>> Agent<G> for TdAgent<F> {
    fn act(&self, game: &G, state: &G::State, player: usize, _r: f64) -> usize {
        greedy(game, &self.0, state, player)
    }
}

fn sigmoid(x: f32) -> f64 {
    1.0 / (1.0 + (-f64::from(x)).exp())
}

/// `w` is one longer than `phi`; the extra trailing weight is the bias.
fn dot_bias(w: &[f32], phi: &[f32]) -> f32 {
    let (bias, w) = w.split_last().expect("weights include a bias");
    w.iter().zip(phi).map(|(a, b)| a * b).sum::<f32>() + bias
}

fn greedy<G: Game, F: StateFeatures<G>>(
    game: &G,
    eval: &TdEval<F>,
    state: &G::State,
    player: usize,
) -> usize {
    let actions = game.legal_actions(state);
    let mut best = 0;
    let mut best_v = f64::NEG_INFINITY;
    for (i, &a) in actions.iter().enumerate() {
        let mut child = state.clone();
        game.apply(&mut child, a);
        let v = if game.is_terminal(&child) {
            (game.returns(&child, player) + 1.0) / 2.0
        } else {
            eval.value(game, &child, player)
        };
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// Online TD(λ) self-play learner. Weights start at zero (v = 0.5 everywhere)
/// unless resumed via [`TdLearner::with_eval`].
pub struct TdLearner<'g, G: Game, F: StateFeatures<G>> {
    game: &'g G,
    eval: TdEval<F>,
    pub cfg: TdConfig,
    traces: [Vec<f32>; 2],
}

impl<'g, G: Game, F: StateFeatures<G>> TdLearner<'g, G, F> {
    pub fn new(game: &'g G, feats: F, cfg: TdConfig) -> Self {
        let n = feats.len() + 1;
        Self::with_eval(game, TdEval::new(feats, vec![0.0; n]), cfg)
    }

    /// Resume from previously learned weights (e.g. [`TdEval::load`]).
    pub fn with_eval(game: &'g G, eval: TdEval<F>, cfg: TdConfig) -> Self {
        assert_eq!(game.num_players(), 2, "TD self-play is two-player only");
        assert_eq!(
            eval.weights.len(),
            eval.feats.len() + 1,
            "weight count must be feature count + bias"
        );
        let n = eval.weights.len();
        Self {
            game,
            eval,
            cfg,
            traces: [vec![0.0; n], vec![0.0; n]],
        }
    }

    pub fn eval(&self) -> &TdEval<F> {
        &self.eval
    }

    pub fn into_eval(self) -> TdEval<F> {
        self.eval
    }

    /// Run `episodes` self-play games, updating weights online.
    pub fn train(&mut self, episodes: u64, seed: u64) {
        let mut rng = Rng::new(seed);
        for _ in 0..episodes {
            self.episode(&mut rng);
        }
    }

    /// One ε-greedy self-play game with online TD(λ) updates.
    pub fn episode(&mut self, rng: &mut Rng) {
        for t in &mut self.traces {
            t.fill(0.0);
        }
        let mut s = self.game.initial_state();
        self.sample_chance(&mut s, rng);
        if self.game.is_terminal(&s) {
            return;
        }
        let mut y = [self.observe(&s, 0), self.observe(&s, 1)];
        loop {
            let mover = match self.game.turn(&s) {
                Turn::Player(p) => p,
                Turn::Chance => unreachable!("chance was just sampled"),
            };
            let actions = self.game.legal_actions(&s);
            let i = if rng.unit() < self.cfg.epsilon {
                ((rng.unit() * actions.len() as f64) as usize).min(actions.len() - 1)
            } else {
                greedy(self.game, &self.eval, &s, mover)
            };
            self.game.apply(&mut s, actions[i]);
            self.sample_chance(&mut s, rng);
            if self.game.is_terminal(&s) {
                for (p, yp) in y.iter().enumerate() {
                    let z = (self.game.returns(&s, p) + 1.0) / 2.0;
                    self.reinforce(p, z - yp);
                }
                return;
            }
            let targets = [
                self.eval.value(self.game, &s, 0),
                self.eval.value(self.game, &s, 1),
            ];
            for (p, yp) in y.iter().enumerate() {
                self.reinforce(p, targets[p] - yp);
            }
            y = [self.observe(&s, 0), self.observe(&s, 1)];
        }
    }

    /// Predict v(s, p) and fold its gradient into `p`'s eligibility trace.
    fn observe(&mut self, s: &G::State, p: usize) -> f64 {
        let phi = self.eval.feats.features(self.game, s, p);
        let v = sigmoid(dot_bias(&self.eval.weights, &phi));
        let g = (v * (1.0 - v)) as f32;
        let lambda = self.cfg.lambda;
        let trace = &mut self.traces[p];
        let bias_at = trace.len() - 1;
        for (i, e) in trace.iter_mut().enumerate() {
            let fi = if i < bias_at { phi[i] } else { 1.0 };
            *e = lambda * *e + g * fi;
        }
        v
    }

    fn reinforce(&mut self, p: usize, delta: f64) {
        let step = self.cfg.alpha * delta as f32;
        for (w, &e) in self.eval.weights.iter_mut().zip(&self.traces[p]) {
            *w += step * e;
        }
    }

    fn sample_chance(&self, s: &mut G::State, rng: &mut Rng) {
        while !self.game.is_terminal(s) && matches!(self.game.turn(s), Turn::Chance) {
            let outs = self.game.chance_outcomes(s);
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
            self.game.apply(s, chosen);
        }
    }
}
