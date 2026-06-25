//! Deep CFR (Brown, Lerer, Gross & Sandholm, "Deep Counterfactual Regret
//! Minimization", ICML 2019) — counterfactual regret minimization with the
//! per-infoset regret/strategy tables replaced by neural networks that key on
//! *features*, so the learned strategy generalizes across the parameterized
//! game family instead of being bounded by any one game's infoset abstraction.
//!
//! ## The algorithm (as implemented)
//!
//! An **advantage network** `D: features -> per-action regret` is a *linear*
//! per-action head ([`Mlp::head_values`]) fit by MSE to sampled instantaneous
//! regrets. The current strategy at an infoset is regret matching over
//! `relu(D(features))` on the legal-action support (uniform if all are `<= 0`) —
//! exactly CFR's regret matching, with `D` standing in for the cumulative-regret
//! table. There are `adv_nets` advantage networks: one per player (Brown 2019,
//! for asymmetric seats such as Kuhn) or a single net when the encoder is
//! seat-relative (the acting seat is reference index 0), which lets one net
//! serve every seat of every config in a parameterized family.
//!
//! Each iteration `t`, for each traverser, runs one (or more) **external-
//! sampling MCCFR** traversals: at the traverser's infosets every action is
//! expanded, the strategy value `v = sum_a sigma(a) v(a)` is formed, and the
//! instantaneous regrets `v(a) - v` are stored as a [`RegretSample`] into the
//! traverser's advantage reservoir; chance and the opponents are sampled to a
//! single action. At every infoset (own or opponent's, traverser's perspective)
//! the current strategy `sigma` is also stored into a shared **strategy
//! reservoir**. Both reservoirs carry the iteration `t` for **linear (Linear
//! CFR) weighting**.
//!
//! Periodically each advantage net is **retrained from scratch** on its
//! reservoir (a fresh net so stale early regrets do not bias the fit), with the
//! per-sample loss weighted by `t`. At the end the **average-strategy network**
//! is trained on the strategy reservoir with `t`-weighted cross-entropy; that
//! net is the deployable policy.
//!
//! ## Generality
//!
//! The engine is generic over a [`Game`] `G` and an [`Encoder<G>`] that maps
//! `(game, state, player) -> features` and gives the legal-action support
//! indices into a fixed policy space. The same engine therefore validates on
//! Kuhn poker (a tiny hand-written encoder) and runs on Liar's Dice round
//! subgames (`features::encode` + `features::support`). Correctness is proven on
//! Kuhn: its average-strategy exploitability via
//! [`nash_conv`](crate::nash_conv) must drive toward zero.

use game_core::{Game, Rng, Turn};

use crate::azero::{InferCache, Mlp, RegretSample, Sample, SgdMomentum};
use crate::tabular::regret_match;

/// Maps a game's decision node into the net's fixed feature/action space.
///
/// `features` and `support` are the two halves the advantage and strategy nets
/// need: a fixed-length input vector for `(state, player)`, and the policy-head
/// indices of the legal actions in `legal_actions` order. The engine never
/// inspects the game's `Action` type, so one engine serves every game with an
/// encoder.
pub trait Encoder<G: Game>: Sync {
    /// Length of the [`Encoder::features`] vector (the net's input width).
    fn feature_len(&self) -> usize;
    /// Size of the fixed action space (the net's policy-head width).
    fn policy_len(&self) -> usize;
    /// Fixed-length features of `(state, player)`. Must encode the side to move.
    fn features(&self, game: &G, state: &G::State, player: usize) -> Vec<f32>;
    /// Policy-head indices of the legal actions at `state`, in
    /// `legal_actions(state)` order (parallel to it).
    fn support(&self, game: &G, state: &G::State) -> Vec<usize>;
}

/// Deep CFR hyperparameters.
#[derive(Clone)]
pub struct DeepCfrConfig {
    /// CFR iterations (each runs one traversal per player as traverser).
    pub iters: usize,
    /// Traversals per (iteration, traverser) — more lowers regret-target noise.
    pub traversals: usize,
    /// Retrain every advantage net from scratch every `train_every` iterations.
    pub train_every: usize,
    /// Hidden width of the advantage and strategy nets.
    pub hidden: usize,
    /// Reservoir capacity per advantage net.
    pub adv_reservoir: usize,
    /// Reservoir capacity of the (shared) strategy buffer.
    pub strat_reservoir: usize,
    /// SGD steps when retraining an advantage net.
    pub adv_steps: usize,
    /// SGD steps when training the final average-strategy net.
    pub strat_steps: usize,
    pub batch: usize,
    pub lr: f32,
    pub momentum: f32,
    pub l2: f32,
    pub seed: u64,
    /// Number of advantage nets/reservoirs. `0` (the default) means one per
    /// player (Brown 2019's per-player nets — right when seats are asymmetric,
    /// e.g. Kuhn). Set to `1` for a single feature-keyed net when the encoder
    /// already presents every seat from the acting player's perspective (Liar's
    /// Dice rotates the acting seat to reference index 0, so one net covers all
    /// seats *and* all player counts in the config family). The acting seat `p`
    /// uses net/reservoir `p % adv_nets`.
    pub adv_nets: usize,
    /// Also emit a value target at each traversal root (opening public features,
    /// `z` = the traverser's root counterfactual value) into the strategy
    /// reservoir, so the average-strategy net's value head learns the
    /// round-opening equity — the signal a Liar's Dice `NetValue` continuation
    /// reads. Off for games with no continuation to learn (e.g. Kuhn).
    pub collect_root_value: bool,
    /// ε-greedy exploration weight for the sampled *opponent* branches of an
    /// external-sampling traversal. At an opponent node the action is drawn from
    /// `q(a) = ε/n + (1-ε)·σ(a)` instead of pure σ, so a current strategy that
    /// rarely plays an action still reaches — and trains — the infosets beyond
    /// it. The returned subtree value is importance-weighted by `σ(a)/q(a)`
    /// (compounded along the path) so the traverser's regret estimates stay
    /// unbiased in expectation. The traverser's own nodes still expand every
    /// action (external sampling) and chance is still sampled from its true
    /// distribution (already unbiased) — both unaffected. `0.0` recovers pure
    /// on-policy sampling.
    pub explore_eps: f64,
}

impl Default for DeepCfrConfig {
    fn default() -> Self {
        Self {
            iters: 400,
            traversals: 1,
            train_every: 1,
            hidden: 64,
            adv_reservoir: 1_000_000,
            strat_reservoir: 1_000_000,
            adv_steps: 400,
            strat_steps: 4000,
            batch: 256,
            lr: 0.01,
            momentum: 0.9,
            l2: 1e-4,
            seed: 0xDEEC_F00D,
            adv_nets: 0,
            collect_root_value: false,
            explore_eps: 0.6,
        }
    }
}

/// A reservoir sample: a payload plus the CFR iteration it was generated on
/// (the Linear CFR weight).
struct Reserved<T> {
    item: T,
    iter: f32,
}

/// Reservoir-sampling buffer (Vitter's Algorithm R): a uniform sample of the
/// stream of items seen, capped at `cap`. Each item keeps its generating
/// iteration for linear (Linear CFR) loss weighting.
struct Reservoir<T> {
    buf: Vec<Reserved<T>>,
    cap: usize,
    seen: u64,
}

impl<T> Reservoir<T> {
    fn new(cap: usize) -> Self {
        Self {
            buf: Vec::new(),
            cap: cap.max(1),
            seen: 0,
        }
    }

    fn push(&mut self, item: T, iter: f32, rng: &mut Rng) {
        self.seen += 1;
        if self.buf.len() < self.cap {
            self.buf.push(Reserved { item, iter });
        } else {
            // Replace a uniformly chosen slot with probability cap/seen.
            let j = (rng.unit() * self.seen as f64) as usize;
            if j < self.cap {
                self.buf[j] = Reserved { item, iter };
            }
        }
    }

    fn len(&self) -> usize {
        self.buf.len()
    }
}

/// The current regret-matched strategy at an infoset given an advantage net.
///
/// Reads the linear advantage head over `support`, takes `relu` (only positive
/// predicted advantages shape the strategy — CFR+ regret matching), and
/// normalizes; uniform when no action has a positive advantage.
fn strategy_from_advantage(
    net: &Mlp,
    cache: &InferCache,
    x: &[f32],
    support: &[usize],
) -> Vec<f64> {
    let (adv, _) = net.head_values_cached(cache, x, support);
    let pos: Vec<f64> = adv.iter().map(|&a| f64::from(a).max(0.0)).collect();
    regret_match(&pos)
}

/// A snapshot of an advantage net plus its inference cache, used as the fixed
/// strategy during a batch of traversals.
struct AdvNet {
    net: Mlp,
    cache: InferCache,
}

impl AdvNet {
    fn new(net: Mlp) -> Self {
        let cache = net.infer_cache();
        Self { net, cache }
    }

    fn strategy(&self, x: &[f32], support: &[usize]) -> Vec<f64> {
        strategy_from_advantage(&self.net, &self.cache, x, support)
    }
}

/// Deep CFR trainer keyed on features through an [`Encoder`].
///
/// The engine holds only persistent, game-agnostic state — the advantage nets,
/// the reservoirs (plain feature/index/value tuples), the RNG. The game `G` and
/// its encoder `E` are supplied per call, not baked into the type, so a
/// parameterized family of games (Liar's Dice round subgames with per-round
/// configs *and* per-block continuation nets — each a different borrow) feeds
/// one set of feature-keyed nets and reservoirs. Generalization across configs
/// comes for free: the nets see only features. For a single fixed game,
/// [`DeepCfr::run`] supplies it every iteration.
pub struct DeepCfr {
    cfg: DeepCfrConfig,
    feature_len: usize,
    policy_len: usize,
    /// One advantage net per "advantage slot" (`cfg.adv_nets`): Brown 2019's
    /// per-player advantage nets when slots = players (each player minimizes its
    /// own regret), or a single seat-relative net when slots = 1 (the encoder
    /// presents every seat from the acting player's view).
    advantage: Vec<AdvNet>,
    adv_buf: Vec<Reservoir<RegretSample>>,
    strat_buf: Reservoir<Sample>,
    rng: Rng,
    adv_nets: usize,
    iter: usize,
}

impl DeepCfr {
    /// A trainer for `players`-player games whose encoder reports
    /// `feature_len`/`policy_len`. The number of advantage nets is
    /// `cfg.adv_nets` (defaulting to one per player). The encoder itself is
    /// passed to [`DeepCfr::run`] / [`DeepCfr::run_family`], so one engine can
    /// serve a family of differently-borrowing game instances.
    pub fn new<G: Game, E: Encoder<G>>(players: usize, enc: &E, cfg: DeepCfrConfig) -> Self {
        let adv_nets = if cfg.adv_nets == 0 {
            players
        } else {
            cfg.adv_nets
        };
        let (feature_len, policy_len) = (enc.feature_len(), enc.policy_len());
        let advantage = (0..adv_nets)
            .map(|p| {
                AdvNet::new(Mlp::new(
                    feature_len,
                    cfg.hidden,
                    policy_len,
                    cfg.seed ^ (0x100 + p as u64),
                ))
            })
            .collect();
        let adv_buf = (0..adv_nets)
            .map(|_| Reservoir::new(cfg.adv_reservoir))
            .collect();
        let rng = Rng::new(cfg.seed);
        let strat_buf = Reservoir::new(cfg.strat_reservoir);
        Self {
            cfg,
            feature_len,
            policy_len,
            advantage,
            adv_buf,
            strat_buf,
            rng,
            adv_nets,
            iter: 0,
        }
    }

    /// Run all configured iterations on a single fixed `game` (encoded by
    /// `enc`). Returns the deployable average-strategy net (input =
    /// `feature_len`, policy = `policy_len`).
    pub fn run<G: Game, E: Encoder<G>>(&mut self, game: &G, enc: &E) -> Mlp {
        for _ in 0..self.cfg.iters {
            self.iteration(game, enc);
        }
        self.train_average_strategy()
    }

    /// Run `n` iterations, each on a freshly sampled game instance (the config
    /// family). `sample` draws a game from a sub-stream of the engine RNG; the
    /// drawn game's own player count drives that iteration's traversers, so the
    /// family may mix player counts (a single seat-relative advantage net,
    /// `adv_nets = 1`, then handles every seat of every config). Call repeatedly
    /// to checkpoint between blocks; the engine carries reservoir/iteration
    /// state across calls.
    pub fn run_family<G: Game, E: Encoder<G>>(
        &mut self,
        n: usize,
        enc: &E,
        mut sample: impl FnMut(&mut Rng) -> G,
    ) -> Mlp {
        for _ in 0..n {
            let mut sub = Rng::new(self.rng.next_u64());
            let game = sample(&mut sub);
            self.iteration(&game, enc);
        }
        self.train_average_strategy()
    }

    /// Run iterations until `self.iter` reaches `to` on a single fixed `game`,
    /// returning the average-strategy net at that checkpoint. Lets a caller
    /// probe convergence; the engine carries state across calls.
    pub fn run_through<G: Game, E: Encoder<G>>(&mut self, to: usize, game: &G, enc: &E) -> Mlp {
        while self.iter < to {
            self.iteration(game, enc);
        }
        self.train_average_strategy()
    }

    /// One CFR iteration on `game`: a batch of external-sampling traversals per
    /// traverser, then (every `train_every` iters) a from-scratch advantage-net
    /// retrain. The Linear CFR weight is the global iteration count.
    fn iteration<G: Game, E: Encoder<G>>(&mut self, game: &G, enc: &E) {
        self.iter += 1;
        let iter = self.iter as f32;
        // Take the RNG out so `traverse` can hold `&mut self` and a `&mut Rng`
        // without a borrow conflict; the stream advances across all traversals.
        let mut rng = std::mem::replace(&mut self.rng, Rng::new(0));
        let root = game.initial_state();
        let players = game.num_players();
        for traverser in 0..players {
            let mut root_v = 0.0;
            for _ in 0..self.cfg.traversals {
                root_v += self.traverse(game, enc, &root, traverser, iter, &mut rng);
            }
            // A value target for the root *public* state (features at the
            // opening node, before any chance roll), so the average-strategy
            // net's value head learns the game equity from the round-opening
            // state — the signal a `NetValue` continuation reads. Skipped unless
            // requested (Kuhn has no continuation to learn).
            if self.cfg.collect_root_value {
                let z = (root_v / self.cfg.traversals.max(1) as f64) as f32;
                let x = enc.features(game, &root, traverser);
                self.strat_buf.push(
                    Sample {
                        x,
                        policy: Vec::new(),
                        z,
                    },
                    iter,
                    &mut rng,
                );
            }
        }
        self.rng = rng;
        if self.iter.is_multiple_of(self.cfg.train_every) {
            for p in 0..self.adv_nets {
                self.retrain_advantage(p);
            }
        }
    }

    /// External-sampling traversal returning the (importance-weighted)
    /// counterfactual value to `traverser`. At the traverser's nodes every action
    /// is expanded and the instantaneous regrets are reservoired; chance and
    /// opponents are sampled. The current strategy at *every* decision node is
    /// reservoired into the shared strategy buffer with the Linear CFR weight
    /// `iter`.
    ///
    /// ## ε-greedy exploration with importance weighting
    ///
    /// At a sampled (opponent) node the on-policy value is the expectation
    /// `Σ_a σ(a)·V(a)`. To reach rarely-played actions we instead draw the
    /// continuation from `q(a) = ε/n + (1-ε)·σ(a)` and return the single sample
    /// `V(a)·σ(a)/q(a)`. That is unbiased — `E_{a~q}[V(a)·σ(a)/q(a)] =
    /// Σ_a q(a)·V(a)·σ(a)/q(a) = Σ_a σ(a)·V(a)` — so the traverser's value `v`
    /// (and hence every regret `v(a)−v` it stores) is unbiased in expectation.
    /// The `σ(a)/q(a)` factors compound multiplicatively along a path, exactly as
    /// outcome sampling threads its sample probability; here each sampled node
    /// folds its factor into the value it returns upward, so a parent simply
    /// multiplies child values by `σ` as usual. Only the *value* (and hence the
    /// regret) is importance-weighted; the strategy-reservoir push is left as-is
    /// (the average strategy at every visited infoset is still recorded with the
    /// plain Linear-CFR weight).
    fn traverse<G: Game, E: Encoder<G>>(
        &mut self,
        game: &G,
        enc: &E,
        state: &G::State,
        traverser: usize,
        iter: f32,
        rng: &mut Rng,
    ) -> f64 {
        if game.is_terminal(state) {
            return game.returns(state, traverser);
        }
        match game.turn(state) {
            Turn::Chance => {
                let (action, _p) = game.sample_chance(state, rng);
                let mut child = state.clone();
                game.apply(&mut child, action);
                self.traverse(game, enc, &child, traverser, iter, rng)
            }
            Turn::Player(p) => {
                let actions = game.legal_actions(state);
                let n = actions.len();
                let x = enc.features(game, state, p);
                let support = enc.support(game, state);
                debug_assert_eq!(support.len(), n);
                let net = p % self.adv_nets;
                let sigma = self.advantage[net].strategy(&x, &support);

                // Record the current strategy for the average-strategy net.
                // `z = NaN`: a decision node carries a policy label but no
                // equity target, so it trains the policy head only — leaving the
                // value head to the dedicated root value samples (a `z = 0` here
                // would instead drag the value head toward zero everywhere).
                let policy: Vec<(usize, f32)> = support
                    .iter()
                    .zip(&sigma)
                    .map(|(&i, &q)| (i, q as f32))
                    .collect();
                self.strat_buf.push(
                    Sample {
                        x: x.clone(),
                        policy,
                        z: f32::NAN,
                    },
                    iter,
                    rng,
                );

                if p == traverser {
                    // Expand every action: external sampling at the traverser.
                    let mut child_v = vec![0.0; n];
                    let mut v = 0.0;
                    for (i, &a) in actions.iter().enumerate() {
                        let mut child = state.clone();
                        game.apply(&mut child, a);
                        child_v[i] = self.traverse(game, enc, &child, traverser, iter, rng);
                        v += sigma[i] * child_v[i];
                    }
                    // Instantaneous regret r(a) = v(a) - v.
                    let target: Vec<f32> = child_v.iter().map(|&cv| (cv - v) as f32).collect();
                    self.adv_buf[net].push(RegretSample { x, support, target }, iter, rng);
                    v
                } else {
                    // Opponent node: ε-greedy sampling so rarely-played lines are
                    // still reached, with the subtree value importance-weighted by
                    // σ(a)/q(a) to keep the traverser's regret estimates unbiased.
                    let eps = self.cfg.explore_eps;
                    let q: Vec<f64> = sigma
                        .iter()
                        .map(|&pr| eps / n as f64 + (1.0 - eps) * pr)
                        .collect();
                    let i = rng.pick(&q);
                    let iw = sigma[i] / q[i];
                    let mut child = state.clone();
                    game.apply(&mut child, actions[i]);
                    let v = self.traverse(game, enc, &child, traverser, iter, rng);
                    // Fold this node's σ(a)/q(a) into the value returned up.
                    v * iw
                }
            }
        }
    }

    /// Retrain player `p`'s advantage net from scratch on its reservoir, fitting
    /// the linear head to the stored regrets by MSE with Linear CFR (per-sample
    /// `iter`) weighting. A from-scratch net each time keeps stale early regrets
    /// from biasing the current fit (Brown 2019).
    fn retrain_advantage(&mut self, p: usize) {
        let len = self.adv_buf[p].len();
        if len == 0 {
            return;
        }
        let mut net = Mlp::new(
            self.feature_len,
            self.cfg.hidden,
            self.policy_len,
            self.cfg.seed ^ (0x2000 + p as u64),
        );
        let mut opt = SgdMomentum::new(self.cfg.lr, self.cfg.momentum, self.cfg.l2);
        let mut grad = Vec::new();
        let buf = &self.adv_buf[p].buf;
        // Normalize the linear weights to a peak of 1 so the rejection sampler's
        // acceptance and the effective learning rate do not scale with `iter`.
        let max_iter = buf.iter().map(|r| r.iter).fold(0.0_f32, f32::max).max(1.0);
        let b = self.cfg.batch.min(len);
        let mut rng = std::mem::replace(&mut self.rng, Rng::new(0));
        for _ in 0..self.cfg.adv_steps {
            let idx = weighted_indices(buf, b, max_iter, &mut rng);
            let refs: Vec<&RegretSample> = idx.iter().map(|&i| &buf[i].item).collect();
            #[cfg(feature = "parallel")]
            net.regret_grad_par(&refs, &mut grad);
            #[cfg(not(feature = "parallel"))]
            net.regret_grad(&refs, &mut grad);
            opt.step(&mut net, &grad);
        }
        self.rng = rng;
        self.advantage[p] = AdvNet::new(net);
    }

    /// Train the deployable average-strategy net on the strategy reservoir with
    /// `iter`-weighted (Linear CFR) cross-entropy. The result is a
    /// `feature_len -> policy_len` [`Mlp`] (NetAgent-compatible).
    fn train_average_strategy(&mut self) -> Mlp {
        let mut net = Mlp::new(
            self.feature_len,
            self.cfg.hidden,
            self.policy_len,
            self.cfg.seed ^ 0xA5A5_5A5A,
        );
        if self.strat_buf.len() == 0 {
            return net;
        }
        let mut opt = SgdMomentum::new(self.cfg.lr, self.cfg.momentum, self.cfg.l2);
        let mut grad = Vec::new();
        let len = self.strat_buf.len();
        let buf = &self.strat_buf.buf;
        let max_iter = buf.iter().map(|r| r.iter).fold(0.0_f32, f32::max).max(1.0);
        let b = self.cfg.batch.min(len);
        let mut rng = std::mem::replace(&mut self.rng, Rng::new(0));
        for _ in 0..self.cfg.strat_steps {
            let idx = weighted_indices(buf, b, max_iter, &mut rng);
            let refs: Vec<&Sample> = idx.iter().map(|&i| &buf[i].item).collect();
            #[cfg(feature = "parallel")]
            net.grad_par(&refs, &mut grad);
            #[cfg(not(feature = "parallel"))]
            net.grad(&refs, &mut grad);
            opt.step(&mut net, &grad);
        }
        self.rng = rng;
        net
    }

    pub fn strat_reservoir_len(&self) -> usize {
        self.strat_buf.len()
    }

    pub fn advantage_reservoir_len(&self, player: usize) -> usize {
        self.adv_buf[player].len()
    }
}

/// `b` reservoir indices drawn with probability proportional to each sample's
/// stored iteration weight (Linear CFR), by rejection sampling against the peak
/// weight — cheap, exact, and table-free.
fn weighted_indices<T>(buf: &[Reserved<T>], b: usize, max_iter: f32, rng: &mut Rng) -> Vec<usize> {
    (0..b)
        .map(|_| {
            loop {
                let i = rng.below(buf.len());
                if rng.unit() < (buf[i].iter / max_iter) as f64 {
                    return i;
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nash_conv;
    use game_core::{Game, Turn};

    /// Kuhn poker, mirrored from the solver tests so this module is
    /// self-contained (the shared `tests/common` module is integration-only).
    #[derive(Clone, Default)]
    struct KuhnState {
        cards: [i8; 2],
        dealt: u8,
        history: Vec<u8>,
    }

    struct Kuhn;

    impl Kuhn {
        fn betting_terminal(&self, h: &[u8]) -> bool {
            matches!(h, [0, 0] | [0, 1, 0] | [0, 1, 1] | [1, 0] | [1, 1])
        }
    }

    impl Game for Kuhn {
        type State = KuhnState;
        type Action = u8;

        fn initial_state(&self) -> KuhnState {
            KuhnState {
                cards: [-1, -1],
                dealt: 0,
                history: Vec::new(),
            }
        }

        fn turn(&self, s: &KuhnState) -> Turn {
            if s.dealt < 2 {
                Turn::Chance
            } else {
                Turn::Player(s.history.len() % 2)
            }
        }

        fn is_terminal(&self, s: &KuhnState) -> bool {
            s.dealt == 2 && self.betting_terminal(&s.history)
        }

        fn returns(&self, s: &KuhnState, player: usize) -> f64 {
            let h = &s.history;
            let p0_high = s.cards[0] > s.cards[1];
            let to_p0 = match h.as_slice() {
                [0, 0] => {
                    if p0_high {
                        1.0
                    } else {
                        -1.0
                    }
                }
                [1, 0] => 1.0,
                [1, 1] => {
                    if p0_high {
                        2.0
                    } else {
                        -2.0
                    }
                }
                [0, 1, 0] => -1.0,
                [0, 1, 1] => {
                    if p0_high {
                        2.0
                    } else {
                        -2.0
                    }
                }
                _ => 0.0,
            };
            if player == 0 { to_p0 } else { -to_p0 }
        }

        fn max_return(&self) -> f64 {
            2.0
        }

        fn legal_actions(&self, _s: &KuhnState) -> Vec<u8> {
            vec![0, 1]
        }

        fn chance_outcomes(&self, s: &KuhnState) -> Vec<(u8, f64)> {
            let taken = if s.dealt == 1 { s.cards[0] } else { -1 };
            let avail: Vec<u8> = (0..3u8).filter(|&c| c as i8 != taken).collect();
            let p = 1.0 / avail.len() as f64;
            avail.into_iter().map(|c| (c, p)).collect()
        }

        fn apply(&self, s: &mut KuhnState, a: u8) {
            if s.dealt < 2 {
                s.cards[s.dealt as usize] = a as i8;
                s.dealt += 1;
            } else {
                s.history.push(a);
            }
        }

        fn infoset_key(&self, s: &KuhnState, player: usize) -> u64 {
            let mut k = (s.cards[player] + 1) as u64;
            for &a in &s.history {
                k = k * 3 + 1 + a as u64;
            }
            k
        }

        fn state_key(&self, s: &KuhnState) -> Option<u64> {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            s.cards.hash(&mut h);
            s.history.hash(&mut h);
            Some(h.finish())
        }
    }

    /// A tiny Kuhn encoder: card one-hot (3), history-length one-hot (up to 3),
    /// the two betting flags, and a side-to-move bit. The action space is the
    /// two betting moves (check/fold = 0, bet/call = 1). Distinct infosets map
    /// to distinct features, so the net can represent the equilibrium exactly.
    struct KuhnEnc;

    impl Encoder<Kuhn> for KuhnEnc {
        fn feature_len(&self) -> usize {
            3 + 4 + 2 + 1
        }
        fn policy_len(&self) -> usize {
            2
        }
        fn features(&self, _g: &Kuhn, s: &KuhnState, player: usize) -> Vec<f32> {
            let mut x = vec![0.0f32; self.feature_len()];
            let card = s.cards[player];
            if card >= 0 {
                x[card as usize] = 1.0;
            }
            x[3 + s.history.len().min(3)] = 1.0;
            for (i, &a) in s.history.iter().take(2).enumerate() {
                if a == 1 {
                    x[7 + i] = 1.0;
                }
            }
            x[9] = (s.history.len() % 2) as f32;
            x
        }
        fn support(&self, _g: &Kuhn, _s: &KuhnState) -> Vec<usize> {
            vec![0, 1]
        }
    }

    /// THE CORRECTNESS GATE. Deep CFR on Kuhn must drive its average strategy's
    /// exploitability toward the Nash value (~0). If this fails, the engine is
    /// buggy — do not trust it on Liar's Dice.
    #[test]
    fn kuhn_deep_cfr_reaches_low_exploitability() {
        let game = Kuhn;
        let enc = KuhnEnc;
        let cfg = DeepCfrConfig {
            iters: 250,
            traversals: 24,
            train_every: 1,
            hidden: 64,
            adv_reservoir: 400_000,
            strat_reservoir: 800_000,
            adv_steps: 300,
            strat_steps: 6000,
            batch: 512,
            lr: 0.02,
            momentum: 0.9,
            l2: 1e-5,
            seed: 0xC0FFEE,
            adv_nets: 0,
            collect_root_value: false,
            // Exploration ON. The importance weighting keeps the estimator
            // unbiased, so the gate still reaches near-Nash with ε-greedy
            // sampling — the proof the IW math is right. `0.1` here (vs the
            // production `0.6`/`0.5`) is a budget choice, not a correctness one:
            // ε-greedy raises the per-iteration regret-target variance, and on
            // this tiny gate (250 iters) a large ε would need many more
            // iterations to average that variance back down. At ε=0.1 the gate
            // reaches the same ~0.04 floor as ε=0 (no exploration), confirming
            // exploration adds variance but no bias.
            explore_eps: 0.1,
        };
        let mut solver = DeepCfr::new(game.num_players(), &enc, cfg);
        let net = solver.run(&game, &enc);
        let cache = net.infer_cache();
        let policy = |_g: &Kuhn, s: &KuhnState, player: usize| {
            let x = enc.features(&game, s, player);
            let (probs, _) = net.policy_value_cached(&cache, &x, &[0, 1]);
            probs.iter().map(|&p| f64::from(p)).collect::<Vec<f64>>()
        };
        let (br0, br1, nashconv) = nash_conv(&game, &policy);
        let exploitability = nashconv / 2.0;
        println!(
            "Kuhn Deep CFR: br0={br0:.4} br1={br1:.4} nashconv={nashconv:.4} \
             exploitability={exploitability:.4}"
        );
        assert!(
            exploitability < 0.05,
            "Deep CFR must reach near-Nash on Kuhn (correctness gate): \
             exploitability={exploitability} (br0={br0} br1={br1})"
        );
    }

    /// The advantage net's linear head fits sampled regrets by MSE: after a few
    /// hundred steps on a one-infoset stream it recovers the target advantages,
    /// so regret matching over it gives the right strategy.
    #[test]
    fn advantage_head_fits_regrets() {
        let enc = KuhnEnc;
        let mut net = Mlp::new(enc.feature_len(), 32, enc.policy_len(), 7);
        let mut opt = SgdMomentum::new(0.05, 0.9, 0.0);
        let mut grad = Vec::new();
        let x = vec![1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let sample = RegretSample {
            x: x.clone(),
            support: vec![0, 1],
            target: vec![0.3, -0.1],
        };
        for _ in 0..2000 {
            let refs = [&sample];
            net.regret_grad(&refs, &mut grad);
            opt.step(&mut net, &grad);
        }
        let (adv, _) = net.head_values(&x, &[0, 1]);
        assert!((adv[0] - 0.3).abs() < 0.02, "adv[0]={}", adv[0]);
        assert!((adv[1] - (-0.1)).abs() < 0.02, "adv[1]={}", adv[1]);
    }
}
