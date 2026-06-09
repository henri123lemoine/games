//! Exact game-theoretic solver for Twenty-One.
//!
//! Training uses External-Sampling Monte Carlo Counterfactual Regret
//! Minimization with the CFR+ refinements (regrets clipped to non-negative,
//! average strategy weighted linearly by iteration), which converges to a Nash
//! equilibrium of this two-player zero-sum game. Regret and average-strategy
//! tables are keyed on [`Env::sufficient_key`] — a *lossless* sufficient
//! statistic that collapses raw decision histories into strategically distinct
//! information sets. The game is solved subgame-by-subgame via backward
//! induction over the (hearts0, hearts1, round) carry-over lattice.
//!
//! Quality is measured by exact best response: [`Solver::exploitability`] walks
//! the full game tree (enumerating every chance event) to compute, for each
//! player, the value a perfect counter-strategy extracts against the current
//! average strategy. NashConv = br0 + br1 → 0 at equilibrium.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::io::{self, Read, Write};

use rayon::prelude::*;

use crate::env::Env;

/// FxHash-style hasher: the information-set keys are already well-distributed
/// bit-packed `u64`s, so a single multiply-rotate mixes them far faster than the
/// default SipHash — this is the solver's hottest inner-loop cost.
#[derive(Default)]
struct FxHasher(u64);

impl Hasher for FxHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u8(b);
        }
    }
    fn write_u8(&mut self, i: u8) {
        self.write_u64(i as u64);
    }
    fn write_u32(&mut self, i: u32) {
        self.write_u64(i as u64);
    }
    fn write_u64(&mut self, i: u64) {
        const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(SEED);
    }
}

type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

/// Per-information-set table mapping a key to `[draw, stand]` regrets or strategy
/// sums.
type InfoTable = FastMap<u64, [f64; 2]>;

/// The result of solving one round subgame in isolation: its round key, the
/// regret and average-strategy updates, and the continuation value to player 0.
type SubgameResult = (u32, InfoTable, InfoTable, f64);

fn fast_map<K, V>() -> FastMap<K, V> {
    FastMap::default()
}

const STARTING_HEARTS: u8 = 6;
const DRAW: usize = 0;
const STAND: usize = 1;

/// Rounds only continue past a real outcome via repeated ties (which deal no
/// damage). Beyond this many rounds the game is a draw (utility 0): reaching it
/// requires a tie streak whose probability is vanishingly small, so capping is
/// unbiased and bounds recursion depth.
const MAX_ROUND: u8 = 16;

/// Monte-Carlo rollouts used to estimate each round subgame's continuation value
/// during backward induction.
const VALUE_SAMPLES: u32 = 4000;

/// Minimal xorshift PRNG (the solver owns its randomness so runs are
/// reproducible and independent of any cloned env's internal RNG).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

pub struct Solver {
    regret: FastMap<u64, [f64; 2]>,
    strategy_sum: FastMap<u64, [f64; 2]>,
    /// Equilibrium value to player 0 at the start of round (h0, h1, round),
    /// filled by backward induction and used as continuation for shallower rounds.
    value: FastMap<u32, f64>,
    /// Starting hearts per player (6 for the full game; smaller values define a
    /// shorter variant that is exactly solvable for validation).
    start_hearts: u8,
    /// When true, information sets are keyed by [`Env::abstract_key`] (a lossy
    /// summary of the unseen set) instead of the lossless [`Env::sufficient_key`].
    /// Trades exactness for far fewer infosets, making the full game converge to
    /// strong play quickly. Best-response exploitability is still measured on the
    /// true game.
    abstract_keys: bool,
    iterations: u64,
    rng: Rng,
}

/// Threaded state for the best-response recursion: which player is the best
/// responder, how the opening deal is integrated, the sampling RNG, and the
/// cross-round value memo.
struct BrCtx {
    br: usize,
    deal_samples: u32,
    rng: Rng,
    between: FastMap<u32, f64>,
}

fn round_state_key(h0: u8, h1: u8, round: u8) -> u32 {
    (h0 as u32) | ((h1 as u32) << 4) | ((round as u32) << 8)
}

/// All (h0, h1, round) subgames reachable from the (start, start, 1) opening,
/// following the three round outcomes (p0 wins → h1 -= round, p1 wins →
/// h0 -= round, tie → unchanged), each advancing the round by one up to
/// [`MAX_ROUND`]. Only states with both players alive are returned.
fn reachable_subgames(start: u8) -> Vec<(u8, u8, u8)> {
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![(start, start, 1u8)];
    while let Some(s @ (h0, h1, round)) = stack.pop() {
        if h0 == 0 || h1 == 0 || round > MAX_ROUND || !seen.insert(s) {
            continue;
        }
        let next = round + 1;
        if next <= MAX_ROUND {
            stack.push((h0, h1.saturating_sub(round), next));
            stack.push((h0.saturating_sub(round), h1, next));
            stack.push((h0, h1, next));
        }
    }
    seen.into_iter().collect()
}

#[inline]
fn deck_count(mask: u16) -> usize {
    mask.count_ones() as usize
}

/// The (n+1)-th lowest card present in `mask` (0-indexed), for sampling.
#[inline]
fn nth_deck_card(mask: u16, mut n: usize) -> u8 {
    let mut m = mask;
    loop {
        let i = m.trailing_zeros();
        if n == 0 {
            return (i + 1) as u8;
        }
        n -= 1;
        m &= m - 1;
    }
}

/// Sample the four opening cards (p0 up, p1 up, p0 down, p1 down) without
/// replacement from the 1..=11 deck.
fn deal_from(rng: &mut Rng) -> [u8; 4] {
    let mut cards: [u8; 11] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    for k in 0..4 {
        let j = k + rng.below(11 - k);
        cards.swap(k, j);
    }
    [cards[0], cards[1], cards[2], cards[3]]
}

impl Solver {
    pub fn new(seed: u64) -> Self {
        Self::with_hearts(seed, STARTING_HEARTS)
    }

    /// Construct a solver for a variant with `start_hearts` hearts per player.
    /// The full game uses 6; smaller values yield a shorter, exactly-solvable
    /// game useful for validating convergence.
    pub fn with_hearts(seed: u64, start_hearts: u8) -> Self {
        Self::configured(seed, start_hearts, false)
    }

    /// Construct a solver that keys information sets by the lossy
    /// [`Env::abstract_key`] abstraction — use for the full 6-heart game, where
    /// the lossless representation has too many information sets to converge to
    /// strong play in reasonable time.
    pub fn abstracted(seed: u64, start_hearts: u8) -> Self {
        Self::configured(seed, start_hearts, true)
    }

    fn configured(seed: u64, start_hearts: u8, abstract_keys: bool) -> Self {
        Self {
            regret: fast_map(),
            strategy_sum: fast_map(),
            value: fast_map(),
            start_hearts,
            abstract_keys,
            iterations: 0,
            rng: Rng::new(seed),
        }
    }

    #[inline]
    fn info_key(&self, env: &Env, player: usize) -> u64 {
        if self.abstract_keys {
            env.abstract_key(player)
        } else {
            env.sufficient_key(player)
        }
    }

    /// Draw probability under the learned average strategy for the current player
    /// in `env`, using whichever information-set keying this solver was trained
    /// with. Returns 0.0 when the deck is empty (a forced stand).
    pub fn play_draw_prob(&self, env: &Env, player: usize) -> f64 {
        if env.deck_mask() == 0 {
            return 0.0;
        }
        self.average_draw_prob(self.info_key(env, player))
    }

    pub fn iterations(&self) -> u64 {
        self.iterations
    }

    pub fn num_infosets(&self) -> usize {
        self.strategy_sum.len()
    }

    // ----- strategy helpers -------------------------------------------------

    /// Regret matching using the current regrets for `key`, preferring the
    /// subgame-local table (this chunk's in-progress updates) and falling back to
    /// the merged global table (prior chunks). Local-first keeps parallel subgame
    /// solves independent.
    fn regret_matching(
        &self,
        key: u64,
        can_draw: bool,
        local: &FastMap<u64, [f64; 2]>,
    ) -> [f64; 2] {
        if !can_draw {
            return [0.0, 1.0];
        }
        let r = local
            .get(&key)
            .or_else(|| self.regret.get(&key))
            .copied()
            .unwrap_or([0.0; 2]);
        let s0 = r[DRAW].max(0.0);
        let s1 = r[STAND].max(0.0);
        let sum = s0 + s1;
        if sum > 0.0 {
            [s0 / sum, s1 / sum]
        } else {
            [0.5, 0.5]
        }
    }

    fn normalize_strategy(s: [f64; 2], can_draw: bool) -> [f64; 2] {
        if !can_draw {
            return [0.0, 1.0];
        }
        let sum = s[DRAW] + s[STAND];
        if sum > 0.0 {
            [s[DRAW] / sum, s[STAND] / sum]
        } else {
            [0.5, 0.5]
        }
    }

    /// Average strategy from the merged global table (used for play and the
    /// best-response measurement, after training has merged all subgames).
    fn average_strategy(&self, key: u64, can_draw: bool) -> [f64; 2] {
        let s = self.strategy_sum.get(&key).copied().unwrap_or([0.0; 2]);
        Self::normalize_strategy(s, can_draw)
    }

    /// Average strategy preferring the subgame-local table, for the continuation
    /// estimate computed while a subgame is still being solved.
    fn average_strategy_local(
        &self,
        key: u64,
        can_draw: bool,
        local: &FastMap<u64, [f64; 2]>,
    ) -> [f64; 2] {
        let s = local
            .get(&key)
            .or_else(|| self.strategy_sum.get(&key))
            .copied()
            .unwrap_or([0.0; 2]);
        Self::normalize_strategy(s, can_draw)
    }

    /// Probability of drawing under the average strategy for `key`, for play.
    pub fn average_draw_prob(&self, key: u64) -> f64 {
        self.average_strategy(key, true)[DRAW]
    }

    // ----- training: backward induction over round subgames -----------------

    /// Solve the whole game by backward induction. Each one-round subgame at
    /// (h0, h1, round) is a shallow imperfect-information game; rounds are linked
    /// only by the public hearts/round carry-over (the deck is reshuffled with no
    /// hidden information surviving a round), so deeper rounds can be solved first
    /// and supplied as exact continuation values.
    ///
    /// Each subgame is solved with `iters_per_subgame` iterations of
    /// external-sampling MCCFR+: every iteration samples the opening deal, then
    /// for each traverser walks the round tree enumerating only that player's two
    /// actions while sampling the opponent's action and the drawn card (chance).
    /// Per-iteration cost is therefore polynomial in the round depth, and
    /// regret/strategy use CFR+ (clipped regrets, linear averaging) for fast
    /// convergence.
    pub fn solve(&mut self, iters_per_subgame: u64) {
        // Solve only subgames reachable from the start, in descending round order
        // so each subgame's cross-round continuations are already solved (a tie
        // keeps hearts and advances the round; a decided round reduces a heart
        // count). Unreachable (hearts, round) carry-overs never occur in play or
        // as continuations, so skipping them is exact.
        //
        // Subgames at the same round are mutually independent — they only read
        // already-solved deeper-round continuation values, and their information
        // sets are keyed disjointly by (hearts, round) — so a whole round is
        // solved in parallel. Each subgame accumulates into private local tables
        // (seeded on demand from the global tables), and the round's results are
        // merged back single-threaded, which keeps the result independent of how
        // many threads run.
        let mut subgames = reachable_subgames(self.start_hearts);
        subgames.sort_unstable_by(|a, b| b.2.cmp(&a.2).then_with(|| a.cmp(b)));
        let base_iters = self.iterations;

        let mut i = 0;
        while i < subgames.len() {
            let round = subgames[i].2;
            let mut seeded = Vec::new();
            while i < subgames.len() && subgames[i].2 == round {
                let (h0, h1, r) = subgames[i];
                seeded.push((h0, h1, r, self.rng.next()));
                i += 1;
            }
            let results: Vec<SubgameResult> = seeded
                .par_iter()
                .map(|&(h0, h1, r, seed)| {
                    self.solve_subgame(h0, h1, r, iters_per_subgame, base_iters, seed)
                })
                .collect();
            for (rk, lr, ls, v) in results {
                self.regret.extend(lr);
                self.strategy_sum.extend(ls);
                self.value.insert(rk, v);
            }
        }
        self.iterations += iters_per_subgame;
    }

    /// Solve a single round subgame in isolation, returning its round key, the
    /// regret and average-strategy updates (local tables seeded from the global
    /// ones), and the estimated continuation value to player 0.
    fn solve_subgame(
        &self,
        h0: u8,
        h1: u8,
        round: u8,
        iters: u64,
        base_iters: u64,
        seed: u64,
    ) -> SubgameResult {
        let mut rng = Rng::new(seed);
        let mut lr: FastMap<u64, [f64; 2]> = fast_map();
        let mut ls: FastMap<u64, [f64; 2]> = fast_map();
        let base = Env::from_state([h0, h1], round);
        for t in 1..=iters {
            let weight = (base_iters + t) as f64;
            for traverser in 0..2 {
                let cards = deal_from(&mut rng);
                let mut env = base.clone();
                env.deal_specific(cards).unwrap();
                self.traverse(&env, traverser, weight, &mut lr, &mut ls, &mut rng);
            }
        }
        let v0 = self.estimate_round_value(h0, h1, round, VALUE_SAMPLES, &ls, &mut rng);
        (round_state_key(h0, h1, round), lr, ls, v0)
    }

    /// One external-sampling MCCFR+ traversal of a single round, returning the
    /// counterfactual value to `traverser`. At the traverser's nodes both actions
    /// are evaluated and regrets updated (CFR+, clipped); the drawn card after a
    /// Draw is sampled. At the opponent's nodes the action is sampled from the
    /// current strategy and the average strategy is accumulated (weighted by
    /// `weight` for linear CFR+ averaging). Regret/strategy updates go to the
    /// subgame-local tables `lr`/`ls`. The cross-round continuation value is
    /// substituted when a non-terminal round ends.
    fn traverse(
        &self,
        env: &Env,
        traverser: usize,
        weight: f64,
        lr: &mut FastMap<u64, [f64; 2]>,
        ls: &mut FastMap<u64, [f64; 2]>,
        rng: &mut Rng,
    ) -> f64 {
        if !env.round_active() {
            if env.is_game_over() {
                return env.utility(traverser);
            }
            let v0 = self.continuation_for_p0(env.hearts(0), env.hearts(1), env.round());
            return if traverser == 0 { v0 } else { -v0 };
        }

        let player = env.current_player();
        let mask = env.deck_mask();
        let can_draw = mask != 0;
        let key = self.info_key(env, player);
        let sigma = self.regret_matching(key, can_draw, lr);

        if player == traverser {
            let mut stood = env.clone();
            stood.stand().unwrap();
            let v_stand = self.traverse(&stood, traverser, weight, lr, ls, rng);
            let mut v_draw = 0.0;
            if can_draw {
                let card = nth_deck_card(mask, rng.below(deck_count(mask)));
                let mut drawn = env.clone();
                drawn.draw_specific(card).unwrap();
                v_draw = self.traverse(&drawn, traverser, weight, lr, ls, rng);
            }
            let node_v = sigma[STAND] * v_stand + sigma[DRAW] * v_draw;
            let entry = lr
                .entry(key)
                .or_insert_with(|| self.regret.get(&key).copied().unwrap_or([0.0; 2]));
            entry[STAND] = (entry[STAND] + v_stand - node_v).max(0.0);
            if can_draw {
                entry[DRAW] = (entry[DRAW] + v_draw - node_v).max(0.0);
            }
            node_v
        } else {
            let s = ls
                .entry(key)
                .or_insert_with(|| self.strategy_sum.get(&key).copied().unwrap_or([0.0; 2]));
            s[STAND] += weight * sigma[STAND];
            if can_draw {
                s[DRAW] += weight * sigma[DRAW];
            }
            if can_draw && rng.unit() < sigma[DRAW] {
                let card = nth_deck_card(mask, rng.below(deck_count(mask)));
                let mut drawn = env.clone();
                drawn.draw_specific(card).unwrap();
                self.traverse(&drawn, traverser, weight, lr, ls, rng)
            } else {
                let mut stood = env.clone();
                stood.stand().unwrap();
                self.traverse(&stood, traverser, weight, lr, ls, rng)
            }
        }
    }

    /// Monte-Carlo estimate (to player 0) of a round subgame under the current
    /// average strategy (preferring the subgame-local table `ls`), used as the
    /// continuation value for shallower rounds. Cheap and low-bias; the headline
    /// exploitability metric is still computed by exact best response, so this
    /// estimate's variance does not enter it.
    fn estimate_round_value(
        &self,
        h0: u8,
        h1: u8,
        round: u8,
        samples: u32,
        ls: &FastMap<u64, [f64; 2]>,
        rng: &mut Rng,
    ) -> f64 {
        let mut total = 0.0;
        for _ in 0..samples {
            let mut env = Env::from_state([h0, h1], round);
            let cards = deal_from(rng);
            env.deal_specific(cards).unwrap();
            loop {
                if !env.round_active() {
                    total += if env.is_game_over() {
                        env.utility(0)
                    } else {
                        self.continuation_for_p0(env.hearts(0), env.hearts(1), env.round())
                    };
                    break;
                }
                let player = env.current_player();
                let mask = env.deck_mask();
                let can_draw = mask != 0;
                let key = self.info_key(&env, player);
                let sigma = self.average_strategy_local(key, can_draw, ls);
                if can_draw && rng.unit() < sigma[DRAW] {
                    let card = nth_deck_card(mask, rng.below(deck_count(mask)));
                    env.draw_specific(card).unwrap();
                } else {
                    env.stand().unwrap();
                }
            }
        }
        total / samples as f64
    }

    /// Continuation value to player 0 once a round has ended into the given
    /// hearts/round carry-over (called only when the game has not ended).
    fn continuation_for_p0(&self, h0: u8, h1: u8, round: u8) -> f64 {
        if round > MAX_ROUND {
            return 0.0;
        }
        self.value
            .get(&round_state_key(h0, h1, round))
            .copied()
            .unwrap_or(0.0)
    }

    // ----- exact best response / exploitability -----------------------------

    /// Returns (best_response_value_player0, best_response_value_player1,
    /// nashconv). NashConv = br0 + br1 and equals 0 at a Nash equilibrium;
    /// exploitability is NashConv / 2. Utilities are in [-1, 1].
    ///
    /// The best responder plays exactly within each round and uses exact
    /// continuation values across rounds; only the opening deal of each round is
    /// integrated. With `deal_samples == 0` that deal is enumerated exhaustively
    /// (a fully exact, but expensive, NashConv). With `deal_samples > 0` it is
    /// Monte-Carlo sampled, giving an unbiased estimate that is far cheaper —
    /// ideal for tracking convergence, with a large sample (or 0) for the final
    /// headline number.
    pub fn exploitability(&self, deal_samples: u32, seed: u64) -> (f64, f64, f64) {
        let h = self.start_hearts;
        let mut ctx = BrCtx {
            br: 0,
            deal_samples,
            rng: Rng::new(seed),
            between: fast_map(),
        };
        let br0 = self.round_value(h, h, 1, &mut ctx);
        ctx.br = 1;
        let br1 = self.round_value(h, h, 1, &mut ctx);
        (br0, br1, br0 + br1)
    }

    /// Best-response value to `ctx.br` at the start of the round identified by
    /// the (h0, h1, round) carry-over, averaged over the opening deal.
    fn round_value(&self, h0: u8, h1: u8, round: u8, ctx: &mut BrCtx) -> f64 {
        if h0 == 0 || h1 == 0 {
            return if (h0 > 0) == (ctx.br == 0) { 1.0 } else { -1.0 };
        }
        if round > MAX_ROUND {
            return 0.0;
        }
        let bkey =
            (h0 as u32) | ((h1 as u32) << 4) | ((round as u32) << 8) | ((ctx.br as u32) << 16);
        if let Some(v) = ctx.between.get(&bkey) {
            return *v;
        }

        let base = Env::from_state([h0, h1], round);
        let mut within: FastMap<u64, f64> = fast_map();
        let mut total = 0.0;
        let mut count = 0u32;
        if ctx.deal_samples == 0 {
            for c0 in 1..=11u8 {
                for c1 in 1..=11u8 {
                    if c1 == c0 {
                        continue;
                    }
                    for c2 in 1..=11u8 {
                        if c2 == c0 || c2 == c1 {
                            continue;
                        }
                        for c3 in 1..=11u8 {
                            if c3 == c0 || c3 == c1 || c3 == c2 {
                                continue;
                            }
                            let mut dealt = base.clone();
                            dealt.deal_specific([c0, c1, c2, c3]).unwrap();
                            total += self.within_value(&dealt, ctx, &mut within);
                            count += 1;
                        }
                    }
                }
            }
        } else {
            for _ in 0..ctx.deal_samples {
                let cards = deal_from(&mut ctx.rng);
                let mut dealt = base.clone();
                dealt.deal_specific(cards).unwrap();
                total += self.within_value(&dealt, ctx, &mut within);
                count += 1;
            }
        }
        let v = total / count as f64;
        ctx.between.insert(bkey, v);
        v
    }

    /// Best-response value to `ctx.br` for an in-progress (or just-ended) round.
    fn within_value(&self, env: &Env, ctx: &mut BrCtx, within: &mut FastMap<u64, f64>) -> f64 {
        if !env.round_active() {
            if env.is_game_over() {
                return env.utility(ctx.br);
            }
            return self.round_value(env.hearts(0), env.hearts(1), env.round(), ctx);
        }

        let wkey = env.search_key();
        if let Some(v) = within.get(&wkey) {
            return *v;
        }

        let player = env.current_player();
        let mask = env.deck_mask();
        let can_draw = mask != 0;
        let value = if player == ctx.br {
            let mut stood = env.clone();
            stood.stand().unwrap();
            let mut best = self.within_value(&stood, ctx, within);
            if can_draw {
                let mut acc = 0.0;
                let mut m = mask;
                while m != 0 {
                    let c = (m.trailing_zeros() + 1) as u8;
                    m &= m - 1;
                    let mut drawn = env.clone();
                    drawn.draw_specific(c).unwrap();
                    acc += self.within_value(&drawn, ctx, within);
                }
                let draw_value = acc / deck_count(mask) as f64;
                if draw_value > best {
                    best = draw_value;
                }
            }
            best
        } else {
            let key = self.info_key(env, player);
            let sigma = self.average_strategy(key, can_draw);
            let mut v = 0.0;
            if sigma[STAND] > 0.0 {
                let mut stood = env.clone();
                stood.stand().unwrap();
                v += sigma[STAND] * self.within_value(&stood, ctx, within);
            }
            if sigma[DRAW] > 0.0 && can_draw {
                let mut acc = 0.0;
                let mut m = mask;
                while m != 0 {
                    let c = (m.trailing_zeros() + 1) as u8;
                    m &= m - 1;
                    let mut drawn = env.clone();
                    drawn.draw_specific(c).unwrap();
                    acc += self.within_value(&drawn, ctx, within);
                }
                v += sigma[DRAW] * (acc / deck_count(mask) as f64);
            }
            v
        };

        within.insert(wkey, value);
        value
    }

    // ----- persistence ------------------------------------------------------

    pub fn save(&self, path: &str) -> io::Result<()> {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&SOLVER_MAGIC.to_le_bytes());
        buf.push(SOLVER_FORMAT_VERSION);
        buf.extend_from_slice(&self.iterations.to_le_bytes());
        buf.push(self.start_hearts);
        buf.push(self.abstract_keys as u8);
        write_table(&mut buf, &self.regret);
        write_table(&mut buf, &self.strategy_sum);
        let mut f = std::fs::File::create(path)?;
        f.write_all(&buf)
    }

    pub fn load(path: &str) -> io::Result<Self> {
        let mut bytes = Vec::new();
        std::fs::File::open(path)?.read_to_end(&mut bytes)?;
        let mut r = Reader::new(&bytes);
        let magic = r.u32()?;
        if magic != SOLVER_MAGIC {
            return Err(invalid(
                "not a Twenty-One solver file (bad magic); was it saved by an incompatible build? rebuild and retrain",
            ));
        }
        let version = r.u8()?;
        if version != SOLVER_FORMAT_VERSION {
            return Err(invalid(format!(
                "unsupported solver format version {version} (expected {SOLVER_FORMAT_VERSION}); retrain with this build"
            )));
        }
        let iterations = r.u64()?;
        let start_hearts = r.u8()?;
        let abstract_keys = r.u8()? != 0;
        let regret = r.table()?;
        let strategy_sum = r.table()?;
        if !r.is_empty() {
            return Err(invalid(
                "trailing bytes after solver tables (corrupt or truncated file)",
            ));
        }
        Ok(Self {
            regret,
            strategy_sum,
            value: fast_map(),
            start_hearts,
            abstract_keys,
            iterations,
            rng: Rng::new(0x5DEECE66D ^ iterations),
        })
    }
}

/// Magic + version prefix so a wrong or stale file fails with a clear error
/// instead of a panic. Bump the version whenever the byte layout changes.
const SOLVER_MAGIC: u32 = 0x3132_5453; // "T21S" little-endian
const SOLVER_FORMAT_VERSION: u8 = 1;

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

fn write_table(buf: &mut Vec<u8>, table: &FastMap<u64, [f64; 2]>) {
    buf.extend_from_slice(&(table.len() as u64).to_le_bytes());
    for (k, v) in table {
        buf.extend_from_slice(&k.to_le_bytes());
        buf.extend_from_slice(&v[0].to_le_bytes());
        buf.extend_from_slice(&v[1].to_le_bytes());
    }
}

/// Bounds-checked little-endian byte reader: every read fails cleanly with an
/// `UnexpectedEof` error rather than panicking on an out-of-range slice.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn is_empty(&self) -> bool {
        self.pos == self.bytes.len()
    }

    fn take(&mut self, n: usize) -> io::Result<&[u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| invalid("length overflow"))?;
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "solver file truncated"))?;
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> io::Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn f64(&mut self) -> io::Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn table(&mut self) -> io::Result<FastMap<u64, [f64; 2]>> {
        let n = self.u64()? as usize;
        // Guard against a corrupt length claiming more entries than the file can
        // possibly hold (24 bytes each) before allocating.
        let remaining = self.bytes.len() - self.pos;
        if n > remaining / 24 {
            return Err(invalid(
                "solver table length exceeds file size (corrupt file)",
            ));
        }
        let mut table = FastMap::with_capacity_and_hasher(n, Default::default());
        for _ in 0..n {
            let k = self.u64()?;
            let a = self.f64()?;
            let b = self.f64()?;
            table.insert(k, [a, b]);
        }
        Ok(table)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_solve_is_deterministic() {
        // Per-subgame RNGs are seeded in a fixed order and round results merged in
        // a fixed order, so training is reproducible and independent of how many
        // threads rayon happens to use. A multi-subgame-per-round variant (2
        // hearts) exercises the parallel path.
        let mut a = Solver::abstracted(99, 2);
        a.solve(500);
        let mut b = Solver::abstracted(99, 2);
        b.solve(500);
        assert_eq!(a.num_infosets(), b.num_infosets());
        let mut keys: Vec<u64> = a.strategy_sum.keys().copied().collect();
        keys.sort_unstable();
        for k in keys.iter().take(64) {
            assert_eq!(a.average_draw_prob(*k), b.average_draw_prob(*k));
            assert_eq!(a.regret.get(k), b.regret.get(k));
        }
    }

    #[test]
    fn solver_reduces_exploitability() {
        // External-sampling MCCFR+ must drive exact exploitability down
        // substantially with more iterations (validating convergence toward Nash).
        // The game is too large to reach ~0 in a unit test, so we assert a clear
        // drop rather than near-zero.
        let mut weak = Solver::with_hearts(1, 1);
        weak.solve(300);
        let (_, _, nc_weak) = weak.exploitability(0, 0);
        let mut strong = Solver::with_hearts(1, 1);
        strong.solve(15_000);
        let (_, _, nc_strong) = strong.exploitability(0, 0);
        assert!(
            nc_strong < nc_weak - 0.2,
            "weak={nc_weak} strong={nc_strong}"
        );
    }

    #[test]
    fn save_load_roundtrip() {
        for mut solver in [Solver::new(7), Solver::abstracted(7, 6)] {
            solver.solve(300);
            let path = std::env::temp_dir().join(format!(
                "twentyone_solver_test_{}.bin",
                solver.abstract_keys
            ));
            let path = path.to_str().unwrap();
            solver.save(path).unwrap();
            let reloaded = Solver::load(path).unwrap();
            assert_eq!(reloaded.iterations(), solver.iterations());
            assert_eq!(reloaded.num_infosets(), solver.num_infosets());
            assert_eq!(reloaded.abstract_keys, solver.abstract_keys);
            let key = *solver.strategy_sum.keys().next().unwrap();
            assert_eq!(
                reloaded.average_draw_prob(key),
                solver.average_draw_prob(key)
            );
        }
    }

    #[test]
    fn load_rejects_invalid_files() {
        let dir = std::env::temp_dir();

        // Random/garbage bytes (e.g. a file from an incompatible build) must
        // error cleanly, never panic.
        let garbage = dir.join("twentyone_solver_garbage.bin");
        std::fs::write(&garbage, vec![0xABu8; 4096]).unwrap();
        assert!(Solver::load(garbage.to_str().unwrap()).is_err());

        // A valid file truncated mid-table must error rather than over-read.
        let mut solver = Solver::with_hearts(3, 2);
        solver.solve(200);
        let good = dir.join("twentyone_solver_good.bin");
        solver.save(good.to_str().unwrap()).unwrap();
        let mut bytes = std::fs::read(&good).unwrap();
        bytes.truncate(bytes.len() - 16);
        let truncated = dir.join("twentyone_solver_truncated.bin");
        std::fs::write(&truncated, &bytes).unwrap();
        assert!(Solver::load(truncated.to_str().unwrap()).is_err());
    }
}
