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
use std::io::{self, Read, Write};

use crate::env::Env;

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
    regret: HashMap<u64, [f64; 2]>,
    strategy_sum: HashMap<u64, [f64; 2]>,
    /// Equilibrium value to player 0 at the start of round (h0, h1, round),
    /// filled by backward induction and used as continuation for shallower rounds.
    value: HashMap<u32, f64>,
    /// Starting hearts per player (6 for the full game; smaller values define a
    /// shorter variant that is exactly solvable for validation).
    start_hearts: u8,
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
    between: HashMap<u32, f64>,
}

fn round_state_key(h0: u8, h1: u8, round: u8) -> u32 {
    (h0 as u32) | ((h1 as u32) << 4) | ((round as u32) << 8)
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
        Self {
            regret: HashMap::new(),
            strategy_sum: HashMap::new(),
            value: HashMap::new(),
            start_hearts,
            iterations: 0,
            rng: Rng::new(seed),
        }
    }

    pub fn iterations(&self) -> u64 {
        self.iterations
    }

    pub fn num_infosets(&self) -> usize {
        self.strategy_sum.len()
    }

    // ----- strategy helpers -------------------------------------------------

    fn regret_matching(&self, key: u64, can_draw: bool) -> [f64; 2] {
        if !can_draw {
            return [0.0, 1.0];
        }
        let r = self.regret.get(&key).copied().unwrap_or([0.0; 2]);
        let s0 = r[DRAW].max(0.0);
        let s1 = r[STAND].max(0.0);
        let sum = s0 + s1;
        if sum > 0.0 {
            [s0 / sum, s1 / sum]
        } else {
            [0.5, 0.5]
        }
    }

    fn average_strategy(&self, key: u64, can_draw: bool) -> [f64; 2] {
        if !can_draw {
            return [0.0, 1.0];
        }
        let s = self.strategy_sum.get(&key).copied().unwrap_or([0.0; 2]);
        let sum = s[DRAW] + s[STAND];
        if sum > 0.0 {
            [s[DRAW] / sum, s[STAND] / sum]
        } else {
            [0.5, 0.5]
        }
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
        for round in (1..=MAX_ROUND).rev() {
            for h0 in 1..=self.start_hearts {
                for h1 in 1..=self.start_hearts {
                    self.solve_round(h0, h1, round, iters_per_subgame);
                }
            }
        }
        self.iterations += iters_per_subgame;
    }

    fn solve_round(&mut self, h0: u8, h1: u8, round: u8, iters: u64) {
        let base = Env::from_state([h0, h1], round);
        for t in 1..=iters {
            let weight = (self.iterations + t) as f64;
            for traverser in 0..2 {
                let cards = deal_from(&mut self.rng);
                let mut env = base.clone();
                env.deal_specific(cards).unwrap();
                self.traverse(&env, traverser, weight);
            }
        }
        let v0 = self.estimate_round_value(h0, h1, round, VALUE_SAMPLES);
        self.value.insert(round_state_key(h0, h1, round), v0);
    }

    /// One external-sampling MCCFR+ traversal of a single round, returning the
    /// counterfactual value to `traverser`. At the traverser's nodes both actions
    /// are evaluated and regrets updated (CFR+, clipped); the drawn card after a
    /// Draw is sampled. At the opponent's nodes the action is sampled from the
    /// current strategy and the average strategy is accumulated (weighted by
    /// `weight` for linear CFR+ averaging). The cross-round continuation value is
    /// substituted when a non-terminal round ends.
    fn traverse(&mut self, env: &Env, traverser: usize, weight: f64) -> f64 {
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
        let key = env.sufficient_key(player);
        let sigma = self.regret_matching(key, can_draw);

        if player == traverser {
            let mut stood = env.clone();
            stood.stand().unwrap();
            let v_stand = self.traverse(&stood, traverser, weight);
            let mut v_draw = 0.0;
            if can_draw {
                let card = nth_deck_card(mask, self.rng.below(deck_count(mask)));
                let mut drawn = env.clone();
                drawn.draw_specific(card).unwrap();
                v_draw = self.traverse(&drawn, traverser, weight);
            }
            let node_v = sigma[STAND] * v_stand + sigma[DRAW] * v_draw;
            let entry = self.regret.entry(key).or_insert([0.0; 2]);
            entry[STAND] = (entry[STAND] + v_stand - node_v).max(0.0);
            if can_draw {
                entry[DRAW] = (entry[DRAW] + v_draw - node_v).max(0.0);
            }
            node_v
        } else {
            let s = self.strategy_sum.entry(key).or_insert([0.0; 2]);
            s[STAND] += weight * sigma[STAND];
            if can_draw {
                s[DRAW] += weight * sigma[DRAW];
            }
            if can_draw && self.rng.unit() < sigma[DRAW] {
                let card = nth_deck_card(mask, self.rng.below(deck_count(mask)));
                let mut drawn = env.clone();
                drawn.draw_specific(card).unwrap();
                self.traverse(&drawn, traverser, weight)
            } else {
                let mut stood = env.clone();
                stood.stand().unwrap();
                self.traverse(&stood, traverser, weight)
            }
        }
    }

    /// Monte-Carlo estimate (to player 0) of a round subgame under the current
    /// average strategy, used as the continuation value for shallower rounds.
    /// Cheap and low-bias; the headline exploitability metric is still computed
    /// by exact best response, so this estimate's variance does not enter it.
    fn estimate_round_value(&mut self, h0: u8, h1: u8, round: u8, samples: u32) -> f64 {
        let mut total = 0.0;
        for _ in 0..samples {
            let mut env = Env::from_state([h0, h1], round);
            let cards = self.sample_deal();
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
                let key = env.sufficient_key(player);
                let sigma = self.average_strategy(key, can_draw);
                if can_draw && self.rng.unit() < sigma[DRAW] {
                    let card = nth_deck_card(mask, self.rng.below(deck_count(mask)));
                    env.draw_specific(card).unwrap();
                } else {
                    env.stand().unwrap();
                }
            }
        }
        total / samples as f64
    }

    fn sample_deal(&mut self) -> [u8; 4] {
        deal_from(&mut self.rng)
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
            between: HashMap::new(),
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
        let mut within: HashMap<u64, f64> = HashMap::new();
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
    fn within_value(&self, env: &Env, ctx: &mut BrCtx, within: &mut HashMap<u64, f64>) -> f64 {
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
            let key = env.sufficient_key(player);
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
        buf.extend_from_slice(&self.iterations.to_le_bytes());
        buf.push(self.start_hearts);
        write_table(&mut buf, &self.regret);
        write_table(&mut buf, &self.strategy_sum);
        let mut f = std::fs::File::create(path)?;
        f.write_all(&buf)
    }

    pub fn load(path: &str) -> io::Result<Self> {
        let mut bytes = Vec::new();
        std::fs::File::open(path)?.read_to_end(&mut bytes)?;
        let mut pos = 0usize;
        let iterations = read_u64(&bytes, &mut pos);
        let start_hearts = bytes[pos];
        pos += 1;
        let regret = read_table(&bytes, &mut pos);
        let strategy_sum = read_table(&bytes, &mut pos);
        Ok(Self {
            regret,
            strategy_sum,
            value: HashMap::new(),
            start_hearts,
            iterations,
            rng: Rng::new(0x5DEECE66D ^ iterations),
        })
    }
}

fn write_table(buf: &mut Vec<u8>, table: &HashMap<u64, [f64; 2]>) {
    buf.extend_from_slice(&(table.len() as u64).to_le_bytes());
    for (k, v) in table {
        buf.extend_from_slice(&k.to_le_bytes());
        buf.extend_from_slice(&v[0].to_le_bytes());
        buf.extend_from_slice(&v[1].to_le_bytes());
    }
}

fn read_u64(bytes: &[u8], pos: &mut usize) -> u64 {
    let v = u64::from_le_bytes(bytes[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    v
}

fn read_f64(bytes: &[u8], pos: &mut usize) -> f64 {
    let v = f64::from_le_bytes(bytes[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    v
}

fn read_table(bytes: &[u8], pos: &mut usize) -> HashMap<u64, [f64; 2]> {
    let n = read_u64(bytes, pos) as usize;
    let mut table = HashMap::with_capacity(n);
    for _ in 0..n {
        let k = read_u64(bytes, pos);
        let a = read_f64(bytes, pos);
        let b = read_f64(bytes, pos);
        table.insert(k, [a, b]);
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let mut solver = Solver::new(7);
        solver.solve(300);
        let path = std::env::temp_dir().join("twentyone_solver_test.bin");
        let path = path.to_str().unwrap();
        solver.save(path).unwrap();
        let reloaded = Solver::load(path).unwrap();
        assert_eq!(reloaded.iterations(), solver.iterations());
        assert_eq!(reloaded.num_infosets(), solver.num_infosets());
        let key = *solver.strategy_sum.keys().next().unwrap();
        assert_eq!(
            reloaded.average_draw_prob(key),
            solver.average_draw_prob(key)
        );
    }
}
