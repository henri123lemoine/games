//! Hand-crafted agents for Liar's Dice, primarily the probabilistic belief
//! player: it reasons exactly about the unknown dice with the binomial
//! distribution (each unknown die shows a given face with probability `1/faces`,
//! since 1s are not wild), which scales to any number of players and dice.

use cfr_core::{Agent, Game};

use crate::{Action, LdState, LiarsDice};

/// `P(Binomial(n, p) >= k)`.
fn binom_sf(n: u32, p: f64, k: i64) -> f64 {
    if k <= 0 {
        return 1.0;
    }
    if k as u32 > n {
        return 0.0;
    }
    let mut term = (1.0 - p).powi(n as i32); // P(X = 0)
    let mut cdf_below = 0.0; // P(X <= k-1)
    for i in 0..k as u32 {
        cdf_below += term;
        term *= p * (n - i) as f64 / ((i + 1) as f64 * (1.0 - p));
    }
    (1.0 - cdf_below).clamp(0.0, 1.0)
}

/// `P(Binomial(n, p) == k)`.
fn binom_pmf(n: u32, p: f64, k: i64) -> f64 {
    if k < 0 || k as u32 > n {
        return 0.0;
    }
    let mut term = (1.0 - p).powi(n as i32);
    for i in 0..k as u32 {
        term *= p * (n - i) as f64 / ((i + 1) as f64 * (1.0 - p));
    }
    term
}

/// Tunable thresholds for the probabilistic player. Defaults are sane; self-play
/// search can refine them.
#[derive(Clone, Copy, Debug)]
pub struct ProbConfig {
    /// Call LIAR when the current bid's probability of being true drops below this.
    pub liar_cut: f64,
    /// Call EXACT when the probability the bid is exactly right exceeds this.
    pub exact_cut: f64,
    /// When raising, accept a bid only if its truth probability is at least this;
    /// otherwise prefer to call rather than make an implausible bid.
    pub safety: f64,
    /// Probability of a deliberate bluff raise (using the supplied randomness).
    pub bluff: f64,
    /// Opponent inference: a bidder credibly holds this many of the bid's face,
    /// so we discount the required count by it when judging their bid's truth.
    pub bidder_bias: f64,
    /// Opening aggression: fraction of the *expected* unknown count of my best
    /// face to fold into the opening bid (0 = bid only what I hold).
    pub open_frac: f64,
    /// Soft calling band: randomize LIAR over a window above `liar_cut` so the
    /// agent isn't a deterministic, readable caller.
    pub mix: f64,
}

impl Default for ProbConfig {
    /// League-tuned on 5p5d6f (see `examples/league`): aggressive bids, eager
    /// EXACT calls (this variant punishes neither when right), ready to call liar.
    fn default() -> Self {
        Self {
            liar_cut: 0.275,
            exact_cut: 0.500,
            safety: 0.191,
            bluff: 0.046,
            bidder_bias: 0.383,
            open_frac: 0.5,
            mix: 0.06,
        }
    }
}

impl ProbConfig {
    /// The original hand-set baseline, kept for regression comparison.
    pub fn baseline() -> Self {
        Self {
            liar_cut: 0.32,
            exact_cut: 0.32,
            safety: 0.42,
            bluff: 0.08,
            bidder_bias: 0.6,
            open_frac: 0.0,
            mix: 0.0,
        }
    }
}

pub struct ProbabilisticAgent {
    pub cfg: ProbConfig,
}

impl ProbabilisticAgent {
    pub fn new(cfg: ProbConfig) -> Self {
        Self { cfg }
    }
    pub fn default_agent() -> Self {
        Self {
            cfg: ProbConfig::default(),
        }
    }

    /// Probability the bid `(q, face)` is true given my hand. `signal` discounts
    /// the count we must find among unknown dice — used to credit a bidder for
    /// credibly holding their own face.
    fn p_true(
        &self,
        game: &LiarsDice,
        s: &LdState,
        player: usize,
        q: u8,
        face: u8,
        signal: f64,
    ) -> f64 {
        let total: u8 = s.dice_left().iter().sum();
        let my_dice = s.dice_left()[player];
        let unknown = (total - my_dice) as u32;
        let need = (q as f64 - s.my_count(player, face) as f64 - signal).ceil() as i64;
        binom_sf(unknown, 1.0 / game.faces as f64, need)
    }

    fn p_exact(&self, game: &LiarsDice, s: &LdState, player: usize, q: u8, face: u8) -> f64 {
        let total: u8 = s.dice_left().iter().sum();
        let my_dice = s.dice_left()[player];
        let unknown = (total - my_dice) as u32;
        let need = q as i64 - s.my_count(player, face) as i64;
        binom_pmf(unknown, 1.0 / game.faces as f64, need)
    }

    /// The bid that results from a raise action, if any.
    fn raised_bid(&self, game: &LiarsDice, q: u8, face: u8, a: Action) -> Option<(u8, u8)> {
        match a {
            Action::RaiseQuantity => Some((q + 1, face)),
            Action::RaiseFace => {
                if face < game.faces {
                    Some((q, face + 1))
                } else {
                    Some((q + 1, 1))
                }
            }
            _ => None,
        }
    }

    fn choose(&self, game: &LiarsDice, s: &LdState, player: usize, r: f64) -> Action {
        let actions = game.legal_actions(s);
        let (q, face) = s.current_bid();

        if q == 0 {
            // Opening: bid honestly around my strongest face, with occasional reach.
            let mut best_face = 1u8;
            let mut best_count = 0u8;
            for f in 1..=game.faces {
                let c = s.my_count(player, f);
                if c >= best_count {
                    best_count = c;
                    best_face = f;
                }
            }
            let total: u8 = s.dice_left().iter().sum();
            let my_dice = s.dice_left()[player];
            let unknown = (total - my_dice) as f64;
            let expected_extra = unknown / game.faces as f64;
            let mut q0 = (best_count as f64 + expected_extra * self.cfg.open_frac).round() as u8;
            q0 = q0.clamp(1, total);
            if r < self.cfg.bluff && q0 < total {
                q0 += 1; // a light bluff
            }
            return Action::Open(q0, best_face);
        }

        let p_true = self.p_true(game, s, player, q, face, self.cfg.bidder_bias);
        let p_exact = self.p_exact(game, s, player, q, face);
        let can_exact = actions.contains(&Action::CallExact);
        let can_liar = actions.contains(&Action::CallLiar);

        // Strong exact read takes precedence (it risks nothing when right).
        if can_exact && p_exact > self.cfg.exact_cut {
            return Action::CallExact;
        }
        // Bid looks like a lie: call it, with a soft randomized band so the
        // calling threshold isn't perfectly readable.
        if can_liar {
            let call_p = if p_true < self.cfg.liar_cut {
                1.0
            } else if self.cfg.mix > 0.0 && p_true < self.cfg.liar_cut + self.cfg.mix {
                (self.cfg.liar_cut + self.cfg.mix - p_true) / self.cfg.mix
            } else {
                0.0
            };
            if r < call_p {
                return Action::CallLiar;
            }
        }

        // Otherwise raise to the most plausible reachable bid.
        let mut best: Option<(Action, f64)> = None;
        for &a in &actions {
            if let Some((nq, nf)) = self.raised_bid(game, q, face, a) {
                let pt = self.p_true(game, s, player, nq, nf, 0.0);
                if best.is_none_or(|(_, b)| pt > b) {
                    best = Some((a, pt));
                }
            }
        }
        match best {
            Some((a, pt)) if pt >= self.cfg.safety || r < self.cfg.bluff => a,
            // No safe raise and not bluffing: prefer to call the current bid.
            _ if can_liar => Action::CallLiar,
            Some((a, _)) => a,
            None => Action::CallLiar,
        }
    }
}

impl Agent<LiarsDice> for ProbabilisticAgent {
    fn act(&self, game: &LiarsDice, state: &LdState, player: usize, r: f64) -> usize {
        let desired = self.choose(game, state, player, r);
        let actions = game.legal_actions(state);
        actions.iter().position(|&a| a == desired).unwrap_or(0)
    }
}

/// A baseline that bids/looks-up at random among legal actions (for evaluation).
pub struct RandomAgent;
impl Agent<LiarsDice> for RandomAgent {
    fn act(&self, game: &LiarsDice, state: &LdState, _player: usize, r: f64) -> usize {
        let n = game.legal_actions(state).len();
        ((r * n as f64) as usize).min(n - 1)
    }
}
