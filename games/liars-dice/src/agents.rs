//! Hand-crafted agents for Liar's Dice, primarily the probabilistic belief
//! player: it reasons exactly about the unknown dice with the binomial
//! distribution (each unknown die shows a given face with probability `1/faces`,
//! since 1s are not wild), which scales to any number of players and dice.

use game_core::{Agent, Determinizer, Rng};

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
    /// League-tuned on 5p5d6f: aggressive bids, moderate exact calls, ready to
    /// call liar. After cap ties were scored as draws, independent same-game
    /// validation favored `exact_cut=0.70` over the legacy eager-exact setting.
    fn default() -> Self {
        Self {
            liar_cut: 0.238,
            exact_cut: 0.70,
            safety: 0.129,
            bluff: 0.005,
            bidder_bias: 0.412,
            open_frac: 0.346,
            mix: 0.051,
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

    /// Exact-Bayes-style caller: no deliberate bluffing, conservative exact
    /// calls, and raises only while the bid still has decent posterior support.
    pub fn honest_bayes() -> Self {
        Self {
            liar_cut: 0.30,
            exact_cut: 0.92,
            safety: 0.42,
            bluff: 0.0,
            bidder_bias: 0.0,
            open_frac: 0.0,
            mix: 0.0,
        }
    }

    /// High-pressure styled baseline for league and tournament fields.
    pub fn aggressive_bluffer() -> Self {
        Self {
            liar_cut: 0.42,
            exact_cut: 0.55,
            safety: 0.18,
            bluff: 0.22,
            bidder_bias: 0.25,
            open_frac: 0.75,
            mix: 0.12,
        }
    }

    /// Trusting, low-bluff baseline that calls liar only on stronger evidence.
    pub fn conservative_caller() -> Self {
        Self {
            liar_cut: 0.18,
            exact_cut: 0.75,
            safety: 0.58,
            bluff: 0.0,
            bidder_bias: 0.85,
            open_frac: 0.20,
            mix: 0.02,
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

    fn choose(&self, game: &LiarsDice, s: &LdState, player: usize, rng: &mut Rng) -> Action {
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
            if rng.unit() < self.cfg.bluff && q0 < total {
                q0 += 1; // a light bluff
            }
            return Action::Open(q0, best_face);
        }

        let p_true = self.p_true(game, s, player, q, face, self.cfg.bidder_bias);
        let p_exact = self.p_exact(game, s, player, q, face);

        // Strong exact read takes precedence (it risks nothing when right).
        if p_exact > self.cfg.exact_cut {
            return Action::CallExact;
        }
        // Bid looks like a lie: call it, with a soft randomized band so the
        // calling threshold isn't perfectly readable.
        let call_p = if p_true < self.cfg.liar_cut {
            1.0
        } else if self.cfg.mix > 0.0 && p_true < self.cfg.liar_cut + self.cfg.mix {
            (self.cfg.liar_cut + self.cfg.mix - p_true) / self.cfg.mix
        } else {
            0.0
        };
        // A fresh draw per stochastic decision: reusing one sample across
        // the bluff/call/raise thresholds correlates them and distorts
        // the tuned marginal probabilities.
        if rng.unit() < call_p {
            return Action::CallLiar;
        }

        // Otherwise raise to the most plausible reachable bid.
        let mut best: Option<(Action, f64)> = None;
        let total: u8 = s.dice_left().iter().sum();
        if q < total {
            let a = Action::RaiseQuantity;
            if let Some((nq, nf)) = self.raised_bid(game, q, face, a) {
                let pt = self.p_true(game, s, player, nq, nf, 0.0);
                best = Some((a, pt));
            }
        }
        if face < game.faces || q < total {
            let a = Action::RaiseFace;
            if let Some((nq, nf)) = self.raised_bid(game, q, face, a) {
                let pt = self.p_true(game, s, player, nq, nf, 0.0);
                if best.is_none_or(|(_, b)| pt > b) {
                    best = Some((a, pt));
                }
            }
        }
        match best {
            Some((a, pt)) if pt >= self.cfg.safety || rng.unit() < self.cfg.bluff => a,
            // No safe raise and not bluffing: prefer to call the current bid.
            _ => Action::CallLiar,
        }
    }

    fn action_index(&self, game: &LiarsDice, s: &LdState, a: Action) -> Option<usize> {
        let total: u8 = s.dice_left().iter().sum();
        let (q, face) = s.current_bid();
        if q == 0 {
            return match a {
                Action::Open(oq, of)
                    if (1..=total).contains(&oq) && (1..=game.faces).contains(&of) =>
                {
                    Some(usize::from(oq - 1) * usize::from(game.faces) + usize::from(of - 1))
                }
                _ => None,
            };
        }

        let mut idx = 0usize;
        if q < total {
            if a == Action::RaiseQuantity {
                return Some(idx);
            }
            idx += 1;
        }
        if face < game.faces || q < total {
            if a == Action::RaiseFace {
                return Some(idx);
            }
            idx += 1;
        }
        match a {
            Action::CallLiar => Some(idx),
            Action::CallExact => Some(idx + 1),
            _ => None,
        }
    }
}

impl Agent<LiarsDice> for ProbabilisticAgent {
    fn act(&self, game: &LiarsDice, state: &LdState, player: usize, rng: &mut Rng) -> usize {
        let desired = self.choose(game, state, player, rng);
        if let Some(i) = self.action_index(game, state, desired) {
            return i;
        }
        // `choose` should always return a legal action; if a future edit ever
        // breaks that, fall back to the first legal action (index 0 is always a
        // legal index when any action exists) rather than emitting a stale index.
        debug_assert!(
            false,
            "ProbabilisticAgent chose {desired:?}, not legal in the current state"
        );
        0
    }
}

/// Determinization knowledge for [`solvers`' rollout agent]: re-roll hidden
/// hands uniformly, crediting bidders with plausibly holding the face they bid
/// (see [`LiarsDice::resample_hidden`]). Defaults are the A/B winners: credit
/// the current bidder only.
pub struct BidConditioned {
    pub bidder_bias: f64,
    pub endorser_bias: f64,
}

impl Default for BidConditioned {
    fn default() -> Self {
        Self {
            bidder_bias: 0.6,
            endorser_bias: 0.0,
        }
    }
}

impl Determinizer<LiarsDice> for BidConditioned {
    fn determinize(&self, game: &LiarsDice, state: &mut LdState, observer: usize, rng: &mut Rng) {
        game.resample_hidden(state, observer, rng, self.bidder_bias, self.endorser_bias);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_core::Game;

    /// Reference `P(Binomial(n, p) == k)` by direct evaluation of C(n,k).
    fn pmf_ref(n: u32, p: f64, k: i64) -> f64 {
        if k < 0 || k as u32 > n {
            return 0.0;
        }
        let k = k as u32;
        let mut c = 1.0;
        for i in 0..k {
            c *= (n - i) as f64 / (i + 1) as f64;
        }
        c * p.powi(k as i32) * (1.0 - p).powi((n - k) as i32)
    }

    fn sf_ref(n: u32, p: f64, k: i64) -> f64 {
        (k.max(0)..=n as i64).map(|i| pmf_ref(n, p, i)).sum()
    }

    #[test]
    fn binomials_match_direct_enumeration() {
        for n in 0..=12u32 {
            for &p in &[1.0 / 6.0, 1.0 / 3.0, 0.5, 0.9] {
                for k in -2..=(n as i64 + 2) {
                    let pmf = binom_pmf(n, p, k);
                    let sf = binom_sf(n, p, k);
                    assert!(
                        (pmf - pmf_ref(n, p, k)).abs() < 1e-12,
                        "pmf(n={n}, p={p}, k={k}) = {pmf}, want {}",
                        pmf_ref(n, p, k)
                    );
                    assert!(
                        (sf - sf_ref(n, p, k)).abs() < 1e-12,
                        "sf(n={n}, p={p}, k={k}) = {sf}, want {}",
                        sf_ref(n, p, k)
                    );
                    assert!(pmf.is_finite() && sf.is_finite());
                }
            }
        }
    }

    #[test]
    fn binomial_edge_cases() {
        assert_eq!(binom_sf(0, 1.0 / 6.0, 0), 1.0);
        assert_eq!(binom_sf(0, 1.0 / 6.0, 1), 0.0);
        assert_eq!(binom_pmf(0, 1.0 / 6.0, 0), 1.0);
        assert_eq!(binom_sf(10, 1.0 / 6.0, -3), 1.0);
        assert_eq!(binom_pmf(10, 1.0 / 6.0, 11), 0.0);
    }

    #[test]
    fn default_exact_threshold_matches_validated_rollout_base() {
        assert!(
            (ProbConfig::default().exact_cut - 0.70).abs() < 1e-12,
            "the deployed rollout base policy should stay on the validated moderate exact threshold"
        );
    }

    #[test]
    fn probabilistic_agent_returns_legal_indices() {
        let agent = ProbabilisticAgent::default_agent();
        let mut rng = Rng::new(0xA91CE);
        for &(players, dice, faces) in &[(2, 1, 6), (3, 3, 4), (5, 5, 6)] {
            let game = LiarsDice::new(players, dice, faces);
            for _ in 0..20 {
                let mut s = game.initial_state();
                let mut steps = 0;
                while !game.is_terminal(&s) {
                    steps += 1;
                    assert!(steps < 100_000, "probabilistic games should terminate");
                    match game.turn(&s) {
                        game_core::Turn::Chance => {
                            let a = game.sample_chance_action(&s, &mut rng);
                            game.apply(&mut s, a);
                        }
                        game_core::Turn::Player(p) => {
                            let actions = game.legal_actions(&s);
                            for (expected, &action) in actions.iter().enumerate() {
                                assert_eq!(
                                    agent.action_index(&game, &s, action),
                                    Some(expected),
                                    "direct action index must match legal_actions order"
                                );
                            }
                            let i = agent.act(&game, &s, p, &mut rng);
                            assert!(
                                i < actions.len(),
                                "agent returned index {i} for {} legal actions",
                                actions.len()
                            );
                            game.apply(&mut s, actions[i]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn opening_action_indices_scale_to_large_legal_configs() {
        let agent = ProbabilisticAgent::default_agent();
        let game = LiarsDice::new(8, 6, 6);
        let mut s = game.initial_state();
        s.qty = 0;
        s.face = 0;
        let actions = game.legal_actions(&s);
        assert_eq!(actions.len(), 8 * 6 * 6);
        for (expected, &action) in actions.iter().enumerate() {
            assert_eq!(
                agent.action_index(&game, &s, action),
                Some(expected),
                "large opening action indices must not overflow"
            );
        }
    }

    #[test]
    #[should_panic(expected = "faces must be at least 2")]
    fn one_faced_dice_are_rejected() {
        LiarsDice::new(2, 5, 1);
    }
}
