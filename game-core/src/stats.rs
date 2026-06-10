//! Statistics for bot comparison: Elo with confidence intervals, the
//! generalized SPRT used by engine-testing frameworks (fishtest-style), and a
//! plain binomial SPRT for hero-vs-field win-probability tests.
//!
//! Everything here is pure math over match counts — no games, no agents — so
//! any harness (the lab CLI, an experiment binary) can drive it.

/// Expected score of a player whose Elo advantage is `elo` (logistic model).
pub fn score_from_elo(elo: f64) -> f64 {
    1.0 / (1.0 + 10f64.powf(-elo / 400.0))
}

/// Elo advantage that produces expected score `score`; the inverse of
/// [`score_from_elo`]. Scores are clamped away from 0/1 so the result stays
/// finite (±2400 at the extremes).
pub fn elo_from_score(score: f64) -> f64 {
    let s = score.clamp(1e-6, 1.0 - 1e-6);
    -400.0 * (1.0 / s - 1.0).log10()
}

/// Wilson score interval for `successes` out of `n` Bernoulli trials at
/// normal quantile `z` (1.96 for 95%). Fractional successes are allowed so a
/// draw can count as half a win.
pub fn wilson(successes: f64, n: f64, z: f64) -> (f64, f64) {
    if n <= 0.0 {
        return (0.0, 1.0);
    }
    let p = (successes / n).clamp(0.0, 1.0);
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denom;
    let half = z / denom * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt();
    ((center - half).max(0.0), (center + half).min(1.0))
}

/// An Elo point estimate with a 95% confidence interval, both obtained by
/// mapping the score fraction (and its Wilson interval) through the logistic.
#[derive(Debug, Clone, Copy)]
pub struct EloEstimate {
    pub elo: f64,
    pub lo: f64,
    pub hi: f64,
}

impl EloEstimate {
    /// Half-width of the confidence interval — the `+/-` to print.
    pub fn margin(&self) -> f64 {
        (self.hi - self.lo) / 2.0
    }
}

/// Elo estimate (with 95% CI) from a W-D-L record, draws scoring half.
pub fn elo_estimate(wins: u64, draws: u64, losses: u64) -> EloEstimate {
    let n = (wins + draws + losses) as f64;
    if n == 0.0 {
        return EloEstimate {
            elo: 0.0,
            lo: elo_from_score(0.0),
            hi: elo_from_score(1.0),
        };
    }
    let points = wins as f64 + draws as f64 / 2.0;
    let (lo, hi) = wilson(points, n, 1.96);
    EloEstimate {
        elo: elo_from_score(points / n),
        lo: elo_from_score(lo),
        hi: elo_from_score(hi),
    }
}

/// Outcome of a sequential test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Keep playing — neither bound crossed.
    Open,
    /// Evidence favors H1 (the LLR crossed the upper bound).
    AcceptH1,
    /// Evidence favors H0 (the LLR crossed the lower bound).
    RejectH1,
}

fn sprt_bounds(alpha: f64, beta: f64) -> (f64, f64) {
    ((beta / (1.0 - alpha)).ln(), ((1.0 - beta) / alpha).ln())
}

fn verdict_for(llr: f64, lower: f64, upper: f64) -> Verdict {
    if llr >= upper {
        Verdict::AcceptH1
    } else if llr <= lower {
        Verdict::RejectH1
    } else {
        Verdict::Open
    }
}

/// Generalized SPRT log-likelihood ratio for a W-D-L record, testing
/// H0: elo = `elo0` against H1: elo = `elo1` (fishtest's approximation:
/// per-game scores treated as i.i.d. with their empirical mean and variance).
/// Each cell is regularized by +0.5 so degenerate records stay finite.
pub fn gsprt_llr(wins: u64, draws: u64, losses: u64, elo0: f64, elo1: f64) -> f64 {
    let w = wins as f64 + 0.5;
    let d = draws as f64 + 0.5;
    let l = losses as f64 + 0.5;
    let n = w + d + l;
    let mean = (w + 0.5 * d) / n;
    let mean_sq = (w + 0.25 * d) / n;
    let var = (mean_sq - mean * mean).max(1e-12);
    let s0 = score_from_elo(elo0);
    let s1 = score_from_elo(elo1);
    n * (s1 - s0) * (2.0 * mean - s0 - s1) / (2.0 * var)
}

/// Sequential probability ratio test on a trinomial (win/draw/loss) record,
/// testing H0: elo = `elo0` vs H1: elo = `elo1` via [`gsprt_llr`], with
/// stopping bounds `log(beta/(1-alpha))` and `log((1-beta)/alpha)`.
pub struct Sprt {
    elo0: f64,
    elo1: f64,
    lower: f64,
    upper: f64,
    wins: u64,
    draws: u64,
    losses: u64,
}

impl Sprt {
    pub fn new(elo0: f64, elo1: f64, alpha: f64, beta: f64) -> Self {
        let (lower, upper) = sprt_bounds(alpha, beta);
        Self {
            elo0,
            elo1,
            lower,
            upper,
            wins: 0,
            draws: 0,
            losses: 0,
        }
    }

    pub fn update(&mut self, wins: u64, draws: u64, losses: u64) {
        self.wins += wins;
        self.draws += draws;
        self.losses += losses;
    }

    pub fn counts(&self) -> (u64, u64, u64) {
        (self.wins, self.draws, self.losses)
    }

    pub fn games(&self) -> u64 {
        self.wins + self.draws + self.losses
    }

    /// The stopping bounds `(lower, upper)`.
    pub fn bounds(&self) -> (f64, f64) {
        (self.lower, self.upper)
    }

    pub fn llr(&self) -> f64 {
        gsprt_llr(self.wins, self.draws, self.losses, self.elo0, self.elo1)
    }

    pub fn verdict(&self) -> Verdict {
        verdict_for(self.llr(), self.lower, self.upper)
    }
}

/// Plain binomial SPRT testing H0: p = `p0` vs H1: p = `p1` on win/loss
/// counts — for hero-vs-field comparisons where "win the game" is the event
/// (fair share is `1/players`).
pub struct BinomialSprt {
    win_term: f64,
    loss_term: f64,
    lower: f64,
    upper: f64,
    wins: u64,
    losses: u64,
}

impl BinomialSprt {
    pub fn new(p0: f64, p1: f64, alpha: f64, beta: f64) -> Self {
        assert!(0.0 < p0 && p0 < 1.0 && 0.0 < p1 && p1 < 1.0 && p0 != p1);
        let (lower, upper) = sprt_bounds(alpha, beta);
        Self {
            win_term: (p1 / p0).ln(),
            loss_term: ((1.0 - p1) / (1.0 - p0)).ln(),
            lower,
            upper,
            wins: 0,
            losses: 0,
        }
    }

    pub fn update(&mut self, wins: u64, losses: u64) {
        self.wins += wins;
        self.losses += losses;
    }

    pub fn counts(&self) -> (u64, u64) {
        (self.wins, self.losses)
    }

    pub fn games(&self) -> u64 {
        self.wins + self.losses
    }

    pub fn bounds(&self) -> (f64, f64) {
        (self.lower, self.upper)
    }

    pub fn llr(&self) -> f64 {
        self.wins as f64 * self.win_term + self.losses as f64 * self.loss_term
    }

    pub fn verdict(&self) -> Verdict {
        verdict_for(self.llr(), self.lower, self.upper)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elo_at_three_quarters_score_is_191() {
        let elo = elo_from_score(0.75);
        assert!((elo - 190.85).abs() < 0.1, "got {elo}");
        assert!((elo.round() - 191.0).abs() < f64::EPSILON);
        assert!((score_from_elo(elo) - 0.75).abs() < 1e-12);
    }

    #[test]
    fn elo_is_symmetric_and_clamped() {
        assert!((elo_from_score(0.5)).abs() < 1e-12);
        assert!((elo_from_score(0.25) + elo_from_score(0.75)).abs() < 1e-9);
        assert!(elo_from_score(1.0).is_finite());
        assert!(elo_from_score(0.0).is_finite());
    }

    #[test]
    fn wilson_matches_known_values() {
        let (lo, hi) = wilson(75.0, 100.0, 1.96);
        assert!((lo - 0.65696).abs() < 1e-3, "lo {lo}");
        assert!((hi - 0.82455).abs() < 1e-3, "hi {hi}");
        let (lo, hi) = wilson(0.0, 10.0, 1.96);
        assert!(lo.abs() < 1e-12);
        assert!(hi > 0.0 && hi < 0.35);
    }

    #[test]
    fn elo_estimate_brackets_the_point() {
        let e = elo_estimate(75, 0, 25);
        assert!(e.lo < e.elo && e.elo < e.hi);
        assert!((e.elo - 190.85).abs() < 0.1);
        assert!(e.margin() > 0.0);
    }

    #[test]
    fn gsprt_known_value() {
        let llr = gsprt_llr(200, 0, 100, 0.0, 10.0);
        assert!((llr - 3.098).abs() < 0.01, "got {llr}");
    }

    #[test]
    fn sprt_accepts_heavy_one_sided_results() {
        let mut s = Sprt::new(0.0, 10.0, 0.05, 0.05);
        assert_eq!(s.verdict(), Verdict::Open);
        s.update(200, 0, 100);
        assert!(s.llr() >= s.bounds().1);
        assert_eq!(s.verdict(), Verdict::AcceptH1);
    }

    #[test]
    fn sprt_balanced_results_drift_to_reject() {
        let mut s = Sprt::new(0.0, 10.0, 0.05, 0.05);
        s.update(500, 0, 500);
        assert!(s.llr() < 0.0, "balanced play must drift negative");
        assert_eq!(s.verdict(), Verdict::Open);
        s.update(4500, 0, 4500);
        assert!((s.llr() + 4.14).abs() < 0.02, "got {}", s.llr());
        assert_eq!(s.verdict(), Verdict::RejectH1);
    }

    #[test]
    fn sprt_bounds_are_wald() {
        let s = Sprt::new(0.0, 10.0, 0.05, 0.05);
        let (lo, hi) = s.bounds();
        assert!((hi - 19f64.ln()).abs() < 1e-12);
        assert!((lo + 19f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn binomial_sprt_known_increments() {
        let mut s = BinomialSprt::new(0.2, 0.25, 0.05, 0.05);
        s.update(1, 0);
        assert!((s.llr() - 1.25f64.ln()).abs() < 1e-12);
        s.update(0, 1);
        assert!((s.llr() - (1.25f64.ln() + 0.9375f64.ln())).abs() < 1e-12);
    }

    #[test]
    fn binomial_sprt_decides_both_ways() {
        let mut acc = BinomialSprt::new(0.2, 0.25, 0.05, 0.05);
        acc.update(100, 100);
        assert_eq!(acc.verdict(), Verdict::AcceptH1);
        let mut rej = BinomialSprt::new(0.2, 0.25, 0.05, 0.05);
        rej.update(0, 300);
        assert_eq!(rej.verdict(), Verdict::RejectH1);
    }
}
