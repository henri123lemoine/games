//! Reproducible distribution sampling over [`Rng`](crate::Rng), shared by
//! the self-play and search crates (each used to carry its own copy, and the
//! copies drifted).

use crate::Rng;

/// Samples an index proportionally to non-negative weights. Normalizes by the
/// total, so weights need not sum to 1; floating-point shortfall lands on the
/// last index. The one weighted sampler behind [`Rng::pick`],
/// [`sample_outcome`], and [`sample_visits`].
pub fn pick_weighted<I>(weights: I, rng: &mut Rng) -> usize
where
    I: Iterator<Item = f64> + Clone,
{
    let total: f64 = weights.clone().sum();
    let mut target = rng.unit() * total;
    let mut last = 0;
    for (i, w) in weights.enumerate() {
        target -= w;
        if target < 0.0 {
            return i;
        }
        last = i;
    }
    last
}

/// Samples an index from `(outcome, probability)` pairs — the shape of
/// [`Game::chance_outcomes`](crate::Game::chance_outcomes).
pub fn sample_outcome<A>(outs: &[(A, f64)], rng: &mut Rng) -> usize {
    debug_assert!(!outs.is_empty());
    pick_weighted(outs.iter().map(|(_, p)| *p), rng)
}

/// Samples an index proportionally to visit counts (all-zero counts fall back
/// to the last index).
pub fn sample_visits(visits: &[u32], rng: &mut Rng) -> usize {
    debug_assert!(!visits.is_empty());
    pick_weighted(visits.iter().map(|&n| f64::from(n)), rng)
}

/// Advances `state` through any chance nodes by sampling outcomes.
pub fn step_chance<G: crate::Game>(game: &G, state: &mut G::State, rng: &mut Rng) {
    while !game.is_terminal(state) && matches!(game.turn(state), crate::Turn::Chance) {
        let outs = game.chance_outcomes(state);
        let i = sample_outcome(&outs, rng);
        game.apply(state, outs[i].0);
    }
}

/// Standard normal via Box-Muller.
pub fn normal(rng: &mut Rng) -> f64 {
    let u1 = rng.unit().max(1e-12);
    let u2 = rng.unit();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Gamma(shape, 1) via Marsaglia-Tsang, with the boost for shape < 1.
pub fn gamma(shape: f64, rng: &mut Rng) -> f64 {
    if shape < 1.0 {
        return gamma(shape + 1.0, rng) * rng.unit().max(1e-12).powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = normal(rng);
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u = rng.unit();
        if u < 1.0 - 0.0331 * x.powi(4)
            || u.max(f64::MIN_POSITIVE).ln() < 0.5 * x * x + d * (1.0 - v + v.ln())
        {
            return d * v;
        }
    }
}

/// One sample from a symmetric Dirichlet(α, …, α) of dimension `k`.
pub fn dirichlet(alpha: f64, k: usize, rng: &mut Rng) -> Vec<f64> {
    let mut v: Vec<f64> = (0..k).map(|_| gamma(alpha, rng)).collect();
    let total: f64 = v.iter().sum::<f64>().max(1e-12);
    for x in &mut v {
        *x /= total;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dirichlet_sums_to_one_and_stays_positive() {
        let mut rng = Rng::new(7);
        for &alpha in &[0.3, 1.0, 5.0] {
            let v = dirichlet(alpha, 8, &mut rng);
            assert!((v.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            assert!(v.iter().all(|&x| x >= 0.0));
        }
    }
}
