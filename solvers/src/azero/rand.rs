//! Reproducible sampling helpers on top of `game_core::Rng`.

use game_core::Rng;

pub(crate) fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

pub(crate) fn mix(a: u64, b: u64) -> u64 {
    splitmix64(a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// Standard normal via Box–Muller.
pub(crate) fn normal(rng: &mut Rng) -> f64 {
    let u1 = rng.unit().max(1e-12);
    let u2 = rng.unit();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Gamma(shape, 1) via Marsaglia–Tsang, with the boost for shape < 1.
pub(crate) fn gamma(shape: f64, rng: &mut Rng) -> f64 {
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
pub(crate) fn dirichlet(alpha: f64, k: usize, rng: &mut Rng) -> Vec<f64> {
    let mut v: Vec<f64> = (0..k).map(|_| gamma(alpha, rng)).collect();
    let total: f64 = v.iter().sum::<f64>().max(1e-12);
    for x in &mut v {
        *x /= total;
    }
    v
}
