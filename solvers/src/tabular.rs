//! Helpers shared by the tabular regret-minimization family
//! ([`crate::Cfr`], [`crate::Mccfr`], [`crate::os_mccfr::OsMccfr`]).

/// Index of the maximum element (first on ties).
pub(crate) fn argmax(v: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

/// Regret-matching strategy from a regret vector (CFR+: only positive
/// regrets), uniform when no action has positive regret.
pub(crate) fn regret_match(regret: &[f64]) -> Vec<f64> {
    let sum: f64 = regret.iter().map(|r| r.max(0.0)).sum();
    if sum > 0.0 {
        regret.iter().map(|r| r.max(0.0) / sum).collect()
    } else {
        uniform(regret.len())
    }
}

/// `sums` normalized to a distribution, or uniform over `n` when absent or
/// all-zero — the average-strategy read common to every tabular solver.
pub(crate) fn normalized_or_uniform(sums: Option<&Vec<f64>>, n: usize) -> Vec<f64> {
    match sums {
        Some(s) => {
            let sum: f64 = s.iter().sum();
            if sum > 0.0 {
                s.iter().map(|x| x / sum).collect()
            } else {
                uniform(n)
            }
        }
        None => uniform(n),
    }
}

pub(crate) fn uniform(n: usize) -> Vec<f64> {
    vec![1.0 / n as f64; n]
}
