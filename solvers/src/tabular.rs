//! Scaffolding shared by the tabular regret-minimization family
//! ([`crate::Cfr`], [`crate::Mccfr`], [`crate::os_mccfr::OsMccfr`]): the
//! regret/strategy tables with their reads and the regret-matching math.
//! The traversal logic — where the variants actually differ — stays with
//! each solver.

use crate::FastMap;

/// Cumulative regrets and cumulative strategy weights, keyed by infoset.
#[derive(Default)]
pub(crate) struct Tabular {
    pub regret: FastMap<u64, Vec<f64>>,
    pub strategy: FastMap<u64, Vec<f64>>,
}

impl Tabular {
    pub fn num_infosets(&self) -> usize {
        self.strategy.len()
    }

    /// Regret-matched current strategy at `key`, creating the regret vector
    /// on first visit.
    pub fn sigma(&mut self, key: u64, n: usize) -> Vec<f64> {
        let r = self.regret.entry(key).or_insert_with(|| vec![0.0; n]);
        debug_assert_eq!(
            r.len(),
            n,
            "action count changed for infoset {key:#x} — legal_actions must be \
             stable per information set"
        );
        regret_match(r)
    }

    /// Accumulates `w·sigma` into the average strategy at `key`.
    pub fn accumulate(&mut self, key: u64, sigma: &[f64], w: f64) {
        let s = self
            .strategy
            .entry(key)
            .or_insert_with(|| vec![0.0; sigma.len()]);
        for (si, p) in s.iter_mut().zip(sigma) {
            *si += w * p;
        }
    }

    /// Average-strategy distribution at `key` (uniform when unvisited).
    pub fn average(&self, key: u64, n: usize) -> Vec<f64> {
        normalized_or_uniform(self.strategy.get(&key), n)
    }
}

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
