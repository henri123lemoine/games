//! Tabular fitted value iteration over the dice lattice for 2-player Liar's
//! Dice, and the value-level proof that the per-round decomposition is exact.
//!
//! Every round re-rolls all hands, so the only thing that survives a round is
//! the post-round dice vector and who opens next. A *scalar* continuation value
//! per such state is therefore lossless once it has converged. [`LatticeValue`]
//! is that table; [`fit_two_player`] fills it by repeatedly solving each
//! one-round [`RoundSubgame`] exactly (with [`Cfr`](solvers::Cfr)) against the
//! current table, sweeping the lattice until the values stop moving — the
//! *infinite-horizon* fixed point, which is the shipped continuation table.
//!
//! ## Proving the decomposition (bias-free)
//!
//! The full multi-round game has *unbounded* depth — a correct `Call Exact`
//! loses no die and re-rolls into a fresh round — so exact full-tree CFR
//! explodes and cannot be the ground truth except at tiny dice/round budgets.
//! The clean proof is the *finite-horizon* one ([`decomposed_value_capped`]):
//! the `cap`-round game's decomposition (solved by backward induction,
//! [`fit_capped`]) is the *same game* as the `cap`-round full game, just
//! factored, so the two values must agree to solver noise with **no truncation
//! bias**. That equality (verified in `examples/decomp_verify.rs` and the
//! `decomp_gate_*` tests, gaps ≤ ~1e-5 to exactly 0) is the value-level proof:
//! solving rounds independently against a continuation value loses nothing.

use std::collections::HashMap;

use solvers::Cfr;

use crate::{ContinuationValue, DiceShareValue, MAX_PLAYERS, RoundSubgame};

/// A converged (or in-progress) table of continuation values, keyed by the
/// post-round per-seat dice vector and who opens the next round.
///
/// For the exact 2-player case the table is stored as a single scalar `v0` per
/// key and read back as `(+v0, -v0)`, which keeps it *exactly* zero-sum by
/// construction regardless of solver noise. Keys that were never set fall back
/// to [`DiceShareValue`], so the table is always a total function and partial
/// tables stay usable as bootstrapping targets.
#[derive(Clone, Default)]
pub struct LatticeValue {
    /// `(dice_vector, next_opener) -> value to seat 0`.
    v0: HashMap<([u8; MAX_PLAYERS], u8), f64>,
}

impl LatticeValue {
    pub fn new() -> Self {
        Self::default()
    }

    fn key(dice_left: &[u8], next_opener: usize) -> ([u8; MAX_PLAYERS], u8) {
        let mut v = [0u8; MAX_PLAYERS];
        v[..dice_left.len()].copy_from_slice(dice_left);
        (v, next_opener as u8)
    }

    /// Store the seat-0 value for the 2-player state `(dice_left, next_opener)`.
    /// Seat 1's value is implied as `-v0`, so the table is always zero-sum.
    pub fn set_two_player(&mut self, dice_left: &[u8], next_opener: usize, value0: f64) {
        self.v0.insert(Self::key(dice_left, next_opener), value0);
    }

    /// The stored seat-0 value for a 2-player state, if present.
    pub fn get_two_player(&self, dice_left: &[u8], next_opener: usize) -> Option<f64> {
        self.v0.get(&Self::key(dice_left, next_opener)).copied()
    }

    /// Number of stored states (for diagnostics / convergence reporting).
    pub fn len(&self) -> usize {
        self.v0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.v0.is_empty()
    }
}

impl ContinuationValue for LatticeValue {
    fn value(&self, faces: u8, dice_left: &[u8], next_opener: usize, player: usize) -> f64 {
        // Stored scalars are 2-player only; everything else (and any unset key)
        // defers to the zero-sum heuristic, so the table is a total function.
        if dice_left.len() == 2
            && let Some(v0) = self.get_two_player(dice_left, next_opener)
        {
            return if player == 0 { v0 } else { -v0 };
        }
        DiceShareValue.value(faces, dice_left, next_opener, player)
    }
}

/// One 2-player lattice state to solve: the post-round dice vector `(a, b)` and
/// the seat opening that round.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LatticeState {
    a: u8,
    b: u8,
    opener: u8,
}

/// The round-cap adjudication outcome as a [`ContinuationValue`]: the seat with
/// the most dice wins outright (ties broken toward the *highest* seat index),
/// exactly mirroring [`LiarsDice`](crate::LiarsDice)'s `resolve_after_call`
/// when `rounds` exceeds `max_rounds`. This closes the *last* round of a
/// finite-horizon (cap-`K`) game, where it must match the full game's own cap
/// behaviour so the two values are the *same game* (see [`fit_capped`]).
#[derive(Clone)]
struct AdjudicationValue;

impl ContinuationValue for AdjudicationValue {
    fn value(&self, _faces: u8, dice_left: &[u8], _next_opener: usize, player: usize) -> f64 {
        let n = dice_left.len();
        // `max_by_key` returns the last maximum, i.e. the highest seat on ties —
        // matching `(0..players).max_by_key(|p| dice_left[p])` in the game.
        let winner = (0..n).max_by_key(|&p| dice_left[p]).unwrap();
        if player == winner {
            1.0
        } else {
            -1.0 / (n as f64 - 1.0)
        }
    }
}

/// Outcome of [`fit_two_player`]: the converged table plus convergence
/// diagnostics for the unit tests and the verification example.
pub struct FitResult {
    pub lattice: LatticeValue,
    /// Max |ΔV| observed in each completed sweep (strictly shrinking ⇒ the
    /// iteration is contracting toward a fixed point).
    pub sweep_deltas: Vec<f64>,
    /// Max within-round exact best-response exploitability over the lattice
    /// states (only filled when [`FitConfig::measure_exploitability`], else
    /// `NaN`). It is floored by the game's deliberately lossy infoset
    /// abstraction (position-relative key, dropped round counter), not driven to
    /// 0 — but that floor is shared with the full-game CFR, so it does not move
    /// the value-level gate (see the module docs).
    pub max_within_round_exploitability: f64,
    /// CFR iterations used per round-subgame solve.
    pub iters_per_solve: u64,
}

/// Tunables for the fitted value iteration. Defaults are chosen so the small
/// verification configs converge to well under `1e-6` max-delta per sweep.
#[derive(Clone, Copy)]
pub struct FitConfig {
    /// CFR iterations per round-subgame solve.
    pub iters_per_solve: u64,
    /// Stop once a full sweep moves every value by less than this.
    pub tol: f64,
    /// Hard cap on sweeps (a safety bound; convergence is the real stop).
    pub max_sweeps: usize,
    /// After convergence, also measure each round's exact exploitability (two
    /// best-response passes per state — pricier than the solves themselves, so
    /// off by default; the verification example turns it on).
    pub measure_exploitability: bool,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            iters_per_solve: 4000,
            tol: 1e-6,
            max_sweeps: 200,
            measure_exploitability: false,
        }
    }
}

/// Enumerate the reachable continuing-round states for a 2-player `dice`-die
/// game: every `(a, b)` with `1 <= a, b <= dice`, each opened by either live
/// seat. Ordered by increasing total `a + b` so a Gauss-Seidel-ish sweep visits
/// the few-dice endgame (which the rest depends on) first.
fn lattice_states(dice: u8) -> Vec<LatticeState> {
    let mut states = Vec::new();
    for a in 1..=dice {
        for b in 1..=dice {
            for opener in 0..2u8 {
                states.push(LatticeState { a, b, opener });
            }
        }
    }
    states.sort_by_key(|s| u16::from(s.a) + u16::from(s.b));
    states
}

/// Build the one-round subgame for a continuing lattice state closed by `value`.
fn round_subgame<V: ContinuationValue>(
    dice: u8,
    faces: u8,
    state: LatticeState,
    value: V,
) -> RoundSubgame<V> {
    let mut dice_left = [0u8; MAX_PLAYERS];
    dice_left[0] = state.a;
    dice_left[1] = state.b;
    RoundSubgame::new(
        2,
        dice,
        faces,
        dice_left,
        state.opener,
        false, // continuing round: free open
        1,     // start_round: single round, never reaches the cap
        value,
    )
}

/// Solve one continuing round exactly with CFR against the continuation `value`,
/// returning the round value to seat 0. Exploitability is *not* computed here —
/// it costs more than the solve itself (two best-response passes) and is only a
/// diagnostic, gathered separately by [`round_exploitabilities`].
fn solve_round_value<V: ContinuationValue>(
    dice: u8,
    faces: u8,
    state: LatticeState,
    value: V,
    iters: u64,
) -> f64 {
    let mut cfr = Cfr::new(round_subgame(dice, faces, state, value));
    cfr.solve(iters);
    cfr.expected_value()
}

/// The within-round exact best-response exploitability of every lattice state
/// solved against the (converged) `lattice`, returning the max. ~0 confirms each
/// per-round subgame is itself solved to equilibrium — the other half of "the
/// decomposition is exact" (each round is solved well; the fixed point ties them
/// together). Run once after convergence, not per sweep.
pub fn round_exploitabilities(dice: u8, faces: u8, lattice: &LatticeValue, iters: u64) -> f64 {
    let mut max_expl = 0.0_f64;
    for st in lattice_states(dice) {
        let mut cfr = Cfr::new(round_subgame(dice, faces, st, lattice.clone()));
        cfr.solve(iters);
        let (_, _, nashconv) = cfr.exploitability();
        max_expl = max_expl.max(nashconv / 2.0);
    }
    max_expl
}

/// Fit the continuation-value lattice for 2-player `dice`×`faces` Liar's Dice by
/// repeated exact per-round CFR solves (fitted value iteration).
///
/// Each sweep snapshots the current table (Jacobi update), solves every lattice
/// state against that snapshot, and writes the fresh seat-0 values back. A
/// correct `CallExact` re-rolls the *same* dice vector with the caller opening,
/// so equal-total states reference each other and a single sweep is not enough;
/// sweeps repeat until the largest value change drops below `cfg.tol`.
pub fn fit_two_player(dice: u8, faces: u8, cfg: FitConfig) -> FitResult {
    let states = lattice_states(dice);
    let mut lattice = LatticeValue::new();
    let mut sweep_deltas = Vec::new();

    for _ in 0..cfg.max_sweeps {
        let snapshot = lattice.clone();
        let mut max_delta = 0.0_f64;
        let mut updated = LatticeValue::new();
        for &st in &states {
            let dice_left = [st.a, st.b];
            let v0 = solve_round_value(dice, faces, st, snapshot.clone(), cfg.iters_per_solve);
            let prev = snapshot
                .get_two_player(&dice_left, st.opener as usize)
                .unwrap_or_else(|| DiceShareValue.value(faces, &dice_left, st.opener as usize, 0));
            max_delta = max_delta.max((v0 - prev).abs());
            updated.set_two_player(&dice_left, st.opener as usize, v0);
        }
        lattice = updated;
        sweep_deltas.push(max_delta);
        if max_delta < cfg.tol {
            break;
        }
    }

    let max_within_round_exploitability = if cfg.measure_exploitability {
        round_exploitabilities(dice, faces, &lattice, cfg.iters_per_solve)
    } else {
        f64::NAN
    };
    FitResult {
        lattice,
        sweep_deltas,
        max_within_round_exploitability,
        iters_per_solve: cfg.iters_per_solve,
    }
}

/// Value of the game's *entry* round (forced `1×1` open, full dice, seat 0
/// opens) closed by the continuation `value`. This is `V_decomposed(entry)`: the
/// decomposed estimate of the full-game value to seat 0.
pub fn entry_round_value<V: ContinuationValue>(dice: u8, faces: u8, value: V, iters: u64) -> f64 {
    let mut dice_left = [0u8; MAX_PLAYERS];
    dice_left[0] = dice;
    dice_left[1] = dice;
    let subgame = RoundSubgame::new(
        2, dice, faces, dice_left, 0, true, // entry round: forced 1x1 open
        1, value,
    );
    let mut cfr = Cfr::new(subgame);
    cfr.solve(iters);
    cfr.expected_value()
}

/// Convenience: fit the lattice then return `V_decomposed(entry)` with it,
/// alongside the fit diagnostics.
pub fn decomposed_game_value(dice: u8, faces: u8, cfg: FitConfig) -> (f64, FitResult) {
    let fit = fit_two_player(dice, faces, cfg);
    let entry = entry_round_value(dice, faces, fit.lattice.clone(), cfg.iters_per_solve);
    (entry, fit)
}

/// Solve one *backward induction* step: every free-open lattice state closed by
/// `next` (the already-solved values one horizon shorter), returning the new
/// lattice for this horizon.
fn backward_step<V>(dice: u8, faces: u8, next: &V, iters: u64) -> LatticeValue
where
    V: ContinuationValue + Clone,
{
    let mut lattice = LatticeValue::new();
    for st in lattice_states(dice) {
        let v0 = solve_round_value(dice, faces, st, next.clone(), iters);
        lattice.set_two_player(&[st.a, st.b], st.opener as usize, v0);
    }
    lattice
}

/// Finite-horizon (cap-`K`) decomposition: the *exact* decomposition of the
/// `cap`-round full game, solved round-by-round by backward induction instead of
/// as one tree. Unlike the infinite-horizon [`fit_two_player`] fixed point this
/// has **no truncation bias** relative to a `cap`-round full game — it *is* that
/// game, just factored — so [`decomposed_value_capped`] must equal
/// `Cfr::new(LiarsDice::two_player(d,f).with_max_rounds(cap)).expected_value()`
/// to solver tolerance. That exact equality is the bias-free decomposition proof
/// (the infinite-horizon `cap` is intractable to enumerate as one tree).
///
/// Horizon `h` = rounds remaining including the current one. The last round
/// (`h = 1`) is closed by [`AdjudicationValue`] (the cap's dice-count winner);
/// each earlier round is closed by the next-shorter horizon's lattice. Returns
/// the lattice for horizon `cap - 1` (what the entry round, at horizon `cap`,
/// is closed by).
pub fn fit_capped(dice: u8, faces: u8, cap: u16, iters: u64) -> LatticeValue {
    assert!(cap >= 1, "a game has at least the entry round");
    // Horizon 1: the last allowed round; any continuation hits the cap.
    let mut lattice = backward_step(dice, faces, &AdjudicationValue, iters);
    // Horizons 2..=cap-1: each closed by the next-shorter horizon.
    for _ in 2..cap {
        lattice = backward_step(dice, faces, &lattice, iters);
    }
    lattice
}

/// `V_decomposed(entry)` for the finite-horizon (cap-`K`) game: the entry round
/// (forced `1×1`, full dice, seat 0 opens) at horizon `cap`, closed by the
/// backward-induction lattice for horizon `cap - 1`. Equals the cap-`K`
/// full-game CFR value (bias-free); see [`fit_capped`].
pub fn decomposed_value_capped(dice: u8, faces: u8, cap: u16, iters: u64) -> f64 {
    if cap == 1 {
        // A single-round game: the entry round closed directly by adjudication.
        return entry_round_value(dice, faces, AdjudicationValue, iters);
    }
    let lattice = fit_capped(dice, faces, cap, iters);
    entry_round_value(dice, faces, lattice, iters)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LiarsDice;

    /// CFR iters small enough to keep the test suite fast but large enough that
    /// these tiny configs reach the convergence tolerance. The value-level gaps
    /// shrink with iters; a few hundred already pins the values to ~1e-3.
    const TEST_CFG: FitConfig = FitConfig {
        iters_per_solve: 800,
        tol: 1e-5,
        max_sweeps: 100,
        measure_exploitability: false,
    };

    #[test]
    fn lattice_value_is_zero_sum() {
        let mut lat = LatticeValue::new();
        lat.set_two_player(&[2, 3], 0, 0.137);
        lat.set_two_player(&[1, 1], 1, -0.42);
        for (dl, op) in [([2u8, 3u8], 0usize), ([1, 1], 1)] {
            let v0 = lat.value(6, &dl, op, 0);
            let v1 = lat.value(6, &dl, op, 1);
            assert!((v0 + v1).abs() < 1e-12, "dl={dl:?} op={op} v0={v0} v1={v1}");
        }
        // An unset key falls back to the (also zero-sum) heuristic.
        let v0 = lat.value(6, &[2, 2], 0, 0);
        let v1 = lat.value(6, &[2, 2], 0, 1);
        assert!((v0 + v1).abs() < 1e-12);
    }

    #[test]
    fn adjudication_value_matches_game_cap_tiebreak() {
        // The game breaks dice-count ties toward the *highest* seat.
        let v = AdjudicationValue;
        assert_eq!(v.value(6, &[2, 1], 0, 0), 1.0); // seat 0 has more
        assert_eq!(v.value(6, &[2, 1], 0, 1), -1.0);
        assert_eq!(v.value(6, &[1, 1], 0, 1), 1.0); // tie -> highest seat (1)
        assert_eq!(v.value(6, &[1, 1], 0, 0), -1.0);
    }

    #[test]
    fn value_iteration_converges() {
        // The infinite-horizon fixed point: max-delta must shrink across sweeps
        // and reach the tolerance (the CallExact coupling needs several sweeps).
        let fit = fit_two_player(2, 3, TEST_CFG);
        assert!(
            fit.sweep_deltas.len() >= 2,
            "coupling via CallExact needs multiple sweeps"
        );
        let last = *fit.sweep_deltas.last().unwrap();
        assert!(
            last < TEST_CFG.tol,
            "did not converge: deltas={:?}",
            fit.sweep_deltas
        );
        // Contraction: the final delta is far below the first.
        assert!(
            last < fit.sweep_deltas[0],
            "no contraction: deltas={:?}",
            fit.sweep_deltas
        );
    }

    #[test]
    fn entry_value_in_range() {
        // 1d6 is the cheapest fixed point (2 lattice states per sweep).
        let (entry, _) = decomposed_game_value(1, 6, TEST_CFG);
        assert!(
            (-1.0..=1.0).contains(&entry),
            "entry value out of range: {entry}"
        );
    }

    /// DECOMPOSITION GATE (single round). A 1-round (`cap = 1`) game is the
    /// entry round closed directly by adjudication; the decomposition then *is*
    /// the full game, so the values must match to solver noise (two independent
    /// CFR runs of the same tree). Guarded in CI for two configs.
    #[test]
    fn decomp_gate_single_round() {
        for (dice, faces) in [(2u8, 3u8), (2, 4)] {
            let iters = 1500;
            let decomp = decomposed_value_capped(dice, faces, 1, iters);
            let mut full = Cfr::new(LiarsDice::two_player(dice, faces).with_max_rounds(1));
            full.solve(iters);
            let full_value = full.expected_value();
            let gap = (decomp - full_value).abs();
            assert!(
                gap < 5e-3,
                "2p{dice}d{faces}f cap=1 gap too large: decomp={decomp} full={full_value} gap={gap}"
            );
        }
    }

    /// DECOMPOSITION GATE (multi-round, bias-free). A 2-round (`cap = 2`) game
    /// exercises the backward-induction chaining: the entry round is closed by
    /// the horizon-1 lattice, itself solved against adjudication. This factored
    /// value must equal the `cap = 2` full game solved as one tree — *exactly*
    /// the same game, with **no truncation bias** — proving the per-round
    /// decomposition at the value level. (2p1d6f: the full `cap = 2` tree is the
    /// largest one that stays enumerable; higher dice/caps explode, which is the
    /// whole reason the decomposition exists.)
    #[test]
    fn decomp_gate_two_round_bias_free() {
        let (dice, faces, iters) = (1u8, 6u8, 700u64);
        let decomp = decomposed_value_capped(dice, faces, 2, iters);
        let mut full = Cfr::new(LiarsDice::two_player(dice, faces).with_max_rounds(2));
        full.solve(iters);
        let full_value = full.expected_value();
        let gap = (decomp - full_value).abs();
        assert!(
            gap < 5e-3,
            "2p{dice}d{faces}f cap=2 decomposition gap too large: \
             decomp={decomp} full={full_value} gap={gap}"
        );
    }
}
