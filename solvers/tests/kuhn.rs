//! Kuhn poker is the canonical CFR correctness test: a tiny imperfect-information
//! game with a known Nash equilibrium where exploitability drives to zero. If the
//! generic solver converges here, the algorithm is implemented correctly.

use solvers::Cfr;

mod common;
use common::Kuhn;

#[test]
fn kuhn_converges_to_nash() {
    let mut solver = Cfr::new(Kuhn, 1);
    solver.solve(50_000);
    assert_eq!(solver.num_infosets(), 12, "Kuhn has 12 information sets");

    // CFR+ drives exact best-response exploitability to ~0 (Nash).
    let (br0, br1, nashconv) = solver.exploitability();
    assert!(
        nashconv < 0.01,
        "Kuhn should converge to Nash: br0={br0} br1={br1} nashconv={nashconv}"
    );

    // And it converges to the known game value to player 0, -1/18.
    assert!(
        (solver.expected_value() - (-1.0 / 18.0)).abs() < 0.01,
        "value to P0 should be -1/18, got {}",
        solver.expected_value()
    );
}
