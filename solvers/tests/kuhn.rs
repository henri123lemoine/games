//! Kuhn poker is the canonical CFR correctness test: a tiny imperfect-information
//! game with a known Nash equilibrium where exploitability drives to zero. If the
//! generic solver converges here, the algorithm is implemented correctly.

use game_core::Game;
use solvers::{Cfr, DeepCfr, DeepCfrConfig, Encoder, nash_conv};

mod common;
use common::{Kuhn, KuhnState};

#[test]
fn kuhn_converges_to_nash() {
    let mut solver = Cfr::new(Kuhn);
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

/// Tiny config-agnostic Kuhn encoder for the generic Deep CFR engine: card
/// one-hot, history-length one-hot, the two betting flags, and a side-to-move
/// bit. Distinct infosets map to distinct features, so a net can represent the
/// equilibrium exactly. The action space is the two betting moves.
struct KuhnEnc;

impl Encoder<Kuhn> for KuhnEnc {
    fn feature_len(&self) -> usize {
        3 + 4 + 2 + 1
    }
    fn policy_len(&self) -> usize {
        2
    }
    fn features(&self, _g: &Kuhn, s: &KuhnState, player: usize) -> Vec<f32> {
        let mut x = vec![0.0f32; self.feature_len()];
        if s.cards[player] >= 0 {
            x[s.cards[player] as usize] = 1.0;
        }
        x[3 + s.history.len().min(3)] = 1.0;
        for (i, &a) in s.history.iter().take(2).enumerate() {
            if a == 1 {
                x[7 + i] = 1.0;
            }
        }
        x[9] = (s.history.len() % 2) as f32;
        x
    }
    fn support(&self, _g: &Kuhn, _s: &KuhnState) -> Vec<usize> {
        vec![0, 1]
    }
}

/// Deep CFR correctness on Kuhn: the average-strategy net's exact exploitability
/// must drive toward the Nash value (~0) over CFR iterations. The curve is
/// printed (the correctness proof); the gate asserts the final value is small.
/// `--ignored` because the checkpointed run is slow; the fast guarded gate is
/// the `solvers::deepcfr` unit test `kuhn_deep_cfr_reaches_low_exploitability`.
#[test]
#[ignore = "slow convergence-curve demonstration; run with --ignored"]
fn kuhn_deep_cfr_exploitability_curve() {
    let game = Kuhn;
    let enc = KuhnEnc;
    let cfg = DeepCfrConfig {
        iters: 300,
        traversals: 24,
        train_every: 1,
        hidden: 64,
        adv_reservoir: 400_000,
        strat_reservoir: 1_000_000,
        adv_steps: 300,
        strat_steps: 6000,
        batch: 512,
        lr: 0.02,
        momentum: 0.9,
        l2: 1e-5,
        seed: 0xC0FFEE,
        adv_nets: 0,
        collect_root_value: false,
        // Exploration on; small ε so the limited-budget curve still converges
        // (the IW keeps it unbiased — see the unit-test note).
        explore_eps: 0.1,
    };
    let mut solver = DeepCfr::new(game.num_players(), &enc, cfg);
    println!("{:>6} {:>14}", "iters", "exploitability");
    let mut last = f64::INFINITY;
    for &to in &[20usize, 40, 60, 100, 150, 200, 300] {
        let net = solver.run_through(to, &game, &enc);
        let cache = net.infer_cache();
        let policy = |_g: &Kuhn, s: &KuhnState, player: usize| {
            let x = enc.features(&game, s, player);
            let (probs, _) = net.policy_value_cached(&cache, &x, &[0, 1]);
            probs.iter().map(|&p| f64::from(p)).collect::<Vec<f64>>()
        };
        last = nash_conv(&game, &policy).2 / 2.0;
        println!("{to:>6} {last:>14.4}");
    }
    assert!(
        last < 0.05,
        "Deep CFR must reach near-Nash on Kuhn: final exploitability={last}"
    );
}
