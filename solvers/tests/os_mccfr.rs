//! Outcome-sampling MCCFR correctness on Kuhn poker (convergence toward Nash,
//! checked with an exact pure-strategy best response) and the property that
//! motivates it: per-iteration cost linear in trajectory length, so deep
//! action ladders — where external sampling's traversal is exponential — train
//! in milliseconds.

use game_core::{Game, Turn};
use solvers::os_mccfr::OsMccfr;

mod common;
use common::{Kuhn, KuhnState};

/// Expected value to player 0 when both players play the solver's average
/// strategy, chance integrated exactly.
fn self_play_value(g: &Kuhn, solver: &OsMccfr<Kuhn>, s: &KuhnState) -> f64 {
    if g.is_terminal(s) {
        return g.returns(s, 0);
    }
    match g.turn(s) {
        Turn::Chance => g
            .chance_outcomes(s)
            .iter()
            .map(|&(a, p)| {
                let mut c = s.clone();
                g.apply(&mut c, a);
                p * self_play_value(g, solver, &c)
            })
            .sum(),
        Turn::Player(pl) => {
            let sigma = solver.policy(s, pl);
            g.legal_actions(s)
                .iter()
                .enumerate()
                .map(|(i, &a)| {
                    let mut c = s.clone();
                    g.apply(&mut c, a);
                    sigma[i] * self_play_value(g, solver, &c)
                })
                .sum()
        }
    }
}

fn collect_infosets(g: &Kuhn, s: &KuhnState, br: usize, keys: &mut Vec<u64>) {
    if g.is_terminal(s) {
        return;
    }
    if let Turn::Player(pl) = g.turn(s)
        && pl == br
    {
        let k = g.infoset_key(s, br);
        if !keys.contains(&k) {
            keys.push(k);
        }
    }
    let actions: Vec<u8> = match g.turn(s) {
        Turn::Chance => g.chance_outcomes(s).iter().map(|&(a, _)| a).collect(),
        Turn::Player(_) => g.legal_actions(s),
    };
    for a in actions {
        let mut c = s.clone();
        g.apply(&mut c, a);
        collect_infosets(g, &c, br, keys);
    }
}

/// Value to `br` when `br` plays the pure strategy `choice` (action index per
/// infoset key) and the opponent plays the solver's average strategy.
fn pure_vs_avg(
    g: &Kuhn,
    solver: &OsMccfr<Kuhn>,
    s: &KuhnState,
    br: usize,
    choice: &std::collections::HashMap<u64, usize>,
) -> f64 {
    if g.is_terminal(s) {
        return g.returns(s, br);
    }
    match g.turn(s) {
        Turn::Chance => g
            .chance_outcomes(s)
            .iter()
            .map(|&(a, p)| {
                let mut c = s.clone();
                g.apply(&mut c, a);
                p * pure_vs_avg(g, solver, &c, br, choice)
            })
            .sum(),
        Turn::Player(pl) if pl == br => {
            let actions = g.legal_actions(s);
            let i = choice[&g.infoset_key(s, br)];
            let mut c = s.clone();
            g.apply(&mut c, actions[i]);
            pure_vs_avg(g, solver, &c, br, choice)
        }
        Turn::Player(pl) => {
            let sigma = solver.policy(s, pl);
            g.legal_actions(s)
                .iter()
                .enumerate()
                .map(|(i, &a)| {
                    let mut c = s.clone();
                    g.apply(&mut c, a);
                    sigma[i] * pure_vs_avg(g, solver, &c, br, choice)
                })
                .sum()
        }
    }
}

/// Exact best-response value for `br` against the solver's average strategy by
/// enumerating all of `br`'s pure strategies (Kuhn: 6 binary infosets → 64).
/// A best response over infosets is realized by some pure strategy, so the max
/// over all of them is exact.
fn best_response_value(g: &Kuhn, solver: &OsMccfr<Kuhn>, br: usize) -> f64 {
    let mut keys = Vec::new();
    collect_infosets(g, &g.initial_state(), br, &mut keys);
    assert_eq!(keys.len(), 6, "each Kuhn player has 6 infosets");
    let mut best = f64::NEG_INFINITY;
    for assignment in 0..(1u32 << keys.len()) {
        let choice: std::collections::HashMap<u64, usize> = keys
            .iter()
            .enumerate()
            .map(|(i, &k)| (k, (assignment >> i & 1) as usize))
            .collect();
        let v = pure_vs_avg(g, solver, &g.initial_state(), br, &choice);
        best = best.max(v);
    }
    best
}

/// P(action 1) at the infoset where `player` holds `card` after `history`.
fn p_bet(solver: &OsMccfr<Kuhn>, player: usize, card: i8, history: &[u8]) -> f64 {
    let mut cards = [-1i8; 2];
    cards[player] = card;
    cards[1 - player] = (card + 1) % 3; // any other card: infoset key ignores it
    let s = KuhnState {
        cards,
        dealt: 2,
        history: history.to_vec(),
    };
    solver.policy(&s, player)[1]
}

#[test]
fn kuhn_converges_to_nash() {
    let mut solver = OsMccfr::new(Kuhn, 7);
    solver.run(300_000);
    assert_eq!(solver.num_infosets(), 12, "Kuhn has 12 information sets");

    // Self-play value of the average strategy approximates the game value -1/18.
    let value = self_play_value(solver.game(), &solver, &solver.game().initial_state());
    assert!(
        (value - (-1.0 / 18.0)).abs() < 0.02,
        "value to P0 should be ~-1/18, got {value}"
    );

    // Exact best response against the average strategy: NashConv = br0 + br1
    // is 0 at Nash; outcome sampling should drive it small.
    let br0 = best_response_value(solver.game(), &solver, 0);
    let br1 = best_response_value(solver.game(), &solver, 1);
    let nashconv = br0 + br1;
    assert!(
        nashconv < 0.08,
        "should approach Nash: br0={br0} br1={br1} nashconv={nashconv}"
    );

    // Known equilibrium structure. P0 opens: bet(Q)=0, bet(K)=3·bet(J) with
    // bet(J)=α ∈ [0, 1/3]. P1 facing a bet: call(J)=0, call(Q)=1/3, call(K)=1.
    // P1 after a check: bet(J)=1/3, bet(Q)=0, bet(K)=1.
    let alpha = p_bet(&solver, 0, 0, &[]);
    assert!(alpha < 1.0 / 3.0 + 0.08, "P0 bet(J)=α≤1/3, got {alpha}");
    assert!(p_bet(&solver, 0, 1, &[]) < 0.08, "P0 should not open-bet Q");
    let bet_k = p_bet(&solver, 0, 2, &[]);
    assert!(
        (bet_k - 3.0 * alpha).abs() < 0.12,
        "P0 bet(K) should be 3α: α={alpha} bet(K)={bet_k}"
    );
    assert!(
        p_bet(&solver, 1, 0, &[1]) < 0.08,
        "P1 should fold J to a bet"
    );
    let call_q = p_bet(&solver, 1, 1, &[1]);
    assert!(
        (call_q - 1.0 / 3.0).abs() < 0.2,
        "P1 call(Q) should be 1/3, got {call_q}"
    );
    assert!(p_bet(&solver, 1, 2, &[1]) > 0.92, "P1 should call with K");
    let bluff_j = p_bet(&solver, 1, 0, &[0]);
    assert!(
        (bluff_j - 1.0 / 3.0).abs() < 0.12,
        "P1 bet(J) after check should be 1/3, got {bluff_j}"
    );
    assert!(p_bet(&solver, 1, 1, &[0]) < 0.08, "P1 should check Q back");
    assert!(
        p_bet(&solver, 1, 2, &[0]) > 0.92,
        "P1 should bet K after check"
    );
}

/// An escalation ladder: at level `l < TOP` the mover may stop (terminal) or
/// raise by 1 or 2. External sampling expands both raises at every traverser
/// node, so its traversal visits T(l) = T(l+1) + T(l+2) nodes — Fibonacci in
/// the remaining depth, ~10^41 from level 0. Outcome sampling walks one
/// trajectory of at most TOP steps.
const TOP: u32 = 200;

#[derive(Clone)]
struct LadderState {
    level: u32,
    mover: usize,
    stopped: bool,
    key: u64,
}

struct Ladder;

impl Game for Ladder {
    type State = LadderState;
    type Action = u8; // 0 = stop, 1 = raise 1, 2 = raise 2

    fn initial_state(&self) -> LadderState {
        LadderState {
            level: 0,
            mover: 0,
            stopped: false,
            key: 1,
        }
    }

    fn turn(&self, s: &LadderState) -> Turn {
        Turn::Player(s.mover)
    }

    fn is_terminal(&self, s: &LadderState) -> bool {
        s.stopped || s.level >= TOP
    }

    fn returns(&self, s: &LadderState, player: usize) -> f64 {
        if !s.stopped {
            return 0.0; // ran off the top: draw
        }
        // The player who stopped is `1 - s.mover` (mover flipped on apply).
        let stopper = 1 - s.mover;
        let v = if s.level.is_multiple_of(3) { 1.0 } else { -0.5 };
        if player == stopper { v } else { -v }
    }

    fn legal_actions(&self, _s: &LadderState) -> Vec<u8> {
        vec![0, 1, 2]
    }

    fn chance_outcomes(&self, _s: &LadderState) -> Vec<(u8, f64)> {
        unreachable!("no chance nodes")
    }

    fn apply(&self, s: &mut LadderState, a: u8) {
        if a == 0 {
            s.stopped = true;
        } else {
            s.level = (s.level + a as u32).min(TOP);
        }
        s.mover = 1 - s.mover;
        s.key = s.key.wrapping_mul(4).wrapping_add(1 + a as u64);
    }

    fn infoset_key(&self, s: &LadderState, _player: usize) -> u64 {
        s.key // perfect information: the public action history
    }
}

#[test]
fn deep_ladder_is_linear_per_iteration() {
    let start = std::time::Instant::now();
    let mut solver = OsMccfr::new(Ladder, 42);
    solver.run(5_000);
    assert!(
        start.elapsed() < std::time::Duration::from_secs(10),
        "outcome sampling must stay O(trajectory length) per iteration"
    );
    assert!(solver.num_infosets() > 0);
    let root = solver.game().initial_state();
    let policy = solver.policy(&root, 0);
    assert_eq!(policy.len(), 3);
    let sum: f64 = policy.iter().sum();
    assert!((sum - 1.0).abs() < 1e-9, "policy must be a distribution");
    assert!(policy.iter().all(|&p| p.is_finite() && p >= 0.0));
}
