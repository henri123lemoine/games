//! Outcome-sampling MCCFR correctness on Kuhn poker (convergence toward Nash,
//! checked with an exact pure-strategy best response) and the property that
//! motivates it: per-iteration cost linear in trajectory length, so deep
//! action ladders — where external sampling's traversal is exponential — train
//! in milliseconds.

use game_core::{Game, Turn};
use solvers::os_mccfr::OsMccfr;

mod common;
use common::{Kuhn, KuhnState, kuhn_nashconv, kuhn_value};

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
    let value = kuhn_value(&|s, p| solver.policy(s, p));
    assert!(
        (value - (-1.0 / 18.0)).abs() < 0.02,
        "value to P0 should be ~-1/18, got {value}"
    );

    // Exact best response against the average strategy: NashConv is 0 at
    // Nash; outcome sampling should drive it small.
    let nashconv = kuhn_nashconv(&|s, p| solver.policy(s, p));
    assert!(nashconv < 0.08, "should approach Nash: nashconv={nashconv}");

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
