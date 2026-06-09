//! Kuhn poker is the canonical CFR correctness test: a tiny imperfect-information
//! game with a known Nash equilibrium where exploitability drives to zero. If the
//! generic solver converges here, the algorithm is implemented correctly.

use cfr_core::{Game, Solver, Turn};

/// Kuhn poker. Cards J=0, Q=1, K=2; ante 1; one betting round. State is the two
/// dealt cards (filled by two chance steps) plus the public action history.
#[derive(Clone, Default)]
struct KuhnState {
    cards: [i8; 2], // -1 until dealt
    dealt: u8,
    history: Vec<u8>, // 0 = check/fold, 1 = bet/call
}

struct Kuhn;

impl Kuhn {
    fn betting_terminal(&self, h: &[u8]) -> bool {
        matches!(h, [0, 0] | [0, 1, 0] | [0, 1, 1] | [1, 0] | [1, 1])
    }
}

impl Game for Kuhn {
    type State = KuhnState;
    type Action = u8;

    fn initial_state(&self) -> KuhnState {
        KuhnState {
            cards: [-1, -1],
            dealt: 0,
            history: Vec::new(),
        }
    }

    fn turn(&self, s: &KuhnState) -> Turn {
        if s.dealt < 2 {
            Turn::Chance
        } else {
            Turn::Player(s.history.len() % 2)
        }
    }

    fn is_terminal(&self, s: &KuhnState) -> bool {
        s.dealt == 2 && self.betting_terminal(&s.history)
    }

    fn returns(&self, s: &KuhnState, player: usize) -> f64 {
        let h = &s.history;
        let p0_high = s.cards[0] > s.cards[1];
        // Pot/transfer to player 0, then sign for `player`.
        let to_p0 = match h.as_slice() {
            [0, 0] => {
                if p0_high {
                    1.0
                } else {
                    -1.0
                }
            } // check-check: showdown for 1
            [1, 0] => 1.0, // bet, fold: p0 wins 1
            [1, 1] => {
                if p0_high {
                    2.0
                } else {
                    -2.0
                }
            } // bet, call: showdown for 2
            [0, 1, 0] => -1.0, // check, bet, fold: p1 wins 1
            [0, 1, 1] => {
                if p0_high {
                    2.0
                } else {
                    -2.0
                }
            } // check, bet, call
            _ => 0.0,
        };
        if player == 0 { to_p0 } else { -to_p0 }
    }

    fn legal_actions(&self, _s: &KuhnState) -> Vec<u8> {
        vec![0, 1]
    }

    fn chance_outcomes(&self, s: &KuhnState) -> Vec<(u8, f64)> {
        // Deal an undealt card uniformly (3 cards, no replacement).
        let taken = if s.dealt == 1 { s.cards[0] } else { -1 };
        let avail: Vec<u8> = (0..3u8).filter(|&c| c as i8 != taken).collect();
        let p = 1.0 / avail.len() as f64;
        avail.into_iter().map(|c| (c, p)).collect()
    }

    fn apply(&self, s: &mut KuhnState, a: u8) {
        if s.dealt < 2 {
            s.cards[s.dealt as usize] = a as i8;
            s.dealt += 1;
        } else {
            s.history.push(a);
        }
    }

    fn infoset_key(&self, s: &KuhnState, player: usize) -> u64 {
        // Base-3 digits, all >= 1 (so different lengths can't collide): the own
        // card (1..3) followed by each action (1..2).
        let mut k = (s.cards[player] + 1) as u64;
        for &a in &s.history {
            k = k * 3 + 1 + a as u64;
        }
        k
    }

    fn state_key(&self, s: &KuhnState) -> Option<u64> {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        s.cards.hash(&mut h);
        s.history.hash(&mut h);
        Some(h.finish())
    }
}

#[test]
fn kuhn_converges_to_nash() {
    let mut solver = Solver::new(Kuhn, 1);
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
