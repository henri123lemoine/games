//! Test games shared across the solver integration tests: Kuhn poker (the
//! CFR-family correctness yardstick) and tic-tac-toe (the perfect-information
//! yardstick), each defined once instead of per test file.
#![allow(dead_code)]

use game_core::{Game, Turn};
use solvers::azero::PolicyValueEncoder;

/// Kuhn poker. Cards J=0, Q=1, K=2; ante 1; one betting round. State is the two
/// dealt cards (filled by two chance steps) plus the public action history.
#[derive(Clone, Default)]
pub struct KuhnState {
    pub cards: [i8; 2], // -1 until dealt
    pub dealt: u8,
    pub history: Vec<u8>, // 0 = check/fold, 1 = bet/call
}

pub struct Kuhn;

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

    fn max_return(&self) -> f64 {
        2.0
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

/// Tic-tac-toe: the perfect-information yardstick (small enough to solve,
/// drawish enough to expose draw-handling bugs).
#[derive(Clone)]
pub struct TttState {
    pub cells: [u8; 9],
    pub to_move: usize,
}

pub struct Ttt;

pub const LINES: [[usize; 3]; 8] = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
];

pub fn ttt_winner(s: &TttState) -> Option<usize> {
    LINES.iter().find_map(|l| {
        let v = s.cells[l[0]];
        (v != 0 && s.cells[l[1]] == v && s.cells[l[2]] == v).then(|| v as usize - 1)
    })
}

impl Game for Ttt {
    type State = TttState;
    type Action = usize;

    fn initial_state(&self) -> TttState {
        TttState {
            cells: [0; 9],
            to_move: 0,
        }
    }

    fn turn(&self, s: &TttState) -> Turn {
        Turn::Player(s.to_move)
    }

    fn is_terminal(&self, s: &TttState) -> bool {
        ttt_winner(s).is_some() || s.cells.iter().all(|&c| c != 0)
    }

    fn returns(&self, s: &TttState, player: usize) -> f64 {
        match ttt_winner(s) {
            Some(w) if w == player => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        }
    }

    fn legal_actions(&self, s: &TttState) -> Vec<usize> {
        (0..9).filter(|&i| s.cells[i] == 0).collect()
    }

    fn chance_outcomes(&self, _s: &TttState) -> Vec<(usize, f64)> {
        Vec::new()
    }

    fn apply(&self, s: &mut TttState, a: usize) {
        s.cells[a] = s.to_move as u8 + 1;
        s.to_move ^= 1;
    }

    fn infoset_key(&self, s: &TttState, _player: usize) -> u64 {
        s.cells
            .iter()
            .fold(s.to_move as u64, |k, &c| k * 4 + c as u64)
    }
}

/// One-hot planes per side plus side-to-move — the encoder shared by the
/// policy-gradient and azero tests.
pub struct TttEnc;

impl PolicyValueEncoder<Ttt> for TttEnc {
    fn input_len(&self) -> usize {
        19
    }

    fn policy_len(&self) -> usize {
        9
    }

    fn encode_state(&self, _g: &Ttt, s: &TttState) -> Vec<f32> {
        let mut x = vec![0.0; 19];
        for (i, &c) in s.cells.iter().enumerate() {
            match c {
                1 => x[i] = 1.0,
                2 => x[9 + i] = 1.0,
                _ => {}
            }
        }
        x[18] = s.to_move as f32;
        x
    }

    fn action_index(&self, _g: &Ttt, _s: &TttState, a: usize) -> usize {
        a
    }
}

/// Exact NashConv of a Kuhn policy: for each player, the best pure
/// infoset-strategy response (Kuhn has 6 infosets x 2 actions per player, so
/// 64 pure strategies — small enough to enumerate exactly, with chance
/// integrated). Zero exactly at a Nash equilibrium.
pub fn kuhn_nashconv(policy: &dyn Fn(&KuhnState, usize) -> Vec<f64>) -> f64 {
    (0..2).map(|br| kuhn_best_response(policy, br)).sum()
}

fn kuhn_best_response(policy: &dyn Fn(&KuhnState, usize) -> Vec<f64>, br: usize) -> f64 {
    let g = Kuhn;
    let mut keys = Vec::new();
    collect_infosets(&g, &g.initial_state(), br, &mut keys);
    let n = keys.len();
    let mut best = f64::NEG_INFINITY;
    for mask in 0..(1u32 << n) {
        let choice: std::collections::HashMap<u64, usize> = keys
            .iter()
            .enumerate()
            .map(|(i, &k)| (k, ((mask >> i) & 1) as usize))
            .collect();
        let v = pure_vs_policy(&g, policy, &g.initial_state(), br, &choice);
        best = best.max(v);
    }
    best
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

fn pure_vs_policy(
    g: &Kuhn,
    policy: &dyn Fn(&KuhnState, usize) -> Vec<f64>,
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
                p * pure_vs_policy(g, policy, &c, br, choice)
            })
            .sum(),
        Turn::Player(pl) if pl == br => {
            let a = g.legal_actions(s)[choice[&g.infoset_key(s, br)]];
            let mut c = s.clone();
            g.apply(&mut c, a);
            pure_vs_policy(g, policy, &c, br, choice)
        }
        Turn::Player(pl) => {
            let sigma = policy(s, pl);
            g.legal_actions(s)
                .iter()
                .enumerate()
                .map(|(i, &a)| {
                    if sigma[i] == 0.0 {
                        return 0.0;
                    }
                    let mut c = s.clone();
                    g.apply(&mut c, a);
                    sigma[i] * pure_vs_policy(g, policy, &c, br, choice)
                })
                .sum()
        }
    }
}
