//! Liar's Dice as a [`cfr_core::Game`].
//!
//! This is the special variant used in the companion project: 1s are not wild;
//! a raise is exactly +1 quantity (same face) or +1 face (same quantity, wrapping
//! face `faces`→1 with +1 quantity); the first round opens at a forced `1×1` bid
//! and later rounds open freely; `Call Liar` and `Call Exact` resolve against the
//! actual dice, the loser drops a die, and the game ends when a player reaches 0.
//!
//! Unlike the original Python env (which rolled dice only at call time), this
//! models the real imperfect-information game: each round begins with chance
//! rolling every player's hand privately, so a player's information set is their
//! own hand plus the public bid history.

use std::hash::{Hash, Hasher};

use cfr_core::{Game, Turn};

const MAX_FACES: usize = 6;

/// Default round cap (see [`LiarsDice::max_rounds`]).
const DEFAULT_MAX_ROUNDS: u8 = 3;

/// Standard action ids 0..4 mirror the reference project; openings are explicit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    RaiseQuantity,
    RaiseFace,
    CallLiar,
    CallExact,
    /// Free opening bid `(quantity, face)` (only legal in the opening state).
    Open(u8, u8),
    /// Chance outcome: the rolling player's hand as per-face counts.
    Roll([u8; MAX_FACES]),
}

#[derive(Clone)]
pub struct LdState {
    dice_left: [u8; 2],
    hands: [[u8; MAX_FACES]; 2],
    rolled: u8, // hands rolled so far this round (0,1,2)
    qty: u8,    // current bid quantity; 0 = free-opening state
    face: u8,   // current bid face; 0 in the opening state
    turn: u8,   // player to act at a decision node
    first_round: bool,
    history: Vec<u8>, // encoded actions this round, for the information set
    rounds: u8,       // rounds opened so far (for the no-progress cap)
    done: bool,
    draw: bool, // reached the round cap with both players alive
}

/// Liar's Dice configuration (2 players assumed).
pub struct LiarsDice {
    pub dice: u8,
    pub faces: u8,
    /// A correct `Call Exact` loses nobody a die, so a round can repeat without
    /// progress; reaching this many rounds is a draw (value 0). Normal play loses
    /// a die every round and ends in `< 2*dice` rounds, so this only bounds the
    /// pathological exact-call loop — and keeps the game tree finite and small
    /// enough for exact CFR + best response. Keep it low for tractability.
    pub max_rounds: u8,
}

impl Default for LiarsDice {
    fn default() -> Self {
        // Matches the reference project's DEFAULT_CFG: 2 dice, 4 faces.
        Self::new(2, 4)
    }
}

impl LiarsDice {
    pub fn new(dice: u8, faces: u8) -> Self {
        assert!(faces as usize <= MAX_FACES);
        Self {
            dice,
            faces,
            max_rounds: DEFAULT_MAX_ROUNDS,
        }
    }

    pub fn with_max_rounds(mut self, max_rounds: u8) -> Self {
        self.max_rounds = max_rounds;
        self
    }

    fn count_face(&self, s: &LdState, face: u8) -> u8 {
        s.hands[0][face as usize - 1] + s.hands[1][face as usize - 1]
    }

    /// Open a fresh round (re-roll), or mark terminal, after a die is lost.
    fn open_round(&self, s: &mut LdState, next_opener: u8) {
        if s.dice_left.contains(&0) {
            s.done = true;
            return;
        }
        s.rounds += 1;
        if s.rounds > self.max_rounds {
            s.done = true;
            s.draw = true;
            return;
        }
        s.turn = next_opener;
        s.qty = 0;
        s.face = 0;
        s.first_round = false;
        s.history.clear();
        s.rolled = 0;
        s.hands = [[0; MAX_FACES]; 2];
    }
}

/// All distinct per-face count vectors for `dice` dice over `faces` faces, with
/// their multinomial probabilities (each die uniform over the faces).
fn hand_distribution(dice: u8, faces: u8) -> Vec<([u8; MAX_FACES], f64)> {
    fn fact(n: u8) -> f64 {
        (1..=n as u64).product::<u64>() as f64
    }
    let mut out = Vec::new();
    let mut counts = [0u8; MAX_FACES];
    let p_each = 1.0 / faces as f64;
    fn rec(
        face: usize,
        remaining: u8,
        faces: u8,
        counts: &mut [u8; MAX_FACES],
        dice: u8,
        p_each: f64,
        out: &mut Vec<([u8; MAX_FACES], f64)>,
    ) {
        if face == faces as usize {
            if remaining == 0 {
                let mut ways = fact(dice);
                for &c in counts.iter() {
                    ways /= fact(c);
                }
                let prob = ways * p_each.powi(dice as i32);
                out.push((*counts, prob));
            }
            return;
        }
        for c in 0..=remaining {
            counts[face] = c;
            rec(face + 1, remaining - c, faces, counts, dice, p_each, out);
        }
        counts[face] = 0;
    }
    rec(0, dice, faces, &mut counts, dice, p_each, &mut out);
    out
}

impl Game for LiarsDice {
    type State = LdState;
    type Action = Action;

    fn initial_state(&self) -> LdState {
        LdState {
            dice_left: [self.dice, self.dice],
            hands: [[0; MAX_FACES]; 2],
            rolled: 0,
            qty: 1, // forced 1x1 opening of the first round
            face: 1,
            turn: 0, // player 0 responds to the forced opening first
            first_round: true,
            history: Vec::new(),
            rounds: 1,
            done: false,
            draw: false,
        }
    }

    fn turn(&self, s: &LdState) -> Turn {
        if s.rolled < 2 {
            Turn::Chance
        } else {
            Turn::Player(s.turn as usize)
        }
    }

    fn is_terminal(&self, s: &LdState) -> bool {
        s.done
    }

    fn returns(&self, s: &LdState, player: usize) -> f64 {
        if s.draw {
            0.0
        } else if s.dice_left[player] > 0 {
            1.0
        } else {
            -1.0
        }
    }

    fn chance_outcomes(&self, s: &LdState) -> Vec<(Action, f64)> {
        let d = s.dice_left[s.rolled as usize];
        hand_distribution(d, self.faces)
            .into_iter()
            .map(|(c, p)| (Action::Roll(c), p))
            .collect()
    }

    fn legal_actions(&self, s: &LdState) -> Vec<Action> {
        let total = s.dice_left[0] + s.dice_left[1];
        let mut acts = Vec::new();
        if s.qty == 0 {
            // Free opening (later rounds): any (q, f) with q up to the dice in play.
            for q in 1..=total {
                for f in 1..=self.faces {
                    acts.push(Action::Open(q, f));
                }
            }
            return acts;
        }
        if s.qty < total {
            acts.push(Action::RaiseQuantity);
        }
        if s.face < self.faces || s.qty < total {
            acts.push(Action::RaiseFace);
        }
        acts.push(Action::CallLiar);
        acts.push(Action::CallExact);
        acts
    }

    fn apply(&self, s: &mut LdState, a: Action) {
        match a {
            Action::Roll(counts) => {
                s.hands[s.rolled as usize] = counts;
                s.rolled += 1;
            }
            Action::Open(q, f) => {
                s.qty = q;
                s.face = f;
                s.history.push(encode(a, self.faces));
                s.turn ^= 1;
            }
            Action::RaiseQuantity => {
                s.qty += 1;
                s.history.push(encode(a, self.faces));
                s.turn ^= 1;
            }
            Action::RaiseFace => {
                if s.face < self.faces {
                    s.face += 1;
                } else {
                    s.face = 1;
                    s.qty += 1;
                }
                s.history.push(encode(a, self.faces));
                s.turn ^= 1;
            }
            Action::CallLiar => {
                let caller = s.turn;
                let bidder = caller ^ 1;
                let count = self.count_face(s, s.face);
                let loser = if count < s.qty { bidder } else { caller };
                s.dice_left[loser as usize] -= 1;
                s.history.push(encode(a, self.faces));
                self.open_round(s, loser);
            }
            Action::CallExact => {
                let caller = s.turn;
                let count = self.count_face(s, s.face);
                if count != s.qty {
                    s.dice_left[caller as usize] -= 1;
                }
                s.history.push(encode(a, self.faces));
                self.open_round(s, caller);
            }
        }
    }

    fn infoset_key(&self, s: &LdState, player: usize) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        // What `player` observes: their own hand, both dice counts, the current
        // bid, the first-round flag, and the public action history this round.
        player.hash(&mut h);
        s.hands[player].hash(&mut h);
        s.dice_left.hash(&mut h);
        s.qty.hash(&mut h);
        s.face.hash(&mut h);
        s.first_round.hash(&mut h);
        s.rounds.hash(&mut h);
        s.history.hash(&mut h);
        h.finish()
    }

    fn state_key(&self, s: &LdState) -> Option<u64> {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        s.dice_left.hash(&mut h);
        s.hands.hash(&mut h);
        s.rolled.hash(&mut h);
        s.qty.hash(&mut h);
        s.face.hash(&mut h);
        s.turn.hash(&mut h);
        s.first_round.hash(&mut h);
        s.rounds.hash(&mut h);
        s.history.hash(&mut h);
        s.done.hash(&mut h);
        Some(h.finish())
    }
}

/// Compact encoding of a played action for the history (chance is never stored).
fn encode(a: Action, faces: u8) -> u8 {
    match a {
        Action::RaiseQuantity => 0,
        Action::RaiseFace => 1,
        Action::CallLiar => 2,
        Action::CallExact => 3,
        Action::Open(q, f) => 4 + (q - 1) * faces + (f - 1),
        Action::Roll(_) => unreachable!("chance outcomes are not recorded in history"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hand(counts: &[u8]) -> [u8; MAX_FACES] {
        let mut c = [0u8; MAX_FACES];
        c[..counts.len()].copy_from_slice(counts);
        c
    }

    #[test]
    fn hand_distribution_sums_to_one() {
        for &(dice, faces) in &[(1u8, 4u8), (2, 4), (2, 6), (3, 4)] {
            let dist = hand_distribution(dice, faces);
            let total: f64 = dist.iter().map(|(_, p)| p).sum();
            assert!(
                (total - 1.0).abs() < 1e-9,
                "dice={dice} faces={faces} sum={total}"
            );
        }
    }

    #[test]
    fn call_liar_resolution_and_dice_loss() {
        let g = LiarsDice::new(2, 4);
        // Construct a mid-round state: bid is 3x2, p0 to call. Hands hold one 2 each
        // (count of face 2 = 2 < 3) so Call Liar is correct -> bidder (p1) loses.
        let mut s = LdState {
            dice_left: [2, 2],
            hands: [hand(&[0, 1, 1, 0]), hand(&[0, 1, 1, 0])],
            rolled: 2,
            qty: 3,
            face: 2,
            turn: 0,
            first_round: false,
            history: vec![],
            rounds: 1,
            done: false,
            draw: false,
        };
        g.apply(&mut s, Action::CallLiar);
        assert_eq!(s.dice_left, [2, 1], "bidder (p1) should lose a die");
        assert!(!s.done);
        assert_eq!(s.qty, 0, "a new round opens freely");
    }

    #[test]
    fn call_exact_correct_loses_nobody() {
        let g = LiarsDice::new(2, 4);
        let mut s = LdState {
            dice_left: [2, 2],
            hands: [hand(&[0, 1, 0, 0]), hand(&[0, 1, 0, 0])], // two 2s total
            rolled: 2,
            qty: 2,
            face: 2,
            turn: 1,
            first_round: false,
            history: vec![],
            rounds: 1,
            done: false,
            draw: false,
        };
        g.apply(&mut s, Action::CallExact);
        assert_eq!(s.dice_left, [2, 2], "exact-correct: nobody loses a die");
    }

    #[test]
    fn game_ends_when_a_player_hits_zero() {
        let g = LiarsDice::new(1, 4); // 1 die each: a single wrong call ends it
        let mut s = LdState {
            dice_left: [1, 1],
            hands: [hand(&[1, 0, 0, 0]), hand(&[1, 0, 0, 0])], // two 1s
            rolled: 2,
            qty: 3,
            face: 1,
            turn: 0,
            first_round: false,
            history: vec![],
            rounds: 1,
            done: false,
            draw: false,
        };
        g.apply(&mut s, Action::CallLiar); // count(1)=2 < 3 -> bidder p1 loses last die
        assert!(g.is_terminal(&s));
        assert_eq!(g.returns(&s, 0), 1.0);
        assert_eq!(g.returns(&s, 1), -1.0);
    }
}
