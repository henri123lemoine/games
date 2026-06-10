//! Liar's Dice — N players, D dice, F faces — as a [`game_core::Game`].
//!
//! Faithful to the companion project's non-standard rules: 1s are not wild; a
//! raise is exactly +1 quantity (same face) or +1 face (same quantity, wrapping
//! `faces`→1 with +1 quantity); the first round opens at a forced `1×1` bid and
//! later rounds open freely; `Call Liar` and `Call Exact` resolve against the
//! actual dice across *all* live players, the loser drops a die, and a player at
//! zero dice is eliminated. Last player standing wins.
//!
//! Hidden dice are rolled by chance at the start of each round, so a player's
//! information set is their own hand plus the public bidding context. To keep
//! learning tractable on large configurations the information-set key abstracts
//! the bid history to the last few actions (full history is infeasible for, e.g.,
//! 5 players × 5 dice).

use std::hash::{Hash, Hasher};

use game_core::{Game, Turn};

mod agents;
mod ui;
pub use agents::{BidConditioned, ProbConfig, ProbabilisticAgent, RandomAgent};

pub const MAX_FACES: usize = 6;
pub const MAX_PLAYERS: usize = 8;
/// Bid-history actions retained in the information-set key (an abstraction).
const HIST_K: usize = 6;
const DEFAULT_MAX_ROUNDS: u8 = 24;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    RaiseQuantity,
    RaiseFace,
    CallLiar,
    CallExact,
    Open(u8, u8),
    /// Chance: the rolling player's hand as per-face counts.
    Roll([u8; MAX_FACES]),
}

#[derive(Clone)]
pub struct LdState {
    dice_left: [u8; MAX_PLAYERS],
    hands: [[u8; MAX_FACES]; MAX_PLAYERS],
    rolled: u8, // players whose hands are rolled this round
    qty: u8,    // current bid quantity; 0 = opening state
    face: u8,
    turn: u8,        // current actor (a live player)
    last_bidder: u8, // who owns the current bid (for call resolution)
    first_round: bool,
    hist: [u8; HIST_K],
    endorsed: [u8; MAX_PLAYERS], // face each player last bid this round (0 = none)
    rounds: u8,
    done: bool,
    winner: u8,
}

pub struct LiarsDice {
    pub players: u8,
    pub dice: u8,
    pub faces: u8,
    pub max_rounds: u8,
}

impl LiarsDice {
    pub fn new(players: u8, dice: u8, faces: u8) -> Self {
        assert!(faces as usize <= MAX_FACES && players as usize <= MAX_PLAYERS && players >= 2);
        Self {
            players,
            dice,
            faces,
            max_rounds: DEFAULT_MAX_ROUNDS,
        }
    }

    /// The common two-player configuration.
    pub fn two_player(dice: u8, faces: u8) -> Self {
        Self::new(2, dice, faces)
    }

    pub fn with_max_rounds(mut self, m: u8) -> Self {
        self.max_rounds = m;
        self
    }

    fn alive(&self, s: &LdState, p: u8) -> bool {
        s.dice_left[p as usize] > 0
    }

    fn num_alive(&self, s: &LdState) -> u8 {
        (0..self.players).filter(|&p| self.alive(s, p)).count() as u8
    }

    fn total_dice(&self, s: &LdState) -> u8 {
        s.dice_left[..self.players as usize].iter().sum()
    }

    fn next_alive(&self, s: &LdState, from: u8) -> u8 {
        let mut p = (from + 1) % self.players;
        while !self.alive(s, p) {
            p = (p + 1) % self.players;
        }
        p
    }

    fn count_face(&self, s: &LdState, face: u8) -> u8 {
        (0..self.players as usize)
            .map(|p| s.hands[p][face as usize - 1])
            .sum()
    }

    fn push_hist(&self, s: &mut LdState, code: u8) {
        s.hist.copy_within(1..HIST_K, 0);
        s.hist[HIST_K - 1] = code;
    }

    /// After a die is lost: eliminate at zero, end the game if one player
    /// remains, otherwise open the next round (re-roll) from `next_opener`.
    fn resolve_after_call(&self, s: &mut LdState, next_opener: u8) {
        if self.num_alive(s) <= 1 {
            s.done = true;
            s.winner = (0..self.players).find(|&p| self.alive(s, p)).unwrap_or(0);
            return;
        }
        s.rounds += 1;
        if s.rounds > self.max_rounds {
            s.done = true;
            s.winner = (0..self.players)
                .max_by_key(|&p| s.dice_left[p as usize])
                .unwrap();
            return;
        }
        let opener = if self.alive(s, next_opener) {
            next_opener
        } else {
            self.next_alive(s, next_opener)
        };
        s.turn = opener;
        s.qty = 0;
        s.face = 0;
        s.first_round = false;
        s.hist = [0; HIST_K];
        s.endorsed = [0; MAX_PLAYERS];
        s.rolled = 0;
        s.hands = [[0; MAX_FACES]; MAX_PLAYERS];
    }

    /// Replace every player's hand *except* `observer`'s with a fresh roll of
    /// their remaining dice — a determinization consistent with what `observer`
    /// knows (their own hand and the public dice counts), for Monte-Carlo
    /// rollouts. Players who bid this round are biased toward credibly holding
    /// the face they last bid: with probability `bidder_bias` (current bidder)
    /// or `endorser_bias` (earlier bidders), one die is converted to that face
    /// if they hold none. The forced 1×1 opener has no endorsement, so nobody is
    /// credited with a face they never chose.
    pub fn resample_hidden(
        &self,
        s: &mut LdState,
        observer: usize,
        rng: &mut game_core::Rng,
        bidder_bias: f64,
        endorser_bias: f64,
    ) {
        for p in 0..self.players as usize {
            if p == observer {
                continue;
            }
            let mut counts = [0u8; MAX_FACES];
            for _ in 0..s.dice_left[p] {
                let face = ((rng.unit() * self.faces as f64) as usize).min(self.faces as usize - 1);
                counts[face] += 1;
            }
            let endorsed = s.endorsed[p];
            if endorsed > 0 && s.dice_left[p] > 0 {
                let f = (endorsed - 1) as usize;
                let strength = if p == s.last_bidder as usize {
                    bidder_bias
                } else {
                    endorser_bias
                };
                if counts[f] == 0 && rng.unit() < strength {
                    // Convert one random die into the endorsed face.
                    for c in counts.iter_mut() {
                        if *c > 0 {
                            *c -= 1;
                            break;
                        }
                    }
                    counts[f] += 1;
                }
            }
            s.hands[p] = counts;
        }
    }

    pub fn action_label(&self, a: Action) -> String {
        match a {
            Action::RaiseQuantity => "raise quantity".into(),
            Action::RaiseFace => "raise face".into(),
            Action::CallLiar => "call LIAR".into(),
            Action::CallExact => "call EXACT".into(),
            Action::Open(q, f) => format!("open {q}x{f}"),
            Action::Roll(_) => "roll".into(),
        }
    }
}

impl LdState {
    pub fn hand(&self, player: usize) -> Vec<u8> {
        let mut dice = Vec::new();
        for (i, &c) in self.hands[player].iter().enumerate() {
            for _ in 0..c {
                dice.push(i as u8 + 1);
            }
        }
        dice
    }
    /// Count of `face` (1-based) in `player`'s own hand.
    pub fn my_count(&self, player: usize, face: u8) -> u8 {
        self.hands[player][face as usize - 1]
    }
    pub fn current_bid(&self) -> (u8, u8) {
        (self.qty, self.face)
    }
    pub fn dice_left(&self) -> &[u8] {
        &self.dice_left[..]
    }
    pub fn turn(&self) -> usize {
        self.turn as usize
    }
    pub fn last_bidder(&self) -> usize {
        self.last_bidder as usize
    }
}

/// Per-face count vectors for `dice` dice over `faces` faces with multinomial
/// probabilities (each die uniform).
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
                out.push((*counts, ways * p_each.powi(dice as i32)));
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

fn encode(a: Action, faces: u8) -> u8 {
    match a {
        Action::RaiseQuantity => 1,
        Action::RaiseFace => 2,
        Action::CallLiar => 3,
        Action::CallExact => 4,
        Action::Open(q, f) => 5 + (q - 1) * faces + (f - 1),
        Action::Roll(_) => unreachable!(),
    }
}

impl Game for LiarsDice {
    type State = LdState;
    type Action = Action;

    fn num_players(&self) -> usize {
        self.players as usize
    }

    fn initial_state(&self) -> LdState {
        let mut dice_left = [0u8; MAX_PLAYERS];
        for d in dice_left.iter_mut().take(self.players as usize) {
            *d = self.dice;
        }
        LdState {
            dice_left,
            hands: [[0; MAX_FACES]; MAX_PLAYERS],
            rolled: 0,
            qty: 1, // forced 1x1 first round
            face: 1,
            turn: 0,
            last_bidder: self.players - 1, // phantom owner of the forced 1x1
            first_round: true,
            hist: [0; HIST_K],
            endorsed: [0; MAX_PLAYERS],
            rounds: 1,
            done: false,
            winner: 0,
        }
    }

    fn turn(&self, s: &LdState) -> Turn {
        if s.rolled < self.players {
            Turn::Chance
        } else {
            Turn::Player(s.turn as usize)
        }
    }

    fn is_terminal(&self, s: &LdState) -> bool {
        s.done
    }

    fn returns(&self, s: &LdState, player: usize) -> f64 {
        // Win the game: +1 to the last player standing, shared -1 to the rest.
        if s.winner as usize == player {
            1.0
        } else {
            -1.0 / (self.players as f64 - 1.0)
        }
    }

    fn chance_outcomes(&self, s: &LdState) -> Vec<(Action, f64)> {
        let d = s.dice_left[s.rolled as usize];
        if d == 0 {
            return vec![(Action::Roll([0; MAX_FACES]), 1.0)];
        }
        hand_distribution(d, self.faces)
            .into_iter()
            .map(|(c, pr)| (Action::Roll(c), pr))
            .collect()
    }

    fn legal_actions(&self, s: &LdState) -> Vec<Action> {
        let total = self.total_dice(s);
        let mut acts = Vec::new();
        if s.qty == 0 {
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
                self.push_hist(s, encode(a, self.faces));
                s.endorsed[s.turn as usize] = s.face;
                s.last_bidder = s.turn;
                s.turn = self.next_alive(s, s.turn);
            }
            Action::RaiseQuantity => {
                s.qty += 1;
                self.push_hist(s, encode(a, self.faces));
                s.endorsed[s.turn as usize] = s.face;
                s.last_bidder = s.turn;
                s.turn = self.next_alive(s, s.turn);
            }
            Action::RaiseFace => {
                if s.face < self.faces {
                    s.face += 1;
                } else {
                    s.face = 1;
                    s.qty += 1;
                }
                self.push_hist(s, encode(a, self.faces));
                s.endorsed[s.turn as usize] = s.face;
                s.last_bidder = s.turn;
                s.turn = self.next_alive(s, s.turn);
            }
            Action::CallLiar => {
                let caller = s.turn;
                let bidder = s.last_bidder;
                let count = self.count_face(s, s.face);
                let loser = if count < s.qty { bidder } else { caller };
                s.dice_left[loser as usize] -= 1;
                self.resolve_after_call(s, loser);
            }
            Action::CallExact => {
                let caller = s.turn;
                let count = self.count_face(s, s.face);
                if count != s.qty {
                    s.dice_left[caller as usize] -= 1;
                }
                self.resolve_after_call(s, caller);
            }
        }
    }

    fn infoset_key(&self, s: &LdState, player: usize) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        player.hash(&mut h);
        s.hands[player].hash(&mut h);
        s.dice_left[..self.players as usize].hash(&mut h);
        s.qty.hash(&mut h);
        s.face.hash(&mut h);
        s.first_round.hash(&mut h);
        // position relative to the bid owner conveys turn order without the path.
        ((s.turn + self.players - s.last_bidder) % self.players).hash(&mut h);
        s.hist.hash(&mut h);
        h.finish()
    }

    fn state_key(&self, s: &LdState) -> Option<u64> {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        s.dice_left[..self.players as usize].hash(&mut h);
        s.hands[..self.players as usize].hash(&mut h);
        s.rolled.hash(&mut h);
        s.qty.hash(&mut h);
        s.face.hash(&mut h);
        s.turn.hash(&mut h);
        s.last_bidder.hash(&mut h);
        s.first_round.hash(&mut h);
        s.hist.hash(&mut h);
        s.endorsed[..self.players as usize].hash(&mut h);
        s.done.hash(&mut h);
        Some(h.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    #[test]
    fn n_player_games_terminate_with_one_winner() {
        for &players in &[2u8, 3, 5] {
            let game = LiarsDice::new(players, 2, 6);
            let mut seed = 0x1234 + players as u64;
            for _ in 0..100 {
                let mut s = game.initial_state();
                let mut steps = 0;
                while !game.is_terminal(&s) {
                    steps += 1;
                    assert!(steps < 100_000, "must terminate (players={players})");
                    match game.turn(&s) {
                        Turn::Chance => {
                            let o = game.chance_outcomes(&s);
                            let a = o[(rng(&mut seed) as usize) % o.len()].0;
                            game.apply(&mut s, a);
                        }
                        Turn::Player(_) => {
                            let acts = game.legal_actions(&s);
                            let a = acts[(rng(&mut seed) as usize) % acts.len()];
                            game.apply(&mut s, a);
                        }
                    }
                }
                let total: f64 = (0..players as usize).map(|p| game.returns(&s, p)).sum();
                assert!(total.abs() < 1e-9, "zero-sum, got {total}");
            }
        }
    }

    #[test]
    fn hand_distribution_sums_to_one() {
        for &(d, f) in &[(2u8, 6u8), (5, 6), (3, 4)] {
            let t: f64 = hand_distribution(d, f).iter().map(|(_, p)| p).sum();
            assert!((t - 1.0).abs() < 1e-9);
        }
    }
}
