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
//! information set is their own hand plus the public bidding context. Bids are
//! monotonic (+1 quantity or +1 face), so the *current* bid already carries the
//! round's high-water mark; what a full history adds is the *path* — who raised
//! and which face each seat committed to (hand signaling). The information-set
//! key captures that path structurally with per-seat raise counts and endorsed
//! faces (both position-relative) rather than retaining a window of raw action
//! codes, which keeps it bounded yet history-aware on large configurations like
//! 5 players × 5 dice.

use game_core::hash::{combine, splitmix64};
use game_core::{Game, Turn};

mod agents;
pub mod deepcfr;
pub mod features;
pub mod net_search;
pub mod online_solve;
pub mod rebel;
mod solve;
mod subgame;
pub mod train;
mod ui;
pub use agents::{BidConditioned, ProbConfig, ProbabilisticAgent};
pub use features::{
    NetAgent, action_index, feature_len, legal_actions_and_support, net_policy, policy_len, support,
};
pub use net_search::NetTruncRollout;
pub use online_solve::{NetOnlineSolveAgent, OnlineSolveAgent, OnlineSolveConfig};
pub use solve::{
    FitConfig, FitResult, LatticeValue, decomposed_game_value, decomposed_value_capped,
    entry_round_value, fit_capped, fit_two_player, round_exploitabilities,
};
pub use subgame::{ContinuationValue, DiceShareValue, NetValue, RoundSubgame};

pub const MAX_FACES: usize = 6;
pub const MAX_PLAYERS: usize = 8;
/// Raise codes retained for the UI's bid-trail reconstruction (see `ui.rs`); not
/// part of the information-set key, which carries history structurally instead.
const HIST_K: usize = 6;

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
    hist: [u16; HIST_K], // last `HIST_K` raise codes, for UI bid-trail reconstruction
    endorsed: [u8; MAX_PLAYERS], // face each player last bid this round (0 = none)
    raises_this_round: [u8; MAX_PLAYERS], // bids each seat has made this round
    rounds: u16,
    done: bool,
    winner: u8,
}

pub struct LiarsDice {
    pub players: u8,
    pub dice: u8,
    pub faces: u8,
    pub max_rounds: u16,
}

impl LiarsDice {
    pub fn new(players: u8, dice: u8, faces: u8) -> Self {
        assert!(faces as usize <= MAX_FACES && players as usize <= MAX_PLAYERS && players >= 2);
        assert!(
            faces >= 2,
            "faces must be at least 2: the belief agents' binomials divide by 1 - 1/faces"
        );
        assert!(
            players as u16 * dice as u16 <= u8::MAX as u16,
            "dice counts are u8: players x dice must not exceed 255"
        );
        // A natural game needs up to `players x dice - 1` die-loss rounds, and
        // correct Exact calls add no-loss rounds that cluster in the few-dice
        // endgame. Truncating a legitimate game is catastrophic while a longer
        // pathological stall only costs rollout time, so the cap doubles the
        // loss bound and adds a flat buffer on top.
        let loss_rounds = u16::from(players) * u16::from(dice) - 1;
        let max_rounds = loss_rounds * 2 + 50;
        Self {
            players,
            dice,
            faces,
            max_rounds,
        }
    }

    /// The common two-player configuration.
    pub fn two_player(dice: u8, faces: u8) -> Self {
        Self::new(2, dice, faces)
    }

    pub fn with_max_rounds(mut self, m: u16) -> Self {
        assert!(m < u16::MAX, "the round counter must fit max_rounds + 1");
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

    /// The seat that opened the round `s` is currently in — the reference point an
    /// online round-subgame solve must rebuild the round from.
    ///
    /// Real bids are placed in turn order starting at the opener and cycling
    /// through the live seats, so with `b = sum(raises_this_round)` bids placed,
    /// the last bidder sits `b - 1` live steps after the opener. Walking back
    /// `b - 1` live steps from `last_bidder` recovers the opener. With no bids yet
    /// placed this round the seat to act *is* the opener (the forced `1×1` first
    /// round is the phantom seat's bid, not a live seat's, so it does not count).
    pub fn round_opener(&self, s: &LdState) -> u8 {
        let bids: u32 = s.raises_this_round[..self.players as usize]
            .iter()
            .map(|&r| u32::from(r))
            .sum();
        if bids == 0 {
            return s.turn;
        }
        let mut p = s.last_bidder;
        for _ in 0..bids - 1 {
            p = (p + self.players - 1) % self.players;
            while !self.alive(s, p) {
                p = (p + self.players - 1) % self.players;
            }
        }
        p
    }

    fn count_face(&self, s: &LdState, face: u8) -> u8 {
        (0..self.players as usize)
            .map(|p| s.hands[p][face as usize - 1])
            .sum()
    }

    /// Terminal because the round cap forced a dice-count adjudication rather
    /// than elimination down to one player.
    fn adjudicated(&self, s: &LdState) -> bool {
        s.done && self.num_alive(s) > 1
    }

    fn cap_leaders(&self, s: &LdState) -> Vec<usize> {
        let live: Vec<usize> = (0..self.players as usize)
            .filter(|&p| s.dice_left[p] > 0)
            .collect();
        let max_dice = live.iter().map(|&p| s.dice_left[p]).max().unwrap_or(0);
        live.into_iter()
            .filter(|&p| s.dice_left[p] == max_dice)
            .collect()
    }

    fn push_hist(&self, s: &mut LdState, code: u16) {
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
            // Round-cap adjudication: most dice wins, ties broken toward the
            // highest seat (an arbitrary but fixed convention; the cap exists
            // only to bound pathological stalls, not as a real rule).
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
        s.raises_this_round = [0; MAX_PLAYERS];
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
                counts[rng.below(self.faces as usize)] += 1;
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
                    // Convert one uniformly chosen die into the endorsed face.
                    let mut k = rng.below(s.dice_left[p] as usize);
                    for c in counts.iter_mut() {
                        if (*c as usize) > k {
                            *c -= 1;
                            break;
                        }
                        k -= *c as usize;
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

    fn sample_roll_counts(&self, dice: u8, rng: &mut game_core::Rng) -> [u8; MAX_FACES] {
        let mut counts = [0u8; MAX_FACES];
        for _ in 0..dice {
            counts[rng.below(self.faces as usize)] += 1;
        }
        counts
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
    /// True only during the game's very first round (the forced `1×1` open
    /// convention); every later round opens freely. Read by online solving to
    /// reconstruct this round's opening subgame with the right open convention.
    pub fn first_round(&self) -> bool {
        self.first_round
    }
    /// Number of bids each seat has made *this round* (index = seat). The sum is
    /// the count of real bids placed since the round opened; the round opener is
    /// recovered from it (see [`LiarsDice::round_opener`]).
    pub fn raises_this_round(&self) -> &[u8] {
        &self.raises_this_round[..]
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

/// Little-endian byte pack (≤ 8 bytes) — fixed-size fields fold into the
/// stable position keys without per-byte hashing.
fn pack(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .enumerate()
        .fold(0, |a, (i, &b)| a | u64::from(b) << (8 * i))
}

/// History code for the infoset key. `u16` because the `Open` range scales
/// with the total dice in play (`5 + (q-1)*faces + (f-1)`), which overflows
/// `u8` on large-but-legal configurations like 8 players x 6 dice.
fn encode(a: Action, faces: u8) -> u16 {
    match a {
        Action::RaiseQuantity => 1,
        Action::RaiseFace => 2,
        Action::CallLiar => 3,
        Action::CallExact => 4,
        Action::Open(q, f) => 5 + u16::from(q - 1) * u16::from(faces) + u16::from(f - 1),
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
            // The forced 1x1 is the phantom seat's; no live seat has bid yet.
            raises_this_round: [0; MAX_PLAYERS],
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
        if self.adjudicated(s) {
            let live: Vec<usize> = (0..self.players as usize)
                .filter(|&p| s.dice_left[p] > 0)
                .collect();
            let leaders = self.cap_leaders(s);
            if leaders.len() == live.len() {
                return 0.0;
            }
            if leaders.contains(&player) {
                return 1.0 / leaders.len() as f64;
            }
            return -1.0 / (live.len() - leaders.len()) as f64;
        }
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

    /// Roll the rolling player's dice directly instead of enumerating the full
    /// multinomial: O(dice) rather than O(C(dice + faces - 1, faces - 1)) — the
    /// difference between fast and unusable on large hands for outcome-sampling
    /// solvers. Returns the rolled counts and their (multinomial) probability,
    /// matching the distribution of `chance_outcomes` exactly.
    fn sample_chance(&self, s: &LdState, rng: &mut game_core::Rng) -> (Action, f64) {
        let d = s.dice_left[s.rolled as usize];
        if d == 0 {
            return (Action::Roll([0; MAX_FACES]), 1.0);
        }
        let counts = self.sample_roll_counts(d, rng);
        let mut ways = (1..=u64::from(d)).product::<u64>() as f64;
        for &c in &counts {
            for k in 1..=u64::from(c) {
                ways /= k as f64;
            }
        }
        let prob = ways * (1.0 / f64::from(self.faces)).powi(i32::from(d));
        (Action::Roll(counts), prob)
    }

    fn sample_chance_action(&self, s: &LdState, rng: &mut game_core::Rng) -> Action {
        let d = s.dice_left[s.rolled as usize];
        if d == 0 {
            return Action::Roll([0; MAX_FACES]);
        }
        Action::Roll(self.sample_roll_counts(d, rng))
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

    fn num_actions(&self, s: &LdState) -> usize {
        let total = self.total_dice(s);
        if s.qty == 0 {
            return usize::from(total) * usize::from(self.faces);
        }
        let mut n = 2; // CallLiar, CallExact.
        if s.qty < total {
            n += 1;
        }
        if s.face < self.faces || s.qty < total {
            n += 1;
        }
        n
    }

    fn action_at(&self, s: &LdState, i: usize) -> Action {
        let total = self.total_dice(s);
        if s.qty == 0 {
            let faces = usize::from(self.faces);
            let n = usize::from(total) * faces;
            assert!(
                i < n,
                "action index {i} out of range for {} opening actions",
                n
            );
            return Action::Open((i / faces + 1) as u8, (i % faces + 1) as u8);
        }

        let mut idx = 0usize;
        if s.qty < total {
            if i == idx {
                return Action::RaiseQuantity;
            }
            idx += 1;
        }
        if s.face < self.faces || s.qty < total {
            if i == idx {
                return Action::RaiseFace;
            }
            idx += 1;
        }
        if i == idx {
            return Action::CallLiar;
        }
        if i == idx + 1 {
            return Action::CallExact;
        }
        panic!("action index {i} out of range for Liar's Dice state");
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
                s.raises_this_round[s.turn as usize] += 1;
                s.last_bidder = s.turn;
                s.turn = self.next_alive(s, s.turn);
            }
            Action::RaiseQuantity => {
                s.qty += 1;
                self.push_hist(s, encode(a, self.faces));
                s.endorsed[s.turn as usize] = s.face;
                s.raises_this_round[s.turn as usize] += 1;
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
                s.raises_this_round[s.turn as usize] += 1;
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

    /// Own hand plus the public bidding context, keyed *structurally* and
    /// position-relative to `player` (see the module docs for why the bid path,
    /// not a raw-action window, is what a history needs to capture).
    ///
    /// Folds: own hand, the full dice vector, the current bid
    /// `(qty, face, first_round)`, the relative turn-vs-bid-owner position, and —
    /// rotated so `player` is reference seat 0, matching `features::encode` — the
    /// per-seat `raises_this_round` and per-seat `endorsed` face. The rotation
    /// makes the key position-relative (not seat-absolute), so equivalent
    /// situations under different seatings share a key. The round number is
    /// excluded: near `max_rounds` the cap adjudication is an anti-stall guard,
    /// not a rule worth key entropy. Deterministic and stable.
    fn infoset_key(&self, s: &LdState, player: usize) -> u64 {
        let p = self.players as usize;
        let seat = |k: usize| (player + k) % p;
        // position relative to the bid owner conveys turn order.
        let rel = (s.turn + self.players - s.last_bidder) % self.players;
        let bid = u64::from(s.qty) << 24
            | u64::from(s.face) << 16
            | u64::from(s.first_round) << 8
            | u64::from(rel);
        let mut raises_rot = [0u8; MAX_PLAYERS];
        let mut endorsed_rot = [0u8; MAX_PLAYERS];
        for k in 0..p {
            raises_rot[k] = s.raises_this_round[seat(k)];
            endorsed_rot[k] = s.endorsed[seat(k)];
        }
        [
            pack(&s.hands[player]),
            pack(&s.dice_left),
            bid,
            pack(&raises_rot),
            pack(&endorsed_rot),
        ]
        .into_iter()
        .fold(splitmix64(player as u64 + 1), combine)
    }

    fn state_key(&self, s: &LdState) -> Option<u64> {
        let fields = u64::from(s.qty) << 40
            | u64::from(s.face) << 32
            | u64::from(s.turn) << 24
            | u64::from(s.last_bidder) << 16
            | u64::from(s.rolled) << 8
            | u64::from(s.first_round) << 1
            | u64::from(s.done);
        let hands = s.hands.iter().fold(0, |h, hand| combine(h, pack(hand)));
        // `raises_this_round` distinguishes genuinely different histories that
        // share the same current bid; the god's-eye best-response memo must keep
        // them distinct.
        Some(
            [
                pack(&s.dice_left),
                hands,
                fields,
                pack(&s.raises_this_round),
                pack(&s.endorsed),
            ]
            .into_iter()
            .fold(splitmix64(0x11A5), combine),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_core::Rng;

    #[test]
    fn n_player_games_terminate_with_one_winner() {
        for &players in &[2u8, 3, 5] {
            let game = LiarsDice::new(players, 2, 6);
            let mut rng = Rng::new(0x1234 + u64::from(players));
            for _ in 0..100 {
                let mut s = game.initial_state();
                let mut steps = 0;
                while !game.is_terminal(&s) {
                    steps += 1;
                    assert!(steps < 100_000, "must terminate (players={players})");
                    match game.turn(&s) {
                        Turn::Chance => {
                            let o = game.chance_outcomes(&s);
                            let a = o[rng.below(o.len())].0;
                            game.apply(&mut s, a);
                        }
                        Turn::Player(_) => {
                            let acts = game.legal_actions(&s);
                            let a = acts[rng.below(acts.len())];
                            game.apply(&mut s, a);
                        }
                    }
                }
                let total: f64 = (0..players as usize).map(|p| game.returns(&s, p)).sum();
                assert!(total.abs() < 1e-9, "zero-sum, got {total}");
            }
        }
    }

    /// Two players, one die each: a lost die is elimination, so every call
    /// resolution ends the game with an inspectable winner.
    fn rolled(game: &LiarsDice, hands: &[[u8; MAX_FACES]]) -> LdState {
        let mut s = game.initial_state();
        for &h in hands {
            game.apply(&mut s, Action::Roll(h));
        }
        s
    }

    #[test]
    fn call_liar_against_a_false_bid_charges_the_bid_owner() {
        let game = LiarsDice::new(2, 1, 6);
        // No 1s anywhere: the forced 1x1 opener (owned by the last seat) is
        // a lie, so player 0's immediate call costs player 1 their die.
        let mut s = rolled(&game, &[[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]);
        game.apply(&mut s, Action::CallLiar);
        assert!(game.is_terminal(&s));
        assert_eq!(game.returns(&s, 0), 1.0);
    }

    #[test]
    fn call_liar_against_a_true_bid_charges_the_caller() {
        let game = LiarsDice::new(2, 1, 6);
        let mut s = rolled(&game, &[[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]);
        game.apply(&mut s, Action::CallLiar);
        assert!(game.is_terminal(&s));
        assert_eq!(game.returns(&s, 1), 1.0);
    }

    #[test]
    fn call_exact_costs_nothing_when_right_and_a_die_when_wrong() {
        let game = LiarsDice::new(2, 1, 6);
        let mut s = rolled(&game, &[[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]);
        game.apply(&mut s, Action::CallExact);
        assert!(!game.is_terminal(&s), "exactly one 1: nobody loses a die");
        assert_eq!(s.dice_left[..2], [1, 1]);
        assert!(matches!(game.turn(&s), Turn::Chance), "next round re-rolls");

        let game = LiarsDice::new(2, 1, 6);
        let mut s = rolled(&game, &[[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]]);
        game.apply(&mut s, Action::CallExact);
        assert!(game.is_terminal(&s), "two 1s != qty 1: the caller loses");
        assert_eq!(game.returns(&s, 1), 1.0);
    }

    #[test]
    fn round_cap_scales_with_the_dice_in_play() {
        assert_eq!(LiarsDice::new(2, 2, 6).max_rounds, 56);
        assert_eq!(LiarsDice::new(5, 5, 6).max_rounds, 98);
        assert_eq!(LiarsDice::new(8, 6, 6).max_rounds, 144);
    }

    #[test]
    fn tied_round_cap_is_scored_as_draw() {
        let game = LiarsDice::new(2, 1, 6).with_max_rounds(1);
        let mut s = rolled(&game, &[[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]);
        game.apply(&mut s, Action::CallExact);
        assert!(game.is_terminal(&s));
        assert!(game.adjudicated(&s));
        assert_eq!(s.dice_left[..2], [1, 1]);
        assert_eq!(game.returns(&s, 0), 0.0);
        assert_eq!(game.returns(&s, 1), 0.0);
    }

    #[test]
    fn unique_round_cap_leader_keeps_winning_returns() {
        let game = LiarsDice::new(2, 2, 6).with_max_rounds(1);
        let mut s = rolled(&game, &[[0, 2, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]);
        game.apply(&mut s, Action::CallLiar);
        assert!(game.is_terminal(&s));
        assert!(game.adjudicated(&s));
        assert_eq!(s.dice_left[..2], [2, 1]);
        assert_eq!(game.returns(&s, 0), 1.0);
        assert_eq!(game.returns(&s, 1), -1.0);
    }

    /// A 5p5d game played to the natural end — one die lost per round plus a
    /// few no-loss exact rounds — must finish by elimination, not round-cap
    /// adjudication. Under the old fixed cap of 24 this game was cut off with
    /// several players still alive.
    #[test]
    fn natural_5p5d_games_outlive_a_fixed_round_cap() {
        let game = LiarsDice::new(5, 5, 6);
        let mut s = game.initial_state();
        let mut rounds_played = 0u32;
        while !game.is_terminal(&s) {
            while matches!(game.turn(&s), Turn::Chance) {
                // First outcome: every die on the top face.
                let a = game.chance_outcomes(&s)[0].0;
                game.apply(&mut s, a);
            }
            rounds_played += 1;
            if s.qty == 0 {
                if rounds_played <= 7 {
                    // True exact: bid the full count of 6s, next player calls it.
                    let total = game.total_dice(&s);
                    game.apply(&mut s, Action::Open(total, 6));
                    game.apply(&mut s, Action::CallExact);
                    continue;
                }
                game.apply(&mut s, Action::Open(1, 6));
            }
            // True bid (or the false forced 1x1): the call costs one die.
            game.apply(&mut s, Action::CallLiar);
        }
        assert!(rounds_played > 24, "the scenario must exceed the old cap");
        assert_eq!(game.num_alive(&s), 1, "must end by elimination");
        assert!(game.alive(&s, s.winner));
        assert_eq!(game.returns(&s, s.winner as usize), 1.0);
    }

    /// The key is now history-aware: two states with the same hand and current
    /// bid but a different raise *path* (one seat raised twice vs once) must hash
    /// to distinct infoset and state keys — the structured fields carry the path
    /// the monotonic current bid alone omits.
    #[test]
    fn raise_path_distinguishes_keys() {
        let game = LiarsDice::new(3, 4, 6);
        let base = rolled(
            &game,
            &[[4, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0], [0, 0, 4, 0, 0, 0]],
        );
        let mut a = base.clone();
        a.qty = 3;
        a.face = 4;
        a.last_bidder = 2;
        a.turn = 0;
        a.raises_this_round = [1, 0, 2, 0, 0, 0, 0, 0];
        // Same hand, dice, bid, turn order — only the raise path differs.
        let mut b = a.clone();
        b.raises_this_round = [2, 0, 1, 0, 0, 0, 0, 0];
        for p in 0..3 {
            assert_ne!(
                game.infoset_key(&a, p),
                game.infoset_key(&b, p),
                "differing raise paths must give distinct infoset keys (seat {p})"
            );
        }
        assert_ne!(
            game.state_key(&a),
            game.state_key(&b),
            "differing raise paths must give distinct state keys"
        );
    }

    /// `round_opener` must recover the seat that opened the live round, for the
    /// forced first round and for free-open continuing rounds, after an arbitrary
    /// bid sequence — the reference point online solving rebuilds the round from.
    #[test]
    fn round_opener_recovers_the_opening_seat() {
        // Drive a real game and, at every in-round decision node, check that
        // `round_opener` returns the seat that actually opened the current round
        // (tracked independently as we play).
        for &players in &[2u8, 3, 4] {
            let game = LiarsDice::new(players, 3, 6);
            let mut rng = Rng::new(0xB1D + u64::from(players));
            for _ in 0..40 {
                let mut s = game.initial_state();
                // The first round's opener is the first live seat to act: seat 0.
                let mut expected_opener = 0u8;
                let mut bids_in_round = 0u32;
                while !game.is_terminal(&s) {
                    match game.turn(&s) {
                        Turn::Chance => {
                            let a = game.sample_chance_action(&s, &mut rng);
                            game.apply(&mut s, a);
                        }
                        Turn::Player(_) => {
                            // The opener of a free-open round is the seat to act
                            // with no bid yet standing.
                            if s.qty == 0 && bids_in_round == 0 {
                                expected_opener = s.turn;
                            }
                            assert_eq!(
                                game.round_opener(&s),
                                expected_opener,
                                "players={players} qty={} bids={bids_in_round}",
                                s.qty
                            );
                            let acts = game.legal_actions(&s);
                            let a = acts[rng.below(acts.len())];
                            let was_round = s.rounds;
                            game.apply(&mut s, a);
                            match a {
                                Action::CallLiar | Action::CallExact => {
                                    // The call may end the round (re-roll) or the
                                    // game. If a new round opened, its opener is
                                    // the seat now to act.
                                    if !game.is_terminal(&s) && s.rounds != was_round {
                                        expected_opener = s.turn;
                                        bids_in_round = 0;
                                    }
                                }
                                _ => bids_in_round += 1,
                            }
                        }
                    }
                }
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

    #[test]
    fn chance_action_sampler_matches_probability_sampler_rolls() {
        for &(players, dice, faces) in &[(2, 1, 6), (5, 5, 6), (8, 6, 6)] {
            let game = LiarsDice::new(players, dice, faces);
            let s = game.initial_state();
            for seed in 0..20 {
                let mut with_prob = Rng::new(seed);
                let mut action_only = Rng::new(seed);
                assert_eq!(
                    game.sample_chance(&s, &mut with_prob).0,
                    game.sample_chance_action(&s, &mut action_only)
                );
            }
        }
    }

    #[test]
    fn action_at_matches_legal_actions_order() {
        for &(players, dice, faces) in &[(2, 1, 6), (3, 3, 4), (5, 5, 6), (8, 6, 6)] {
            let game = LiarsDice::new(players, dice, faces);
            let mut rng = Rng::new(0xAC7100 + u64::from(players));
            for _ in 0..20 {
                let mut s = game.initial_state();
                let mut steps = 0;
                while !game.is_terminal(&s) {
                    steps += 1;
                    assert!(steps < 100_000, "random game should terminate");
                    match game.turn(&s) {
                        Turn::Chance => {
                            let a = game.sample_chance_action(&s, &mut rng);
                            game.apply(&mut s, a);
                        }
                        Turn::Player(_) => {
                            let actions = game.legal_actions(&s);
                            assert_eq!(game.num_actions(&s), actions.len());
                            for (i, &a) in actions.iter().enumerate() {
                                assert_eq!(game.action_at(&s, i), a);
                            }
                            let a = game.action_at(&s, rng.below(actions.len()));
                            game.apply(&mut s, a);
                        }
                    }
                }
            }
        }
    }
}
