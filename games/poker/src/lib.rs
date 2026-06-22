//! No-Limit Texas Hold'em (2–9 seats, 6-max by default) as a
//! [`game_core::Game`]. One game is one hand: chance deals the hole cards and
//! the board card by card, players act in betting rounds (fold / check / call /
//! bet / raise / all-in), and `returns` is each seat's net chip change for the
//! hand in big blinds — the natural zero-sum poker scale, so an arena's mean
//! return is directly the bb/hand win rate.
//!
//! A seat's information set is its two hole cards plus the public board and
//! betting history; the rest of the deck is hidden, so the determinized-rollout
//! bot (see [`agents`]) fills in opponents' cards and the undealt board to
//! estimate equity. Dealing one card per chance node keeps the chance space
//! enumerable (≤ 52 outcomes per node) and exact.

use game_core::hash::{combine, splitmix64};
use game_core::{Game, Turn};

pub mod agents;
pub mod cards;
mod ui;

pub use agents::{AlwaysCall, EquityRollout, HoleSampler, PokerBot, PokerStyle};
pub use cards::{Card, Category, HandRank, card_str, evaluate, parse_card};

/// Seats supported. Two for heads-up, up to nine for a full ring.
pub const MAX_SEATS: usize = 9;

/// Raise sizes a player may choose, as fractions of the pot. A small finite
/// menu keeps the game tree manageable and the rollout bot's per-candidate
/// budget meaningful, while still covering the sizes casual play uses (a call,
/// a half/full/over-pot bet, and a shove).
const POT_FRACTIONS: [f64; 3] = [0.5, 1.0, 2.0];

/// Sentinel for an undealt or mucked hole card.
const NO_CARD: Card = 0xFF;

/// Returns are reported in big blinds; this fixed-point scale keeps split-pot
/// and integer-chip math exact through the f64 conversion in [`Game::returns`]
/// (equal payoffs compare exactly, which the arena's tie detection relies on).
const BB_SCALE: i64 = 1000;

/// Which betting round we are in. The board is dealt up to the matching count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Street {
    Preflop,
    Flop,
    Turn,
    River,
    Showdown,
}

impl Street {
    fn board_cards(self) -> usize {
        match self {
            Street::Preflop => 0,
            Street::Flop => 3,
            Street::Turn => 4,
            Street::River | Street::Showdown => 5,
        }
    }
    fn next(self) -> Street {
        match self {
            Street::Preflop => Street::Flop,
            Street::Flop => Street::Turn,
            Street::Turn => Street::River,
            Street::River | Street::Showdown => Street::Showdown,
        }
    }
}

/// A player or chance action. A `Raise(to)` names the seat's new total street
/// commitment in chips (the "to" amount), so the offered menu is self-describing
/// and the state never has to reconstruct sizing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Fold,
    Check,
    Call,
    Raise(u32),
    AllIn,
    /// Chance: deal `card` to the next undealt position.
    Deal(Card),
}

#[derive(Clone)]
pub struct PokerState {
    hole: [[Card; 2]; MAX_SEATS],
    board: [Card; 5],
    board_len: u8,
    stack: [u32; MAX_SEATS],
    /// Chips each seat has put in the pot across all streets this hand.
    committed: [u32; MAX_SEATS],
    /// Chips each seat has put in *this street* (for matching the current bet).
    street_bet: [u32; MAX_SEATS],
    folded: [bool; MAX_SEATS],
    all_in: [bool; MAX_SEATS],
    /// Net chip result per seat once `done`, in big blinds × [`BB_SCALE`].
    payoff: [i64; MAX_SEATS],
    street: Street,
    to_act: u8,
    /// Highest street commitment any seat has made — the amount to match.
    current_bet: u32,
    /// Size of the last legal raise increment, for min-raise enforcement.
    last_raise: u32,
    /// Whether each seat has acted since the last bet/raise this street. The
    /// round closes once every live, non-all-in seat has acted and matched the
    /// current bet; a raise clears this for everyone but the raiser.
    acted: [bool; MAX_SEATS],
    /// All but one seat are all-in: deal the remaining board with no betting.
    run_out: bool,
    /// 52-bit set of cards already dealt, so chance never repeats one.
    dealt: u64,
    /// Next undealt position: 0..2*seats are hole cards, then board.
    deal_idx: u8,
    button: u8,
    done: bool,
}

/// The table parameters for a hand. Stacks and blinds are in chips; the natural
/// unit for [`Game::returns`] is the big blind.
pub struct Poker {
    pub seats: u8,
    pub starting_stack: u32,
    pub small_blind: u32,
    pub big_blind: u32,
    pub button: u8,
}

impl Poker {
    pub fn new(seats: u8) -> Self {
        assert!(
            (2..=MAX_SEATS as u8).contains(&seats),
            "poker supports 2..=9 seats"
        );
        Self {
            seats,
            starting_stack: 200,
            small_blind: 1,
            big_blind: 2,
            button: 0,
        }
    }

    pub fn with_stack(mut self, stack: u32) -> Self {
        assert!(stack >= self.big_blind, "a stack must cover the big blind");
        self.starting_stack = stack;
        self
    }

    pub fn with_blinds(mut self, sb: u32, bb: u32) -> Self {
        assert!(bb > 0 && sb <= bb, "need 0 < small blind <= big blind");
        self.small_blind = sb;
        self.big_blind = bb;
        self
    }

    pub fn with_button(mut self, button: u8) -> Self {
        assert!(button < self.seats, "button seat out of range");
        self.button = button;
        self
    }

    pub fn seats(&self) -> usize {
        self.seats as usize
    }

    fn next_seat(&self, from: u8) -> u8 {
        (from + 1) % self.seats
    }

    /// Next seat after `from` that can still act (live, with chips behind).
    fn next_to_act(&self, s: &PokerState, from: u8) -> Option<u8> {
        let mut p = self.next_seat(from);
        for _ in 0..self.seats {
            if !s.folded[p as usize] && !s.all_in[p as usize] {
                return Some(p);
            }
            p = self.next_seat(p);
        }
        None
    }

    fn live_count(&self, s: &PokerState) -> usize {
        (0..self.seats()).filter(|&p| !s.folded[p]).count()
    }

    /// Live seats that still have chips to act with.
    fn actionable_count(&self, s: &PokerState) -> usize {
        (0..self.seats())
            .filter(|&p| !s.folded[p] && !s.all_in[p])
            .count()
    }

    pub fn pot(&self, s: &PokerState) -> u32 {
        s.committed[..self.seats()].iter().sum()
    }

    /// Total deals (hole + board) that must precede whatever happens next.
    fn deals_needed(&self, s: &PokerState) -> usize {
        let board = if s.run_out { 5 } else { s.street.board_cards() };
        2 * self.seats() + board
    }
}

impl PokerState {
    pub fn board(&self) -> &[Card] {
        &self.board[..self.board_len as usize]
    }
    pub fn hole(&self, seat: usize) -> Option<[Card; 2]> {
        let h = self.hole[seat];
        (h[0] != NO_CARD).then_some(h)
    }
    pub fn stack(&self, seat: usize) -> u32 {
        self.stack[seat]
    }
    pub fn committed(&self, seat: usize) -> u32 {
        self.committed[seat]
    }
    pub fn street_bet(&self, seat: usize) -> u32 {
        self.street_bet[seat]
    }
    pub fn folded(&self, seat: usize) -> bool {
        self.folded[seat]
    }
    pub fn all_in(&self, seat: usize) -> bool {
        self.all_in[seat]
    }
    pub fn street(&self) -> Street {
        self.street
    }
    pub fn to_act(&self) -> usize {
        self.to_act as usize
    }
    pub fn button(&self) -> usize {
        self.button as usize
    }
    pub fn current_bet(&self) -> u32 {
        self.current_bet
    }
    pub fn done(&self) -> bool {
        self.done
    }
    pub fn payoff_bb(&self, seat: usize) -> f64 {
        self.payoff[seat] as f64 / BB_SCALE as f64
    }
    /// Chips the seat must add to call the current bet.
    pub fn to_call(&self, seat: usize) -> u32 {
        self.current_bet.saturating_sub(self.street_bet[seat])
    }

    #[cfg(test)]
    pub(crate) fn test_set_stack(&mut self, seat: usize, chips: u32) {
        self.stack[seat] = chips;
        self.all_in[seat] = chips == 0;
    }

    #[cfg(test)]
    pub(crate) fn debug_last_raise(&self) -> u32 {
        self.last_raise
    }
}

impl Game for Poker {
    type State = PokerState;
    type Action = Action;

    fn num_players(&self) -> usize {
        self.seats()
    }

    /// Returns are in big blinds; the largest swing is one seat winning every
    /// other seat's whole starting stack.
    fn max_return(&self) -> f64 {
        (self.starting_stack as f64 / self.big_blind as f64) * (self.seats() as f64 - 1.0)
    }

    fn initial_state(&self) -> PokerState {
        let n = self.seats();
        let mut stack = [0u32; MAX_SEATS];
        for s in stack.iter_mut().take(n) {
            *s = self.starting_stack;
        }
        let mut s = PokerState {
            hole: [[NO_CARD; 2]; MAX_SEATS],
            board: [NO_CARD; 5],
            board_len: 0,
            stack,
            committed: [0; MAX_SEATS],
            street_bet: [0; MAX_SEATS],
            folded: [false; MAX_SEATS],
            all_in: [false; MAX_SEATS],
            payoff: [0; MAX_SEATS],
            street: Street::Preflop,
            to_act: 0,
            current_bet: 0,
            last_raise: self.big_blind,
            acted: [false; MAX_SEATS],
            run_out: false,
            dealt: 0,
            deal_idx: 0,
            button: self.button,
            done: false,
        };
        self.post_blinds(&mut s);
        s
    }

    fn turn(&self, s: &PokerState) -> Turn {
        if s.done {
            return Turn::Player(0);
        }
        if (s.deal_idx as usize) < self.deals_needed(s) {
            Turn::Chance
        } else {
            Turn::Player(s.to_act as usize)
        }
    }

    fn is_terminal(&self, s: &PokerState) -> bool {
        s.done
    }

    fn returns(&self, s: &PokerState, player: usize) -> f64 {
        s.payoff[player] as f64 / BB_SCALE as f64
    }

    fn chance_outcomes(&self, s: &PokerState) -> Vec<(Action, f64)> {
        let remaining: Vec<Card> = (0..cards::NUM_CARDS as u8)
            .filter(|&c| s.dealt & (1u64 << c) == 0)
            .collect();
        let p = 1.0 / remaining.len() as f64;
        remaining
            .into_iter()
            .map(|c| (Action::Deal(c), p))
            .collect()
    }

    fn legal_actions(&self, s: &PokerState) -> Vec<Action> {
        let mut acts = Vec::new();
        let p = s.to_act as usize;
        let to_call = s.to_call(p);
        let stack = s.stack[p];
        if to_call > 0 {
            acts.push(Action::Fold);
            if stack > to_call {
                acts.push(Action::Call);
            }
        } else {
            acts.push(Action::Check);
        }
        // Raises need chips beyond a call and an opponent who can still respond.
        let can_aggress = stack > to_call && self.actionable_count(s) > 1;
        let max_to = s.street_bet[p] + stack;
        if can_aggress {
            let min_to = (s.current_bet + s.last_raise).max(self.big_blind);
            // Sized raises exist only when a full min-raise still leaves the
            // seat with chips; otherwise the all-in below is the only raise.
            if min_to < max_to {
                for &frac in &POT_FRACTIONS {
                    let target = s.current_bet + ((self.pot(s) as f64) * frac).round() as u32;
                    let to = target.clamp(min_to, max_to);
                    if to < max_to && !acts.contains(&Action::Raise(to)) {
                        acts.push(Action::Raise(to));
                    }
                }
            }
        }
        // All-in is always available while the seat has chips (it is the call
        // when a call would exhaust the stack, and the shove otherwise).
        if stack > 0 && (to_call == 0 || stack <= to_call || can_aggress) {
            acts.push(Action::AllIn);
        }
        acts
    }

    fn apply(&self, s: &mut PokerState, action: Action) {
        match action {
            Action::Deal(card) => self.apply_deal(s, card),
            other => self.apply_bet(s, other),
        }
    }

    fn infoset_key(&self, s: &PokerState, player: usize) -> u64 {
        let mut k = splitmix64(player as u64 + 0xA5);
        k = combine(
            k,
            s.hole[player][0] as u64 | (s.hole[player][1] as u64) << 8,
        );
        for &c in s.board() {
            k = combine(k, c as u64);
        }
        let fields = (s.street as u64) << 40
            | (s.to_act as u64) << 32
            | (s.current_bet as u64) << 16
            | s.button as u64;
        k = combine(k, fields);
        for p in 0..self.seats() {
            k = combine(
                k,
                s.street_bet[p] as u64
                    | (s.committed[p] as u64) << 20
                    | (s.folded[p] as u64) << 40
                    | (s.all_in[p] as u64) << 41,
            );
        }
        k
    }

    fn action_id(&self, action: &Action) -> u64 {
        match action {
            Action::Fold => 1,
            Action::Check => 2,
            Action::Call => 3,
            Action::AllIn => 4,
            Action::Raise(to) => 5 + *to as u64,
            Action::Deal(c) => 1 << 32 | *c as u64,
        }
    }
}

impl Poker {
    fn post_blinds(&self, s: &mut PokerState) {
        let n = self.seats() as u8;
        // Heads-up: the button posts the small blind and acts first preflop.
        let (sb_seat, bb_seat) = if n == 2 {
            (s.button, self.next_seat(s.button))
        } else {
            let sb = self.next_seat(s.button);
            (sb, self.next_seat(sb))
        };
        self.put(s, sb_seat, self.small_blind);
        self.put(s, bb_seat, self.big_blind);
        s.current_bet = self.big_blind;
        s.last_raise = self.big_blind;
        s.to_act = self.next_to_act(s, bb_seat).unwrap_or(bb_seat);
    }

    /// Move `amount` (capped by stack) into the pot, flagging all-in if it
    /// empties the stack.
    fn put(&self, s: &mut PokerState, seat: u8, amount: u32) {
        let p = seat as usize;
        let amt = amount.min(s.stack[p]);
        s.stack[p] -= amt;
        s.committed[p] += amt;
        s.street_bet[p] += amt;
        if s.stack[p] == 0 && amt > 0 {
            s.all_in[p] = true;
        }
    }

    fn apply_deal(&self, s: &mut PokerState, card: Card) {
        debug_assert!(s.dealt & (1u64 << card) == 0, "dealt a repeated card");
        s.dealt |= 1u64 << card;
        let idx = s.deal_idx as usize;
        let n = self.seats();
        if idx < 2 * n {
            // Two cards per seat, dealt round by round starting left of button.
            let round = idx / n;
            let seat = (self.next_seat(s.button) as usize + (idx % n)) % n;
            s.hole[seat][round] = card;
        } else {
            s.board[s.board_len as usize] = card;
            s.board_len += 1;
        }
        s.deal_idx += 1;
        // A board run-out settles the moment the river is complete.
        if s.run_out && s.board_len == 5 {
            self.settle(s);
        }
    }

    fn apply_bet(&self, s: &mut PokerState, action: Action) {
        let p = s.to_act;
        let pi = p as usize;
        let prev_bet = s.current_bet;
        match action {
            Action::Fold => s.folded[pi] = true,
            Action::Check => {}
            Action::Call => self.put(s, p, s.to_call(pi)),
            Action::AllIn => self.put(s, p, s.stack[pi]),
            Action::Raise(to) => self.put(s, p, to - s.street_bet[pi]),
            Action::Deal(_) => unreachable!("deal handled in apply_deal"),
        }
        s.acted[pi] = true;
        let raised = s.street_bet[pi] > prev_bet;
        if raised {
            // A bet/raise reopens the action: everyone else must respond again.
            // (Reopening on any increase, even a sub-min all-in, is always safe —
            // it never lets a seat skip a legal response — and casual play never
            // hinges on the sub-min-raise exception.)
            s.last_raise = (s.street_bet[pi] - prev_bet).max(self.big_blind);
            s.current_bet = s.street_bet[pi];
            for q in 0..self.seats() {
                if q != pi && !s.folded[q] && !s.all_in[q] {
                    s.acted[q] = false;
                }
            }
        }
        self.advance_action(s);
    }

    /// True once every live, non-all-in seat has acted this street and matched
    /// the current bet — i.e. the betting round is complete.
    fn round_closed(&self, s: &PokerState) -> bool {
        (0..self.seats())
            .all(|p| s.folded[p] || s.all_in[p] || (s.acted[p] && s.street_bet[p] == s.current_bet))
    }

    /// After an action, move to the next actor, the next street, or settlement.
    fn advance_action(&self, s: &mut PokerState) {
        if self.live_count(s) <= 1 {
            self.settle(s);
            return;
        }
        if self.round_closed(s) {
            self.close_street(s);
            return;
        }
        match self.next_to_act(s, s.to_act) {
            Some(next) => s.to_act = next,
            None => self.close_street(s),
        }
    }

    /// End the betting round: clear street bets, then advance the board or
    /// settle. If at most one seat can still act, flag a no-betting run-out.
    fn close_street(&self, s: &mut PokerState) {
        for p in 0..self.seats() {
            s.street_bet[p] = 0;
            s.acted[p] = false;
        }
        s.current_bet = 0;
        s.last_raise = self.big_blind;
        if s.street == Street::River {
            self.settle(s);
            return;
        }
        if self.actionable_count(s) <= 1 {
            s.run_out = true;
            return;
        }
        s.street = s.street.next();
        s.to_act = self.next_to_act(s, s.button).unwrap_or(s.button);
    }

    /// Award the pot (with side pots) and record net payoffs in big blinds.
    fn settle(&self, s: &mut PokerState) {
        let n = self.seats();
        let live: Vec<usize> = (0..n).filter(|&p| !s.folded[p]).collect();
        let mut won = [0u32; MAX_SEATS];
        if live.len() == 1 {
            won[live[0]] = self.pot(s);
        } else {
            self.award_side_pots(s, &live, &mut won);
        }
        let bb = self.big_blind as i64;
        for (p, &w) in won.iter().enumerate().take(n) {
            s.payoff[p] = (w as i64 - s.committed[p] as i64) * BB_SCALE / bb;
        }
        s.done = true;
        s.street = Street::Showdown;
    }

    /// Split the pot into side pots by commitment level; each layer goes to the
    /// best live hand(s) eligible for it.
    fn award_side_pots(&self, s: &PokerState, live: &[usize], won: &mut [u32; MAX_SEATS]) {
        let n = self.seats();
        let mut levels: Vec<u32> = (0..n).map(|p| s.committed[p]).filter(|&c| c > 0).collect();
        levels.sort_unstable();
        levels.dedup();
        let ranks: Vec<Option<HandRank>> = (0..n)
            .map(|p| {
                (!s.folded[p]).then(|| {
                    let mut seven = s.board().to_vec();
                    seven.extend_from_slice(&s.hole[p]);
                    evaluate(&seven)
                })
            })
            .collect();
        let mut prev = 0u32;
        for &level in &levels {
            let slice = level - prev;
            let contributors = (0..n).filter(|&p| s.committed[p] >= level).count() as u32;
            let pot = slice * contributors;
            let eligible: Vec<usize> = live
                .iter()
                .copied()
                .filter(|&p| s.committed[p] >= level)
                .collect();
            if let Some(best) = eligible.iter().map(|&p| ranks[p].unwrap()).max() {
                let winners: Vec<usize> = eligible
                    .into_iter()
                    .filter(|&p| ranks[p].unwrap() == best)
                    .collect();
                let share = pot / winners.len() as u32;
                // Odd chips go to the earliest seats by index — a fixed, minor
                // convention (a chip or two).
                let mut remainder = pot - share * winners.len() as u32;
                for &w in &winners {
                    won[w] += share;
                    if remainder > 0 {
                        won[w] += 1;
                        remainder -= 1;
                    }
                }
            }
            prev = level;
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod adversarial_tests;
