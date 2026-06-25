//! Hand-crafted poker agents — no training, no CFR, pure CPU.
//!
//! The headline bot is [`PokerBot`]: tight, position-aware "ABC poker". It
//! estimates its hand's equity against the live field by Monte-Carlo (deal the
//! opponents' hole cards and the missing board a few thousand times, count wins),
//! then decides on pot odds plus a little randomized aggression and bluffing.
//! That is enough to crush a casual player and runs instantly.
//!
//! [`HoleSampler`] is the determinizer that lets the generic
//! [`solvers::Rollout`] play poker: it fills every unseen card uniformly from
//! the remaining deck. Either bot beats the always-call and random baselines
//! decisively (see `examples/bot_eval`).

use game_core::{Agent, Determinizer, Game, Rng};

use crate::cards::{self, Card};
use crate::{Action, NO_CARD, Poker, PokerState};

/// The cards `observer` can see: its own hole cards plus the public board.
/// Everything else is a candidate for sampling.
fn seen_mask(s: &PokerState, observer: usize) -> u64 {
    let mut mask = 0u64;
    for &c in s.board() {
        mask |= 1 << c;
    }
    if let Some(h) = s.hole(observer) {
        mask |= 1 << h[0];
        mask |= 1 << h[1];
    }
    mask
}

/// The deck of cards not visible to `observer`, shuffled in place into `buf`.
fn shuffled_unseen(seen: u64, buf: &mut Vec<Card>, rng: &mut Rng) {
    buf.clear();
    for c in 0..cards::NUM_CARDS as u8 {
        if seen & (1 << c) == 0 {
            buf.push(c);
        }
    }
    for i in (1..buf.len()).rev() {
        buf.swap(i, rng.below(i + 1));
    }
}

/// Determinizer for [`solvers::Rollout`]: replace every card `observer` cannot
/// see — opponents' hole cards and the undealt board — with a fresh uniform
/// draw from the remaining deck. A consistent concrete world to play out.
pub struct HoleSampler;

impl Determinizer<Poker> for HoleSampler {
    fn determinize(&self, game: &Poker, s: &mut PokerState, observer: usize, rng: &mut Rng) {
        let seen = seen_mask(s, observer);
        let mut deck = Vec::with_capacity(cards::NUM_CARDS);
        shuffled_unseen(seen, &mut deck, rng);
        // Opponents still holding cards get a fresh two-card hand from the
        // unseen deck; folded or undealt seats keep their slots. The undealt
        // board is left to the engine's chance nodes during the playout.
        let mut next = 0;
        for p in 0..game.seats() {
            if p != observer && s.hole(p).is_some() && !s.folded(p) {
                s.set_hole(p, [deck[next], deck[next + 1]]);
                next += 2;
            }
        }
    }
}

/// Personality knobs. Defaults are a solid, slightly aggressive TAG (tight-
/// aggressive) profile that beats casual players comfortably.
#[derive(Clone, Copy, Debug)]
pub struct PokerStyle {
    /// Monte-Carlo samples per equity estimate. A few thousand is plenty.
    pub samples: u32,
    /// Continue (call/raise) only when equity clears pot odds by this margin.
    pub call_margin: f64,
    /// Raise for value when equity exceeds this against the field.
    pub value_raise: f64,
    /// Probability of a bluff raise when equity is too low to continue.
    pub bluff: f64,
    /// Extra equity credited for acting in late position (per the fraction of
    /// seats still to act behind), making the bot looser on the button.
    pub position_bonus: f64,
    /// Fraction of the pot to size a value/bluff raise.
    pub raise_frac: f64,
}

impl Default for PokerStyle {
    fn default() -> Self {
        Self {
            samples: 2000,
            call_margin: 0.0,
            value_raise: 0.62,
            bluff: 0.07,
            position_bonus: 0.06,
            raise_frac: 0.75,
        }
    }
}

/// Tight-aggressive equity bot. Decides from Monte-Carlo hand equity, pot odds,
/// and position, with light randomized aggression so it isn't perfectly
/// readable. No training, no search tree — just a few thousand rollouts of the
/// board and the opponents' cards per decision.
pub struct PokerBot {
    pub style: PokerStyle,
}

impl PokerBot {
    pub fn new(style: PokerStyle) -> Self {
        Self { style }
    }

    pub fn default_bot() -> Self {
        Self {
            style: PokerStyle::default(),
        }
    }

    /// Probability this seat's hand is best at showdown against `opponents`
    /// random hands, completing the board uniformly. Ties count as a fractional
    /// win (`1/k`), so the estimate is true equity, not just outright wins.
    fn equity(&self, s: &PokerState, seat: usize, opponents: usize, rng: &mut Rng) -> f64 {
        let hole = match s.hole(seat) {
            Some(h) => h,
            None => return 0.0,
        };
        if opponents == 0 {
            return 1.0;
        }
        let seen = seen_mask(s, seat);
        let board_have = s.board().len();
        let mut deck = Vec::with_capacity(cards::NUM_CARDS);
        let mut equity = 0.0;
        for _ in 0..self.style.samples {
            shuffled_unseen(seen, &mut deck, rng);
            let mut idx = 0;
            // Complete the five-card board.
            let mut board = [NO_CARD; 5];
            board[..board_have].copy_from_slice(s.board());
            for b in board.iter_mut().take(5).skip(board_have) {
                *b = deck[idx];
                idx += 1;
            }
            let mut hero_cards = board.to_vec();
            hero_cards.push(hole[0]);
            hero_cards.push(hole[1]);
            let hero = cards::evaluate(&hero_cards);
            // Best opponent hand among `opponents` random two-card hands.
            let mut ties = 1;
            let mut beaten = false;
            for _ in 0..opponents {
                let opp = [deck[idx], deck[idx + 1]];
                idx += 2;
                let mut oc = board.to_vec();
                oc.push(opp[0]);
                oc.push(opp[1]);
                let ov = cards::evaluate(&oc);
                match ov.cmp(&hero) {
                    std::cmp::Ordering::Greater => {
                        beaten = true;
                        break;
                    }
                    std::cmp::Ordering::Equal => ties += 1,
                    std::cmp::Ordering::Less => {}
                }
            }
            if !beaten {
                equity += 1.0 / ties as f64;
            }
        }
        equity / self.style.samples as f64
    }

    fn live_opponents(&self, game: &Poker, s: &PokerState, seat: usize) -> usize {
        (0..game.seats())
            .filter(|&p| p != seat && !s.folded(p))
            .count()
    }

    /// Fraction of opponents still to act behind this seat — a cheap proxy for
    /// position (high on the button, zero in the blinds preflop after a raise).
    fn position(&self, game: &Poker, s: &PokerState, seat: usize) -> f64 {
        let n = game.seats();
        let mut behind = 0;
        let mut p = (seat + 1) % n;
        while p != seat {
            if !s.folded(p) && !s.all_in(p) {
                behind += 1;
            }
            p = (p + 1) % n;
        }
        let opp = self.live_opponents(game, s, seat).max(1);
        behind as f64 / opp as f64
    }

    fn choose(&self, game: &Poker, s: &PokerState, seat: usize, rng: &mut Rng) -> Action {
        let actions = game.legal_actions(s);
        let opponents = self.live_opponents(game, s, seat);
        let raw_equity = self.equity(s, seat, opponents, rng);
        let equity =
            (raw_equity + self.position(game, s, seat) * self.style.position_bonus).min(1.0);

        let to_call = s.to_call(seat);
        let pot = game.pot(s) as f64;
        // Pot odds: the equity a call needs to break even.
        let needed = if to_call == 0 {
            0.0
        } else {
            to_call as f64 / (pot + to_call as f64)
        };

        let can = |a: &Action| actions.contains(a);
        let raise_action = || best_raise(&actions, s, game, self.style.raise_frac);

        if to_call == 0 {
            // No bet to face: value-bet strong hands, occasionally bluff, else
            // check.
            if equity >= self.style.value_raise
                && let Some(r) = raise_action()
            {
                return r;
            }
            if rng.unit() < self.style.bluff
                && let Some(r) = raise_action()
            {
                return r;
            }
            return if can(&Action::Check) {
                Action::Check
            } else {
                fallback(&actions)
            };
        }

        // Facing a bet: raise for clear value, call when equity beats pot odds,
        // bluff-raise a little, otherwise fold.
        if equity >= self.style.value_raise
            && let Some(r) = raise_action()
        {
            return r;
        }
        if equity >= needed + self.style.call_margin {
            if can(&Action::Call) {
                return Action::Call;
            }
            if can(&Action::AllIn) && s.stack(seat) <= to_call {
                return Action::AllIn; // calling all-in
            }
        }
        if rng.unit() < self.style.bluff
            && let Some(r) = raise_action()
        {
            return r;
        }
        if can(&Action::Fold) {
            Action::Fold
        } else {
            fallback(&actions)
        }
    }
}

/// The raise/all-in from `actions` whose chip total is closest to `target`.
pub(crate) fn closest_raise(actions: &[Action], s: &PokerState, target: u32) -> Option<Action> {
    let p = s.to_act();
    let mut best: Option<(Action, u32)> = None;
    for &a in actions {
        let to = match a {
            Action::Raise(to) => to,
            Action::AllIn => s.street_bet(p) + s.stack(p),
            _ => continue,
        };
        let dist = to.abs_diff(target);
        if best.is_none_or(|(_, d)| dist < d) {
            best = Some((a, dist));
        }
    }
    best.map(|(a, _)| a)
}

/// Pick the raise closest to `frac` of the pot from the offered menu, else the
/// smallest raise/all-in available.
fn best_raise(actions: &[Action], s: &PokerState, game: &Poker, frac: f64) -> Option<Action> {
    let target = s.current_bet() + (game.pot(s) as f64 * frac).round() as u32;
    closest_raise(actions, s, target)
}

/// Last-resort legal action when the preferred one is unavailable: check or
/// call if possible, else fold, else the first legal action.
fn fallback(actions: &[Action]) -> Action {
    for pref in [Action::Check, Action::Call, Action::Fold] {
        if actions.contains(&pref) {
            return pref;
        }
    }
    actions[0]
}

impl Agent<Poker> for PokerBot {
    fn act(&self, game: &Poker, state: &PokerState, player: usize, rng: &mut Rng) -> usize {
        let desired = self.choose(game, state, player, rng);
        let actions = game.legal_actions(state);
        actions
            .iter()
            .position(|&a| a == desired)
            .unwrap_or_else(|| {
                actions
                    .iter()
                    .position(|&a| a == fallback(&actions))
                    .unwrap_or(0)
            })
    }
}

/// A pure equity check-or-fold baseline used for tests: never bluffs, never
/// raises, just calls when equity beats pot odds. Useful as a measuring stick.
pub struct EquityRollout {
    pub samples: u32,
}

impl Default for EquityRollout {
    fn default() -> Self {
        Self { samples: 1500 }
    }
}

impl Agent<Poker> for EquityRollout {
    fn act(&self, game: &Poker, s: &PokerState, player: usize, rng: &mut Rng) -> usize {
        let bot = PokerBot {
            style: PokerStyle {
                samples: self.samples,
                bluff: 0.0,
                value_raise: 2.0, // never raises
                ..PokerStyle::default()
            },
        };
        bot.act(game, s, player, rng)
    }
}

impl PokerState {
    /// Overwrite a seat's hole cards — used only by [`HoleSampler`] when
    /// determinizing a hidden world for rollouts.
    pub(crate) fn set_hole(&mut self, seat: usize, cards: [Card; 2]) {
        self.hole[seat] = cards;
    }
}

/// Always-call baseline: never folds, never raises (calls or checks). A casual
/// "calling station". Used to measure the equity bot's edge.
pub struct AlwaysCall;

impl Agent<Poker> for AlwaysCall {
    fn act(&self, game: &Poker, s: &PokerState, _player: usize, _rng: &mut Rng) -> usize {
        let actions = game.legal_actions(s);
        for pref in [Action::Check, Action::Call] {
            if let Some(i) = actions.iter().position(|&a| a == pref) {
                return i;
            }
        }
        // Forced all-in to call (stack <= to_call) or fold as last resort.
        actions
            .iter()
            .position(|&a| a == Action::AllIn)
            .unwrap_or(0)
    }
}
