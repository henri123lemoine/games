//! Per-round decomposition of Liar's Dice for offline solving.
//!
//! Every round of Liar's Dice re-rolls all hands, so given a continuation
//! value `V(dice_vector)` for the next round the rounds are independent: solving
//! "the current round" under uniform hand priors with `V` at the leaves loses
//! nothing. [`RoundSubgame`] is exactly that — one round of an inner
//! [`LiarsDice`], wrapped (not reimplemented) so the bidding and call resolution
//! are byte-for-byte the real rules, with a [`ContinuationValue`] standing in for
//! the rest of the game whenever a round ends without an outright winner.
//!
//! The continuation value is the extension point for the learned value net (the
//! Stage-B fitted-value-iteration target); [`DiceShareValue`] is the Stage-A
//! heuristic placeholder.

use game_core::hash::combine;
use game_core::{Game, Turn};

use crate::{Action, HIST_K, LdState, LiarsDice, MAX_FACES, MAX_PLAYERS};

/// Expected game return for `player` given the post-round dice vector and who
/// opens the *next* round — the continuation value that closes a
/// [`RoundSubgame`]'s leaves.
///
/// `dice_left` is the per-seat remaining dice (index = seat, `0` = eliminated);
/// `n = dice_left.len()` is the player count. `next_opener` is the seat that
/// will open the next round: continuation equity is not a pure function of the
/// dice vector, because the opener enjoys a positional effect (acting first /
/// the loser of the last round opens), so the converged value differs between
/// the same dice vector opened by different seats. The returned value is on the
/// game's outcome scale (+1 for an eventual win, `-1/(n-1)` for a loss), and the
/// values across the `n` seats must sum to ~0 for any vector — the rest of the
/// game is zero-sum.
pub trait ContinuationValue: Sync {
    fn value(&self, faces: u8, dice_left: &[u8], next_opener: usize, player: usize) -> f64;
}

/// Stage-A heuristic: a seat's continuation value is proportional to its share
/// of the remaining dice. A placeholder for the Stage-B value net (which will
/// implement this same trait), it is exactly zero-sum and respects elimination.
///
/// With one seat alive that seat wins outright (+1, others `-1/(n-1)`).
/// Otherwise a seat's win probability is its dice share `dice_left[p]/total`,
/// and its return is `win_prob - (1 - win_prob)/(n - 1)`, which sums to 0
/// across seats because the win probabilities sum to 1.
///
/// The opener has no effect on a pure dice-share heuristic, so `next_opener` is
/// ignored — only the converged [`LatticeValue`](crate::LatticeValue) captures
/// the positional effect.
pub struct DiceShareValue;

impl ContinuationValue for DiceShareValue {
    fn value(&self, _faces: u8, dice_left: &[u8], _next_opener: usize, player: usize) -> f64 {
        let n = dice_left.len();
        let total: u32 = dice_left.iter().map(|&d| u32::from(d)).sum();
        let alive = dice_left.iter().filter(|&&d| d > 0).count();
        let loser_share = -1.0 / (n as f64 - 1.0);
        if alive <= 1 {
            return if dice_left[player] > 0 {
                1.0
            } else {
                loser_share
            };
        }
        // total > 0 here: alive >= 2 means at least two seats hold a die.
        let win_prob = f64::from(dice_left[player]) / total as f64;
        win_prob - (1.0 - win_prob) / (n as f64 - 1.0)
    }
}

/// One round of Liar's Dice as a standalone [`Game`], wrapping an inner
/// [`LiarsDice`]. Bidding, call resolution, chance rolls, and the information-set
/// and state keys are delegated to the inner game (zero rule divergence); the
/// only departures are the start state (reconstructed to open the round) and the
/// terminal/return logic (the round ends as a leaf, valued by `V` when the game
/// itself has not ended).
pub struct RoundSubgame<V: ContinuationValue> {
    inner: LiarsDice,
    start: LdState,
    start_round: u16,
    value: V,
}

impl<V: ContinuationValue> RoundSubgame<V> {
    /// Build the subgame for the round that opens with `dice_left` on the table,
    /// `opener` to act, and the inner `LiarsDice` configured for `players` /
    /// `faces` (the inner game's per-seat dice count is irrelevant — the table
    /// state lives in `dice_left`, so it is passed through unchanged).
    ///
    /// `first_round` selects the opening convention exactly as
    /// [`LiarsDice::initial_state`]: the game's very first round is a forced
    /// `1×1` bid, every later round opens freely (`qty = 0`).
    ///
    /// `start_round` is the inner game's round counter at this round's start; it
    /// only matters for the round-cap adjudication far from any real play, and is
    /// used here to detect "the round we started has ended" without reading the
    /// (delegated, abstracted) keys.
    // The configuration (players/dice/faces) plus the round table state
    // (dice_left/opener/first_round/start_round) are each load-bearing and
    // independent; bundling them into a struct would only move the argument
    // list to the caller.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        players: u8,
        dice_ignored: u8,
        faces: u8,
        dice_left: [u8; MAX_PLAYERS],
        opener: u8,
        first_round: bool,
        start_round: u16,
        value: V,
    ) -> Self {
        let inner = LiarsDice::new(players, dice_ignored, faces);
        let start = round_opening_state(&inner, dice_left, opener, first_round, start_round);
        Self {
            inner,
            start,
            start_round,
            value,
        }
    }
}

/// Reconstruct the state at the *opening* of the round (chance about to roll all
/// live hands, the opener about to bid), matching the conventions of
/// [`LiarsDice::initial_state`] for the first round and of
/// [`LiarsDice::resolve_after_call`]'s next-round setup for a free open.
fn round_opening_state(
    inner: &LiarsDice,
    dice_left: [u8; MAX_PLAYERS],
    opener: u8,
    first_round: bool,
    start_round: u16,
) -> LdState {
    let mut s = inner.initial_state();
    s.dice_left = dice_left;
    s.hands = [[0; MAX_FACES]; MAX_PLAYERS];
    s.rolled = 0;
    s.turn = opener;
    s.first_round = first_round;
    s.hist = [0; HIST_K];
    s.endorsed = [0; MAX_PLAYERS];
    s.raises_this_round = [0; MAX_PLAYERS];
    s.rounds = start_round;
    s.done = false;
    s.winner = 0;
    if first_round {
        // Forced 1x1, owned by the phantom seat just before the opener, exactly
        // as `initial_state` (opener 0, last_bidder = players - 1).
        s.qty = 1;
        s.face = 1;
        s.last_bidder = prev_seat(inner.players, opener);
    } else {
        // Free open. `resolve_after_call` zeroes the bid and leaves `last_bidder`
        // carrying the previous round's bid owner; the opener then sits one step
        // past that owner in turn order. We have no previous round, so we pin
        // `last_bidder` to the live seat immediately before the opener — the
        // relative position (`rel = 1` in `infoset_key`) a fresh opener always
        // occupies, keeping the delegated key sane and config-independent.
        s.qty = 0;
        s.face = 0;
        s.last_bidder = prev_alive_seat(inner, &s, opener);
    }
    s
}

/// The seat immediately before `seat` in raw player order (the phantom owner of
/// the forced first-round bid, matching `initial_state`'s `players - 1` when
/// `opener == 0`).
fn prev_seat(players: u8, seat: u8) -> u8 {
    (seat + players - 1) % players
}

/// The nearest live seat strictly before `seat` in turn order — the seat a fresh
/// opener follows, mirroring `next_alive` walked backwards.
fn prev_alive_seat(inner: &LiarsDice, s: &LdState, seat: u8) -> u8 {
    let mut p = prev_seat(inner.players, seat);
    while s.dice_left[p as usize] == 0 {
        p = prev_seat(inner.players, p);
    }
    p
}

impl<V: ContinuationValue> Game for RoundSubgame<V> {
    type State = LdState;
    type Action = Action;

    fn num_players(&self) -> usize {
        self.inner.num_players()
    }

    fn initial_state(&self) -> LdState {
        self.start.clone()
    }

    fn turn(&self, s: &LdState) -> Turn {
        self.inner.turn(s)
    }

    fn is_terminal(&self, s: &LdState) -> bool {
        // The round we opened is over once the inner game ends it (`done`) or
        // advances past it (`rounds` ticks up in `resolve_after_call`).
        s.done || s.rounds > self.start_round
    }

    fn returns(&self, s: &LdState, player: usize) -> f64 {
        // The game itself ended this round (one seat left standing): the real
        // outcome is exact, so defer to the inner game's returns.
        if s.done && self.inner.num_alive(s) == 1 {
            return self.inner.returns(s, player);
        }
        // The round ended but the game continues — including a correct
        // `CallExact`, where no die is lost and the dice vector is unchanged.
        // `resolve_after_call` has already advanced `s.turn` to the seat that
        // opens the next round, so it is the continuation's `next_opener`. The
        // continuation value closes the leaf over the post-round dice.
        self.value.value(
            self.inner.faces,
            &s.dice_left[..self.inner.players as usize],
            s.turn as usize,
            player,
        )
    }

    fn max_return(&self) -> f64 {
        self.inner.max_return()
    }

    fn legal_actions(&self, s: &LdState) -> Vec<Action> {
        self.inner.legal_actions(s)
    }

    fn chance_outcomes(&self, s: &LdState) -> Vec<(Action, f64)> {
        self.inner.chance_outcomes(s)
    }

    fn sample_chance(&self, s: &LdState, rng: &mut game_core::Rng) -> (Action, f64) {
        self.inner.sample_chance(s, rng)
    }

    fn apply(&self, s: &mut LdState, a: Action) {
        self.inner.apply(s, a);
    }

    fn infoset_key(&self, s: &LdState, player: usize) -> u64 {
        self.inner.infoset_key(s, player)
    }

    fn state_key(&self, s: &LdState) -> Option<u64> {
        // The inner `state_key` excludes the round counter (deliberate, for the
        // full game's cross-round strategy sharing). In a *single*-round subgame
        // that aliases a round-end leaf with an in-round node: a correct
        // `CallExact` by the opener resets to `qty=0`, `rolled=0`, empty hands —
        // byte-identical to this round's *opening* node except that
        // `resolve_after_call` ticked `rounds` past `start_round`. The two are a
        // terminal leaf and a live node, so the best-response state memo must not
        // confuse them; fold the round-ended flag into the key.
        let key = self.inner.state_key(s)?;
        let round_ended = s.rounds > self.start_round || s.done;
        Some(combine(key, u64::from(round_ended)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_with(players: usize, counts: &[u8]) -> [u8; MAX_PLAYERS] {
        let mut v = [0u8; MAX_PLAYERS];
        v[..players].copy_from_slice(&counts[..players]);
        v
    }

    /// Drive every live seat's chance roll using the supplied per-seat hands (in
    /// seat order; eliminated seats are auto-rolled to the empty hand by the
    /// inner game). Returns once all hands are rolled.
    fn roll_all<V: ContinuationValue>(
        g: &RoundSubgame<V>,
        s: &mut LdState,
        hands: &[[u8; MAX_FACES]],
    ) {
        let mut next = 0usize;
        while let Turn::Chance = g.turn(s) {
            if s.dice_left[next] == 0 {
                g.apply(s, Action::Roll([0; MAX_FACES]));
            } else {
                g.apply(s, Action::Roll(hands[next]));
            }
            next += 1;
        }
    }

    #[test]
    fn dice_share_value_is_zero_sum() {
        let v = DiceShareValue;
        let cases: &[(usize, &[u8])] = &[
            (2, &[1, 1]),
            (2, &[2, 0]),
            (3, &[3, 2, 1]),
            (3, &[0, 4, 2]),
            (3, &[5, 0, 0]),
            (5, &[5, 4, 3, 2, 1]),
            (5, &[1, 0, 3, 0, 2]),
            (6, &[2, 2, 2, 2, 2, 2]),
            (4, &[0, 0, 0, 7]),
        ];
        for &(n, dl) in cases {
            // The opener never changes a zero-sum heuristic; check a couple.
            for opener in 0..n {
                let sum: f64 = (0..n).map(|p| v.value(6, dl, opener, p)).sum();
                assert!(
                    sum.abs() < 1e-9,
                    "n={n} dl={dl:?} opener={opener} sum={sum}"
                );
            }
        }
    }

    #[test]
    fn dice_share_value_one_alive_wins() {
        let v = DiceShareValue;
        let dl = &[0u8, 3, 0];
        assert_eq!(v.value(6, dl, 1, 1), 1.0);
        assert_eq!(v.value(6, dl, 1, 0), -0.5);
        assert_eq!(v.value(6, dl, 1, 2), -0.5);
    }

    #[test]
    fn first_round_re_rolls_then_opener_acts() {
        let dl = vec_with(3, &[2, 2, 2]);
        let g = RoundSubgame::new(3, 2, 6, dl, 0, true, 1, DiceShareValue);
        let mut s = g.initial_state();
        // First round opens at the forced 1x1, owned by the phantom prior seat.
        assert_eq!(s.current_bid(), (1, 1));
        assert_eq!(s.last_bidder(), 2);
        let mut chance_steps = 0;
        while let Turn::Chance = g.turn(&s) {
            let o = g.chance_outcomes(&s);
            g.apply(&mut s, o[0].0);
            chance_steps += 1;
        }
        assert_eq!(chance_steps, 3, "all three live seats roll");
        assert_eq!(g.turn(&s), Turn::Player(0));
        assert!(!g.is_terminal(&s));
    }

    #[test]
    fn free_open_round_re_rolls_with_eliminated_seat_skipped() {
        // Seat 1 eliminated: only seats 0 and 2 are live.
        let dl = vec_with(3, &[2, 0, 2]);
        let g = RoundSubgame::new(3, 2, 6, dl, 2, false, 4, DiceShareValue);
        let mut s = g.initial_state();
        assert_eq!(s.current_bid(), (0, 0), "free open: no standing bid");
        // Opener is seat 2; the relative-position anchor is the live seat before
        // it (seat 0), so the delegated infoset key stays sane.
        assert_eq!(s.last_bidder(), 0);
        let mut rolled_seats = 0;
        while let Turn::Chance = g.turn(&s) {
            let seat = rolled_seats;
            let outcome = g.chance_outcomes(&s);
            // Eliminated seat 1 has a single forced empty roll.
            if seat == 1 {
                assert_eq!(outcome.len(), 1);
            } else {
                assert!(outcome.len() > 1);
            }
            g.apply(&mut s, outcome[0].0);
            rolled_seats += 1;
        }
        assert_eq!(rolled_seats, 3, "every seat index advances `rolled`");
        assert_eq!(g.turn(&s), Turn::Player(2));
    }

    #[test]
    fn round_end_without_game_end_is_terminal_with_continuation_value() {
        // 3 players, 2 dice each: a lost die ends the round but not the game.
        let dl = vec_with(3, &[2, 2, 2]);
        let g = RoundSubgame::new(3, 2, 6, dl, 0, false, 4, DiceShareValue);
        let mut s = g.initial_state();
        // Hands: nobody holds a 3. Seat 0 opens an over-bid on 3s; seat 1 calls
        // liar and is right, so seat 0 (the bid owner) loses a die.
        let hands = [[2, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0]];
        roll_all(&g, &mut s, &hands);
        assert!(!g.is_terminal(&s));
        g.apply(&mut s, Action::Open(2, 3)); // false: zero 3s exist
        assert!(!g.is_terminal(&s));
        g.apply(&mut s, Action::CallLiar);
        assert!(g.is_terminal(&s), "round ended -> subgame terminal");
        // Seat 0 lost the die; the game continues (all still alive).
        let expect = vec_with(3, &[1, 2, 2]);
        assert_eq!(&s.dice_left()[..3], &expect[..3]);
        let cv = DiceShareValue;
        let next_opener = s.turn();
        for p in 0..3 {
            assert_eq!(
                g.returns(&s, p),
                cv.value(6, &expect[..3], next_opener, p),
                "returns delegate to continuation value for seat {p}"
            );
        }
    }

    #[test]
    fn game_ending_call_matches_real_liars_dice_returns() {
        // 2 players, 1 die each: a lost die is elimination, so the call ends the
        // game and returns must equal the inner game's real returns.
        let dl = vec_with(2, &[1, 1]);
        let g = RoundSubgame::new(2, 1, 6, dl, 0, false, 4, DiceShareValue);
        let mut s = g.initial_state();
        // Both hold a 2; nobody holds a 3.
        let hands = [[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]];
        roll_all(&g, &mut s, &hands);
        g.apply(&mut s, Action::Open(1, 3)); // false: zero 3s
        g.apply(&mut s, Action::CallLiar); // seat 1 calls, right -> seat 0 out
        assert!(g.is_terminal(&s));
        assert_eq!(g.inner.num_alive(&s), 1);
        // Real game-ending returns: winner +1, loser -1/(n-1).
        assert_eq!(g.returns(&s, 1), 1.0);
        assert_eq!(g.returns(&s, 0), -1.0);
        assert_eq!(g.returns(&s, 1), g.inner.returns(&s, 1));
        assert_eq!(g.returns(&s, 0), g.inner.returns(&s, 0));
    }

    #[test]
    fn correct_call_exact_ends_round_with_unchanged_dice() {
        // A correct CallExact loses no die: the round ends, dice unchanged, and
        // the continuation value is over the unchanged vector.
        let dl = vec_with(3, &[2, 2, 2]);
        let g = RoundSubgame::new(3, 2, 6, dl, 0, false, 4, DiceShareValue);
        let mut s = g.initial_state();
        // Exactly one 3 across all hands.
        let hands = [[2, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0], [2, 0, 0, 0, 0, 0]];
        roll_all(&g, &mut s, &hands);
        g.apply(&mut s, Action::Open(1, 3)); // claim exactly one 3
        g.apply(&mut s, Action::CallExact); // correct: there is exactly one 3
        assert!(g.is_terminal(&s));
        assert!(!s.done, "game not over; only the round ended");
        let unchanged = vec_with(3, &[2, 2, 2]);
        assert_eq!(&s.dice_left()[..3], &unchanged[..3]);
        let cv = DiceShareValue;
        let next_opener = s.turn();
        for p in 0..3 {
            assert_eq!(
                g.returns(&s, p),
                cv.value(6, &unchanged[..3], next_opener, p)
            );
        }
    }

    /// Playing the same fixed hands and the same action sequence through one
    /// round must be byte-for-byte identical between `RoundSubgame` and a real
    /// `LiarsDice` (the subgame wraps it). We compare the per-seat infoset keys
    /// at every decision node and the underlying state key. The subgame's state
    /// key folds in a round-ended flag (so a round-end leaf never aliases an
    /// in-round node in the best-response memo), so we compare the *inner* key,
    /// recovering the flag the subgame would add.
    #[test]
    fn one_round_is_rule_identical_to_real_liars_dice() {
        let players = 3u8;
        let faces = 6u8;
        let real = LiarsDice::new(players, 2, faces);
        // Real first-round opening state (forced 1x1).
        let mut rs = real.initial_state();
        let dl = rs.dice_left;
        let g = RoundSubgame::new(players, 2, faces, dl, 0, true, 1, DiceShareValue);
        let mut ss = g.initial_state();

        // The subgame state key is the inner key with the round-ended flag
        // folded in; in-round states carry `false`.
        let expect_key =
            |s: &LdState, ended: bool| real.state_key(s).map(|k| combine(k, u64::from(ended)));

        // Start states agree on every delegated key (in-round: flag false).
        assert_eq!(expect_key(&rs, false), g.state_key(&ss));

        let hands = [[1, 1, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0]];
        // Identical chance rolls.
        let mut idx = 0;
        while let Turn::Chance = real.turn(&rs) {
            let a = if rs.dice_left[idx] == 0 {
                Action::Roll([0; MAX_FACES])
            } else {
                Action::Roll(hands[idx])
            };
            real.apply(&mut rs, a);
            g.apply(&mut ss, a);
            idx += 1;
        }
        assert_eq!(expect_key(&rs, false), g.state_key(&ss));

        // A fixed bidding line ending in a call within this one round.
        let line = [
            Action::RaiseQuantity, // 2x1 (raise the forced 1x1)
            Action::RaiseFace,     // 2x2
            Action::CallLiar,      // resolve
        ];
        for a in line {
            // Decision-node parity: same legal actions and same keys for all.
            assert_eq!(real.legal_actions(&rs), g.legal_actions(&ss));
            for p in 0..players as usize {
                assert_eq!(
                    real.infoset_key(&rs, p),
                    g.infoset_key(&ss, p),
                    "infoset key mismatch for seat {p} before {a:?}"
                );
            }
            real.apply(&mut rs, a);
            g.apply(&mut ss, a);
        }
        // The call (a die-loss) ended the round: the subgame marks the leaf as
        // round-ended, while the underlying inner state matches the real game.
        assert!(g.is_terminal(&ss));
        assert_eq!(expect_key(&rs, true), g.state_key(&ss));
    }
}
