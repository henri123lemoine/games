//! AlphaBeta behaviors the chess suite cannot reach: non-alternating turns
//! (the same-mover branch of `child_value`), margin/fractional terminal
//! returns, and the preference for faster wins.

use game_core::{Eval, Game, NoSpec, Turn};
use solvers::AlphaBeta;

struct Zero;
impl<G: Game> Eval<G> for Zero {
    fn eval(&self, _game: &G, _state: &G::State, _player: usize) -> f64 {
        0.0
    }
}

/// P0 picks a row, P1 a column; the payoff is a fractional margin to P0.
struct MarginDuel;

const MARGINS: [[f64; 2]; 3] = [[0.30, 0.80], [0.55, 0.60], [-0.20, 0.90]];

impl Game for MarginDuel {
    type State = (Option<u8>, Option<u8>);
    type Action = u8;

    fn initial_state(&self) -> Self::State {
        (None, None)
    }
    fn turn(&self, s: &Self::State) -> Turn {
        Turn::Player(s.0.is_some() as usize)
    }
    fn is_terminal(&self, s: &Self::State) -> bool {
        s.1.is_some()
    }
    fn returns(&self, s: &Self::State, player: usize) -> f64 {
        let m = MARGINS[s.0.unwrap() as usize][s.1.unwrap() as usize];
        if player == 0 { m } else { -m }
    }
    fn legal_actions(&self, s: &Self::State) -> Vec<u8> {
        if s.0.is_none() {
            vec![0, 1, 2]
        } else {
            vec![0, 1]
        }
    }
    fn chance_outcomes(&self, _s: &Self::State) -> Vec<(u8, f64)> {
        Vec::new()
    }
    fn apply(&self, s: &mut Self::State, a: u8) {
        if s.0.is_none() {
            s.0 = Some(a);
        } else {
            s.1 = Some(a);
        }
    }
    fn infoset_key(&self, s: &Self::State, _player: usize) -> u64 {
        self.state_key(s).unwrap()
    }
    fn state_key(&self, s: &Self::State) -> Option<u64> {
        Some((s.0.map(|r| r as u64 + 1).unwrap_or(0) << 8) | s.1.map(|c| c as u64 + 1).unwrap_or(0))
    }
}

/// P0 moves twice in a row, then P1 closes; payoffs make the minimax-correct
/// opening differ from the one a sign-flip-per-move search would pick.
struct DoubleMove;

const PAY: [[[f64; 2]; 2]; 2] = [
    [[0.0, 1.0], [1.0, -1.0]], // a = 0: v = max(min(0,1), min(1,-1)) = 0
    [[-1.0, 1.0], [1.0, 1.0]], // a = 1: v = max(min(-1,1), min(1,1)) = 1
];

impl Game for DoubleMove {
    type State = Vec<u8>;
    type Action = u8;

    fn initial_state(&self) -> Self::State {
        Vec::new()
    }
    fn turn(&self, s: &Self::State) -> Turn {
        Turn::Player((s.len() == 2) as usize)
    }
    fn is_terminal(&self, s: &Self::State) -> bool {
        s.len() == 3
    }
    fn returns(&self, s: &Self::State, player: usize) -> f64 {
        let v = PAY[s[0] as usize][s[1] as usize][s[2] as usize];
        if player == 0 { v } else { -v }
    }
    fn legal_actions(&self, _s: &Self::State) -> Vec<u8> {
        vec![0, 1]
    }
    fn chance_outcomes(&self, _s: &Self::State) -> Vec<(u8, f64)> {
        Vec::new()
    }
    fn apply(&self, s: &mut Self::State, a: u8) {
        s.push(a);
    }
    fn infoset_key(&self, s: &Self::State, _player: usize) -> u64 {
        s.iter().fold(1, |k, &b| (k << 2) | (b as u64 + 1))
    }
    fn state_key(&self, s: &Self::State) -> Option<u64> {
        Some(s.iter().fold(1, |k, &b| (k << 2) | (b as u64 + 1)))
    }
}

/// The mover can win now or delay and win later; both lines win, so only the
/// faster-win preference separates them. The delaying action is listed first
/// so ordering alone cannot produce the right answer.
struct DelayedWin;

impl Game for DelayedWin {
    /// (plies elapsed, won?)
    type State = (u8, bool);
    type Action = u8;

    fn initial_state(&self) -> Self::State {
        (0, false)
    }
    fn turn(&self, _s: &Self::State) -> Turn {
        Turn::Player(0)
    }
    fn is_terminal(&self, s: &Self::State) -> bool {
        s.1 || s.0 >= 4
    }
    fn returns(&self, s: &Self::State, player: usize) -> f64 {
        let v = if s.1 { 1.0 } else { 0.0 };
        if player == 0 { v } else { -v }
    }
    fn legal_actions(&self, _s: &Self::State) -> Vec<u8> {
        vec![0, 1] // 0 = delay, 1 = win now
    }
    fn chance_outcomes(&self, _s: &Self::State) -> Vec<(u8, f64)> {
        Vec::new()
    }
    fn apply(&self, s: &mut Self::State, a: u8) {
        s.0 += 1;
        if a == 1 {
            s.1 = true;
        }
    }
    fn infoset_key(&self, s: &Self::State, _player: usize) -> u64 {
        (s.0 as u64) << 1 | s.1 as u64
    }
    fn state_key(&self, s: &Self::State) -> Option<u64> {
        Some((s.0 as u64) << 1 | s.1 as u64)
    }
}

#[test]
fn margin_returns_pick_the_best_guaranteed_margin() {
    // Row maximin: min over columns is [0.30, 0.55, -0.20] — row 1 wins,
    // even though row 2 holds the single largest entry (0.90).
    let bot = AlphaBeta::new(2, Zero, NoSpec);
    assert_eq!(bot.best_action(&MarginDuel, &(None, None)), 1);
    // And from P1's seat: facing row 1, P1 prefers the smaller margin (0.55).
    assert_eq!(bot.best_action(&MarginDuel, &(Some(1), None)), 0);
}

#[test]
fn non_alternating_turns_keep_perspective_straight() {
    let bot = AlphaBeta::new(3, Zero, NoSpec);
    // A search that sign-flips per move (instead of per mover change) values
    // a=0 at 0 ≥ a=1 at -1 and opens wrong.
    assert_eq!(bot.best_action(&DoubleMove, &Vec::new()), 1);
    assert_eq!(bot.best_action(&DoubleMove, &vec![1]), 1);
}

#[test]
fn faster_wins_are_preferred() {
    let bot = AlphaBeta::new(4, Zero, NoSpec);
    assert_eq!(
        bot.best_action(&DelayedWin, &(0, false)),
        1,
        "both actions win eventually; only mate-distance scoring prefers now"
    );
}
