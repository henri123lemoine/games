//! Pente as a [`game_core::Game`].
//!
//! Square board of configurable size (default 13), Black is player 0 and moves
//! first. Perfect information, no chance nodes. Rules:
//!
//! * A move places a stone on an empty intersection; players alternate.
//! * **Tournament first move**: Black's opening stone must be the center
//!   intersection (the sole legal action at the empty board).
//! * **Custodial capture**: after a placement, each of the eight directions is
//!   probed for the pattern `[YOU][OPP][OPP][YOU]` starting at the placed
//!   stone — exactly two flanked opponent stones bracketed by your own. Those
//!   two stones are removed and the mover's pair count rises by one. A single
//!   move can capture in several directions at once. Capture happens *only* on
//!   the flanking (placing) move: moving *into* a bracket
//!   (`[OPP][YOU][YOU][OPP]`) is safe, so placement never removes the stone
//!   just played.
//! * **Win**: five (or more) of your stones in a line — horizontal, vertical,
//!   or either diagonal — *or* five captured pairs (ten enemy stones).
//! * **Draw**: the board fills with no winner.
//!
//! Five-in-a-row and the fifth pair are both checked only against the stone
//! just placed (a line through it) and the mover's running pair count, so
//! terminal detection is O(1) in the board after each move and recorded on the
//! state.

pub mod encode;
mod knowledge;
pub mod solver;
mod ui;

pub use encode::PenteEncoder;
pub use knowledge::{PenteEval, PenteSpec};
pub use solver::{VcfConfig, hybrid_move, winning_move};

use game_core::hash::splitmix64;
use game_core::{Game, Turn};

const BLACK: u8 = 0;
const WHITE: u8 = 1;
const EMPTY: u8 = 2;

/// Captured pairs needed to win (ten enemy stones).
pub const PAIRS_TO_WIN: u8 = 5;
/// Stones in a line needed to win.
pub const LINE_TO_WIN: usize = 5;
/// Move generation considers only empty intersections within this Chebyshev
/// distance of a stone. Two reaches a capture flank (`[me][opp][opp][me]`
/// places three away) and the far end of an open three, which is every move
/// that matters; it keeps the branching factor low enough for deep alpha-beta.
const RELEVANCE_RADIUS: i32 = 2;

/// The four line orientations as `(drow, dcol)`; each is probed in both
/// directions for line and capture scans.
const DIRECTIONS: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];

/// Black is player 0 and moves first; both win conditions (five in a row, five
/// captured pairs) apply to either side.
#[derive(Clone, Copy)]
pub struct Pente {
    size: usize,
}

impl Default for Pente {
    fn default() -> Self {
        Self::new(13)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PenteAction(pub u16);

#[derive(Clone)]
pub struct PenteState {
    cells: Vec<u8>,
    to_move: usize,
    /// Captured *pairs* by each player (a pair = two enemy stones).
    pairs: [u8; 2],
    /// Stones placed so far — `0` means the empty board (center-only opening).
    moves: u32,
    /// The winner once decided; `None` while playing or on a full-board draw.
    winner: Option<usize>,
    over: bool,
    /// Board index of the last placement, for UI/encoder last-move highlighting.
    last: Option<u16>,
}

impl PenteState {
    /// The stone at board index `p`: `Some(0)` Black, `Some(1)` White, `None`
    /// empty.
    pub fn stone(&self, p: usize) -> Option<usize> {
        match self.cells[p] {
            EMPTY => None,
            c => Some(c as usize),
        }
    }

    /// Captured pairs by each player so far, `[black, white]`.
    pub fn pairs(&self) -> [u8; 2] {
        self.pairs
    }

    /// The player to move: 0 Black, 1 White.
    pub fn to_move(&self) -> usize {
        self.to_move
    }

    /// Stones placed so far (the empty board is `0`).
    pub fn moves(&self) -> u32 {
        self.moves
    }

    /// Stones currently on the board. Each placement adds one and each captured
    /// pair removes two, so this is `moves − 2·(captured pairs)` — never the raw
    /// placement count, which overcounts by the stones captures have removed.
    fn occupied(&self) -> usize {
        (self.moves as usize).saturating_sub(2 * (self.pairs[0] as usize + self.pairs[1] as usize))
    }

    /// Board index of the most recent placement, if any.
    pub fn last_move(&self) -> Option<u16> {
        self.last
    }

    /// The winner once the game is decided by a win condition; `None` while
    /// playing and on a full-board draw.
    pub fn winner(&self) -> Option<usize> {
        self.winner
    }

    fn key(&self) -> u64 {
        let mut h = 0u64;
        for (p, &c) in self.cells.iter().enumerate() {
            if c != EMPTY {
                h ^= splitmix64((p * 2 + c as usize) as u64 + 1);
            }
        }
        if self.to_move == 1 {
            h ^= splitmix64(0x517c_c1b7_2722_0a95);
        }
        // Pair counts are part of the position — two boards that look identical
        // but sit at different capture scores are genuinely different states
        // (one may be a fifth-pair win away).
        h ^= splitmix64(0x1234_5678_9abc_def0 ^ u64::from(self.pairs[0]));
        h ^= splitmix64(0x0fed_cba9_8765_4321 ^ (u64::from(self.pairs[1]) << 8));
        if self.over {
            h ^= splitmix64(0x3c6e_f372_fe94_f82b);
        }
        h
    }
}

impl Pente {
    /// A `size`×`size` board; sizes 5..=19 (five-in-a-row needs at least 5, and
    /// 19 is the upper edge of sane Pente boards). Standard play is 13 or 15;
    /// the smaller default keeps alpha-beta's branching factor searchable.
    pub fn new(size: usize) -> Self {
        assert!(
            (5..=19).contains(&size),
            "board size must be in 5..=19 (five-in-a-row needs ≥5)"
        );
        Self { size }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// The center intersection's board index — Black's forced opening point.
    pub fn center(&self) -> u16 {
        let c = self.size / 2;
        (c * self.size + c) as u16
    }

    /// Board index for a coordinate like `"g7"` (column letters skip `i`,
    /// matching Go's coordinate convention; row 1 is the bottom row).
    pub fn point(&self, coord: &str) -> Option<u16> {
        let mut chars = coord.chars();
        let col = col_index(chars.next()?.to_ascii_lowercase())?;
        let row: usize = chars.as_str().parse().ok()?;
        if col < self.size && (1..=self.size).contains(&row) {
            Some(((row - 1) * self.size + col) as u16)
        } else {
            None
        }
    }

    /// Builds a position from rows of `.`/`X`/`O` characters (top row first,
    /// spaces ignored), with `to_move` to play and the given captured-pair
    /// counts. No move counter is reconstructed (set to a non-zero value so the
    /// center-only opening rule does not re-engage), and no winner is inferred —
    /// the test helper exists to set up tactical positions, not finished games.
    pub fn parse_state(&self, rows: &[&str], to_move: usize, pairs: [u8; 2]) -> PenteState {
        assert_eq!(rows.len(), self.size, "expected {} rows", self.size);
        let mut cells = vec![EMPTY; self.size * self.size];
        let mut placed = 0u32;
        for (i, row) in rows.iter().enumerate() {
            let r = self.size - 1 - i;
            let mut c = 0;
            for ch in row.chars().filter(|ch| !ch.is_whitespace()) {
                assert!(c < self.size, "row {i} has more than {} points", self.size);
                cells[r * self.size + c] = match ch {
                    '.' => EMPTY,
                    'X' => BLACK,
                    'O' => WHITE,
                    _ => panic!("unexpected board character {ch:?}"),
                };
                if cells[r * self.size + c] != EMPTY {
                    placed += 1;
                }
                c += 1;
            }
            assert_eq!(c, self.size, "row {i} has fewer than {} points", self.size);
        }
        PenteState {
            cells,
            to_move,
            pairs,
            moves: placed.max(1),
            winner: None,
            over: false,
            last: None,
        }
    }

    /// The opponent pairs `[YOU][OPP][OPP][YOU]` captured by placing `color` at
    /// `(row, col)`: for each of the eight directions, removes the two flanked
    /// opponent stones and returns how many *pairs* were taken. Mutates `cells`
    /// in place. The placed stone is assumed already set.
    fn resolve_captures(&self, cells: &mut [u8], row: usize, col: usize, color: u8) -> u8 {
        let opp = color ^ 1;
        let mut pairs = 0u8;
        for (dr, dc) in DIRECTIONS {
            for sign in [1, -1] {
                let (dr, dc) = (dr * sign, dc * sign);
                let p1 = step(self.size, row, col, dr, dc, 1);
                let p2 = step(self.size, row, col, dr, dc, 2);
                let p3 = step(self.size, row, col, dr, dc, 3);
                if let (Some(a), Some(b), Some(c)) = (p1, p2, p3)
                    && cells[a] == opp
                    && cells[b] == opp
                    && cells[c] == color
                {
                    cells[a] = EMPTY;
                    cells[b] = EMPTY;
                    pairs += 1;
                }
            }
        }
        pairs
    }

    /// Whether some stone lies within [`RELEVANCE_RADIUS`] (Chebyshev) of `p`.
    fn near_stone(&self, s: &PenteState, p: usize) -> bool {
        let size = self.size as i32;
        let (row, col) = ((p / self.size) as i32, (p % self.size) as i32);
        for dr in -RELEVANCE_RADIUS..=RELEVANCE_RADIUS {
            for dc in -RELEVANCE_RADIUS..=RELEVANCE_RADIUS {
                if dr == 0 && dc == 0 {
                    continue;
                }
                let (r, c) = (row + dr, col + dc);
                if r >= 0
                    && c >= 0
                    && r < size
                    && c < size
                    && s.cells[(r * size + c) as usize] != EMPTY
                {
                    return true;
                }
            }
        }
        false
    }

    /// How many pairs placing `color` at empty `p` would capture — counted by
    /// probing the eight `[p=YOU][OPP][OPP][YOU]` arms directly, without
    /// committing the move or cloning the board (a hot path for move ordering).
    pub(crate) fn capture_pairs_at(&self, s: &PenteState, p: usize, color: u8) -> u8 {
        let mut pairs = 0;
        for_each_captured_pair(&s.cells, self.size, p, color, color ^ 1, |_, _| pairs += 1);
        pairs
    }
}

impl Game for Pente {
    type State = PenteState;
    type Action = PenteAction;

    fn initial_state(&self) -> PenteState {
        PenteState {
            cells: vec![EMPTY; self.size * self.size],
            to_move: 0,
            pairs: [0, 0],
            moves: 0,
            winner: None,
            over: false,
            last: None,
        }
    }

    fn turn(&self, state: &PenteState) -> Turn {
        Turn::Player(state.to_move)
    }

    fn is_terminal(&self, state: &PenteState) -> bool {
        state.over
    }

    fn returns(&self, state: &PenteState, player: usize) -> f64 {
        match state.winner {
            Some(w) if w == player => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        }
    }

    fn legal_actions(&self, state: &PenteState) -> Vec<PenteAction> {
        // Tournament rule: Black's first stone is forced to the center.
        if state.moves == 0 {
            return vec![PenteAction(self.center())];
        }
        // Restrict to empty intersections within Chebyshev distance
        // [`RELEVANCE_RADIUS`] of some stone. In a connection game every
        // meaningful move touches the existing position; a stone in open space
        // far from play is never a capture, a block, or part of a line, so
        // pruning it keeps the action set tractable for deep search without
        // changing best play. Distant empties only re-enter if play ever
        // reaches them (their neighbors fill), so termination is unaffected.
        let mut out = Vec::new();
        for p in 0..state.cells.len() {
            if state.cells[p] == EMPTY && self.near_stone(state, p) {
                out.push(PenteAction(p as u16));
            }
        }
        if out.is_empty() {
            // No stone in range of any empty (only the all-empty board, already
            // handled, or a fully separated remnant): fall back to every empty
            // so the draw clause still sees the board fill.
            for p in 0..state.cells.len() {
                if state.cells[p] == EMPTY {
                    out.push(PenteAction(p as u16));
                }
            }
        }
        out
    }

    fn chance_outcomes(&self, _state: &PenteState) -> Vec<(PenteAction, f64)> {
        vec![]
    }

    fn apply(&self, state: &mut PenteState, action: PenteAction) {
        debug_assert!(!state.over);
        let p = action.0 as usize;
        debug_assert_eq!(state.cells[p], EMPTY, "occupied intersection");
        let color = state.to_move as u8;
        let (row, col) = (p / self.size, p % self.size);
        state.cells[p] = color;
        let captured = self.resolve_captures(&mut state.cells, row, col, color);
        state.pairs[state.to_move] += captured;
        state.last = Some(action.0);
        state.moves += 1;

        if state.pairs[state.to_move] >= PAIRS_TO_WIN
            || completes_line(&state.cells, self.size, row, col, color)
        {
            state.winner = Some(state.to_move);
            state.over = true;
        } else if state.occupied() == self.size * self.size {
            state.over = true; // every intersection filled, no winner: a draw
        }
        state.to_move ^= 1;
    }

    fn infoset_key(&self, state: &PenteState, _player: usize) -> u64 {
        state.key()
    }

    fn state_key(&self, state: &PenteState) -> Option<u64> {
        Some(state.key())
    }

    fn action_id(&self, action: &PenteAction) -> u64 {
        u64::from(action.0) + 1
    }
}

/// The board index `steps` away from `(row, col)` along `(dr, dc)`, or `None`
/// if it walks off the board.
fn step(size: usize, row: usize, col: usize, dr: i32, dc: i32, steps: i32) -> Option<usize> {
    let r = row as i32 + dr * steps;
    let c = col as i32 + dc * steps;
    if r >= 0 && c >= 0 && (r as usize) < size && (c as usize) < size {
        Some(r as usize * size + c as usize)
    } else {
        None
    }
}

/// Count of contiguous `color` stones starting one step from `(row, col)` along
/// `(dr, dc)` (the placed stone itself is excluded).
fn run_length(
    cells: &[u8],
    size: usize,
    row: usize,
    col: usize,
    dr: i32,
    dc: i32,
    color: u8,
) -> usize {
    let mut n = 0;
    let mut step_n = 1;
    while let Some(p) = step(size, row, col, dr, dc, step_n) {
        if cells[p] == color {
            n += 1;
            step_n += 1;
        } else {
            break;
        }
    }
    n
}

/// Whether a `color` stone at `(row, col)` completes a line of at least
/// [`LINE_TO_WIN`]. The single win-by-line scan, shared by `apply`, the
/// eval/ordering knowledge, and the VCF solver.
pub(crate) fn completes_line(cells: &[u8], size: usize, row: usize, col: usize, color: u8) -> bool {
    DIRECTIONS.iter().any(|&(dr, dc)| {
        let run = 1
            + run_length(cells, size, row, col, dr, dc, color)
            + run_length(cells, size, row, col, -dr, -dc, color);
        run >= LINE_TO_WIN
    })
}

/// Invokes `f(a, b)` for each direction in which `placer` placing at empty `p`
/// flanks a `victim` pair — the `[p=placer][a=victim][b=victim][placer]`
/// custodial pattern. The single source of the capture scan, shared by capture
/// counting, the eval's capture tactics, and the encoder's capturable-pair
/// planes. (`resolve_captures` keeps its own copy: it removes pairs as it scans,
/// which would alias this immutable borrow.)
pub(crate) fn for_each_captured_pair(
    cells: &[u8],
    size: usize,
    p: usize,
    placer: u8,
    victim: u8,
    mut f: impl FnMut(usize, usize),
) {
    let (row, col) = (p / size, p % size);
    for (dr, dc) in DIRECTIONS {
        for sign in [1i32, -1] {
            let (dr, dc) = (dr * sign, dc * sign);
            if let (Some(a), Some(b), Some(c)) = (
                step(size, row, col, dr, dc, 1),
                step(size, row, col, dr, dc, 2),
                step(size, row, col, dr, dc, 3),
            ) && cells[a] == victim
                && cells[b] == victim
                && cells[c] == placer
            {
                f(a, b);
            }
        }
    }
}

pub(crate) fn col_letter(col: usize) -> char {
    let skip_i = usize::from(col >= 8);
    (b'a' + (col + skip_i) as u8) as char
}

pub(crate) fn col_index(letter: char) -> Option<usize> {
    match letter {
        'a'..='h' => Some(letter as usize - 'a' as usize),
        'j'..='z' => Some(letter as usize - 'a' as usize - 1),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn place(g: &Pente, s: &mut PenteState, coord: &str) {
        let p = g.point(coord).unwrap();
        g.apply(s, PenteAction(p));
    }

    fn occupied(g: &Pente, s: &PenteState, coord: &str) -> Option<usize> {
        s.stone(g.point(coord).unwrap() as usize)
    }

    #[test]
    fn first_move_is_forced_to_center() {
        let g = Pente::new(13);
        let s = g.initial_state();
        let legal = g.legal_actions(&s);
        assert_eq!(legal, vec![PenteAction(g.center())]);
        assert_eq!(g.center(), g.point("g7").unwrap(), "center of 13x13 is g7");
    }

    #[test]
    fn second_move_is_pruned_to_the_center_neighborhood() {
        let g = Pente::new(13);
        let mut s = g.initial_state();
        g.apply(&mut s, PenteAction(g.center()));
        let legal = g.legal_actions(&s);
        // Only empties within Chebyshev distance 2 of the center stone — a 5x5
        // block minus the occupied center.
        assert_eq!(
            legal.len(),
            5 * 5 - 1,
            "the center's relevance neighborhood"
        );
        assert!(
            !legal.contains(&PenteAction(g.center())),
            "center is occupied"
        );
        // A far corner is not yet a relevant move.
        assert!(!legal.contains(&PenteAction(g.point("a1").unwrap())));
    }

    #[test]
    fn relevance_pruning_only_drops_far_empties() {
        // With a stone in the corner, distant empties are pruned but the
        // neighborhood stays legal.
        let g = Pente::new(9);
        let mut s = g.parse_state(&[". . . . . . . . ."; 9], 0, [0, 0]);
        s.cells[g.point("a1").unwrap() as usize] = BLACK;
        let legal = g.legal_actions(&s);
        assert!(
            legal.contains(&PenteAction(g.point("c3").unwrap())),
            "in range"
        );
        assert!(
            !legal.contains(&PenteAction(g.point("e5").unwrap())),
            "out of range"
        );
    }

    #[test]
    fn custodial_capture_removes_exactly_the_pair() {
        // Black plays the right flank of  X O O .  -> X O O X, capturing the pair.
        let g = Pente::new(9);
        let mut s = g.parse_state(
            &[
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". X O O . . . . .",
                ". . . . . . . . .",
            ],
            0,
            [0, 0],
        );
        place(&g, &mut s, "e2"); // the flanking X
        assert_eq!(occupied(&g, &s, "c2"), None, "first bracketed stone gone");
        assert_eq!(occupied(&g, &s, "d2"), None, "second bracketed stone gone");
        assert_eq!(occupied(&g, &s, "b2"), Some(0), "the original flank stays");
        assert_eq!(s.pairs(), [1, 0], "one pair to Black");
    }

    #[test]
    fn three_in_a_row_is_not_captured() {
        // X O O O X captures nothing — custodial capture is exactly two stones.
        let g = Pente::new(9);
        let mut s = g.parse_state(
            &[
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". X O O O . . . .",
                ". . . . . . . . .",
            ],
            0,
            [0, 0],
        );
        place(&g, &mut s, "f2"); // X O O O X
        assert_eq!(occupied(&g, &s, "c2"), Some(1));
        assert_eq!(occupied(&g, &s, "d2"), Some(1));
        assert_eq!(occupied(&g, &s, "e2"), Some(1));
        assert_eq!(s.pairs(), [0, 0], "a triple is immune");
    }

    #[test]
    fn moving_into_a_bracket_is_safe() {
        // White fills  X . X  to  X O X : the pattern [OPP][YOU][YOU][OPP] is
        // NOT formed (only one O between the X's), but even a full
        // [X][O][O][X] formed by White's own move must not self-capture.
        let g = Pente::new(9);
        let mut s = g.parse_state(
            &[
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". X O . X . . . .",
                ". . . . . . . . .",
            ],
            1,
            [0, 0],
        );
        // White plays d2 to complete  X O O X  by moving into the bracket.
        place(&g, &mut s, "d2");
        assert_eq!(occupied(&g, &s, "c2"), Some(1), "white pair survives...");
        assert_eq!(
            occupied(&g, &s, "d2"),
            Some(1),
            "...the move that formed it"
        );
        assert_eq!(s.pairs(), [0, 0], "no capture on moving into a bracket");
    }

    #[test]
    fn capture_works_in_all_eight_directions() {
        // A black stone at the center with eight  X O O .  arms; placing the
        // black flank on each arm captures that pair. Build the eight arms and
        // verify each direction independently from a fresh state.
        let g = Pente::new(11);
        let center = "f6"; // (5,5) on 0-based -> center of 11x11
        let arms: [(&str, &str, &str); 8] = [
            // (near opp, far opp, flank to place) along each of 8 rays
            ("g6", "h6", "j6"), // E
            ("e6", "d6", "c6"), // W
            ("f7", "f8", "f9"), // N
            ("f5", "f4", "f3"), // S
            ("g7", "h8", "j9"), // NE
            ("e7", "d8", "c9"), // NW
            ("g5", "h4", "j3"), // SE
            ("e5", "d4", "c3"), // SW
        ];
        for (near, far, flank) in arms {
            let mut s = g.parse_state(&[". . . . . . . . . . ."; 11], 0, [0, 0]);
            // Lay the black anchor, the two white stones, then place the flank.
            let set = |s: &mut PenteState, coord: &str, color: u8| {
                s.cells[g.point(coord).unwrap() as usize] = color;
            };
            set(&mut s, center, BLACK);
            set(&mut s, near, WHITE);
            set(&mut s, far, WHITE);
            place(&g, &mut s, flank);
            assert_eq!(occupied(&g, &s, near), None, "{flank}: near captured");
            assert_eq!(occupied(&g, &s, far), None, "{flank}: far captured");
            assert_eq!(s.pairs(), [1, 0], "{flank}: exactly one pair");
        }
    }

    #[test]
    fn one_move_captures_in_multiple_directions() {
        // The placed black stone flanks two white pairs at once: a horizontal
        // and a vertical arm sharing the placement point.
        let g = Pente::new(9);
        let mut s = g.parse_state(
            &[
                ". . . . . . . . .",
                ". . . . X . . . .",
                ". . . . O . . . .",
                ". . . . O . . . .",
                ". X O O . O O X .",
                ". . . . O . . . .",
                ". . . . O . . . .",
                ". . . . X . . . .",
                ". . . . . . . . .",
            ],
            0,
            [0, 0],
        );
        // e5 is empty (the '.' between the two horizontal white pairs and the
        // vertical ones). Placing black there flanks four arms: W, E, N, S.
        place(&g, &mut s, "e5");
        assert_eq!(s.pairs(), [4, 0], "four pairs captured by one stone");
        for gone in ["c5", "d5", "f5", "g5", "e6", "e7", "e3", "e4"] {
            assert_eq!(occupied(&g, &s, gone), None, "{gone} captured");
        }
    }

    #[test]
    fn five_in_a_row_wins() {
        // Four blacks a2..d2; the fifth at e2 completes a horizontal five.
        let g = Pente::new(9);
        let mut s = g.parse_state(
            &[
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                "X X X X . . . . .",
                ". . . . . . . . .",
            ],
            0,
            [0, 0],
        );
        assert!(!g.is_terminal(&s), "not yet won");
        place(&g, &mut s, "e2");
        assert!(g.is_terminal(&s), "five completes the line");
        assert_eq!(g.returns(&s, 0), 1.0);
        assert_eq!(g.returns(&s, 1), -1.0);
    }

    #[test]
    fn five_in_a_row_diagonal_and_vertical() {
        let g = Pente::new(9);
        // Vertical: four blacks in column e, complete with the fifth.
        let mut s = g.parse_state(
            &[
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . X . . . .",
                ". . . . X . . . .",
                ". . . . X . . . .",
                ". . . . X . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
            ],
            0,
            [0, 0],
        );
        place(&g, &mut s, "e2");
        assert!(
            g.is_terminal(&s) && g.returns(&s, 0) == 1.0,
            "vertical five"
        );

        // Diagonal (up-right): a1,b2,c3,d4 black, complete e5. Build it through
        // `point` so the coordinates speak for themselves (row 1 is the bottom).
        let mut s = g.parse_state(&[". . . . . . . . ."; 9], 0, [0, 0]);
        for coord in ["a1", "b2", "c3", "d4"] {
            s.cells[g.point(coord).unwrap() as usize] = BLACK;
        }
        place(&g, &mut s, "e5");
        assert!(
            g.is_terminal(&s) && g.returns(&s, 0) == 1.0,
            "up-right diagonal five"
        );
    }

    #[test]
    fn six_in_a_row_also_wins() {
        let g = Pente::new(9);
        // Black already has X X X X . X horizontally; filling the gap makes six.
        let mut s = g.parse_state(
            &[
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                "X X X . X X . . .",
                ". . . . . . . . .",
            ],
            0,
            [0, 0],
        );
        place(&g, &mut s, "d2");
        assert!(g.is_terminal(&s) && g.returns(&s, 0) == 1.0, "six-in-a-row");
    }

    #[test]
    fn five_captured_pairs_wins() {
        // Black sits at four pairs; one more capture ends it on captures, not a
        // line. Set up an  X O O .  arm and place the flank.
        let g = Pente::new(9);
        let mut s = g.parse_state(
            &[
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". X O O . . . . .",
                ". . . . . . . . .",
            ],
            0,
            [4, 0],
        );
        assert!(!g.is_terminal(&s), "four pairs is not yet a win");
        place(&g, &mut s, "e2");
        assert_eq!(s.pairs(), [5, 0]);
        assert!(g.is_terminal(&s), "fifth pair ends the game");
        assert_eq!(g.returns(&s, 0), 1.0);
        assert_eq!(g.returns(&s, 1), -1.0);
    }

    #[test]
    fn full_board_with_no_winner_is_a_draw() {
        // A 5x5 board filled so no five-line and no fifth pair exist. We hand a
        // pattern that fills completely; place the final stone and assert a draw
        // (winner None, terminal true, returns 0).
        let g = Pente::new(5);
        // Construct a coloring with no 5-in-a-row in any direction. Rows of
        // alternating-ish colors avoid any full line of one color.
        let mut s = g.parse_state(
            &[
                "X O X O X",
                "X O X O X",
                "O X O X O",
                "X O X O X",
                "X O X O .",
            ],
            1, // White to place the last empty cell (e1)
            [0, 0],
        );
        // No column is a single color (column a is X X O X X — broken by O),
        // and the final move must not complete a line for White.
        assert!(!g.is_terminal(&s));
        place(&g, &mut s, "e1");
        assert!(g.is_terminal(&s), "board is full");
        assert_eq!(s.winner, None, "no five-line, no fifth pair");
        assert_eq!(g.returns(&s, 0), 0.0);
        assert_eq!(g.returns(&s, 1), 0.0);
    }

    #[test]
    fn captures_do_not_trigger_a_premature_draw() {
        // A capture removes two stones without rewinding the placement counter,
        // so a board where placements equal the cell count can still hold empty
        // intersections. The draw clause must key on actual occupancy, not on
        // the raw placement count, or it ends a still-playable game as a draw.
        let g = Pente::new(5);
        // 25 cells. Lay a full-but-for-the-capture board: 24 stones placed, then
        // a 25th placement that captures a pair — leaving the board with two
        // empty cells even though 25 placements have happened.
        // Build a position one placement before that capture.
        let mut s = g.parse_state(
            &[
                "X O X O X",
                "O X O X O",
                "X O X O X",
                "O X O X O",
                "X O O . X", // d1 empty; b1,c1 are a white pair flanked by a1=X
            ],
            0, // Black to flank at d1, capturing b1,c1
            [0, 0],
        );
        // Force the placement counter to the brink of a "full board": 24 stones
        // are on the board, so the next placement is the 25th.
        s.moves = 24;
        assert!(!g.is_terminal(&s), "not terminal before the capturing move");
        place(&g, &mut s, "d1");
        assert_eq!(s.pairs(), [1, 0], "the white pair b1,c1 is captured");
        assert!(
            !g.is_terminal(&s),
            "two cells were just emptied by the capture: the game is NOT a draw"
        );
        // The freshly-emptied cells are legal moves again.
        let legal = g.legal_actions(&s);
        assert!(legal.contains(&PenteAction(g.point("b1").unwrap())));
        assert!(legal.contains(&PenteAction(g.point("c1").unwrap())));
    }

    #[test]
    fn state_key_distinguishes_side_and_pairs() {
        let g = Pente::new(9);
        let base = g.parse_state(&[". . . . . . . . ."; 9], 0, [0, 0]);
        let mut other_side = base.clone();
        other_side.to_move = 1;
        assert_ne!(g.state_key(&base), g.state_key(&other_side));
        let mut more_pairs = base.clone();
        more_pairs.pairs = [1, 0];
        assert_ne!(
            g.state_key(&base),
            g.state_key(&more_pairs),
            "pair count is part of the position"
        );
    }

    #[test]
    fn random_playthroughs_terminate() {
        use game_core::Rng;
        let g = Pente::new(9);
        let mut rng = Rng::new(0xC0FFEE);
        for _ in 0..20 {
            let mut s = g.initial_state();
            let mut plies = 0;
            while !g.is_terminal(&s) {
                assert!(plies <= 9 * 9, "must end by a full board at the latest");
                let actions = g.legal_actions(&s);
                assert!(!actions.is_empty());
                let i = rng.below(actions.len());
                g.apply(&mut s, actions[i]);
                plies += 1;
            }
            let r = g.returns(&s, 0);
            assert!(r == 1.0 || r == -1.0 || r == 0.0);
            assert_eq!(g.returns(&s, 1), -r);
        }
    }
}
