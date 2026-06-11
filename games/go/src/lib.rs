//! Go as a [`game_core::Game`].
//!
//! Square board of configurable size (default 9), komi 7.5 for White, Black is
//! player 0. Perfect information, no chance nodes. Rules:
//!
//! * A move places a stone on an empty point or passes. After a placement,
//!   adjacent opponent groups with no liberties are removed; the move is
//!   illegal if the placed stone's own group then has no liberties (suicide).
//! * **Simple ko**: a move may not recreate the whole-board position that
//!   existed immediately before the opponent's last move. The state carries the
//!   hash of that reference position; only capturing moves can violate it,
//!   since any other placement leaves the just-played stones of both sides on
//!   the board.
//! * Two consecutive passes end the game.
//! * **Area (Chinese) scoring**: each side scores its stones on the board plus
//!   the empty regions bordered exclusively by its color; White adds komi 7.5,
//!   so there are no draws.
//!
//! **Draw-guard**: simple ko alone does not forbid long cycles (e.g. triple
//! ko), so after `4 * size * size` plies the game is forcibly ended and scored
//! as-is by area scoring. This guarantees unguided random playouts terminate;
//! it essentially never binds in directed play. The ply counter is *not* part
//! of [`Game::state_key`], so positions identical up to the draw-guard clock
//! share a key.

mod knowledge;
mod ui;

pub use knowledge::{GoEval, GoSpec};

use game_core::hash::splitmix64;
use game_core::{Game, Turn};

pub const KOMI: f64 = 7.5;
const KOMI_X2: u64 = 15;

const BLACK: u8 = 0;
const WHITE: u8 = 1;
const EMPTY: u8 = 2;

/// Black is player 0 and moves first; White (player 1) receives komi 7.5.
pub struct Go {
    size: usize,
}

impl Default for Go {
    fn default() -> Self {
        Self::new(9)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GoAction {
    /// Place a stone at a board index (`row * size + col`, row 0 = row 1).
    Place(u16),
    Pass,
}

#[derive(Clone, Debug)]
pub struct GoState {
    cells: Vec<u8>,
    to_move: usize,
    passes: u8,
    captures: [u32; 2],
    /// Board hash before the last move — the simple-ko reference position.
    prev_key: u64,
    plies: u32,
    over: bool,
}

impl GoState {
    /// The stone at board index `p`: `Some(0)` Black, `Some(1)` White, `None` empty.
    pub fn stone(&self, p: usize) -> Option<usize> {
        match self.cells[p] {
            EMPTY => None,
            c => Some(c as usize),
        }
    }

    /// Stones captured *by* each player so far (display only — area scoring
    /// does not count prisoners).
    pub fn captures(&self) -> [u32; 2] {
        self.captures
    }

    fn key(&self) -> u64 {
        let mut h = board_hash(&self.cells);
        if self.to_move == 1 {
            h ^= splitmix64(0x517c_c1b7_2722_0a95);
        }
        h ^= splitmix64(self.prev_key ^ 0x6a09_e667_f3bc_c909);
        h ^= splitmix64(0xbb67_ae85_84ca_a73b ^ u64::from(self.passes));
        if self.over {
            h ^= splitmix64(0x3c6e_f372_fe94_f82b);
        }
        h
    }
}

impl Go {
    /// A `size`×`size` board; sizes 2..=25 (coordinate letters skip `i`).
    pub fn new(size: usize) -> Self {
        assert!((2..=25).contains(&size), "board size must be in 2..=25");
        Self { size }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn komi(&self) -> f64 {
        KOMI
    }

    fn max_plies(&self) -> u32 {
        (4 * self.size * self.size) as u32
    }

    /// Board index for a coordinate like `"d4"` (column letters skip `i`).
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
    /// spaces ignored), with `to_move` to play, no pass history, and no ko
    /// restriction in force.
    pub fn parse_state(&self, rows: &[&str], to_move: usize) -> GoState {
        assert_eq!(rows.len(), self.size, "expected {} rows", self.size);
        let mut cells = vec![EMPTY; self.size * self.size];
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
                c += 1;
            }
            assert_eq!(c, self.size, "row {i} has fewer than {} points", self.size);
        }
        let prev_key = board_hash(&cells);
        GoState {
            cells,
            to_move,
            passes: 0,
            captures: [0, 0],
            prev_key,
            plies: 0,
            over: false,
        }
    }

    /// Area (Chinese) scores before komi: stones plus empty regions bordered
    /// exclusively by one color, as `(black, white)`.
    pub fn area_scores(&self, s: &GoState) -> (u64, u64) {
        let mut score = [0u64; 2];
        let mut seen = vec![false; s.cells.len()];
        for (p, &cell) in s.cells.iter().enumerate() {
            if cell != EMPTY {
                score[cell as usize] += 1;
                continue;
            }
            if seen[p] {
                continue;
            }
            let mut region = vec![p];
            seen[p] = true;
            let mut borders = [false; 2];
            let mut i = 0;
            while i < region.len() {
                let q = region[i];
                i += 1;
                for n in neighbors(self.size, q) {
                    match s.cells[n] {
                        EMPTY => {
                            if !seen[n] {
                                seen[n] = true;
                                region.push(n);
                            }
                        }
                        c => borders[c as usize] = true,
                    }
                }
            }
            match (borders[0], borders[1]) {
                (true, false) => score[0] += region.len() as u64,
                (false, true) => score[1] += region.len() as u64,
                _ => {}
            }
        }
        (score[0], score[1])
    }

    fn placement_legal(&self, s: &GoState, p: usize) -> bool {
        let color = s.to_move as u8;
        let mut empty_neighbor = false;
        let mut opponent_neighbor = false;
        for n in neighbors(self.size, p) {
            match s.cells[n] {
                EMPTY => empty_neighbor = true,
                c if c != color => opponent_neighbor = true,
                _ => {}
            }
        }
        if empty_neighbor && !opponent_neighbor {
            return true;
        }
        let mut cells = s.cells.clone();
        match place(&mut cells, self.size, p, color) {
            None => false,
            Some(0) => true,
            Some(_) => board_hash(&cells) != s.prev_key,
        }
    }
}

impl Game for Go {
    type State = GoState;
    type Action = GoAction;

    fn initial_state(&self) -> GoState {
        GoState {
            cells: vec![EMPTY; self.size * self.size],
            to_move: 0,
            passes: 0,
            captures: [0, 0],
            prev_key: 0,
            plies: 0,
            over: false,
        }
    }

    fn turn(&self, state: &GoState) -> Turn {
        Turn::Player(state.to_move)
    }

    fn is_terminal(&self, state: &GoState) -> bool {
        state.over
    }

    fn returns(&self, state: &GoState, player: usize) -> f64 {
        let (black, white) = self.area_scores(state);
        let winner = if 2 * black > 2 * white + KOMI_X2 {
            0
        } else {
            1
        };
        if player == winner { 1.0 } else { -1.0 }
    }

    fn legal_actions(&self, state: &GoState) -> Vec<GoAction> {
        let mut out = Vec::new();
        for (p, &cell) in state.cells.iter().enumerate() {
            if cell == EMPTY && self.placement_legal(state, p) {
                out.push(GoAction::Place(p as u16));
            }
        }
        out.push(GoAction::Pass);
        out
    }

    fn chance_outcomes(&self, _state: &GoState) -> Vec<(GoAction, f64)> {
        vec![]
    }

    fn apply(&self, state: &mut GoState, action: GoAction) {
        debug_assert!(!state.over);
        let before = board_hash(&state.cells);
        match action {
            GoAction::Pass => {
                state.passes += 1;
                if state.passes >= 2 {
                    state.over = true;
                }
            }
            GoAction::Place(p) => {
                let captured = place(&mut state.cells, self.size, p as usize, state.to_move as u8)
                    .expect("illegal move: suicide");
                state.captures[state.to_move] += captured as u32;
                state.passes = 0;
            }
        }
        state.prev_key = before;
        state.to_move ^= 1;
        state.plies += 1;
        if state.plies >= self.max_plies() {
            state.over = true;
        }
    }

    fn infoset_key(&self, state: &GoState, _player: usize) -> u64 {
        state.key()
    }

    fn state_key(&self, state: &GoState) -> Option<u64> {
        Some(state.key())
    }
}

/// Sets the stone, removes adjacent opponent groups left without liberties,
/// and returns the number captured — or `None` if the move is suicide (in
/// which case `cells` must be discarded).
fn place(cells: &mut [u8], size: usize, p: usize, color: u8) -> Option<usize> {
    debug_assert_eq!(cells[p], EMPTY);
    cells[p] = color;
    let mut captured = 0;
    for n in neighbors(size, p) {
        if cells[n] == (color ^ 1) {
            let (stones, alive) = group(cells, size, n);
            if !alive {
                captured += stones.len();
                for s in stones {
                    cells[s] = EMPTY;
                }
            }
        }
    }
    if captured == 0 {
        let (_, alive) = group(cells, size, p);
        if !alive {
            return None;
        }
    }
    Some(captured)
}

/// The group containing `start`, plus whether it has any liberty.
fn group(cells: &[u8], size: usize, start: usize) -> (Vec<usize>, bool) {
    let color = cells[start];
    let mut stones = vec![start];
    let mut seen = vec![false; cells.len()];
    seen[start] = true;
    let mut has_liberty = false;
    let mut i = 0;
    while i < stones.len() {
        let p = stones[i];
        i += 1;
        for n in neighbors(size, p) {
            if cells[n] == EMPTY {
                has_liberty = true;
            } else if cells[n] == color && !seen[n] {
                seen[n] = true;
                stones.push(n);
            }
        }
    }
    (stones, has_liberty)
}

fn neighbors(size: usize, p: usize) -> impl Iterator<Item = usize> {
    let (r, c) = (p / size, p % size);
    let mut out = [0usize; 4];
    let mut n = 0;
    if r > 0 {
        out[n] = p - size;
        n += 1;
    }
    if r + 1 < size {
        out[n] = p + size;
        n += 1;
    }
    if c > 0 {
        out[n] = p - 1;
        n += 1;
    }
    if c + 1 < size {
        out[n] = p + 1;
        n += 1;
    }
    out.into_iter().take(n)
}

fn board_hash(cells: &[u8]) -> u64 {
    let mut h = 0;
    for (p, &c) in cells.iter().enumerate() {
        if c != EMPTY {
            h ^= splitmix64((p * 2 + c as usize) as u64 + 1);
        }
    }
    h
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
