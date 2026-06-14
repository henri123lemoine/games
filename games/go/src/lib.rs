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

pub mod encode;
mod knowledge;
mod ui;

use knowledge::is_eyelike;
pub use knowledge::{GoEval, GoSpec};

use game_core::hash::splitmix64;
use game_core::{Game, Turn};

/// Default komi (White's compensation for moving second), area scoring.
pub const KOMI: f64 = 7.5;

/// Number of recent move locations kept for the encoder's move-history planes.
pub const HISTORY: usize = 5;
/// History sentinel: a pass, or a slot not yet filled.
const NO_MOVE: u16 = u16::MAX;

const BLACK: u8 = 0;
const WHITE: u8 = 1;
const EMPTY: u8 = 2;

/// Black is player 0 and moves first; White (player 1) receives `komi`.
#[derive(Clone, Copy)]
pub struct Go {
    size: usize,
    komi: f64,
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
    /// Locations of the last [`HISTORY`] moves, most-recent first; [`NO_MOVE`]
    /// for a pass or an unfilled slot. Feeds the encoder's move-history planes.
    recent: [u16; HISTORY],
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

    /// The player to move: 0 Black, 1 White.
    pub fn to_move(&self) -> usize {
        self.to_move
    }

    /// Plies played so far, passes included.
    pub fn plies(&self) -> u32 {
        self.plies
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
    /// A `size`×`size` board with the default komi; sizes 2..=25 (coordinate
    /// letters skip `i`).
    pub fn new(size: usize) -> Self {
        Self::with_komi(size, KOMI)
    }

    /// A `size`×`size` board with an explicit komi — used by self-play komi
    /// randomization so the net learns *score* across komi rather than a single
    /// fixed-komi win/loss bit.
    pub fn with_komi(size: usize, komi: f64) -> Self {
        assert!((2..=25).contains(&size), "board size must be in 2..=25");
        Self { size, komi }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn komi(&self) -> f64 {
        self.komi
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
            recent: [NO_MOVE; HISTORY],
        }
    }

    /// Area (Chinese) scores before komi: stones plus empty regions bordered
    /// exclusively by one color, as `(black, white)`. No dead-stone
    /// adjudication: a two-pass ending with dead stones on the board scores
    /// them as alive. Bot playouts capture before passing, so this only
    /// surprises humans who pass early in the lab client.
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

    /// Per-point ownership of `state`: `+1.0` a point that scores for Black,
    /// `-1.0` for White, `0.0` neutral (dame, or an empty region bordered by
    /// both). A stone scores for its color; an empty region bordered by one
    /// color scores for that color — the same partition [`Go::area_scores`]
    /// counts. This is the dense per-point territory signal the trainer's
    /// auxiliary ownership head learns from (KataGo-style): it teaches the
    /// trunk territory directly, so even late filling positions are
    /// informative and the net never finds passing-early attractive.
    pub fn ownership(&self, s: &GoState) -> Vec<f32> {
        let n = self.size * self.size;
        let mut own = vec![0.0f32; n];
        let mut seen = vec![false; n];
        for p in 0..n {
            match s.cells[p] {
                BLACK => own[p] = 1.0,
                WHITE => own[p] = -1.0,
                _ => {
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
                        for nb in neighbors(self.size, q) {
                            match s.cells[nb] {
                                EMPTY => {
                                    if !seen[nb] {
                                        seen[nb] = true;
                                        region.push(nb);
                                    }
                                }
                                c => borders[c as usize] = true,
                            }
                        }
                    }
                    let val = match (borders[0], borders[1]) {
                        (true, false) => 1.0,
                        (false, true) => -1.0,
                        _ => 0.0,
                    };
                    for &q in &region {
                        own[q] = val;
                    }
                }
            }
        }
        own
    }

    /// Final area-score margin from Black's view: `black − white − komi`,
    /// positive when Black is ahead. The continuous signal underneath the
    /// win/loss [`Game::returns`] — a denser training target (how *much* one
    /// side won, not just who) that lets the net learn score, not a single
    /// fixed-komi bit.
    pub fn score_margin(&self, s: &GoState) -> f64 {
        let (black, white) = self.area_scores(s);
        black as f64 - white as f64 - self.komi()
    }

    /// True if the mover has a *productive* move: a legal placement that does
    /// not merely fill one of its own true eyes. When this holds, passing is a
    /// strictly wasteful move and self-play/eval/serving should forbid it —
    /// without that guard, area scoring's "a sparse board is a komi win for
    /// White" makes passing early a degenerate self-play equilibrium (White
    /// ends the game before Black can build territory). Pass becomes available
    /// again once only eye-filling moves remain, so finished games still end.
    pub fn has_productive_move(&self, s: &GoState) -> bool {
        let color = s.to_move as u8;
        (0..self.size * self.size).any(|p| {
            s.cells[p] == EMPTY
                && self.placement_legal(s, p)
                && !is_eyelike(&s.cells, self.size, p, color)
        })
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
            recent: [NO_MOVE; HISTORY],
        }
    }

    fn turn(&self, state: &GoState) -> Turn {
        Turn::Player(state.to_move)
    }

    fn is_terminal(&self, state: &GoState) -> bool {
        state.over
    }

    fn returns(&self, state: &GoState, player: usize) -> f64 {
        // Komi carries a half point (no integer ties); Black wins iff its area
        // lead beats komi.
        let winner = if self.score_margin(state) > 0.0 { 0 } else { 1 };
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
        // Slide the move-history ring and record this move's location.
        for i in (1..HISTORY).rev() {
            state.recent[i] = state.recent[i - 1];
        }
        state.recent[0] = match action {
            GoAction::Place(p) => p,
            GoAction::Pass => NO_MOVE,
        };
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

    fn action_id(&self, action: &GoAction) -> u64 {
        match action {
            GoAction::Place(i) => u64::from(*i) + 1,
            GoAction::Pass => 0,
        }
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
