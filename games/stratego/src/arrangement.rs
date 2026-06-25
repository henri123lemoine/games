//! Setup-phase representation: the 40-cell home arrangement and the A-M
//! character bijection (`stratego_board.cu:27-41`, `:249`).
//!
//! An arrangement is 40 [`PieceType`]s in row-major home order
//! (`arrangement[0]` -> bottom-left home cell, `arrangement[39]` -> the cell
//! before the lakes). Red places into cells `0..40`; blue's arrangement is
//! placed point-reflected into cells `60..100` (`stratego_board.cu:484-500`).

use crate::board::{Board, CLASSIC_INITIAL_COUNTS, Color, HOME_CELLS, Piece, PieceType};

/// The A-M arrangement alphabet, indexed by character offset `c - 'A'`:
/// `A=Empty, B=Bomb, C=Spy, D=Scout, E=Miner, F=Sergeant, G=Lieutenant,
/// H=Captain, I=Major, J=Colonel, K=General, L=Marshal, M=Flag`.
pub const CHAR_TO_TYPE: [PieceType; 13] = [
    PieceType::Empty,
    PieceType::Bomb,
    PieceType::Spy,
    PieceType::Scout,
    PieceType::Miner,
    PieceType::Sergeant,
    PieceType::Lieutenant,
    PieceType::Captain,
    PieceType::Major,
    PieceType::Colonel,
    PieceType::General,
    PieceType::Marshal,
    PieceType::Flag,
];

/// The inverse string `type -> char`, ordered by [`PieceType`] value
/// (`stratego_board.cu:249` `"CDEFGHIJKLMB_A"`).
const TYPE_TO_CHAR: [u8; 14] = *b"CDEFGHIJKLMB_A";

/// Maps an arrangement character `'A'..='M'` to its piece type.
pub fn char_to_type(ch: char) -> Option<PieceType> {
    let up = ch.to_ascii_uppercase();
    if !('A'..='M').contains(&up) {
        return None;
    }
    Some(CHAR_TO_TYPE[(up as u8 - b'A') as usize])
}

/// Maps a piece type to its arrangement character.
pub fn type_to_char(t: PieceType) -> char {
    TYPE_TO_CHAR[t as usize] as char
}

/// The 6 home cells from which a piece can move on the very first turn (the two
/// columns flanking each lake gap plus the rightmost pair), in arrangement
/// coordinates (`stratego_board.cu:375`, `test_is_terminal_arrangement.py:39`).
pub const CORRIDOR_CELLS: [usize; 6] = [30, 31, 34, 35, 38, 39];

/// A red home deployment: 40 piece types in row-major home order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Arrangement(pub [PieceType; HOME_CELLS]);

impl Arrangement {
    /// Parses a 40-character `'A'..='M'` arrangement string.
    pub fn from_chars(s: &str) -> Option<Arrangement> {
        let bytes: Vec<char> = s.chars().collect();
        if bytes.len() != HOME_CELLS {
            return None;
        }
        let mut types = [PieceType::Empty; HOME_CELLS];
        for (i, &ch) in bytes.iter().enumerate() {
            types[i] = char_to_type(ch)?;
        }
        Some(Arrangement(types))
    }

    pub fn to_chars(&self) -> String {
        self.0.iter().map(|&t| type_to_char(t)).collect()
    }

    /// Mirrors the deployment left-right across the centre column: the column
    /// order within each home row is reversed (`out[row][col] =
    /// self[row][9 - col]`). This is the faithful equivalent of the reference
    /// `flip_arrangements` (`arrangement/utils.py`, a `.flip(-2)` over the
    /// `(row, col)` grid). Applied with probability 1/2 it cancels the
    /// right-half-flag handedness bias, restoring the left-right symmetry an
    /// equilibrium setup distribution must have.
    pub fn flipped(&self) -> Arrangement {
        let mut out = [PieceType::Empty; HOME_CELLS];
        for row in 0..4 {
            for col in 0..10 {
                out[row * 10 + col] = self.0[row * 10 + (9 - col)];
            }
        }
        Arrangement(out)
    }

    /// Per-type counts over `[0, 14)` (lake/empty included at their indices).
    pub fn type_counts(&self) -> [u8; 14] {
        let mut counts = [0u8; 14];
        for &t in &self.0 {
            counts[t as usize] += 1;
        }
        counts
    }

    /// Whether red (player 0) cannot make a single move from this deployment —
    /// a trivially-lost setup (`stratego_board.cu:356-386`,
    /// `IsTerminalArrangement`). A movable piece must sit next to an empty home
    /// cell or be on a corridor cell.
    pub fn is_terminal(&self) -> bool {
        let movable = |i: usize| self.0[i].is_movable();
        for j in 0..HOME_CELLS {
            let close_to_empty = self.0[j] == PieceType::Empty
                && ((j % 10 != 0 && movable(j - 1))
                    || (j % 10 != 9 && movable(j + 1))
                    || (j >= 10 && movable(j - 10))
                    || (j <= 29 && movable(j + 10)));
            let close_to_corridor = movable(j) && CORRIDOR_CELLS.contains(&j);
            if close_to_empty || close_to_corridor {
                return false;
            }
        }
        true
    }
}

/// Remaining-supply tracker for the serialized per-square deployment used by the
/// [`Game`](game_core::Game) impl: which types may legally fill the next empty
/// home square, given exhausted types and (under handedness) the flag column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeploymentState {
    /// Remaining count per type `[0, 14)`; only `[0, 12)` are ever nonzero plus
    /// `Empty` for the barrage filler.
    pub remaining: [u8; 14],
    /// Pieces already placed, in row-major order.
    pub placed: Vec<PieceType>,
    /// The player currently deploying (0 = red, 1 = blue).
    pub player: usize,
    /// Force the flag onto the right half (columns 5-9, any home row) during
    /// generation (`force_handedness`, `arrangement_transformer.py`).
    pub force_handedness: bool,
}

impl DeploymentState {
    /// Starts a classic deployment for `player`.
    pub fn classic(player: usize, force_handedness: bool) -> DeploymentState {
        let mut remaining = [0u8; 14];
        remaining[..12].copy_from_slice(&CLASSIC_INITIAL_COUNTS);
        DeploymentState {
            remaining,
            placed: Vec::with_capacity(HOME_CELLS),
            player,
            force_handedness,
        }
    }

    /// The home square index `[0, 40)` next to be filled (row-major).
    pub fn next_square(&self) -> usize {
        self.placed.len()
    }

    pub fn is_complete(&self) -> bool {
        self.placed.len() == HOME_CELLS
    }

    /// Total pieces still to place across all types.
    pub fn total_remaining(&self) -> u8 {
        self.remaining.iter().sum()
    }

    /// The piece types that may legally fill [`next_square`](Self::next_square),
    /// in [`PieceType`] order. A type is legal iff its supply is nonzero and,
    /// for the flag under handedness, the square is on the right half (columns
    /// 5-9) of its home row.
    pub fn legal_types(&self) -> Vec<PieceType> {
        let square = self.next_square();
        let mut out = Vec::new();
        for t in 0..14u8 {
            if self.remaining[t as usize] == 0 {
                continue;
            }
            let kind = PieceType::from_u8(t);
            if kind == PieceType::Flag && self.force_handedness && !flag_allowed_on_right(square) {
                continue;
            }
            out.push(kind);
        }
        out
    }

    /// Places `kind` on the next square, consuming supply.
    pub fn place(&mut self, kind: PieceType) {
        debug_assert!(self.remaining[kind as usize] > 0);
        self.remaining[kind as usize] -= 1;
        self.placed.push(kind);
    }

    /// The completed [`Arrangement`]; panics if deployment is unfinished.
    pub fn arrangement(&self) -> Arrangement {
        assert!(self.is_complete(), "deployment not complete");
        let mut types = [PieceType::Empty; HOME_CELLS];
        types.copy_from_slice(&self.placed);
        Arrangement(types)
    }
}

/// Under forced handedness the flag must land on the right half of the home
/// grid — columns 5-9 of *any* home row. This is a pure column constraint that
/// breaks the left-right mirror symmetry while leaving the flag's row free for
/// the net to choose, mirroring the reference `right_side` mask
/// (`arrangement_transformer.py`: `N_ARRANGEMENT_ROW * (5*[False] + 5*[True])`,
/// the flag forbidden wherever `~right_side`). The remaining mirror axis is
/// restored by flipping ~50% of generated setups (`flip_arrangements`); see
/// [`Arrangement::flipped`].
fn flag_allowed_on_right(square: usize) -> bool {
    square % 10 >= 5
}

/// Writes a parsed red/blue arrangement pair onto a fresh board, placing lakes,
/// piece ids (POV starting square), and the hidden counters — the move-phase
/// initial state (`GenerateInitializationBoards`, `stratego_board.cu:467-509`).
pub fn board_from_arrangements(red: &Arrangement, blue: &Arrangement) -> Board {
    let mut board = Board::blank();
    let mut num_hidden = [[0u8; 12]; 2];
    let mut unmoved = [0u8; 2];

    for (j, &kind) in red.0.iter().enumerate() {
        if kind == PieceType::Empty {
            continue;
        }
        board.pieces[j] = Piece::new(kind, Color::Red, j as u8);
        board.zero_types[0][j] = kind as u8;
        num_hidden[0][kind as usize] += 1;
        unmoved[0] += 1;
    }
    for (j, &kind) in blue.0.iter().enumerate() {
        if kind == PieceType::Empty {
            continue;
        }
        let cell = 99 - j;
        board.pieces[cell] = Piece::new(kind, Color::Blue, j as u8);
        board.zero_types[1][j] = kind as u8;
        num_hidden[1][kind as usize] += 1;
        unmoved[1] += 1;
    }

    board.num_hidden = num_hidden;
    board.num_hidden_unmoved = unmoved;
    board
}
