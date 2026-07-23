use std::fmt;
use std::sync::Arc;

use game_core::hash::splitmix64;

pub const SIDE: usize = 14;
pub const CELLS: usize = SIDE * SIDE;
pub const NONE_SQUARE: u8 = u8::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Color {
    Red = 0,
    Blue = 1,
    Yellow = 2,
    Green = 3,
}

impl Color {
    pub const ALL: [Color; 4] = [Color::Red, Color::Blue, Color::Yellow, Color::Green];

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn from_index(index: usize) -> Color {
        Self::ALL[index]
    }

    pub const fn name(self) -> &'static str {
        match self {
            Color::Red => "Red",
            Color::Blue => "Blue",
            Color::Yellow => "Yellow",
            Color::Green => "Green",
        }
    }

    /// Forward from this army's home edge toward the center/opposite arm.
    pub const fn forward(self) -> (i8, i8) {
        match self {
            Color::Red => (0, 1),
            Color::Blue => (1, 0),
            Color::Yellow => (0, -1),
            Color::Green => (-1, 0),
        }
    }

    /// The player's right-hand direction while facing forward.
    pub const fn right(self) -> (i8, i8) {
        match self {
            Color::Red => (1, 0),
            Color::Blue => (0, -1),
            Color::Yellow => (-1, 0),
            Color::Green => (0, 1),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PieceKind {
    Pawn = 1,
    Knight = 2,
    Bishop = 3,
    Rook = 4,
    Queen = 5,
    King = 6,
}

impl PieceKind {
    pub const fn score(self, promoted: bool) -> i16 {
        match self {
            PieceKind::Pawn => 1,
            PieceKind::Knight => 3,
            PieceKind::Bishop | PieceKind::Rook => 5,
            PieceKind::Queen if promoted => 1,
            PieceKind::Queen => 9,
            PieceKind::King => 20,
        }
    }

    pub const fn letter(self) -> char {
        match self {
            PieceKind::Pawn => 'P',
            PieceKind::Knight => 'N',
            PieceKind::Bishop => 'B',
            PieceKind::Rook => 'R',
            PieceKind::Queen => 'Q',
            PieceKind::King => 'K',
        }
    }
}

/// One-byte piece: kind in bits 0..=2, owner in 3..=4, promoted-pawn flag in 5.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct Piece(u8);

impl Piece {
    pub const EMPTY: Piece = Piece(0);

    pub const fn new(color: Color, kind: PieceKind) -> Piece {
        Piece(kind as u8 | ((color as u8) << 3))
    }

    pub const fn promoted_queen(color: Color) -> Piece {
        Piece::new(color, PieceKind::Queen).with_promoted()
    }

    const fn with_promoted(self) -> Piece {
        Piece(self.0 | (1 << 5))
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub const fn color(self) -> Color {
        Color::from_index(((self.0 >> 3) & 3) as usize)
    }

    pub const fn kind(self) -> PieceKind {
        match self.0 & 7 {
            1 => PieceKind::Pawn,
            2 => PieceKind::Knight,
            3 => PieceKind::Bishop,
            4 => PieceKind::Rook,
            5 => PieceKind::Queen,
            6 => PieceKind::King,
            _ => panic!("empty piece has no kind"),
        }
    }

    pub const fn promoted(self) -> bool {
        self.0 & (1 << 5) != 0
    }

    pub const fn raw(self) -> u8 {
        self.0
    }
}

impl fmt::Debug for Piece {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return f.write_str("Empty");
        }
        f.debug_struct("Piece")
            .field("color", &self.color())
            .field("kind", &self.kind())
            .field("promoted", &self.promoted())
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Move {
    pub from: u8,
    pub to: u8,
}

impl Move {
    pub const fn new(from: u8, to: u8) -> Move {
        Move { from, to }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndReason {
    Ongoing,
    LastArmy,
    Repetition,
    FiftyMove,
    InsufficientMaterial,
    PlyCap,
}

#[derive(Debug)]
struct HistoryNode {
    key: u64,
    prev: Option<Arc<HistoryNode>>,
}

/// A perfect-information FFA position. Dead armies stay on the board and block
/// movement, but their pieces neither move nor attack and are worth no points.
#[derive(Debug, Clone)]
pub struct State {
    pub board: [Piece; CELLS],
    pub to_move: Color,
    /// One bit per army that can still move.
    pub active: u8,
    pub scores: [i16; 4],
    /// Two castling bits per army: bit `2*p` king-side, `2*p+1` queen-side.
    pub castling: u8,
    /// Destination of each army's most recent still-en-passant-eligible double push.
    pub en_passant: [u8; 4],
    /// Army credited for a currently checked king, or [`NONE_SQUARE`]. This is
    /// state because checkmate is adjudicated only when the victim's turn is
    /// reached; one or two other armies may move between check and mate.
    pub check_credit: [u8; 4],
    pub halfmove: u16,
    pub ply: u16,
    pub last_move: Option<Move>,
    pub end: EndReason,
    history: Option<Arc<HistoryNode>>,
}

impl State {
    pub fn standard() -> State {
        let mut state = State {
            board: [Piece::EMPTY; CELLS],
            to_move: Color::Red,
            active: 0b1111,
            scores: [0; 4],
            castling: u8::MAX,
            en_passant: [NONE_SQUARE; 4],
            check_credit: [NONE_SQUARE; 4],
            halfmove: 0,
            ply: 0,
            last_move: None,
            end: EndReason::Ongoing,
            history: None,
        };

        // Red is the canonical army; the others are exact quarter-turns.
        let back = [
            PieceKind::Rook,
            PieceKind::Knight,
            PieceKind::Bishop,
            PieceKind::Queen,
            PieceKind::King,
            PieceKind::Bishop,
            PieceKind::Knight,
            PieceKind::Rook,
        ];
        for color in Color::ALL {
            for (file, kind) in back.into_iter().enumerate() {
                let (home, pawn) = army_square(color, file);
                state.board[home as usize] = Piece::new(color, kind);
                state.board[pawn as usize] = Piece::new(color, PieceKind::Pawn);
            }
        }
        let key = state.repetition_key();
        state.history = Some(Arc::new(HistoryNode { key, prev: None }));
        state
    }

    #[cfg(test)]
    pub(crate) fn empty(to_move: Color) -> State {
        State {
            board: [Piece::EMPTY; CELLS],
            to_move,
            active: 0b1111,
            scores: [0; 4],
            castling: 0,
            en_passant: [NONE_SQUARE; 4],
            check_credit: [NONE_SQUARE; 4],
            halfmove: 0,
            ply: 0,
            last_move: None,
            end: EndReason::Ongoing,
            history: None,
        }
    }

    pub const fn is_active(&self, color: Color) -> bool {
        self.active & (1 << color.index()) != 0
    }

    pub fn piece_at(&self, square: u8) -> Option<Piece> {
        let piece = self.board[square as usize];
        (!piece.is_empty()).then_some(piece)
    }

    pub fn king_square(&self, color: Color) -> Option<u8> {
        self.board.iter().enumerate().find_map(|(square, &piece)| {
            (!piece.is_empty() && piece.color() == color && piece.kind() == PieceKind::King)
                .then_some(square as u8)
        })
    }

    pub fn repetition_key(&self) -> u64 {
        let mut key = splitmix64(u64::from(self.to_move as u8) | (u64::from(self.active) << 8));
        for (square, piece) in self.board.iter().enumerate() {
            if !piece.is_empty() {
                key ^= splitmix64(0x1000_0000 ^ (square as u64) ^ (u64::from(piece.raw()) << 16));
            }
        }
        key ^= splitmix64(0x2000_0000 ^ u64::from(self.castling));
        for (color, &square) in self.en_passant.iter().enumerate() {
            if square != NONE_SQUARE {
                key ^= splitmix64(0x3000_0000 ^ ((color as u64) << 8) ^ u64::from(square));
            }
        }
        for (victim, &credit) in self.check_credit.iter().enumerate() {
            if credit != NONE_SQUARE {
                key ^= splitmix64(0x3800_0000 ^ ((victim as u64) << 8) ^ u64::from(credit));
            }
        }
        key
    }

    pub fn state_key(&self) -> u64 {
        let mut key = self.repetition_key();
        for (seat, &score) in self.scores.iter().enumerate() {
            key ^= splitmix64(0x4000_0000 ^ ((seat as u64) << 32) ^ score as u16 as u64);
        }
        key ^= splitmix64(0x5000_0000 ^ u64::from(self.halfmove));
        // Training may opt into a ply cap, making otherwise identical boards
        // at different plies value-distinct. `end` likewise distinguishes an
        // adjudicated terminal from its final board position.
        key ^= splitmix64(0x6000_0000 ^ u64::from(self.ply));
        key ^= splitmix64(0x7000_0000 ^ self.end as u64);
        key
    }

    pub(crate) fn record_position(&mut self, irreversible: bool) -> u8 {
        let key = self.repetition_key();
        let prev = if irreversible {
            None
        } else {
            self.history.clone()
        };
        self.history = Some(Arc::new(HistoryNode { key, prev }));
        let mut count = 0u8;
        let mut node = self.history.as_deref();
        while let Some(history) = node {
            count += u8::from(history.key == key);
            if count >= 3 {
                break;
            }
            node = history.prev.as_deref();
        }
        count
    }
}

/// Board square for one item in an army's own left-to-right back rank and pawn rank.
fn army_square(color: Color, file: usize) -> (u8, u8) {
    let canonical_x = 3 + file as i8;
    match color {
        Color::Red => (
            square(canonical_x, 0).unwrap(),
            square(canonical_x, 1).unwrap(),
        ),
        Color::Blue => (
            square(0, 13 - canonical_x).unwrap(),
            square(1, 13 - canonical_x).unwrap(),
        ),
        Color::Yellow => (
            square(13 - canonical_x, 13).unwrap(),
            square(13 - canonical_x, 12).unwrap(),
        ),
        Color::Green => (
            square(13, canonical_x).unwrap(),
            square(12, canonical_x).unwrap(),
        ),
    }
}

pub const fn is_valid_xy(x: i8, y: i8) -> bool {
    if x < 0 || y < 0 || x >= SIDE as i8 || y >= SIDE as i8 {
        return false;
    }
    !((x < 3 || x > 10) && (y < 3 || y > 10))
}

pub const fn square(x: i8, y: i8) -> Option<u8> {
    if is_valid_xy(x, y) {
        Some((y as usize * SIDE + x as usize) as u8)
    } else {
        None
    }
}

pub const fn xy(square: u8) -> (i8, i8) {
    (
        (square as usize % SIDE) as i8,
        (square as usize / SIDE) as i8,
    )
}

pub fn square_name(square: u8) -> String {
    let (x, y) = xy(square);
    format!("{}{}", (b'a' + x as u8) as char, y + 1)
}

pub fn parse_square(text: &str) -> Option<u8> {
    let bytes = text.as_bytes();
    if !(2..=3).contains(&bytes.len()) || !(b'a'..=b'n').contains(&bytes[0]) {
        return None;
    }
    let rank: i8 = text[1..].parse().ok()?;
    square((bytes[0] - b'a') as i8, rank - 1)
}

pub(crate) fn add(square: u8, dx: i8, dy: i8) -> Option<u8> {
    let (x, y) = xy(square);
    self::square(x + dx, y + dy)
}

pub(crate) fn castle_bit(color: Color, king_side: bool) -> u8 {
    1 << (2 * color.index() + usize::from(!king_side))
}

pub(crate) fn home_king(color: Color) -> u8 {
    army_square(color, 4).0
}

pub(crate) fn home_rook(color: Color, king_side: bool) -> u8 {
    army_square(color, if king_side { 7 } else { 0 }).0
}
