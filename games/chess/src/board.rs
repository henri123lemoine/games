//! Mailbox board: piece placement, FEN, move application, attack detection,
//! and a Zobrist-style position key.

use std::fmt;
use std::str::FromStr;

use game_core::hash::splitmix64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    White,
    Black,
}

impl Color {
    pub fn flip(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    pub fn index(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Piece {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

impl Piece {
    pub fn value(self) -> i32 {
        [100, 320, 330, 500, 900, 0][self as usize]
    }

    fn to_char(self, color: Color) -> char {
        let c = b"pnbrqk"[self as usize] as char;
        match color {
            Color::White => c.to_ascii_uppercase(),
            Color::Black => c,
        }
    }

    fn from_char(c: char) -> Option<(Color, Piece)> {
        let color = if c.is_ascii_uppercase() {
            Color::White
        } else {
            Color::Black
        };
        let piece = match c.to_ascii_lowercase() {
            'p' => Piece::Pawn,
            'n' => Piece::Knight,
            'b' => Piece::Bishop,
            'r' => Piece::Rook,
            'q' => Piece::Queen,
            'k' => Piece::King,
            _ => return None,
        };
        Some((color, piece))
    }
}

pub const CASTLE_WK: u8 = 1;
pub const CASTLE_WQ: u8 = 2;
pub const CASTLE_BK: u8 = 4;
pub const CASTLE_BQ: u8 = 8;

/// Squares are 0..64, a1 = 0, h1 = 7, a8 = 56 (file = sq % 8, rank = sq / 8).
pub fn square_at(file: i8, rank: i8) -> Option<u8> {
    if (0..8).contains(&file) && (0..8).contains(&rank) {
        Some((rank * 8 + file) as u8)
    } else {
        None
    }
}

fn square_name(sq: u8) -> String {
    let file = (b'a' + sq % 8) as char;
    let rank = (b'1' + sq / 8) as char;
    format!("{file}{rank}")
}

fn parse_square(s: &str) -> Option<u8> {
    let bytes = s.as_bytes();
    if bytes.len() != 2 {
        return None;
    }
    let file = bytes[0].checked_sub(b'a')?;
    let rank = bytes[1].checked_sub(b'1')?;
    if file < 8 && rank < 8 {
        Some(rank * 8 + file)
    } else {
        None
    }
}

/// A move in coordinate form. Castling is encoded as the king's two-square
/// move; en passant as the pawn's diagonal step onto the en-passant square.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Move {
    pub from: u8,
    pub to: u8,
    pub promo: Option<Piece>,
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", square_name(self.from), square_name(self.to))?;
        if let Some(p) = self.promo {
            write!(f, "{}", p.to_char(Color::Black))?;
        }
        Ok(())
    }
}

impl FromStr for Move {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.len() != 4 && s.len() != 5 {
            return Err(format!("bad move '{s}': expected UCI like e2e4 or e7e8q"));
        }
        let from = parse_square(&s[0..2]).ok_or_else(|| format!("bad from-square in '{s}'"))?;
        let to = parse_square(&s[2..4]).ok_or_else(|| format!("bad to-square in '{s}'"))?;
        let promo = match s.as_bytes().get(4) {
            None => None,
            Some(b'q') => Some(Piece::Queen),
            Some(b'r') => Some(Piece::Rook),
            Some(b'b') => Some(Piece::Bishop),
            Some(b'n') => Some(Piece::Knight),
            Some(_) => return Err(format!("bad promotion piece in '{s}'")),
        };
        Ok(Move { from, to, promo })
    }
}

#[derive(Debug, Clone)]
pub struct Board {
    pub squares: [Option<(Color, Piece)>; 64],
    pub stm: Color,
    pub castling: u8,
    pub ep: Option<u8>,
    pub halfmove: u16,
    pub fullmove: u16,
    pub kings: [u8; 2],
}

pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

pub(crate) const KNIGHT_DELTAS: [(i8, i8); 8] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
];
pub(crate) const KING_DELTAS: [(i8, i8); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];

/// Ray directions indexed N, S, E, W, NE, NW, SE, SW; the first four are the
/// rook directions. `DIR_DELTAS` is the square-index step per direction and
/// `RAY_LEN[sq][dir]` the number of on-board steps from `sq` to the edge.
pub(crate) mod dir {
    pub const N: usize = 0;
    pub const S: usize = 1;
    pub const E: usize = 2;
    pub const W: usize = 3;
    pub const NE: usize = 4;
    pub const NW: usize = 5;
    pub const SE: usize = 6;
    pub const SW: usize = 7;
}
pub(crate) const DIR_DELTAS: [i8; 8] = [8, -8, 1, -1, 9, 7, -7, -9];
const DIR_STEPS: [(i8, i8); 8] = [
    (0, 1),
    (0, -1),
    (1, 0),
    (-1, 0),
    (1, 1),
    (-1, 1),
    (1, -1),
    (-1, -1),
];

const fn leaper_attacks(deltas: [(i8, i8); 8]) -> [u64; 64] {
    let mut table = [0u64; 64];
    let mut sq = 0;
    while sq < 64 {
        let mut i = 0;
        while i < 8 {
            let f = (sq % 8) as i8 + deltas[i].0;
            let r = (sq / 8) as i8 + deltas[i].1;
            if f >= 0 && f < 8 && r >= 0 && r < 8 {
                table[sq] |= 1u64 << (r * 8 + f);
            }
            i += 1;
        }
        sq += 1;
    }
    table
}

/// `[color][sq]`: squares from which a pawn of `color` attacks `sq`.
const fn pawn_attacker_squares() -> [[u64; 64]; 2] {
    let mut table = [[0u64; 64]; 2];
    let mut sq = 0;
    while sq < 64 {
        let mut color = 0;
        while color < 2 {
            let dr: i8 = if color == 0 { -1 } else { 1 };
            let mut k = 0;
            while k < 2 {
                let df: i8 = if k == 0 { -1 } else { 1 };
                let f = (sq % 8) as i8 + df;
                let r = (sq / 8) as i8 + dr;
                if f >= 0 && f < 8 && r >= 0 && r < 8 {
                    table[color][sq] |= 1u64 << (r * 8 + f);
                }
                k += 1;
            }
            color += 1;
        }
        sq += 1;
    }
    table
}

const fn ray_lengths() -> [[u8; 8]; 64] {
    let mut table = [[0u8; 8]; 64];
    let mut sq = 0;
    while sq < 64 {
        let mut d = 0;
        while d < 8 {
            let mut f = (sq % 8) as i8 + DIR_STEPS[d].0;
            let mut r = (sq / 8) as i8 + DIR_STEPS[d].1;
            while f >= 0 && f < 8 && r >= 0 && r < 8 {
                table[sq][d] += 1;
                f += DIR_STEPS[d].0;
                r += DIR_STEPS[d].1;
            }
            d += 1;
        }
        sq += 1;
    }
    table
}

const KNIGHT_ATTACKS: [u64; 64] = leaper_attacks(KNIGHT_DELTAS);
const KING_ATTACKS: [u64; 64] = leaper_attacks(KING_DELTAS);
const PAWN_ATTACKERS: [[u64; 64]; 2] = pawn_attacker_squares();
pub(crate) const RAY_LEN: [[u8; 8]; 64] = ray_lengths();

fn castle_rights_lost(sq: u8) -> u8 {
    match sq {
        0 => CASTLE_WQ,
        7 => CASTLE_WK,
        4 => CASTLE_WK | CASTLE_WQ,
        56 => CASTLE_BQ,
        63 => CASTLE_BK,
        60 => CASTLE_BK | CASTLE_BQ,
        _ => 0,
    }
}

impl Board {
    pub fn start() -> Board {
        Board::from_fen(START_FEN).expect("start FEN is valid")
    }

    pub fn from_fen(fen: &str) -> Result<Board, String> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(format!("FEN '{fen}' needs at least 4 fields"));
        }

        let mut squares = [None; 64];
        let ranks: Vec<&str> = parts[0].split('/').collect();
        if ranks.len() != 8 {
            return Err(format!("FEN placement needs 8 ranks, got {}", ranks.len()));
        }
        for (i, rank_str) in ranks.iter().enumerate() {
            let rank = 7 - i;
            let mut file = 0usize;
            for c in rank_str.chars() {
                if let Some(skip) = c.to_digit(10) {
                    file += skip as usize;
                } else if let Some(cp) = Piece::from_char(c) {
                    if file >= 8 {
                        return Err(format!("FEN rank '{rank_str}' overflows"));
                    }
                    squares[rank * 8 + file] = Some(cp);
                    file += 1;
                } else {
                    return Err(format!("bad FEN piece char '{c}'"));
                }
            }
            if file != 8 {
                return Err(format!("FEN rank '{rank_str}' has {file} files"));
            }
        }

        let stm = match parts[1] {
            "w" => Color::White,
            "b" => Color::Black,
            other => return Err(format!("bad side-to-move '{other}'")),
        };

        let mut castling = 0;
        if parts[2] != "-" {
            for c in parts[2].chars() {
                castling |= match c {
                    'K' => CASTLE_WK,
                    'Q' => CASTLE_WQ,
                    'k' => CASTLE_BK,
                    'q' => CASTLE_BQ,
                    other => return Err(format!("bad castling char '{other}'")),
                };
            }
        }

        let ep = if parts[3] == "-" {
            None
        } else {
            Some(parse_square(parts[3]).ok_or_else(|| format!("bad ep square '{}'", parts[3]))?)
        };

        let halfmove = parts
            .get(4)
            .map(|s| s.parse().map_err(|_| format!("bad halfmove '{s}'")))
            .transpose()?
            .unwrap_or(0);
        let fullmove = parts
            .get(5)
            .map(|s| s.parse().map_err(|_| format!("bad fullmove '{s}'")))
            .transpose()?
            .unwrap_or(1);

        let mut kings = [64u8; 2];
        for (sq, cell) in squares.iter().enumerate() {
            if let Some((c, Piece::King)) = cell {
                kings[c.index()] = sq as u8;
            }
        }
        if kings[0] == 64 || kings[1] == 64 {
            return Err("FEN must contain both kings".to_string());
        }

        Ok(Board {
            squares,
            stm,
            castling,
            ep,
            halfmove,
            fullmove,
            kings,
        })
    }

    pub fn to_fen(&self) -> String {
        let mut fen = String::new();
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                match self.squares[rank * 8 + file] {
                    None => empty += 1,
                    Some((c, p)) => {
                        if empty > 0 {
                            fen.push_str(&empty.to_string());
                            empty = 0;
                        }
                        fen.push(p.to_char(c));
                    }
                }
            }
            if empty > 0 {
                fen.push_str(&empty.to_string());
            }
            if rank > 0 {
                fen.push('/');
            }
        }
        fen.push(' ');
        fen.push(match self.stm {
            Color::White => 'w',
            Color::Black => 'b',
        });
        fen.push(' ');
        if self.castling == 0 {
            fen.push('-');
        } else {
            for (bit, c) in [
                (CASTLE_WK, 'K'),
                (CASTLE_WQ, 'Q'),
                (CASTLE_BK, 'k'),
                (CASTLE_BQ, 'q'),
            ] {
                if self.castling & bit != 0 {
                    fen.push(c);
                }
            }
        }
        fen.push(' ');
        match self.ep {
            None => fen.push('-'),
            Some(sq) => fen.push_str(&square_name(sq)),
        }
        fen.push_str(&format!(" {} {}", self.halfmove, self.fullmove));
        fen
    }

    /// Applies a move assumed to be legal (or at least pseudo-legal: legality
    /// is then checked by probing whether the mover's king is attacked).
    pub fn apply(&mut self, m: Move) {
        let from = m.from as usize;
        let to = m.to as usize;
        let (color, piece) = self.squares[from].expect("apply: empty from-square");
        let mut capture = self.squares[to].is_some();

        if piece == Piece::Pawn && self.ep == Some(m.to) && m.from % 8 != m.to % 8 {
            let captured_sq = match color {
                Color::White => to - 8,
                Color::Black => to + 8,
            };
            self.squares[captured_sq] = None;
            capture = true;
        }

        self.squares[from] = None;
        self.squares[to] = Some((color, m.promo.unwrap_or(piece)));

        if piece == Piece::King {
            self.kings[color.index()] = m.to;
            if m.to == m.from + 2 {
                self.squares[from + 1] = self.squares[from + 3].take();
            } else if m.to + 2 == m.from {
                self.squares[from - 1] = self.squares[from - 4].take();
            }
        }

        self.castling &= !(castle_rights_lost(m.from) | castle_rights_lost(m.to));

        self.ep = None;
        if piece == Piece::Pawn {
            let (lo, hi) = if from < to { (from, to) } else { (to, from) };
            if hi - lo == 16 {
                self.ep = Some(((lo + hi) / 2) as u8);
            }
        }

        if piece == Piece::Pawn || capture {
            self.halfmove = 0;
        } else {
            self.halfmove += 1;
        }
        if color == Color::Black {
            self.fullmove += 1;
        }
        self.stm = color.flip();
    }

    fn any_piece_on(&self, mut bb: u64, want: (Color, Piece)) -> bool {
        while bb != 0 {
            let s = bb.trailing_zeros() as usize;
            bb &= bb - 1;
            if self.squares[s] == Some(want) {
                return true;
            }
        }
        false
    }

    pub fn is_attacked(&self, sq: u8, by: Color) -> bool {
        let sq = sq as usize;
        if self.any_piece_on(PAWN_ATTACKERS[by.index()][sq], (by, Piece::Pawn))
            || self.any_piece_on(KNIGHT_ATTACKS[sq], (by, Piece::Knight))
            || self.any_piece_on(KING_ATTACKS[sq], (by, Piece::King))
        {
            return true;
        }

        for d in 0..8 {
            let slider = if d < 4 { Piece::Rook } else { Piece::Bishop };
            let mut s = sq as i32;
            for _ in 0..RAY_LEN[sq][d] {
                s += DIR_DELTAS[d] as i32;
                if let Some((c, p)) = self.squares[s as usize] {
                    if c == by && (p == slider || p == Piece::Queen) {
                        return true;
                    }
                    break;
                }
            }
        }

        false
    }

    pub fn in_check(&self, color: Color) -> bool {
        self.is_attacked(self.kings[color.index()], color.flip())
    }

    /// Covers K vs K and K + single minor vs K. Richer dead-position cases
    /// (e.g. K+B vs K+B with same-colored bishops) are intentionally skipped.
    pub fn insufficient_material(&self) -> bool {
        let mut minors = 0;
        for cell in self.squares.iter().flatten() {
            match cell.1 {
                Piece::King => {}
                Piece::Knight | Piece::Bishop => minors += 1,
                _ => return false,
            }
        }
        minors <= 1
    }

    /// 64-bit hash of the full position: placement, side to move, castling
    /// rights, en-passant square, and the halfmove clock. The clock must be
    /// in the key because terminality (the 50-move rule) depends on it — the
    /// same placement one ply from the draw and at clock zero have different
    /// game values, and the search's transposition table keys on this hash.
    /// (The cost is fewer TT hits across lines that differ only in clock.)
    pub fn key(&self) -> u64 {
        let mut h = 0u64;
        for (sq, cell) in self.squares.iter().enumerate() {
            if let Some((c, p)) = cell {
                h ^= splitmix64((sq * 12 + c.index() * 6 + *p as usize) as u64 + 1);
            }
        }
        if self.stm == Color::Black {
            h ^= splitmix64(0x1000);
        }
        h ^= splitmix64(0x2000 + self.castling as u64);
        if let Some(ep) = self.ep {
            h ^= splitmix64(0x3000 + ep as u64);
        }
        h ^= splitmix64(0x4000 + self.halfmove as u64);
        h
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for rank in (0..8).rev() {
            write!(f, "{}  ", rank + 1)?;
            for file in 0..8 {
                let c = match self.squares[rank * 8 + file] {
                    None => '.',
                    Some((color, piece)) => piece.to_char(color),
                };
                write!(f, "{c} ")?;
            }
            writeln!(f)?;
        }
        write!(f, "\n   a b c d e f g h")
    }
}
