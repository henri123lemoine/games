//! The packed Stratego board and its piece/death types.
//!
//! Mirrors `stratego_board.h` from the Ataraxos reference: every field the rule
//! kernels read lives here, in idiomatic packed Rust rather than the byte-exact
//! `alignas(128)` CUDA layout (we never load their raw snapshots). Coordinates
//! are absolute `cell = 10*row + col` over a 10x10 grid; the per-player 180deg
//! point-reflection view (`99 - cell`) is applied at the encoding boundary, not
//! stored here.

pub const BOARD_SIZE: usize = 10;
pub const NUM_CELLS: usize = 100;
pub const HOME_CELLS: usize = 40;

/// Lake cells, baked in as `LAKE`-colored pieces.
pub const LAKES: [usize; 8] = [42, 43, 46, 47, 52, 53, 56, 57];

/// Classic per-type starting counts, indexed by [`PieceType`] value
/// (spy..bomb), summing to 40.
pub const CLASSIC_INITIAL_COUNTS: [u8; 12] = [1, 8, 5, 4, 4, 4, 3, 2, 1, 1, 1, 6];
/// Barrage variant: 8 movable + flag + bomb, the rest empty (32 empties).
pub const BARRAGE_INITIAL_COUNTS: [u8; 12] = [1, 2, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1];

/// Piece ranks and the immovable/terrain sentinels. Numeric value is the rank:
/// higher wins a fair fight, with the documented special-case exceptions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum PieceType {
    Spy = 0,
    Scout = 1,
    Miner = 2,
    Sergeant = 3,
    Lieutenant = 4,
    Captain = 5,
    Major = 6,
    Colonel = 7,
    General = 8,
    Marshal = 9,
    Flag = 10,
    Bomb = 11,
    Lake = 12,
    Empty = 13,
}

/// The "unknown opponent piece" sentinel used by the threat/protection bitsets.
pub const HIDDEN_PIECE: u8 = 15;

impl PieceType {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Spy,
            1 => Self::Scout,
            2 => Self::Miner,
            3 => Self::Sergeant,
            4 => Self::Lieutenant,
            5 => Self::Captain,
            6 => Self::Major,
            7 => Self::Colonel,
            8 => Self::General,
            9 => Self::Marshal,
            10 => Self::Flag,
            11 => Self::Bomb,
            12 => Self::Lake,
            _ => Self::Empty,
        }
    }

    /// A piece that can be ordered to move (rank below `Flag`).
    pub fn is_movable(self) -> bool {
        (self as u8) < PieceType::Flag as u8
    }

    pub fn is_scout(self) -> bool {
        self == PieceType::Scout
    }
}

/// Cell occupancy color. Matches the reference encoding 0=empty, 1=red, 2=blue,
/// 3=lake; player `p` (0/1) owns color `p + 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Color {
    Empty = 0,
    Red = 1,
    Blue = 2,
    Lake = 3,
}

impl Color {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Red,
            2 => Self::Blue,
            3 => Self::Lake,
            _ => Self::Empty,
        }
    }

    /// The color owned by player index `p` (0 -> red, 1 -> blue).
    pub fn of_player(p: usize) -> Self {
        if p == 0 { Color::Red } else { Color::Blue }
    }

    /// Player index for a piece color, or `None` for empty/lake.
    pub fn player(self) -> Option<usize> {
        match self {
            Color::Red => Some(0),
            Color::Blue => Some(1),
            _ => None,
        }
    }
}

/// Why a piece died, used by the M2 death-status feature channels.
/// Mirrors `DeathReason` in `stratego_board.h:71-93`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DeathReason {
    AttackedVisibleStronger = 0,
    AttackedVisibleTie = 1,
    AttackedHidden = 2,
    VisibleDefendedWeaker = 3,
    VisibleDefendedTie = 4,
    HiddenDefended = 5,
}

/// Per-piece-id death record. The `reason` classification feeds the NN and is
/// recorded for completeness; the rules engine only reads `is_dead`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DeathStatus {
    pub is_dead: bool,
    pub reason: u8,
    pub piece_type: u8,
    pub death_location: u8,
}

/// One board square's occupant. The seven 16-bit per-type bitset groups back
/// the threat/protection feature channels; each bit `t` records piece type `t`
/// (`HIDDEN_PIECE = 15` for an unknown opponent), maintained by `rules::apply`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Piece {
    pub kind: PieceType,
    pub color: Color,
    pub visible: bool,
    pub has_moved: bool,
    /// Starting-square identity in `[0, 39]`, or `0xff` for empty/lake.
    pub piece_id: u8,
    pub threatened: u16,
    pub evaded: u16,
    pub actively_adjacent: u16,
    pub protected_: u16,
    pub protected_against: u16,
    pub was_protected_by: u16,
    pub was_protected_against: u16,
}

impl Piece {
    pub const EMPTY: Piece = Piece {
        kind: PieceType::Empty,
        color: Color::Empty,
        visible: true,
        has_moved: false,
        piece_id: 0xff,
        threatened: 0,
        evaded: 0,
        actively_adjacent: 0,
        protected_: 0,
        protected_against: 0,
        was_protected_by: 0,
        was_protected_against: 0,
    };

    pub const LAKE: Piece = Piece {
        kind: PieceType::Lake,
        color: Color::Lake,
        visible: true,
        ..Piece::EMPTY
    };

    pub fn new(kind: PieceType, color: Color, piece_id: u8) -> Piece {
        Piece {
            kind,
            color,
            visible: false,
            has_moved: false,
            piece_id,
            ..Piece::EMPTY
        }
    }

    pub fn is_empty(&self) -> bool {
        self.kind == PieceType::Empty
    }

    /// The type bit the threat/protection trackers key on: the true type when
    /// visible, else [`HIDDEN_PIECE`]. Mirrors `piece.visible ? piece.type :
    /// HIDDEN_PIECE` throughout `action_kernels.cu`.
    #[inline]
    pub fn tracked_type(&self) -> u8 {
        if self.visible {
            self.kind as u8
        } else {
            HIDDEN_PIECE
        }
    }
}

/// Reads one type bit from a 16-bit per-type bitset (`field[t/8] & (1<<(t%8))`).
#[inline]
pub fn bitset_get(field: u16, t: u8) -> bool {
    field & (1 << t) != 0
}

/// Sets one type bit in a 16-bit per-type bitset. The reference stores these as
/// two bytes indexed `field[t/8] |= 1<<(t%8)`; packed into a `u16` that is the
/// same as `field |= 1<<t` for `t in [0,16)`.
#[inline]
pub fn bitset_set(field: &mut u16, t: u8) {
    *field |= 1 << t;
}

/// Whether two absolute cells are orthogonally adjacent. Mirrors the
/// `IS_ADJACENT` macro, including the row-wrap guards.
#[inline]
pub fn is_adjacent(src: usize, dst: usize) -> bool {
    (dst == src + 1 && !dst.is_multiple_of(10))
        || (src >= 1 && dst + 1 == src && !src.is_multiple_of(10))
        || (dst == src + 10)
        || (src >= 10 && dst + 10 == src)
}

/// The packed game state: the grid plus every counter and per-color restriction
/// substate the rule kernels maintain. `Clone` is a cheap memcpy of inline
/// arrays, as required for search/self-play branching.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Board {
    pub pieces: [Piece; NUM_CELLS],
    /// Hidden-piece counts per player, per type `[0, 12)`.
    pub num_hidden: [[u8; 12]; 2],
    /// Hidden pieces that have never moved, per player.
    pub num_hidden_unmoved: [u8; 2],
    /// Bitset (over piece_id) of dead pieces, per player; 5 bytes covers 40 ids.
    pub deaths: [[u8; 5]; 2],
    pub death_status: [[DeathStatus; HOME_CELLS]; 2],
    /// Destination (absolute) of the previous / previous-previous move, or
    /// `0xff` when unset. Used by the feature trackers and chase update.
    pub prev_dst: u8,
    pub prev_prev_dst: u8,
    /// Type of the last moved piece (`HIDDEN_PIECE` if it was hidden, `0xff`
    /// after an attack/death).
    pub last_moved_piece_type: u8,
    /// Total half-moves since the game (move phase) began.
    pub num_moves: u32,
    /// Half-moves since the most recent attack (reset to 0 on any battle).
    pub num_moves_since_last_attack: u32,
    /// Two-square substate, per player.
    pub twosquare: [crate::twosquare::TwosquareState; 2],
    /// The continuous-chase counters/history (`chase_length`, last own
    /// move, and a bounded board-snapshot window), a faithful port of the
    /// reference `chase_state.cu` kernel. Re-seeded from the fully-placed
    /// board once deployment finishes (`board_from_arrangements`).
    pub chase: crate::chase::ChaseState,
    /// The starting type at each `piece_id` slot, per player, or `0xff` for an
    /// empty home cell. This is `d_zero_boards` in the reference: the cemetery
    /// channels read the dead piece's type from the initial arrangement, not the
    /// (now-vacated) board square.
    pub zero_types: [[u8; HOME_CELLS]; 2],
    /// The 1800-slot action index of every move played so far, in the *acting
    /// player's* POV encoding (exactly the integer the reference stores in
    /// `d_action_history`). The encoder reconstructs the src/dst history planes
    /// from the tail of this buffer.
    pub action_history: Vec<u16>,
}

/// The 6-byte per-move record (`action_kernels.cu:159-164`), enough for the
/// chase/two-square machines: source and destination cells (acting POV), the
/// encoded source/destination piece descriptors, and the piece ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoveSummary {
    pub src_rel: u8,
    pub dst_rel: u8,
    pub src_code: u8,
    pub dst_code: u8,
    pub src_id: u8,
    pub dst_id: u8,
}

/// The move-summary destination code for "moved onto an empty square" — the
/// no-attack marker the chase rule keys on (`EMPTY(13) | visible(16) = 29`).
pub const NO_ATTACK_DST_CODE: u8 = 29;

impl MoveSummary {
    pub fn is_attack(&self) -> bool {
        self.dst_code != NO_ATTACK_DST_CODE
    }
}

impl Board {
    /// An all-empty board with lakes placed and counters cleared. Deployment
    /// fills the home rows; [`Board::finish_deployment`] sets the counters.
    pub fn blank() -> Board {
        let mut pieces = [Piece::EMPTY; NUM_CELLS];
        for &c in &LAKES {
            pieces[c] = Piece::LAKE;
        }
        Board {
            pieces,
            num_hidden: [[0; 12]; 2],
            num_hidden_unmoved: [0; 2],
            deaths: [[0; 5]; 2],
            death_status: [[DeathStatus::default(); HOME_CELLS]; 2],
            prev_dst: 0xff,
            prev_prev_dst: 0xff,
            last_moved_piece_type: 0xff,
            num_moves: 0,
            num_moves_since_last_attack: 0,
            twosquare: [crate::twosquare::TwosquareState::default(); 2],
            chase: crate::chase::ChaseState::new_from_board_pieces(&pieces),
            zero_types: [[0xff; HOME_CELLS]; 2],
            action_history: Vec::new(),
        }
    }

    #[inline]
    pub fn at(&self, cell: usize) -> &Piece {
        &self.pieces[cell]
    }

    /// Marks `piece_id` of `player` dead in the bitset.
    pub fn mark_dead(&mut self, player: usize, piece_id: u8) {
        self.deaths[player][piece_id as usize / 8] |= 1 << (piece_id % 8);
    }

    /// Whether `piece_id` of `player` is recorded dead.
    pub fn is_dead(&self, player: usize, piece_id: u8) -> bool {
        self.deaths[player][piece_id as usize / 8] & (1 << (piece_id % 8)) != 0
    }
}
