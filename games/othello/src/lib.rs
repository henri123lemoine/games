//! Othello (Reversi) as a [`game_core::Game`].
//!
//! Perfect information, no chance nodes: `turn` is always a player decision
//! and `infoset_key` equals the canonical position key. Black is player 0 and
//! moves first. A placement must flip at least one line of opponent discs in
//! one of the 8 directions; a player with no placement passes (the single
//! legal action is then [`Move::Pass`]). The game ends when neither player can
//! place; the winner has the most discs (draws possible).
//!
//! The board is a pair of bitboards (bit `row * 8 + col`, row 0 = row 1,
//! col 0 = file a) with the classic shift-and-propagate move generator.

mod eval;
mod ui;

pub use eval::{OthelloEval, OthelloSpec};

use game_core::{Game, Turn};

const NOT_FILE_A: u64 = !0x0101_0101_0101_0101;
const NOT_FILE_H: u64 = !0x8080_8080_8080_8080;

fn shift(b: u64, dir: usize) -> u64 {
    match dir {
        0 => (b << 1) & NOT_FILE_A,
        1 => (b >> 1) & NOT_FILE_H,
        2 => b << 8,
        3 => b >> 8,
        4 => (b << 9) & NOT_FILE_A,
        5 => (b << 7) & NOT_FILE_H,
        6 => (b >> 7) & NOT_FILE_A,
        7 => (b >> 9) & NOT_FILE_H,
        _ => unreachable!(),
    }
}

/// Empty squares where `own` can legally place (each brackets >=1 `opp` disc).
pub(crate) fn placements(own: u64, opp: u64) -> u64 {
    let empty = !(own | opp);
    let mut moves = 0u64;
    for dir in 0..8 {
        let mut ray = shift(own, dir) & opp;
        for _ in 0..5 {
            ray |= shift(ray, dir) & opp;
        }
        moves |= shift(ray, dir) & empty;
    }
    moves
}

/// Discs flipped by `own` placing at `sq` (0 if the placement is illegal).
fn flips(own: u64, opp: u64, sq: u8) -> u64 {
    let placed = 1u64 << sq;
    let mut flipped = 0u64;
    for dir in 0..8 {
        let mut ray = shift(placed, dir) & opp;
        for _ in 0..5 {
            ray |= shift(ray, dir) & opp;
        }
        if shift(ray, dir) & own != 0 {
            flipped |= ray;
        }
    }
    flipped
}

fn mix(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^= x >> 33;
    x
}

/// A move: place a disc on square `row * 8 + col`, or pass (only legal when
/// no placement is).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Move {
    Place(u8),
    Pass,
}

/// Position: disc bitboards for each player plus the side to move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Board {
    black: u64,
    white: u64,
    to_move: usize,
}

impl Board {
    /// The standard start: white d4/e5, black e4/d5, Black to move.
    pub fn start() -> Self {
        Board {
            black: (1 << 28) | (1 << 35),
            white: (1 << 27) | (1 << 36),
            to_move: 0,
        }
    }

    fn own_opp(&self) -> (u64, u64) {
        if self.to_move == 0 {
            (self.black, self.white)
        } else {
            (self.white, self.black)
        }
    }

    /// Bitboard of legal placements for the side to move.
    pub fn placements(&self) -> u64 {
        let (own, opp) = self.own_opp();
        placements(own, opp)
    }

    /// Neither player can place a disc.
    pub fn is_over(&self) -> bool {
        let (own, opp) = self.own_opp();
        placements(own, opp) == 0 && placements(opp, own) == 0
    }

    /// Disc count for `player` (0 = Black, 1 = White).
    pub fn discs(&self, player: usize) -> u32 {
        if player == 0 {
            self.black.count_ones()
        } else {
            self.white.count_ones()
        }
    }

    pub fn side_to_move(&self) -> usize {
        self.to_move
    }

    fn apply(&mut self, action: Move) {
        if let Move::Place(sq) = action {
            let (own, opp) = self.own_opp();
            let flipped = flips(own, opp, sq);
            debug_assert!(flipped != 0, "illegal placement at {sq}");
            debug_assert!((own | opp) & (1 << sq) == 0, "square {sq} occupied");
            let gained = (1u64 << sq) | flipped;
            if self.to_move == 0 {
                self.black |= gained;
                self.white &= !flipped;
            } else {
                self.white |= gained;
                self.black &= !flipped;
            }
        }
        self.to_move ^= 1;
    }

    /// Canonical position hash (discs + side to move).
    pub fn key(&self) -> u64 {
        mix(self.black ^ mix(self.white ^ mix(self.to_move as u64 + 0x9e37_79b9_7f4a_7c15)))
    }

    pub(crate) fn bb(&self, player: usize) -> u64 {
        if player == 0 { self.black } else { self.white }
    }

    pub(crate) fn disc_at(&self, sq: u8) -> Option<usize> {
        let bit = 1u64 << sq;
        if self.black & bit != 0 {
            Some(0)
        } else if self.white & bit != 0 {
            Some(1)
        } else {
            None
        }
    }
}

/// Black is player 0, White is player 1.
pub struct Othello;

impl Game for Othello {
    type State = Board;
    type Action = Move;

    fn initial_state(&self) -> Board {
        Board::start()
    }

    fn turn(&self, state: &Board) -> Turn {
        Turn::Player(state.to_move)
    }

    fn is_terminal(&self, state: &Board) -> bool {
        state.is_over()
    }

    fn returns(&self, state: &Board, player: usize) -> f64 {
        let diff = state.discs(player) as i64 - state.discs(player ^ 1) as i64;
        diff.signum() as f64
    }

    fn legal_actions(&self, state: &Board) -> Vec<Move> {
        let mut bits = state.placements();
        if bits == 0 {
            return vec![Move::Pass];
        }
        let mut actions = Vec::with_capacity(bits.count_ones() as usize);
        while bits != 0 {
            actions.push(Move::Place(bits.trailing_zeros() as u8));
            bits &= bits - 1;
        }
        actions
    }

    fn chance_outcomes(&self, _state: &Board) -> Vec<(Move, f64)> {
        vec![]
    }

    fn apply(&self, state: &mut Board, action: Move) {
        state.apply(action);
    }

    fn infoset_key(&self, state: &Board, _player: usize) -> u64 {
        state.key()
    }

    fn state_key(&self, state: &Board) -> Option<u64> {
        Some(state.key())
    }
}
