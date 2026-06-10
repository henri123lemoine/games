//! Connect-4 as a [`game_core::Game`].
//!
//! Standard 7x6 rules: player 0 moves first, four in a row in any direction
//! wins, a full board with no winner is a draw. Perfect information, no
//! chance nodes. Each player's stones live in a bitboard laid out 7 bits per
//! column (rows 0-5 plus an always-empty spare bit), so the spare row stops
//! shift-based win probes from wrapping between columns.

mod ui;

use game_core::{Eval, Game, Turn};

pub const COLS: usize = 7;
pub const ROWS: usize = 6;
const CELLS: u8 = (COLS * ROWS) as u8;
const BITS_PER_COL: usize = 7;
const CENTER_COL: usize = 3;

#[derive(Clone, Debug)]
pub struct Connect4State {
    stones: [u64; 2],
    heights: [u8; COLS],
    moves: u8,
    winner: Option<usize>,
}

impl Connect4State {
    pub(crate) fn mover(&self) -> usize {
        (self.moves % 2) as usize
    }

    pub(crate) fn stone_at(&self, col: usize, row: usize) -> Option<usize> {
        let bit = cell_bit(col, row);
        if self.stones[0] & bit != 0 {
            Some(0)
        } else if self.stones[1] & bit != 0 {
            Some(1)
        } else {
            None
        }
    }

    fn key(&self) -> u64 {
        let side = (self.moves & 1) as u64;
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for chunk in [self.stones[0], self.stones[1], side] {
            h = (h ^ chunk).wrapping_mul(0x100_0000_01b3);
        }
        h ^ (h >> 31)
    }
}

const fn cell_bit(col: usize, row: usize) -> u64 {
    1u64 << (col * BITS_PER_COL + row)
}

const fn column_mask(col: usize) -> u64 {
    0x3F << (col * BITS_PER_COL)
}

fn four_in_a_row(bb: u64) -> bool {
    const DIRS: [u32; 4] = [
        1,
        BITS_PER_COL as u32,
        BITS_PER_COL as u32 - 1,
        BITS_PER_COL as u32 + 1,
    ];
    DIRS.iter().any(|&d| {
        let pairs = bb & (bb >> d);
        pairs & (pairs >> (2 * d)) != 0
    })
}

/// Player 0 ("X") moves first. Actions are 0-based column indices, and
/// [`Game::legal_actions`] lists them left to right so a menu index is the
/// column itself when no column is full.
pub struct Connect4;

impl Game for Connect4 {
    type State = Connect4State;
    type Action = u8;

    fn initial_state(&self) -> Connect4State {
        Connect4State {
            stones: [0; 2],
            heights: [0; COLS],
            moves: 0,
            winner: None,
        }
    }

    fn turn(&self, state: &Connect4State) -> Turn {
        Turn::Player(state.mover())
    }

    fn is_terminal(&self, state: &Connect4State) -> bool {
        state.winner.is_some() || state.moves == CELLS
    }

    fn returns(&self, state: &Connect4State, player: usize) -> f64 {
        match state.winner {
            Some(w) if w == player => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        }
    }

    fn legal_actions(&self, state: &Connect4State) -> Vec<u8> {
        (0..COLS as u8)
            .filter(|&c| state.heights[c as usize] < ROWS as u8)
            .collect()
    }

    fn chance_outcomes(&self, _state: &Connect4State) -> Vec<(u8, f64)> {
        vec![]
    }

    fn apply(&self, state: &mut Connect4State, action: u8) {
        let col = action as usize;
        debug_assert!(state.heights[col] < ROWS as u8, "column {col} is full");
        let mover = state.mover();
        state.stones[mover] |= cell_bit(col, state.heights[col] as usize);
        state.heights[col] += 1;
        state.moves += 1;
        if four_in_a_row(state.stones[mover]) {
            state.winner = Some(mover);
        }
    }

    fn infoset_key(&self, state: &Connect4State, _player: usize) -> u64 {
        state.key()
    }

    fn state_key(&self, state: &Connect4State) -> Option<u64> {
        Some(state.key())
    }
}

const WINDOWS: usize = 69;

const fn window_masks() -> [u64; WINDOWS] {
    let dirs: [(isize, isize); 4] = [(1, 0), (0, 1), (1, 1), (1, -1)];
    let mut masks = [0u64; WINDOWS];
    let mut n = 0;
    let mut d = 0;
    while d < dirs.len() {
        let (dc, dr) = dirs[d];
        let mut col = 0;
        while col < COLS as isize {
            let mut row = 0;
            while row < ROWS as isize {
                let (end_c, end_r) = (col + 3 * dc, row + 3 * dr);
                if 0 <= end_c && end_c < COLS as isize && 0 <= end_r && end_r < ROWS as isize {
                    let mut m = 0u64;
                    let mut i = 0;
                    while i < 4 {
                        m |= 1u64 << (((col + i * dc) * BITS_PER_COL as isize) + row + i * dr);
                        i += 1;
                    }
                    masks[n] = m;
                    n += 1;
                }
                row += 1;
            }
            col += 1;
        }
        d += 1;
    }
    masks
}

static WINDOW_MASKS: [u64; WINDOWS] = window_masks();

const TWO_WEIGHT: i32 = 2;
const THREE_WEIGHT: i32 = 5;
const CENTER_WEIGHT: i32 = 3;

fn line_weight(stones: u32) -> i32 {
    match stones {
        2 => TWO_WEIGHT,
        3 => THREE_WEIGHT,
        _ => 0,
    }
}

/// Static evaluation: every length-4 window holding only one side's stones
/// scores by how filled it is (open twos and threes), plus a bonus per stone
/// in the center column. Squashed into `(-1, 1)` to stay on the returns scale.
pub struct Connect4Eval;

impl Eval<Connect4> for Connect4Eval {
    fn eval(&self, _game: &Connect4, state: &Connect4State, player: usize) -> f64 {
        let mine = state.stones[player];
        let theirs = state.stones[1 - player];
        let mut net = 0i32;
        for &w in &WINDOW_MASKS {
            let m = (mine & w).count_ones();
            let t = (theirs & w).count_ones();
            if t == 0 {
                net += line_weight(m);
            } else if m == 0 {
                net -= line_weight(t);
            }
        }
        let center = column_mask(CENTER_COL);
        net += CENTER_WEIGHT
            * ((mine & center).count_ones() as i32 - (theirs & center).count_ones() as i32);
        (f64::from(net) / 50.0).tanh()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn play_cols(cols: &[u8]) -> (Connect4, Connect4State) {
        let game = Connect4;
        let mut state = game.initial_state();
        for &c in cols {
            assert!(!game.is_terminal(&state), "premature terminal before {c}");
            game.apply(&mut state, c);
        }
        (game, state)
    }

    fn assert_win(cols: &[u8], winner: usize) {
        let (game, state) = play_cols(cols);
        assert!(game.is_terminal(&state));
        assert_eq!(game.returns(&state, winner), 1.0);
        assert_eq!(game.returns(&state, 1 - winner), -1.0);
    }

    #[test]
    fn vertical_win() {
        assert_win(&[0, 1, 0, 1, 0, 1, 0], 0);
    }

    #[test]
    fn horizontal_win() {
        assert_win(&[0, 0, 1, 1, 2, 2, 3], 0);
    }

    #[test]
    fn diagonal_up_right_win() {
        assert_win(&[0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3], 0);
    }

    #[test]
    fn diagonal_up_left_win() {
        assert_win(&[3, 2, 2, 1, 1, 0, 1, 0, 0, 6, 0], 0);
    }

    #[test]
    fn second_player_can_win() {
        assert_win(&[6, 0, 6, 1, 5, 2, 5, 3], 1);
    }

    #[test]
    fn draw_on_full_board() {
        let drawn_game: [u8; 42] = [
            1, 3, 1, 1, 1, 1, 1, 3, 5, 2, 3, 0, 4, 3, 2, 5, 5, 2, 6, 0, 6, 4, 0, 0, 0, 4, 2, 6, 3,
            0, 5, 4, 2, 2, 4, 4, 6, 5, 5, 6, 6, 3,
        ];
        let (game, state) = play_cols(&drawn_game);
        assert!(game.is_terminal(&state));
        assert!(game.legal_actions(&state).is_empty());
        assert_eq!(game.returns(&state, 0), 0.0);
        assert_eq!(game.returns(&state, 1), 0.0);
    }

    #[test]
    fn legal_actions_are_left_to_right_and_drop_full_columns() {
        let game = Connect4;
        assert_eq!(
            game.legal_actions(&game.initial_state()),
            vec![0, 1, 2, 3, 4, 5, 6]
        );
        let (game, state) = play_cols(&[2, 2, 2, 2, 2, 2]);
        assert_eq!(game.legal_actions(&state), vec![0, 1, 3, 4, 5, 6]);
    }

    #[test]
    fn state_key_distinguishes_side_to_move() {
        let game = Connect4;
        let (_, a) = play_cols(&[0]);
        let (_, b) = play_cols(&[0, 1]);
        assert_ne!(game.state_key(&a), game.state_key(&b));
        assert_ne!(game.state_key(&a), game.state_key(&game.initial_state()));
    }

    #[test]
    fn window_masks_are_all_four_cells() {
        for &w in &WINDOW_MASKS {
            assert_eq!(w.count_ones(), 4);
        }
    }
}
