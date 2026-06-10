//! Snake as a single-player [`game_core::Game`] with chance nodes for food.
//!
//! The snake starts at length 3, centered, heading right. Actions are
//! relative — [`SnakeAction::TurnLeft`], [`SnakeAction::Straight`],
//! [`SnakeAction::TurnRight`] — and all three are always legal; a move into a
//! wall or the body ends the game with the current score. The tail cell
//! vacates on the same tick the head advances, so moving into the tail is
//! safe unless the snake is eating.
//!
//! Food placement is a chance node: uniform over empty cells, at the start
//! and after each meal. Terminal states are death (crash), a full board
//! (win), and a starvation cap of `w*h` consecutive moves without eating.
//! [`Game::returns`] is snake length / board area, in `[0, 1]`.

mod ui;

use std::collections::VecDeque;

use game_core::{Eval, Game, Turn};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dir {
    Up,
    Right,
    Down,
    Left,
}

impl Dir {
    fn left(self) -> Self {
        match self {
            Dir::Up => Dir::Left,
            Dir::Left => Dir::Down,
            Dir::Down => Dir::Right,
            Dir::Right => Dir::Up,
        }
    }

    fn right(self) -> Self {
        self.left().left().left()
    }

    fn delta(self) -> (i32, i32) {
        match self {
            Dir::Up => (0, -1),
            Dir::Right => (1, 0),
            Dir::Down => (0, 1),
            Dir::Left => (-1, 0),
        }
    }
}

/// Why (or whether) the game has ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Alive,
    /// Hit a wall or the body; score frozen at the pre-move length.
    Crashed,
    /// `w*h` consecutive moves without eating.
    Starved,
    /// The snake fills the board.
    Won,
}

/// Relative moves plus the chance outcome that places food on a cell
/// (row-major index `y * w + x`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnakeAction {
    TurnLeft,
    Straight,
    TurnRight,
    Food(u16),
}

#[derive(Clone, Debug)]
pub struct SnakeState {
    /// Front is the head.
    body: VecDeque<(u8, u8)>,
    heading: Dir,
    food: Option<(u8, u8)>,
    hunger: u32,
    status: Status,
}

impl SnakeState {
    /// Current snake length (the score numerator).
    pub fn len(&self) -> usize {
        self.body.len()
    }

    /// A snake always has a body.
    pub fn is_empty(&self) -> bool {
        false
    }

    pub fn head(&self) -> (usize, usize) {
        let (x, y) = self.body[0];
        (x as usize, y as usize)
    }

    pub fn food(&self) -> Option<(usize, usize)> {
        self.food.map(|(x, y)| (x as usize, y as usize))
    }

    /// Moves since the last meal (or the start).
    pub fn hunger(&self) -> usize {
        self.hunger as usize
    }

    pub fn status(&self) -> Status {
        self.status
    }
}

/// The board; `Snake::default()` is 10x10.
pub struct Snake {
    w: usize,
    h: usize,
}

impl Snake {
    /// # Panics
    /// If the board cannot fit the initial horizontal length-3 snake
    /// (`w < 3` or `h < 1`) or exceeds the `u8` coordinate range.
    pub fn new(w: usize, h: usize) -> Self {
        assert!(w >= 3 && h >= 1, "board must fit a length-3 snake");
        assert!(w <= 255 && h <= 255, "coordinates are u8");
        Self { w, h }
    }

    pub fn width(&self) -> usize {
        self.w
    }

    pub fn height(&self) -> usize {
        self.h
    }

    pub fn area(&self) -> usize {
        self.w * self.h
    }

    /// Moves without eating before the game ends as [`Status::Starved`].
    pub fn starvation_cap(&self) -> usize {
        self.area()
    }

    /// Length / area — identical to [`Game::returns`], usable mid-game.
    pub fn score(&self, state: &SnakeState) -> f64 {
        state.len() as f64 / self.area() as f64
    }

    fn cell_xy(&self, cell: u16) -> (u8, u8) {
        (
            (cell as usize % self.w) as u8,
            (cell as usize / self.w) as u8,
        )
    }
}

impl Default for Snake {
    fn default() -> Self {
        Self::new(10, 10)
    }
}

fn fnv(h: u64, b: u64) -> u64 {
    (h ^ b).wrapping_mul(0x100_0000_01b3)
}

impl Game for Snake {
    type State = SnakeState;
    type Action = SnakeAction;

    fn num_players(&self) -> usize {
        1
    }

    fn initial_state(&self) -> SnakeState {
        let head_x = (self.w / 2).max(2) as u8;
        let y = (self.h / 2) as u8;
        let body: VecDeque<_> = (0..3).map(|i| (head_x - i, y)).collect();
        let status = if body.len() == self.area() {
            Status::Won
        } else {
            Status::Alive
        };
        SnakeState {
            body,
            heading: Dir::Right,
            food: None,
            hunger: 0,
            status,
        }
    }

    fn turn(&self, state: &SnakeState) -> Turn {
        if state.food.is_none() {
            Turn::Chance
        } else {
            Turn::Player(0)
        }
    }

    fn is_terminal(&self, state: &SnakeState) -> bool {
        state.status != Status::Alive
    }

    fn returns(&self, state: &SnakeState, _player: usize) -> f64 {
        self.score(state)
    }

    fn legal_actions(&self, _state: &SnakeState) -> Vec<SnakeAction> {
        vec![
            SnakeAction::TurnLeft,
            SnakeAction::Straight,
            SnakeAction::TurnRight,
        ]
    }

    fn chance_outcomes(&self, state: &SnakeState) -> Vec<(SnakeAction, f64)> {
        let mut occupied = vec![false; self.area()];
        for &(x, y) in &state.body {
            occupied[y as usize * self.w + x as usize] = true;
        }
        let empties: Vec<u16> = (0..self.area() as u16)
            .filter(|&c| !occupied[c as usize])
            .collect();
        let p = 1.0 / empties.len() as f64;
        empties
            .into_iter()
            .map(|c| (SnakeAction::Food(c), p))
            .collect()
    }

    fn apply(&self, state: &mut SnakeState, action: SnakeAction) {
        debug_assert_eq!(state.status, Status::Alive);
        if let SnakeAction::Food(cell) = action {
            debug_assert!(state.food.is_none());
            let xy = self.cell_xy(cell);
            debug_assert!(!state.body.contains(&xy));
            state.food = Some(xy);
            return;
        }
        state.heading = match action {
            SnakeAction::TurnLeft => state.heading.left(),
            SnakeAction::TurnRight => state.heading.right(),
            _ => state.heading,
        };
        let (dx, dy) = state.heading.delta();
        let (hx, hy) = state.body[0];
        let nx = hx as i32 + dx;
        let ny = hy as i32 + dy;
        if nx < 0 || ny < 0 || nx >= self.w as i32 || ny >= self.h as i32 {
            state.status = Status::Crashed;
            return;
        }
        let new_head = (nx as u8, ny as u8);
        let eats = state.food == Some(new_head);
        let blocking = if eats {
            state.body.len()
        } else {
            state.body.len() - 1
        };
        if state.body.iter().take(blocking).any(|&c| c == new_head) {
            state.status = Status::Crashed;
            return;
        }
        state.body.push_front(new_head);
        if eats {
            state.food = None;
            state.hunger = 0;
            if state.body.len() == self.area() {
                state.status = Status::Won;
            }
        } else {
            state.body.pop_back();
            state.hunger += 1;
            if state.hunger as usize >= self.starvation_cap() {
                state.status = Status::Starved;
            }
        }
    }

    fn infoset_key(&self, state: &SnakeState, _player: usize) -> u64 {
        let pack = |(x, y): (u8, u8)| 1 + y as u64 * 256 + x as u64;
        let mut h = fnv(0xcbf2_9ce4_8422_2325, state.heading as u64);
        h = fnv(h, state.status as u64);
        h = fnv(h, state.hunger as u64);
        h = fnv(h, state.food.map_or(0, pack));
        h = fnv(h, state.body.len() as u64);
        for &c in &state.body {
            h = fnv(h, pack(c));
        }
        h ^ (h >> 31)
    }

    fn state_key(&self, state: &SnakeState) -> Option<u64> {
        Some(self.infoset_key(state, 0))
    }
}

/// Length plus a small food-proximity shaping term, on the [`Game::returns`]
/// scale: `(len + 1/(1 + manhattan(head, food))) / area`. The shaping is
/// always worth less than one food, so it only breaks ties between
/// equal-length states.
pub struct SnakeEval;

impl Eval<Snake> for SnakeEval {
    fn eval(&self, game: &Snake, state: &SnakeState, _player: usize) -> f64 {
        if game.is_terminal(state) {
            return game.score(state);
        }
        let shaped = match state.food {
            Some((fx, fy)) => {
                let (hx, hy) = state.body[0];
                let d = (hx as i32 - fx as i32).abs() + (hy as i32 - fy as i32).abs();
                1.0 / (1.0 + d as f64)
            }
            None => 0.0,
        };
        (state.len() as f64 + shaped) / game.area() as f64
    }
}
