//! Competitive 1v1 Snake as a 2-player [`game_core::Game`] on a shared grid.
//!
//! Both snakes move one cell per tick. The lab's trait is sequential, so one
//! visual tick is modelled as the REDESIGN's option 1 — alternating
//! sub-ticks: seat 0 commits a heading, then seat 1 commits a heading *seeing
//! seat 0's choice*, and seat 1's commit resolves the tick (both heads
//! advance at once). Food placement is a chance node over empty cells at the
//! start and after any meal.
//!
//! Death: a head leaving the board, entering a body cell, a head-to-head
//! collision (the shorter snake dies; both die if equal length), or running
//! its health to zero (Battlesnake-style starvation). Tails vacate on the same
//! tick the heads advance, so a head may chase a non-eating opponent's vacating
//! tail safely. Each tick costs one health; eating refills it to full and grows
//! the snake, so neither snake can idle — both must keep hunting food, which
//! forces engagement over the single contested morsel. The game ends when a
//! snake dies (the survivor wins) or at a step cap (the higher score wins, else
//! a draw). [`Game::returns`] is the win/loss/draw convention `{1, 0, -1}`.

use std::collections::VecDeque;

use game_core::{Game, Turn};

pub const SIDE: usize = 20;

/// Health a snake starts with and is refilled to on eating; each tick without
/// a meal costs one, so a snake that never eats starves after this many ticks.
pub const MAX_HEALTH: u8 = 100;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    Up,
    Right,
    Down,
    Left,
}

impl Dir {
    fn delta(self) -> (i32, i32) {
        match self {
            Dir::Up => (0, -1),
            Dir::Right => (1, 0),
            Dir::Down => (0, 1),
            Dir::Left => (-1, 0),
        }
    }

    fn opposite(self) -> Self {
        match self {
            Dir::Up => Dir::Down,
            Dir::Right => Dir::Left,
            Dir::Down => Dir::Up,
            Dir::Left => Dir::Right,
        }
    }

    /// All four headings, clockwise from Up.
    pub const ALL: [Dir; 4] = [Dir::Up, Dir::Right, Dir::Down, Dir::Left];
}

/// A seat's chosen heading, or the chance outcome that places food on a cell
/// (row-major index `y * SIDE + x`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DuelAction {
    Move(Dir),
    Food(u16),
}

/// One snake on the board: a body whose front is the head, the heading it last
/// moved (turning back onto itself is disallowed), and whether it is alive.
#[derive(Clone, Debug)]
pub struct Worm {
    body: VecDeque<(u8, u8)>,
    heading: Dir,
    alive: bool,
    health: u8,
}

impl Worm {
    pub fn head(&self) -> (usize, usize) {
        let (x, y) = self.body[0];
        (x as usize, y as usize)
    }

    pub fn heading(&self) -> Dir {
        self.heading
    }

    pub fn len(&self) -> usize {
        self.body.len()
    }

    pub fn is_empty(&self) -> bool {
        self.body.is_empty()
    }

    pub fn alive(&self) -> bool {
        self.alive
    }

    /// Remaining health, in `0..=MAX_HEALTH`; zero means the snake starved.
    pub fn health(&self) -> u8 {
        self.health
    }

    /// Cells head first.
    pub fn cells(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.body.iter().map(|&(x, y)| (x as usize, y as usize))
    }

    /// Rebuilds a worm from observed parts (`cells` head-first), for clients
    /// that reconstruct a board from the view rather than replaying moves.
    pub fn from_parts(cells: &[(usize, usize)], heading: Dir, alive: bool, health: u8) -> Worm {
        Worm {
            body: cells.iter().map(|&(x, y)| (x as u8, y as u8)).collect(),
            heading,
            alive,
            health,
        }
    }
}

/// Why the duel ended (or that it is still running).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Ongoing,
    /// Exactly one snake survived (the seat index).
    Win(usize),
    /// Both snakes died on the same tick, or the step cap split scores evenly.
    Draw,
}

#[derive(Clone, Debug)]
pub struct DuelState {
    worms: [Worm; 2],
    food: Option<(u8, u8)>,
    /// Seat 0's heading once committed this tick; consumed when seat 1 commits.
    pending: Option<Dir>,
    steps: u32,
    outcome: Outcome,
}

impl DuelState {
    pub fn worm(&self, seat: usize) -> &Worm {
        &self.worms[seat]
    }

    pub fn food(&self) -> Option<(usize, usize)> {
        self.food.map(|(x, y)| (x as usize, y as usize))
    }

    /// Seat 0's committed heading this tick, once seat 0 has moved.
    pub fn pending(&self) -> Option<Dir> {
        self.pending
    }

    pub fn steps(&self) -> usize {
        self.steps as usize
    }

    pub fn outcome(&self) -> Outcome {
        self.outcome
    }

    /// A snake's score is its length; the survivor at the cap wins on score,
    /// so length is the resource both sides fight over.
    pub fn score(&self, seat: usize) -> usize {
        self.worms[seat].len()
    }

    /// A snake's remaining health in `0..=MAX_HEALTH`.
    pub fn health(&self, seat: usize) -> u8 {
        self.worms[seat].health()
    }
}

/// The 20x20 duel board. The two snakes start length 3 on opposite sides,
/// heading toward the centre (seat 0 facing right, seat 1 facing left), so
/// neither begins with a wall in front of it.
pub struct Duel {
    step_cap: u32,
}

impl Duel {
    pub fn new() -> Self {
        Self {
            step_cap: (SIDE * SIDE) as u32,
        }
    }

    pub fn step_cap(&self) -> usize {
        self.step_cap as usize
    }

    pub fn side(&self) -> usize {
        SIDE
    }

    pub fn area(&self) -> usize {
        SIDE * SIDE
    }

    /// Rebuilds an ongoing `DuelState` from observed parts — for clients (the
    /// browser bot) that reconstruct the board from the view JSON instead of
    /// replaying moves through the chance-driven engine. `pending` is seat 0's
    /// committed heading when seat 1 is to move (`None` otherwise); the outcome
    /// is always [`Outcome::Ongoing`] since the engine only hands a live
    /// position to a driven seat.
    pub fn state_from_parts(
        worms: [Worm; 2],
        food: Option<(usize, usize)>,
        pending: Option<Dir>,
        steps: u32,
    ) -> DuelState {
        DuelState {
            worms,
            food: food.map(|(x, y)| (x as u8, y as u8)),
            pending,
            steps,
            outcome: Outcome::Ongoing,
        }
    }

    fn in_bounds(x: i32, y: i32) -> bool {
        (0..SIDE as i32).contains(&x) && (0..SIDE as i32).contains(&y)
    }

    fn cell_xy(cell: u16) -> (u8, u8) {
        ((cell as usize % SIDE) as u8, (cell as usize / SIDE) as u8)
    }

    fn occupied(state: &DuelState) -> [bool; SIDE * SIDE] {
        let mut occ = [false; SIDE * SIDE];
        for worm in &state.worms {
            for &(x, y) in &worm.body {
                occ[y as usize * SIDE + x as usize] = true;
            }
        }
        occ
    }

    /// The heading a snake adopts for `dir`, refusing a 180° reversal (which
    /// would instantly self-collide) by keeping the current heading.
    fn resolved_heading(worm: &Worm, dir: Dir) -> Dir {
        if dir == worm.heading.opposite() && worm.len() > 1 {
            worm.heading
        } else {
            dir
        }
    }

    /// Whether `(x, y)` is occupied by a blocking segment of `worm`. The tail
    /// vacates this tick (unless the snake eats), so it never blocks; `skip_head`
    /// drops the head cell too, since two heads meeting is resolved by the
    /// head-to-head length rule, not as a body collision.
    fn hits_body(worm: &Worm, x: i32, y: i32, owner_eats: bool, skip_head: bool) -> bool {
        let end = if owner_eats {
            worm.body.len()
        } else {
            worm.body.len().saturating_sub(1)
        };
        let start = usize::from(skip_head);
        worm.body
            .iter()
            .take(end)
            .skip(start)
            .any(|&(bx, by)| bx as i32 == x && by as i32 == y)
    }

    /// Resolve one tick: both snakes step to their committed heading at once,
    /// then deaths (wall, body, head-to-head, starvation) and meals are
    /// applied. Called when seat 1 commits, using seat 0's `pending` heading.
    fn resolve_tick(&self, state: &mut DuelState, dir1: Dir) {
        let dir0 = state.pending.take().expect("seat 0 commits before seat 1");
        let h0 = Self::resolved_heading(&state.worms[0], dir0);
        let h1 = Self::resolved_heading(&state.worms[1], dir1);

        let next = |worm: &Worm, h: Dir| {
            let (dx, dy) = h.delta();
            let (hx, hy) = worm.body[0];
            (hx as i32 + dx, hy as i32 + dy)
        };
        let old0 = state.worms[0].body[0];
        let old1 = state.worms[1].body[0];
        let (nx0, ny0) = next(&state.worms[0], h0);
        let (nx1, ny1) = next(&state.worms[1], h1);

        let food = state.food;
        let eats0 = food == Some((nx0 as u8, ny0 as u8));
        let eats1 = food == Some((nx1 as u8, ny1 as u8));

        let wall0 = !Self::in_bounds(nx0, ny0);
        let wall1 = !Self::in_bounds(nx1, ny1);

        // A snake that does not eat this tick spends one health; reaching zero
        // starves it, which kills it exactly like a crash.
        let starve0 = !eats0 && state.worms[0].health <= 1;
        let starve1 = !eats1 && state.worms[1].health <= 1;

        let mut dead0 = wall0
            || starve0
            || Self::hits_body(&state.worms[0], nx0, ny0, eats0, false)
            || Self::hits_body(&state.worms[1], nx0, ny0, eats1, true);
        let mut dead1 = wall1
            || starve1
            || Self::hits_body(&state.worms[1], nx1, ny1, eats1, false)
            || Self::hits_body(&state.worms[0], nx1, ny1, eats0, true);

        // Head-to-head: the two heads land on the same cell, or swap cells
        // (passing through each other). Both resolve on length — the shorter
        // snake dies, both if equal.
        let same_cell = !wall0 && !wall1 && nx0 == nx1 && ny0 == ny1;
        let swap =
            !wall0 && !wall1 && (nx0 as u8, ny0 as u8) == old1 && (nx1 as u8, ny1 as u8) == old0;

        // A head landing on the opponent's OLD-head cell — when that is NOT the
        // mutual swap above — is a body collision, not a head-to-head meet: the
        // opponent's head moved off that cell, but the segment behind it shifts
        // in, so the cell stays occupied and is fatal. `skip_head=true` above
        // exempted it (so two NEW heads meeting can be resolved by the length
        // rule); restore the collision for this distinct case. A snake only
        // vacates its old-head cell if its whole body is length 1, which never
        // happens — snakes start length 3 and only grow.
        let vacated0 = state.worms[0].len() <= 1;
        let vacated1 = state.worms[1].len() <= 1;
        dead1 |= !swap && !wall1 && (nx1 as u8, ny1 as u8) == old0 && !vacated0;
        dead0 |= !swap && !wall0 && (nx0 as u8, ny0 as u8) == old1 && !vacated1;

        if same_cell || swap {
            match state.worms[0].len().cmp(&state.worms[1].len()) {
                std::cmp::Ordering::Greater => dead1 = true,
                std::cmp::Ordering::Less => dead0 = true,
                std::cmp::Ordering::Equal => {
                    dead0 = true;
                    dead1 = true;
                }
            }
        }

        if !dead0 {
            Self::advance(&mut state.worms[0], h0, (nx0, ny0), eats0);
        }
        if !dead1 {
            Self::advance(&mut state.worms[1], h1, (nx1, ny1), eats1);
        }
        state.worms[0].alive = !dead0;
        state.worms[1].alive = !dead1;

        if eats0 || eats1 {
            state.food = None;
        }

        state.steps += 1;
        self.settle(state);
    }

    fn advance(worm: &mut Worm, heading: Dir, head: (i32, i32), eats: bool) {
        worm.heading = heading;
        worm.body.push_front((head.0 as u8, head.1 as u8));
        if eats {
            worm.health = MAX_HEALTH;
        } else {
            worm.body.pop_back();
            worm.health = worm.health.saturating_sub(1);
        }
    }

    /// Decide the post-tick outcome: both dead is a draw, one dead is a win
    /// for the survivor, and reaching the step cap scores on length.
    fn settle(&self, state: &mut DuelState) {
        let (a0, a1) = (state.worms[0].alive, state.worms[1].alive);
        state.outcome = match (a0, a1) {
            (false, false) => Outcome::Draw,
            (true, false) => Outcome::Win(0),
            (false, true) => Outcome::Win(1),
            (true, true) if state.steps >= self.step_cap => {
                match state.worms[0].len().cmp(&state.worms[1].len()) {
                    std::cmp::Ordering::Greater => Outcome::Win(0),
                    std::cmp::Ordering::Less => Outcome::Win(1),
                    std::cmp::Ordering::Equal => Outcome::Draw,
                }
            }
            (true, true) => Outcome::Ongoing,
        };
    }

    fn key(&self, state: &DuelState) -> u64 {
        use game_core::hash::combine;
        let pack = |(x, y): (u8, u8)| 1 + y as u64 * 32 + x as u64;
        let mut h = combine(0, state.steps as u64);
        h = combine(h, state.pending.map_or(0, |d| 1 + d as u64));
        h = combine(h, state.food.map_or(0, pack));
        for worm in &state.worms {
            h = combine(h, worm.heading as u64);
            h = combine(h, worm.alive as u64);
            h = combine(h, worm.health as u64);
            h = combine(h, worm.body.len() as u64);
            for &c in &worm.body {
                h = combine(h, pack(c));
            }
        }
        h
    }
}

impl Default for Duel {
    fn default() -> Self {
        Self::new()
    }
}

impl Game for Duel {
    type State = DuelState;
    type Action = DuelAction;

    fn num_players(&self) -> usize {
        2
    }

    fn initial_state(&self) -> DuelState {
        let mid = (SIDE / 2) as u8;
        let left = 4u8;
        let right = (SIDE - 5) as u8;
        let worm0 = Worm {
            body: (0..3).map(|i| (left - i, mid)).collect(),
            heading: Dir::Right,
            alive: true,
            health: MAX_HEALTH,
        };
        let worm1 = Worm {
            body: (0..3).map(|i| (right + i, mid)).collect(),
            heading: Dir::Left,
            alive: true,
            health: MAX_HEALTH,
        };
        DuelState {
            worms: [worm0, worm1],
            food: None,
            pending: None,
            steps: 0,
            outcome: Outcome::Ongoing,
        }
    }

    fn turn(&self, state: &DuelState) -> Turn {
        if state.food.is_none() {
            Turn::Chance
        } else if state.pending.is_none() {
            Turn::Player(0)
        } else {
            Turn::Player(1)
        }
    }

    fn is_terminal(&self, state: &DuelState) -> bool {
        state.outcome != Outcome::Ongoing
    }

    fn returns(&self, state: &DuelState, player: usize) -> f64 {
        match state.outcome {
            Outcome::Win(w) if w == player => 1.0,
            Outcome::Win(_) => -1.0,
            _ => 0.0,
        }
    }

    fn legal_actions(&self, _state: &DuelState) -> Vec<DuelAction> {
        Dir::ALL.map(DuelAction::Move).to_vec()
    }

    fn chance_outcomes(&self, state: &DuelState) -> Vec<(DuelAction, f64)> {
        debug_assert!(state.food.is_none());
        let occ = Self::occupied(state);
        let empties: Vec<u16> = (0..self.area() as u16)
            .filter(|&c| !occ[c as usize])
            .collect();
        let p = 1.0 / empties.len() as f64;
        empties
            .into_iter()
            .map(|c| (DuelAction::Food(c), p))
            .collect()
    }

    fn apply(&self, state: &mut DuelState, action: DuelAction) {
        debug_assert_eq!(state.outcome, Outcome::Ongoing);
        match action {
            DuelAction::Food(cell) => {
                debug_assert!(state.food.is_none());
                state.food = Some(Self::cell_xy(cell));
            }
            DuelAction::Move(dir) if state.pending.is_none() => state.pending = Some(dir),
            DuelAction::Move(dir) => self.resolve_tick(state, dir),
        }
    }

    fn infoset_key(&self, state: &DuelState, _player: usize) -> u64 {
        self.key(state)
    }

    fn state_key(&self, state: &DuelState) -> Option<u64> {
        Some(self.key(state))
    }

    fn action_id(&self, action: &DuelAction) -> u64 {
        match action {
            DuelAction::Move(d) => *d as u64,
            DuelAction::Food(c) => 4 + u64::from(*c),
        }
    }
}

#[cfg(test)]
mod parts_tests {
    use super::*;
    use crate::SnakeEncoder;
    use game_core::PolicyValueEncoder;

    /// Rebuilding a live position from its observed parts (what the browser bot
    /// does from the view JSON) yields an encoder-identical state — the
    /// reconstruction the AlphaZero bot relies on is exact. Covers a seat-1 turn
    /// (food placed, seat 0's heading pending), the lossiest case.
    #[test]
    fn state_from_parts_round_trips_through_the_encoder() {
        let game = Duel::new();
        let mut state = game.initial_state();
        let outs = game.chance_outcomes(&state);
        game.apply(&mut state, outs[3].0); // place food, seat 0 to move
        game.apply(&mut state, DuelAction::Move(Dir::Up)); // seat 0 commits → seat 1 to move
        assert!(
            matches!(game.turn(&state), Turn::Player(1)),
            "seat 1 on the clock"
        );
        assert!(state.pending().is_some(), "seat 0's heading is pending");

        let worm = |seat: usize| {
            let w = state.worm(seat);
            Worm::from_parts(
                &w.cells().collect::<Vec<_>>(),
                w.heading(),
                w.alive(),
                w.health(),
            )
        };
        let rebuilt = Duel::state_from_parts(
            [worm(0), worm(1)],
            state.food(),
            state.pending(),
            state.steps() as u32,
        );

        assert_eq!(game.turn(&rebuilt), game.turn(&state), "same side to move");
        let enc = SnakeEncoder::new();
        assert_eq!(
            enc.encode_state(&game, &rebuilt),
            enc.encode_state(&game, &state),
            "reconstructed planes match the original"
        );
    }
}
