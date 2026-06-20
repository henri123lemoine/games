//! Static evaluation for [`Duel`] search: the standard competitive-snake
//! heuristic — Voronoi territory between the two heads, plus length, plus a
//! food-pull term and a survival bias.

use std::collections::VecDeque;

use game_core::{Eval, Game};

use crate::duel::{Duel, DuelState, Outcome, SIDE};

/// Reachable-territory (Voronoi) advantage from `player`'s perspective.
///
/// A simultaneous BFS from both heads over the free cells labels each cell by
/// whoever reaches it first (ties go to neither); the signed territory
/// difference is the dominant term, since controlling more of the board is
/// what wins competitive snake. Length and food proximity break ties between
/// equal-territory positions, and a small survival bias rewards simply having
/// more room than the opponent.
pub struct DuelEval;

const TERRITORY_W: f64 = 1.0;
const LENGTH_W: f64 = 4.0;
const FOOD_W: f64 = 2.0;
const SQUASH: f64 = 120.0;

impl Eval<Duel> for DuelEval {
    fn eval(&self, game: &Duel, state: &DuelState, player: usize) -> f64 {
        if game.is_terminal(state) {
            return match state.outcome() {
                Outcome::Win(w) if w == player => 1.0,
                Outcome::Win(_) => -1.0,
                Outcome::Draw => 0.0,
                Outcome::Ongoing => unreachable!("terminal state is not Ongoing"),
            };
        }
        let me = player;
        let foe = 1 - player;
        let (mine, theirs) = voronoi(state);
        let territory = mine as f64 - theirs as f64;
        let length = state.score(me) as f64 - state.score(foe) as f64;
        let food = food_pull(state, me) - food_pull(state, foe);
        let score = TERRITORY_W * territory + LENGTH_W * length + FOOD_W * food;
        game_core::eval_squash(score, SQUASH)
    }
}

/// Closeness to the food, in `[0, 1)`, by shortest grid distance from a
/// snake's head — `0` when there is no food or the snake is dead.
fn food_pull(state: &DuelState, seat: usize) -> f64 {
    let Some((fx, fy)) = state.food() else {
        return 0.0;
    };
    if !state.worm(seat).alive() {
        return 0.0;
    }
    let (hx, hy) = state.worm(seat).head();
    let d = hx.abs_diff(fx) + hy.abs_diff(fy);
    1.0 / (1.0 + d as f64)
}

/// Cells each snake reaches strictly first in a flood fill from both heads
/// over the unoccupied board (contested cells count for neither).
fn voronoi(state: &DuelState) -> (u32, u32) {
    let mut owner = [0u8; SIDE * SIDE];
    let mut occupied = [false; SIDE * SIDE];
    for seat in 0..2 {
        for (x, y) in state.worm(seat).cells() {
            occupied[y * SIDE + x] = true;
        }
    }

    const FREE: u8 = 0;
    const A: u8 = 1;
    const B: u8 = 2;
    const TIE: u8 = 3;

    let mut frontier: VecDeque<(usize, usize, u8)> = VecDeque::new();
    for (seat, mark) in [(0usize, A), (1usize, B)] {
        if !state.worm(seat).alive() {
            continue;
        }
        let (hx, hy) = state.worm(seat).head();
        frontier.push_back((hx, hy, mark));
    }

    let mut dist = [u16::MAX; SIDE * SIDE];
    for &(x, y, _) in &frontier {
        dist[y * SIDE + x] = 0;
    }

    while let Some((x, y, mark)) = frontier.pop_front() {
        let d = dist[y * SIDE + x];
        for (nx, ny) in neighbours(x, y) {
            let i = ny * SIDE + nx;
            if occupied[i] {
                continue;
            }
            let nd = d + 1;
            if nd < dist[i] {
                dist[i] = nd;
                owner[i] = mark;
                frontier.push_back((nx, ny, mark));
            } else if nd == dist[i] && owner[i] != mark && owner[i] != FREE {
                owner[i] = TIE;
            }
        }
    }

    let mut a = 0;
    let mut b = 0;
    for &o in &owner {
        match o {
            A => a += 1,
            B => b += 1,
            _ => {}
        }
    }
    (a, b)
}

fn neighbours(x: usize, y: usize) -> impl Iterator<Item = (usize, usize)> {
    const DELTAS: [(i32, i32); 4] = [(0, -1), (1, 0), (0, 1), (-1, 0)];
    DELTAS.into_iter().filter_map(move |(dx, dy)| {
        let (nx, ny) = (x as i32 + dx, y as i32 + dy);
        ((0..SIDE as i32).contains(&nx) && (0..SIDE as i32).contains(&ny))
            .then_some((nx as usize, ny as usize))
    })
}
