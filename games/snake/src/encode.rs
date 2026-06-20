//! AlphaZero-style policy/value encoding for the 1v1 [`Duel`].
//!
//! Planes are all from the mover's perspective ("me" is the seat to move):
//! the two heads and bodies, the food, length scalars for both snakes, the
//! committed-but-unresolved heading of seat 0 when seat 1 is on the clock
//! (the one piece of state the board cannot show — seat 1 chooses seeing it),
//! and a constant ones plane so zero conv padding stays distinguishable from
//! an empty board cell. Policy index is the absolute [`Dir`] (0..4).

use game_core::{Game, PolicyValueEncoder, Turn};

use crate::duel::{Dir, Duel, DuelAction, DuelState, SIDE};

const MY_HEAD: usize = 0;
const MY_BODY: usize = 1;
const OPP_HEAD: usize = 2;
const OPP_BODY: usize = 3;
const FOOD: usize = 4;
const MY_LEN: usize = 5;
const OPP_LEN: usize = 6;
const PENDING: usize = 7;
const ONES: usize = 8;
pub const PLANES: usize = ONES + 1;

pub struct SnakeEncoder;

impl SnakeEncoder {
    pub fn new() -> SnakeEncoder {
        SnakeEncoder
    }
}

impl Default for SnakeEncoder {
    fn default() -> Self {
        SnakeEncoder::new()
    }
}

fn cell(x: usize, y: usize) -> usize {
    y * SIDE + x
}

fn step(x: usize, y: usize, dir: Dir) -> Option<(usize, usize)> {
    let (dx, dy) = match dir {
        Dir::Up => (0, -1),
        Dir::Right => (1, 0),
        Dir::Down => (0, 1),
        Dir::Left => (-1, 0),
    };
    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
    ((0..SIDE as i32).contains(&nx) && (0..SIDE as i32).contains(&ny))
        .then_some((nx as usize, ny as usize))
}

impl PolicyValueEncoder<Duel> for SnakeEncoder {
    fn input_len(&self) -> usize {
        PLANES * SIDE * SIDE
    }

    fn policy_len(&self) -> usize {
        Dir::ALL.len()
    }

    fn encode_state(&self, game: &Duel, state: &DuelState) -> Vec<f32> {
        let n = SIDE * SIDE;
        let mut out = vec![0.0f32; PLANES * n];
        let me = match game.turn(state) {
            Turn::Player(p) => p,
            _ => 0,
        };
        let opp = 1 - me;

        for (seat, head_plane, body_plane) in [(me, MY_HEAD, MY_BODY), (opp, OPP_HEAD, OPP_BODY)] {
            for (i, (x, y)) in state.worm(seat).cells().enumerate() {
                let plane = if i == 0 { head_plane } else { body_plane };
                out[plane * n + cell(x, y)] = 1.0;
            }
        }

        if let Some((fx, fy)) = state.food() {
            out[FOOD * n + cell(fx, fy)] = 1.0;
        }

        let area = n as f32;
        out[MY_LEN * n..(MY_LEN + 1) * n].fill(state.worm(me).len() as f32 / area);
        out[OPP_LEN * n..(OPP_LEN + 1) * n].fill(state.worm(opp).len() as f32 / area);

        if me == 1
            && let Some(dir) = state.pending()
        {
            let (hx, hy) = state.worm(0).head();
            if let Some((nx, ny)) = step(hx, hy, dir) {
                out[PENDING * n + cell(nx, ny)] = 1.0;
            }
        }

        out[ONES * n..(ONES + 1) * n].fill(1.0);
        out
    }

    fn action_index(&self, _game: &Duel, _state: &DuelState, action: DuelAction) -> usize {
        match action {
            DuelAction::Move(dir) => dir as usize,
            DuelAction::Food(_) => unreachable!("food is a chance outcome, never a policy action"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_core::Game;

    #[test]
    fn heads_bodies_food_and_ones_planes() {
        let g = Duel::new();
        let mut s = g.initial_state();
        let outs = g.chance_outcomes(&s);
        g.apply(&mut s, outs[0].0);
        let n = SIDE * SIDE;
        let enc = SnakeEncoder::new();
        let x = enc.encode_state(&g, &s);

        let me = 0;
        let (hx, hy) = s.worm(me).head();
        assert_eq!(x[MY_HEAD * n + cell(hx, hy)], 1.0, "my head");
        let (ox, oy) = s.worm(1).head();
        assert_eq!(x[OPP_HEAD * n + cell(ox, oy)], 1.0, "opp head");
        assert_eq!(
            x[MY_HEAD * n + cell(ox, oy)],
            0.0,
            "opp head not on my plane"
        );
        assert_eq!(x[ONES * n], 1.0, "ones plane filled");
        assert_eq!(x[MY_LEN * n], 3.0 / n as f32, "my length scalar");

        let food_cells: f32 = x[FOOD * n..(FOOD + 1) * n].iter().sum();
        assert_eq!(food_cells, 1.0, "exactly one food cell");
    }

    #[test]
    fn pending_plane_marks_seat0_projected_head() {
        let g = Duel::new();
        let mut s = g.initial_state();
        let outs = g.chance_outcomes(&s);
        g.apply(&mut s, outs[0].0);
        g.apply(&mut s, DuelAction::Move(Dir::Right));
        let n = SIDE * SIDE;
        let enc = SnakeEncoder::new();
        let x = enc.encode_state(&g, &s);
        let (hx, hy) = s.worm(0).head();
        let (nx, ny) = step(hx, hy, Dir::Right).unwrap();
        assert_eq!(x[PENDING * n + cell(nx, ny)], 1.0, "seat 0 projected head");
        let pending_cells: f32 = x[PENDING * n..(PENDING + 1) * n].iter().sum();
        assert_eq!(pending_cells, 1.0, "exactly one pending cell");
    }

    #[test]
    fn perspective_flips_with_mover() {
        let g = Duel::new();
        let mut s = g.initial_state();
        let outs = g.chance_outcomes(&s);
        g.apply(&mut s, outs[0].0);
        g.apply(&mut s, DuelAction::Move(Dir::Right));
        let n = SIDE * SIDE;
        let enc = SnakeEncoder::new();
        let x = enc.encode_state(&g, &s);
        let (h1x, h1y) = s.worm(1).head();
        assert_eq!(x[MY_HEAD * n + cell(h1x, h1y)], 1.0, "seat 1 is now me");
    }

    #[test]
    fn action_index_is_absolute_dir() {
        let g = Duel::new();
        let s = g.initial_state();
        let enc = SnakeEncoder::new();
        for d in Dir::ALL {
            assert_eq!(enc.action_index(&g, &s, DuelAction::Move(d)), d as usize,);
        }
    }
}
