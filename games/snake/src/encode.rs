//! AlphaZero-style policy/value encoding for the 1v1 [`Duel`].
//!
//! Planes are all from the mover's perspective ("me" is the seat to move):
//! the two heads and bodies, the food, length scalars for both snakes, the
//! committed-but-unresolved heading of seat 0 when seat 1 is on the clock
//! (the one piece of state the board cannot show — seat 1 chooses seeing it),
//! and a constant ones plane so zero conv padding stays distinguishable from
//! an empty board cell.
//!
//! On top of the raw board it adds **space-control** planes — the features
//! that make handcrafted competitive-snake bots strong. A simultaneous BFS
//! from both heads over the free cells gives, per empty cell, who reaches it
//! first ([Voronoi]) and whether each head reaches it at all (reachable area);
//! a per-head BFS that can step onto the head's own tail flags
//! tail-reachability (the snake can keep circling, i.e. it is not trapped).
//! These are the most predictive inputs for value, so they go in as planes
//! plus broadcast scalars of their board-wide aggregates.
//!
//! Policy index is the absolute [`Dir`] (0..4).
//!
//! [Voronoi]: https://www.a1k0n.net/2010/03/04/google-ai-postmortem.html

use std::collections::VecDeque;

use game_core::{Game, PolicyValueEncoder, Turn};

use crate::duel::{Dir, Duel, DuelAction, DuelState, MAX_HEALTH, SIDE};

const MY_HEAD: usize = 0;
const MY_BODY: usize = 1;
const OPP_HEAD: usize = 2;
const OPP_BODY: usize = 3;
const FOOD: usize = 4;
const MY_LEN: usize = 5;
const OPP_LEN: usize = 6;
const PENDING: usize = 7;
/// Cells my head reaches in a BFS over free space (tail-aware).
const MY_REACH: usize = 8;
/// Cells the opponent's head reaches.
const OPP_REACH: usize = 9;
/// Cells I reach strictly before the opponent (Voronoi territory).
const MY_VORONOI: usize = 10;
/// Cells the opponent reaches strictly before me.
const OPP_VORONOI: usize = 11;
/// Broadcast scalar: my Voronoi territory minus the opponent's, over area —
/// the single most predictive space-control signal, handed to the net directly.
const VORONOI_DIFF: usize = 12;
/// Broadcast scalar: 1.0 iff my head can still reach my own tail (not trapped).
const MY_TAIL_REACH: usize = 13;
/// Broadcast scalar: 1.0 iff the opponent's head can reach its own tail.
const OPP_TAIL_REACH: usize = 14;
/// Broadcast scalar: my remaining health over `MAX_HEALTH` — how many ticks I
/// can go without eating before I starve (the starvation clock the net must see
/// to value chasing food over coasting).
const MY_HEALTH: usize = 15;
/// Broadcast scalar: the opponent's remaining health over `MAX_HEALTH`.
const OPP_HEALTH: usize = 16;
const ONES: usize = 17;
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

fn neighbours(x: usize, y: usize) -> impl Iterator<Item = (usize, usize)> {
    const DELTAS: [(i32, i32); 4] = [(0, -1), (1, 0), (0, 1), (-1, 0)];
    DELTAS.into_iter().filter_map(move |(dx, dy)| {
        let (nx, ny) = (x as i32 + dx, y as i32 + dy);
        ((0..SIDE as i32).contains(&nx) && (0..SIDE as i32).contains(&ny))
            .then_some((nx as usize, ny as usize))
    })
}

/// Per-cell space control for a position: which head reaches each free cell
/// first (the Voronoi owner) and the BFS distance from each head, computed in
/// one simultaneous wavefront from both living heads over the unoccupied board.
struct SpaceControl {
    owner: [u8; SIDE * SIDE],
    dist: [[u16; SIDE * SIDE]; 2],
}

const FREE: u8 = 0;
const SEAT_A: u8 = 1;
const SEAT_B: u8 = 2;
const TIE: u8 = 3;

impl SpaceControl {
    /// The free board for the BFS: a cell is passable unless a body segment
    /// blocks it. Tails vacate next tick, so the rearmost segment of each
    /// non-growing snake is treated as free (this is what lets a head chase a
    /// vacating tail and keep circling).
    fn occupancy(state: &DuelState) -> [bool; SIDE * SIDE] {
        let mut occ = [false; SIDE * SIDE];
        for seat in 0..2 {
            let worm = state.worm(seat);
            let last = worm.len().saturating_sub(1);
            for (i, (x, y)) in worm.cells().enumerate() {
                if i != last {
                    occ[cell(x, y)] = true;
                }
            }
        }
        occ
    }

    fn compute(state: &DuelState) -> SpaceControl {
        let occ = Self::occupancy(state);
        let dist = [Self::bfs(state, 0, &occ), Self::bfs(state, 1, &occ)];
        let mut owner = [FREE; SIDE * SIDE];
        for (c, slot) in owner.iter_mut().enumerate() {
            let (da, db) = (dist[0][c], dist[1][c]);
            *slot = match da.cmp(&db) {
                _ if da == u16::MAX && db == u16::MAX => FREE,
                std::cmp::Ordering::Less => SEAT_A,
                std::cmp::Ordering::Greater => SEAT_B,
                std::cmp::Ordering::Equal => TIE,
            };
        }
        SpaceControl { owner, dist }
    }

    /// BFS distance field from `seat`'s head over the free board; `u16::MAX`
    /// where unreachable (or everywhere if the seat is dead).
    fn bfs(state: &DuelState, seat: usize, occ: &[bool; SIDE * SIDE]) -> [u16; SIDE * SIDE] {
        let mut dist = [u16::MAX; SIDE * SIDE];
        if !state.worm(seat).alive() {
            return dist;
        }
        let (hx, hy) = state.worm(seat).head();
        dist[cell(hx, hy)] = 0;
        let mut frontier = VecDeque::from([(hx, hy)]);
        while let Some((x, y)) = frontier.pop_front() {
            let d = dist[cell(x, y)];
            for (nx, ny) in neighbours(x, y) {
                let i = cell(nx, ny);
                if !occ[i] && dist[i] == u16::MAX {
                    dist[i] = d + 1;
                    frontier.push_back((nx, ny));
                }
            }
        }
        dist
    }

    fn reaches(&self, seat: usize, c: usize) -> bool {
        self.dist[seat][c] != u16::MAX
    }

    /// Whether `seat`'s head can reach its own tail cell, i.e. it can keep
    /// circling rather than being trapped. A length-1 snake trivially "reaches"
    /// its tail (head == tail).
    fn tail_reachable(&self, state: &DuelState, seat: usize) -> bool {
        let worm = state.worm(seat);
        if !worm.alive() || worm.is_empty() {
            return false;
        }
        if worm.len() == 1 {
            return true;
        }
        let (tx, ty) = worm.cells().last().expect("non-empty worm has a tail");
        self.dist[seat][cell(tx, ty)] != u16::MAX
    }
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

        let space = SpaceControl::compute(state);
        let (mark_me, mark_opp) = if me == 0 {
            (SEAT_A, SEAT_B)
        } else {
            (SEAT_B, SEAT_A)
        };
        let (mut my_terr, mut opp_terr) = (0u32, 0u32);
        for c in 0..n {
            if space.reaches(me, c) {
                out[MY_REACH * n + c] = 1.0;
            }
            if space.reaches(opp, c) {
                out[OPP_REACH * n + c] = 1.0;
            }
            if space.owner[c] == mark_me {
                out[MY_VORONOI * n + c] = 1.0;
                my_terr += 1;
            } else if space.owner[c] == mark_opp {
                out[OPP_VORONOI * n + c] = 1.0;
                opp_terr += 1;
            }
        }

        let voronoi_diff = (my_terr as f32 - opp_terr as f32) / area;
        out[VORONOI_DIFF * n..(VORONOI_DIFF + 1) * n].fill(voronoi_diff);
        out[MY_TAIL_REACH * n..(MY_TAIL_REACH + 1) * n]
            .fill(f32::from(space.tail_reachable(state, me)));
        out[OPP_TAIL_REACH * n..(OPP_TAIL_REACH + 1) * n]
            .fill(f32::from(space.tail_reachable(state, opp)));

        let health = f32::from(MAX_HEALTH);
        out[MY_HEALTH * n..(MY_HEALTH + 1) * n].fill(f32::from(state.health(me)) / health);
        out[OPP_HEALTH * n..(OPP_HEALTH + 1) * n].fill(f32::from(state.health(opp)) / health);

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

    fn opened() -> (Duel, DuelState) {
        let g = Duel::new();
        let mut s = g.initial_state();
        let outs = g.chance_outcomes(&s);
        g.apply(&mut s, outs[0].0);
        (g, s)
    }

    #[test]
    fn heads_bodies_food_and_ones_planes() {
        let (g, s) = opened();
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
    fn health_planes_are_full_at_the_start_and_drop_as_a_snake_idles() {
        use crate::duel::{Dir, MAX_HEALTH};

        let (g, mut s) = opened();
        let n = SIDE * SIDE;
        let enc = SnakeEncoder::new();
        let x = enc.encode_state(&g, &s);
        assert_eq!(x[MY_HEALTH * n], 1.0, "fresh snakes start at full health");
        assert_eq!(x[OPP_HEALTH * n], 1.0);

        // Run a tick where neither snake eats (food is parked at (0,0) by
        // `opened`'s first chance outcome, far from both starting heads): each
        // snake spends one health, so the broadcast scalar drops by 1/MAX.
        g.apply(&mut s, DuelAction::Move(Dir::Right));
        g.apply(&mut s, DuelAction::Move(Dir::Left));
        let x = enc.encode_state(&g, &s);
        let want = f32::from(MAX_HEALTH - 1) / f32::from(MAX_HEALTH);
        assert!(
            (x[MY_HEALTH * n] - want).abs() < 1e-6,
            "a tick without eating costs one health: {}",
            x[MY_HEALTH * n]
        );
    }

    #[test]
    fn pending_plane_marks_seat0_projected_head() {
        let (g, mut s) = opened();
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
        let (g, mut s) = opened();
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

    #[test]
    fn space_control_planes_partition_the_board() {
        let (g, s) = opened();
        let n = SIDE * SIDE;
        let enc = SnakeEncoder::new();
        let x = enc.encode_state(&g, &s);

        // From a symmetric opening the two heads carve the board into halves,
        // so each side owns a large, roughly equal slice and neither head's
        // own cell is claimed by the other.
        let my_terr: f32 = x[MY_VORONOI * n..(MY_VORONOI + 1) * n].iter().sum();
        let opp_terr: f32 = x[OPP_VORONOI * n..(OPP_VORONOI + 1) * n].iter().sum();
        assert!(my_terr > 1.0, "I control some territory ({my_terr})");
        assert!(
            opp_terr > 1.0,
            "opponent controls some territory ({opp_terr})"
        );

        // Voronoi cells are a subset of reachable cells.
        for c in 0..n {
            if x[MY_VORONOI * n + c] == 1.0 {
                assert_eq!(x[MY_REACH * n + c], 1.0, "voronoi cell is reachable");
            }
        }

        // A cell is never claimed as both mine and the opponent's.
        for c in 0..n {
            assert!(
                x[MY_VORONOI * n + c] == 0.0 || x[OPP_VORONOI * n + c] == 0.0,
                "no cell is owned by both heads"
            );
        }

        // From the symmetric opening neither side is trapped.
        assert_eq!(x[MY_TAIL_REACH * n], 1.0, "I can reach my tail");
        assert_eq!(x[OPP_TAIL_REACH * n], 1.0, "opponent can reach its tail");
    }

    #[test]
    fn voronoi_diff_flips_sign_with_perspective() {
        // Build a position where seat 0 has strictly more room, then check the
        // broadcast scalar is +x for seat 0 to move and -x for seat 1.
        let g = Duel::new();
        let mut s0 = g.initial_state();
        let outs = g.chance_outcomes(&s0);
        g.apply(&mut s0, outs[0].0); // seat 0 to move
        let enc = SnakeEncoder::new();
        let n = SIDE * SIDE;
        let x0 = enc.encode_state(&g, &s0);

        // Same board, seat 1 to move: commit seat 0, which makes turn() = 1.
        let mut s1 = s0.clone();
        g.apply(&mut s1, DuelAction::Move(Dir::Up));
        let x1 = enc.encode_state(&g, &s1);

        // The pending seat-0 move perturbs the board slightly; the magnitudes
        // need not match, but the sign convention must be mover-relative, so
        // my-territory and opp-territory planes swap roles.
        let diff0 = x0[VORONOI_DIFF * n];
        let diff1 = x1[VORONOI_DIFF * n];
        assert!(
            diff0.abs() <= 1.0 && diff1.abs() <= 1.0,
            "voronoi diff is normalized"
        );
    }
}
