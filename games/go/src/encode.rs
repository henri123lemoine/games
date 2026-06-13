//! AlphaZero-style policy-value encoding for Go, plus the dihedral board
//! symmetries trainers use to augment samples.
//!
//! Features are 9 planes of `size²`, all from the side to move's
//! perspective: own/opponent stones, each side's groups at one and at two
//! liberties, the empty points illegal for the mover (ko and suicide — the
//! one piece of state the stone planes cannot show), a constant "the mover
//! is White" plane (komi breaks color symmetry), and a constant ones plane
//! (zero conv padding otherwise makes off-board indistinguishable from
//! empty). Policy index `p` is the board index of a placement; `size²` is
//! the pass.

use game_core::{Game, PolicyValueEncoder};

use crate::{EMPTY, Go, GoAction, GoState, group, neighbors};

pub const PLANES: usize = 9;

pub struct GoEncoder;

impl PolicyValueEncoder<Go> for GoEncoder {
    fn input_len(&self) -> usize {
        PLANES * 81
    }

    fn policy_len(&self) -> usize {
        82
    }

    fn encode_state(&self, game: &Go, state: &GoState) -> Vec<f32> {
        let n = game.size() * game.size();
        let mut out = vec![0.0f32; PLANES * n];
        let cells = &state.cells;
        let own = state.to_move as u8;

        let mut seen = vec![false; n];
        for p in 0..n {
            let c = cells[p];
            if c == EMPTY || seen[p] {
                continue;
            }
            let (stones, _) = group(cells, game.size(), p);
            let libs = group_liberty_count(cells, game.size(), &stones);
            let side = usize::from(c != own);
            let lib_plane = match libs {
                1 => Some(2 + side),
                2 => Some(4 + side),
                _ => None,
            };
            for &s in &stones {
                seen[s] = true;
                out[side * n + s] = 1.0;
                if let Some(pl) = lib_plane {
                    out[pl * n + s] = 1.0;
                }
            }
        }

        let mut legal = vec![false; n];
        for a in game.legal_actions(state) {
            if let GoAction::Place(p) = a {
                legal[p as usize] = true;
            }
        }
        for p in 0..n {
            if cells[p] == EMPTY && !legal[p] {
                out[6 * n + p] = 1.0;
            }
        }

        if state.to_move == 1 {
            out[7 * n..8 * n].fill(1.0);
        }
        out[8 * n..9 * n].fill(1.0);
        out
    }

    fn action_index(&self, game: &Go, _state: &GoState, action: GoAction) -> usize {
        match action {
            GoAction::Place(p) => p as usize,
            GoAction::Pass => game.size() * game.size(),
        }
    }
}

fn group_liberty_count(cells: &[u8], size: usize, stones: &[usize]) -> usize {
    let mut seen = vec![false; cells.len()];
    let mut libs = 0;
    for &s in stones {
        for nb in neighbors(size, s) {
            if cells[nb] == EMPTY && !seen[nb] {
                seen[nb] = true;
                libs += 1;
            }
        }
    }
    libs
}

/// Board index `p` under symmetry `t` (0..8): rotation by `t % 4` quarter
/// turns, then a horizontal mirror if `t >= 4`. `d8(p, t)` followed by
/// `d8(p, inverse_d8(t))` is the identity.
pub fn d8(p: usize, t: u8, size: usize) -> usize {
    let (mut r, mut c) = (p / size, p % size);
    for _ in 0..(t % 4) {
        (r, c) = (c, size - 1 - r);
    }
    if t >= 4 {
        c = size - 1 - c;
    }
    r * size + c
}

pub fn inverse_d8(t: u8) -> u8 {
    match t {
        1 => 3,
        3 => 1,
        _ => t,
    }
}

/// Policy index under symmetry `t`: placements move with the board, pass
/// (`size²`) is fixed.
pub fn d8_policy(idx: u16, t: u8, size: usize) -> u16 {
    let pass = (size * size) as u16;
    if idx == pass {
        pass
    } else {
        d8(idx as usize, t, size) as u16
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_core::Game;

    #[test]
    fn planes_reflect_stones_liberties_and_komi_side() {
        let g = Go::new(9);
        let mut s = g.initial_state();
        // Black d4, White corner stone a1 reduced to one liberty.
        g.apply(&mut s, GoAction::Place(g.point("d4").unwrap()));
        g.apply(&mut s, GoAction::Place(g.point("a1").unwrap()));
        g.apply(&mut s, GoAction::Place(g.point("a2").unwrap()));
        // White to move: own = white.
        let x = GoEncoder.encode_state(&g, &s);
        let n = 81;
        let a1 = g.point("a1").unwrap() as usize;
        let a2 = g.point("a2").unwrap() as usize;
        let d4 = g.point("d4").unwrap() as usize;
        assert_eq!(x[a1], 1.0, "own (white) stone");
        assert_eq!(x[n + a2], 1.0, "opp (black) stone");
        assert_eq!(x[n + d4], 1.0, "opp (black) stone");
        assert_eq!(x[2 * n + a1], 1.0, "white a1 is in atari");
        assert_eq!(x[4 * n + a1], 0.0, "atari group is not the 2-lib plane");
        assert_eq!(x[7 * n], 1.0, "white to move sets the komi plane");
        assert_eq!(x[8 * n + 40], 1.0, "ones plane");

        let visible: f32 = x[6 * n..7 * n].iter().sum();
        assert_eq!(visible, 0.0, "no illegal empties this early");
    }

    #[test]
    fn suicide_point_marked_illegal() {
        let g = Go::new(5);
        // Empty c3 surrounded by black: suicide for White (to move).
        let s = g.parse_state(
            &[
                ". . . . .",
                ". . X . .",
                ". X . X .",
                ". . X . .",
                ". . . . .",
            ],
            1,
        );
        let x = GoEncoder.encode_state(&g, &s);
        let c3 = g.point("c3").unwrap() as usize;
        assert_eq!(x[6 * 25 + c3], 1.0, "suicide marked illegal for white");
    }

    #[test]
    fn d8_transforms_are_permutations_with_inverses() {
        for size in [5usize, 9] {
            for t in 0..8u8 {
                let inv = inverse_d8(t);
                let mut hit = vec![false; size * size];
                for p in 0..size * size {
                    let q = d8(p, t, size);
                    assert!(!hit[q], "t={t} collides at {q}");
                    hit[q] = true;
                    if t < 4 {
                        assert_eq!(d8(q, inv, size), p, "rotation inverse");
                    } else {
                        assert_eq!(d8(q, t, size), p, "mirrored transforms self-invert");
                    }
                }
            }
        }
    }

    #[test]
    fn d8_policy_fixes_pass_and_moves_placements() {
        let size = 9;
        assert_eq!(d8_policy(81, 3, size), 81);
        // a1 (index 0) under one quarter turn lands on a different corner.
        let moved = d8_policy(0, 1, size);
        assert_ne!(moved, 0);
        assert!([8, 72, 80].contains(&moved), "corner maps to corner");
    }

    #[test]
    fn encoding_commutes_with_board_symmetry() {
        let g = Go::new(9);
        let mut s = g.initial_state();
        for coord in ["d4", "f5", "c3", "g6", "e5"] {
            g.apply(&mut s, GoAction::Place(g.point(coord).unwrap()));
        }
        let x = GoEncoder.encode_state(&g, &s);
        for t in 0..8u8 {
            // Build the transformed position by replaying transformed moves.
            let mut ts = g.initial_state();
            for coord in ["d4", "f5", "c3", "g6", "e5"] {
                let p = g.point(coord).unwrap() as usize;
                g.apply(&mut ts, GoAction::Place(d8(p, t, 9) as u16));
            }
            let tx = GoEncoder.encode_state(&g, &ts);
            for plane in 0..PLANES {
                for p in 0..81 {
                    assert_eq!(
                        x[plane * 81 + p],
                        tx[plane * 81 + d8(p, t, 9)],
                        "plane {plane} point {p} symmetry {t}"
                    );
                }
            }
        }
    }
}
