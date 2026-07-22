//! AlphaZero-style policy-value encoding for Pente, plus the dihedral board
//! symmetries the trainer augments samples with.
//!
//! Eight planes, all from the side-to-move's perspective: own/opponent stones,
//! the last placement, the own stones standing in a pair the opponent can
//! capture next move and the opponent stones standing in a pair we can capture
//! (the capture tactics a pure stone map hides), two constant planes
//! broadcasting each side's captured-pair count toward the five-pair win (the
//! one part of the position the stone planes cannot show), and a constant ones
//! plane so zero conv padding does not read off-board as empty. Policy index `p`
//! is the board index of a placement; Pente has no pass, so the spatial head's
//! trailing pass logit (index `size²`) is allocated for head reuse and left
//! unused.

use game_core::PolicyValueEncoder;

use crate::{EMPTY, PAIRS_TO_WIN, Pente, PenteAction, PenteState, for_each_captured_pair};

const OWN: usize = 0;
const OPP: usize = 1;
const LAST: usize = 2;
/// Own stones in a pair the opponent can flank-capture on its next move.
const OWN_VULNERABLE: usize = 3;
/// Opponent stones in a pair we can flank-capture on our next move.
const OPP_CAPTURABLE: usize = 4;
const OWN_PAIRS: usize = 5;
const OPP_PAIRS: usize = 6;
const ONES: usize = 7;
pub const PLANES: usize = 8;

/// Board-size-parameterized so the policy-head width and input length are known
/// without a state in hand — the [`PolicyValueEncoder`] length methods take only
/// `&self`. The conv weights are board-size-agnostic (global-pool head), so one
/// trained net runs at any size; only the encoder carries the size.
pub struct PenteEncoder {
    size: usize,
}

impl PenteEncoder {
    pub fn new(size: usize) -> PenteEncoder {
        PenteEncoder { size }
    }
}

impl PolicyValueEncoder<Pente> for PenteEncoder {
    fn input_len(&self) -> usize {
        PLANES * self.size * self.size
    }

    fn policy_len(&self) -> usize {
        // Spatial head width: one logit per point plus the (unused) pass slot.
        self.size * self.size + 1
    }

    fn encode_state(&self, game: &Pente, state: &PenteState) -> Vec<f32> {
        let size = game.size();
        let n = size * size;
        let mut out = vec![0.0f32; PLANES * n];
        let own = state.to_move as u8;
        let opp = own ^ 1;

        for (p, &c) in state.cells[..n].iter().enumerate() {
            if c == own {
                out[OWN * n + p] = 1.0;
            } else if c == opp {
                out[OPP * n + p] = 1.0;
            }
        }
        if let Some(last) = state.last {
            out[LAST * n + last as usize] = 1.0;
        }

        // A capturable pair is two `victim` stones with the placer's stone behind
        // them and an empty flank in front (`[empty][victim][victim][placer]`):
        // the placer takes them by filling the empty flank. Mark the two victim
        // stones so the net sees its own pairs in danger and the opponent's pairs
        // it can take.
        mark_capturable_pairs(
            &state.cells[..n],
            size,
            own,
            opp,
            &mut out[OPP_CAPTURABLE * n..(OPP_CAPTURABLE + 1) * n],
        );
        mark_capturable_pairs(
            &state.cells[..n],
            size,
            opp,
            own,
            &mut out[OWN_VULNERABLE * n..(OWN_VULNERABLE + 1) * n],
        );

        let denom = f32::from(PAIRS_TO_WIN);
        out[OWN_PAIRS * n..(OWN_PAIRS + 1) * n].fill(f32::from(state.pairs[own as usize]) / denom);
        out[OPP_PAIRS * n..(OPP_PAIRS + 1) * n].fill(f32::from(state.pairs[opp as usize]) / denom);
        out[ONES * n..(ONES + 1) * n].fill(1.0);
        out
    }

    fn action_index(&self, _game: &Pente, _state: &PenteState, action: PenteAction) -> usize {
        action.0 as usize
    }
}

/// For every empty flank from which `placer` could capture a `victim` pair
/// (`[empty][victim][victim][placer]` outward), mark the two victim stones in
/// `plane`.
fn mark_capturable_pairs(cells: &[u8], size: usize, placer: u8, victim: u8, plane: &mut [f32]) {
    for p in 0..cells.len() {
        if cells[p] != EMPTY {
            continue;
        }
        for_each_captured_pair(cells, size, p, placer, victim, |a, b| {
            plane[a] = 1.0;
            plane[b] = 1.0;
        });
    }
}

/// Board index `p` under symmetry `t` (0..8): rotation by `t % 4` quarter turns,
/// then a horizontal mirror if `t >= 4`. `d8(p, t)` followed by
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

/// Policy index under symmetry `t`: placements move with the board, the unused
/// pass slot (`size²`) is fixed.
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
    use crate::{BLACK, PenteAction, WHITE};
    use game_core::Game;

    #[test]
    fn planes_reflect_stones_last_and_pairs() {
        let g = Pente::new(9);
        // Black pair b2,c2 with empty flanks (a2, d2) and no white flanker: not
        // yet capturable. White to move, with the pair counts [black 1, white 2].
        let mut s = g.parse_state(
            &[
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". X X . . . . . .",
                ". . . . . . . . .",
            ],
            1,
            [1, 2],
        );
        s.last = Some(g.point("c2").unwrap());
        let n = 81;
        let x = PenteEncoder::new(9).encode_state(&g, &s);

        let b2 = g.point("b2").unwrap() as usize;
        let c2 = g.point("c2").unwrap() as usize;
        // White to move: own = white, opp = black.
        assert_eq!(x[OPP * n + b2], 1.0, "black stone is the opponent's");
        assert_eq!(x[OWN * n + b2], 0.0);
        assert_eq!(x[LAST * n + c2], 1.0, "last move plane");
        let capturable: f32 = x[OPP_CAPTURABLE * n..(OPP_CAPTURABLE + 1) * n].iter().sum();
        assert_eq!(capturable, 0.0, "no capturable pair in this shape");
        // Pair-count planes broadcast pairs/5 from the mover's view.
        assert_eq!(x[OWN_PAIRS * n], 2.0 / 5.0, "white (mover) has 2 pairs");
        assert_eq!(x[OPP_PAIRS * n], 1.0 / 5.0, "black has 1 pair");
        assert_eq!(x[ONES * n + 40], 1.0, "ones plane");
    }

    #[test]
    fn capturable_pair_is_marked() {
        let g = Pente::new(9);
        // X O O . : Black (to move) can play the empty right flank to capture the
        // white pair. From Black's view those white stones are capturable.
        let s = g.parse_state(
            &[
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". X O O . . . . .",
                ". . . . . . . . .",
            ],
            0,
            [0, 0],
        );
        let n = 81;
        let x = PenteEncoder::new(9).encode_state(&g, &s);
        let c2 = g.point("c2").unwrap() as usize; // first O
        let d2 = g.point("d2").unwrap() as usize; // second O
        assert_eq!(x[OPP_CAPTURABLE * n + c2], 1.0, "white pair is capturable");
        assert_eq!(x[OPP_CAPTURABLE * n + d2], 1.0);
        // None of Black's own stones are in danger here.
        let vulnerable: f32 = x[OWN_VULNERABLE * n..(OWN_VULNERABLE + 1) * n].iter().sum();
        assert_eq!(vulnerable, 0.0);
    }

    #[test]
    fn action_index_is_the_board_index() {
        let g = Pente::new(13);
        let s = g.initial_state();
        let enc = PenteEncoder::new(13);
        let a = PenteAction(57);
        assert_eq!(enc.action_index(&g, &s, a), 57);
        assert_eq!(enc.policy_len(), 13 * 13 + 1);
        assert_eq!(enc.input_len(), PLANES * 13 * 13);
    }

    #[test]
    fn d8_transforms_are_permutations_with_inverses() {
        for size in [5usize, 9, 13] {
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
    fn encoding_commutes_with_board_symmetry() {
        let g = Pente::new(9);
        let coords = ["c3", "d3", "e5", "f6", "c4"];
        let mut s = g.parse_state(&[". . . . . . . . ."; 9], 0, [1, 2]);
        for c in coords {
            s.cells[g.point(c).unwrap() as usize] =
                if c == "e5" || c == "f6" { WHITE } else { BLACK };
        }
        s.last = Some(g.point("c4").unwrap());
        let x = PenteEncoder::new(9).encode_state(&g, &s);

        for t in 0..8u8 {
            let mut ts = g.parse_state(&[". . . . . . . . ."; 9], 0, [1, 2]);
            for c in coords {
                let p = g.point(c).unwrap() as usize;
                ts.cells[d8(p, t, 9)] = if c == "e5" || c == "f6" { WHITE } else { BLACK };
            }
            ts.last = Some(d8(g.point("c4").unwrap() as usize, t, 9) as u16);
            let tx = PenteEncoder::new(9).encode_state(&g, &ts);
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
