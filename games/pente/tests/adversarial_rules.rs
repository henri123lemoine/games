//! Adversarial rule tests written to BREAK the Pente implementation:
//! edge/corner captures, the exactly-two custodial rule against longer runs,
//! multi-capture timing, overlines, win-by-capture timing, the tournament
//! opening, draws, and no-wrap-across-edges.

use game_core::Game;
use pente::{Pente, PenteAction, PenteState};

fn place(g: &Pente, s: &mut PenteState, coord: &str) {
    let p = g.point(coord).unwrap();
    assert_eq!(
        s.stone(p as usize),
        None,
        "{coord} must be empty before placing"
    );
    g.apply(s, PenteAction(p));
}

fn at(g: &Pente, s: &PenteState, coord: &str) -> Option<usize> {
    s.stone(g.point(coord).unwrap() as usize)
}

// --- Edge / corner captures -------------------------------------------------

#[test]
fn capture_along_the_bottom_edge() {
    // Bottom row (row 1): X O O .  with the flank placed at d1, on the very edge.
    let g = Pente::new(9);
    let mut s = g.parse_state(
        &[
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            "X O O . . . . . .",
        ],
        0,
        [0, 0],
    );
    place(&g, &mut s, "d1");
    assert_eq!(at(&g, &s, "b1"), None, "edge: near captured");
    assert_eq!(at(&g, &s, "c1"), None, "edge: far captured");
    assert_eq!(s.pairs(), [1, 0]);
}

#[test]
fn capture_into_the_corner() {
    // Corner anchor at a1: place flank by completing X(a1) O(b1) O(c1) X(d1).
    let g = Pente::new(9);
    let mut s = g.parse_state(
        &[
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            "X O O . . . . . .", // row 1: a1=X, b1=O, c1=O, place flank at d1
        ],
        0,
        [0, 0],
    );
    place(&g, &mut s, "d1");
    assert_eq!(at(&g, &s, "b1"), None);
    assert_eq!(at(&g, &s, "c1"), None);
    assert_eq!(at(&g, &s, "a1"), Some(0), "corner anchor stays");
    assert_eq!(s.pairs(), [1, 0]);
}

#[test]
fn no_capture_when_far_flank_is_off_board() {
    // O O X on the bottom-left: the pattern that would capture needs a black
    // stone one past the edge. Placing the only available black flank can never
    // wrap to the other side. Set up . O O X near the left wall and verify that
    // playing to the LEFT of the pair (off-board) is impossible and nothing is
    // captured by an interior move.
    let g = Pente::new(9);
    let mut s = g.parse_state(
        &[
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            "O O X . . . . . .",
            ". . . . . . . . .",
        ],
        0,
        [0, 0],
    );
    // Black to move; there is no legal point to the left of a2 (it is the wall),
    // so the white pair a2,b2 cannot be custodially captured by Black here.
    // Any black move elsewhere captures nothing.
    place(&g, &mut s, "e2");
    assert_eq!(
        at(&g, &s, "a2"),
        Some(1),
        "off-board flank: no wrap capture"
    );
    assert_eq!(at(&g, &s, "b2"), Some(1));
    assert_eq!(s.pairs(), [0, 0]);
}

// --- Exactly-two rule against longer runs -----------------------------------

#[test]
fn flanking_four_in_a_row_captures_nothing() {
    // X O O O O X — placing the second X flank must NOT capture the run of four.
    let g = Pente::new(9);
    let mut s = g.parse_state(
        &[
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". X O O O O . . .",
            ". . . . . . . . .",
        ],
        0,
        [0, 0],
    );
    place(&g, &mut s, "g2"); // X O O O O X
    for c in ["c2", "d2", "e2", "f2"] {
        assert_eq!(at(&g, &s, c), Some(1), "{c}: run of four is immune");
    }
    assert_eq!(s.pairs(), [0, 0]);
}

#[test]
fn placing_inside_a_longer_run_does_not_capture_a_subpair() {
    // X O O O X already on the board with the flanks present; the only empty is
    // not a flank. Verify a placement at the *other* side of a sub-pair never
    // captures two of the three. Build X O O O . and place the right flank: that
    // makes X O O O X, no capture (immune triple).
    let g = Pente::new(9);
    let mut s = g.parse_state(
        &[
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". X O O O . . . .",
            ". . . . . . . . .",
        ],
        0,
        [0, 0],
    );
    place(&g, &mut s, "f2");
    assert_eq!(s.pairs(), [0, 0], "no sub-pair of a triple is captured");
    assert_eq!(at(&g, &s, "c2"), Some(1));
    assert_eq!(at(&g, &s, "d2"), Some(1));
    assert_eq!(at(&g, &s, "e2"), Some(1));
}

// --- Safe move into a bracket -----------------------------------------------

#[test]
fn placing_the_inner_stone_into_a_full_bracket_is_safe() {
    // O X X O on a row with one inner X missing: O X . O. White's stone at the
    // gap forms O X X O around its OWN pair, which must NOT self-capture.
    let g = Pente::new(9);
    let mut s = g.parse_state(
        &[
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". O X . O . . . .",
            ". . . . . . . . .",
        ],
        0, // Black to move, completing its own pair inside the white bracket
        [0, 0],
    );
    place(&g, &mut s, "d2"); // O X X O formed by Black's own move
    assert_eq!(at(&g, &s, "c2"), Some(0), "black inner pair survives...");
    assert_eq!(at(&g, &s, "d2"), Some(0), "...the move that formed it");
    assert_eq!(
        s.pairs(),
        [0, 0],
        "moving into a bracket never self-captures"
    );
}

// --- Multiple captures from one stone, near a corner ------------------------

#[test]
fn multi_direction_capture_at_a_corner_anchor() {
    // Place a black stone that flanks two white pairs reaching out from a corner:
    // one horizontal arm and one vertical arm, both anchored at the same black
    // flank in the corner region.
    let g = Pente::new(9);
    // Flank to be placed at a1. Horizontal arm: a1 b1=O c1=O d1=X.
    // Vertical arm: a1 a2=O a3=O a4=X.
    let mut s = g.parse_state(
        &[
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            "X . . . . . . . .", // row 4: a4=X
            "O . . . . . . . .", // row 3: a3=O
            "O . . . . . . . .", // row 2: a2=O
            ". O O X . . . . .", // row 1: a1=. (flank), b1=O, c1=O, d1=X
        ],
        0,
        [0, 0],
    );
    place(&g, &mut s, "a1");
    assert_eq!(s.pairs(), [2, 0], "two pairs captured from the corner");
    for gone in ["b1", "c1", "a2", "a3"] {
        assert_eq!(at(&g, &s, gone), None, "{gone} captured");
    }
}

// --- Win by captures: exact timing of the fifth pair ------------------------

#[test]
fn fourth_pair_does_not_win_but_fifth_does() {
    let g = Pente::new(9);
    // At 3 pairs, a capturing move reaches 4 (not a win).
    let mut s = g.parse_state(
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
        [3, 0],
    );
    place(&g, &mut s, "e2");
    assert_eq!(s.pairs(), [4, 0]);
    assert!(!g.is_terminal(&s), "four pairs is not a win");

    // From 4 pairs, the capture to 5 wins immediately.
    let mut s = g.parse_state(
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
        [4, 0],
    );
    place(&g, &mut s, "e2");
    assert_eq!(s.pairs(), [5, 0]);
    assert!(g.is_terminal(&s), "fifth pair wins");
    assert_eq!(g.returns(&s, 0), 1.0);
    assert_eq!(g.returns(&s, 1), -1.0);
}

#[test]
fn fifth_pair_via_multi_capture_from_four() {
    // From four pairs, a single stone that captures TWO pairs at once should
    // win (reaching six pairs >= 5). Confirms the >= comparison, not ==.
    let g = Pente::new(9);
    let mut s = g.parse_state(
        &[
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . X . . . .",
            ". X O O . O O X .",
            ". . . . O . . . .",
            ". . . . O . . . .",
            ". . . . X . . . .",
            ". . . . . . . . .",
        ],
        0,
        [4, 0],
    );
    // e5 flanks the W, E, N, S arms; from 4 pairs this jumps well past 5.
    place(&g, &mut s, "e5");
    assert!(
        s.pairs()[0] >= 5,
        "captured enough pairs to win, got {:?}",
        s.pairs()
    );
    assert!(g.is_terminal(&s));
    assert_eq!(g.returns(&s, 0), 1.0);
}

// --- Overlines --------------------------------------------------------------

#[test]
fn overline_of_six_built_by_extending_a_five_still_wins() {
    // A genuine overline: four in a row with both ends open, fill one end making
    // five (win). Then separately, a six formed across a gap.
    let g = Pente::new(11);
    // X X X X X . -> already five; but test the *placement that makes six* from
    // an existing five with a gap pattern: X X X X X . X, fill makes 7. Use a
    // simpler robust check: a run of 6 wins.
    let mut s = g.parse_state(
        &[
            ". . . . . . . . . . .",
            ". . . . . . . . . . .",
            ". . . . . . . . . . .",
            ". . . . . . . . . . .",
            ". . . . . . . . . . .",
            ". X X X X X . . . . .",
            ". . . . . . . . . . .",
            ". . . . . . . . . . .",
            ". . . . . . . . . . .",
            ". . . . . . . . . . .",
            ". . . . . . . . . . .",
        ],
        0,
        [0, 0],
    );
    // This board already has five — but it was placed by parse_state, not via
    // apply, so winner is not set. Confirm: completing a SIXTH via apply still
    // reports terminal (overline must win, not be rejected).
    place(&g, &mut s, "g6"); // extends the five to a six at the right end
    assert!(g.is_terminal(&s), "overline (six) must win");
    assert_eq!(g.returns(&s, 0), 1.0);
}

// --- No wrap across the edge ------------------------------------------------

#[test]
fn five_does_not_wrap_across_a_row_boundary() {
    // Black stones at the right end of row 2 and the left end of row 3 are NOT a
    // line: g2 h2 j2 (cols 6,7,8) then a3 b3 (cols 0,1). Filling a join point
    // must not be read as five.
    let g = Pente::new(9);
    let mut s = g.parse_state(
        &[
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            "X X . . . . . . .", // row 3: a3,b3 = X
            ". . . . . . X X X", // row 2: g2,h2,j2 = X
            ". . . . . . . . .",
        ],
        0,
        [0, 0],
    );
    // There is no single straight line containing all five; placing any stone
    // must not declare a win by "wrapping". Fill f2 (extends row-2 run to four)
    // — still not five, definitely not a wrap.
    place(&g, &mut s, "f2");
    assert!(!g.is_terminal(&s), "no wrap-around five across the edge");
}

// --- Tournament opening -----------------------------------------------------

#[test]
fn black_opening_is_center_only_and_apply_enforces_nothing_else() {
    let g = Pente::new(13);
    let s = g.initial_state();
    let legal = g.legal_actions(&s);
    assert_eq!(
        legal,
        vec![PenteAction(g.center())],
        "only the center is legal"
    );
    assert_eq!(g.center(), g.point("g7").unwrap());
}

// --- Draw -------------------------------------------------------------------

#[test]
fn full_board_no_winner_is_a_draw() {
    let g = Pente::new(5);
    let mut s = g.parse_state(
        &[
            "X O X O X",
            "X O X O X",
            "O X O X O",
            "X O X O X",
            "X O X O .",
        ],
        1,
        [0, 0],
    );
    assert!(!g.is_terminal(&s));
    place(&g, &mut s, "e1");
    assert!(g.is_terminal(&s));
    assert_eq!(s.winner(), None);
    assert_eq!(g.returns(&s, 0), 0.0);
    assert_eq!(g.returns(&s, 1), 0.0);
}
