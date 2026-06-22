//! Deeper adversarial probes: capture/line interactions, legal-move
//! completeness after captures, win-precedence, and contract invariants that
//! the happy-path tests don't exercise.

use game_core::{Game, Rng, Turn};
use pente::{Pente, PenteAction, PenteState};

fn place(g: &Pente, s: &mut PenteState, coord: &str) {
    g.apply(s, PenteAction(g.point(coord).unwrap()));
}

fn at(g: &Pente, s: &PenteState, coord: &str) -> Option<usize> {
    s.stone(g.point(coord).unwrap() as usize)
}

fn legal_has(g: &Pente, s: &PenteState, coord: &str) -> bool {
    let a = PenteAction(g.point(coord).unwrap());
    g.legal_actions(s).contains(&a)
}

// A move that simultaneously completes a five AND captures a pair: the win must
// be reported (five-in-a-row), and the capture still resolved.
#[test]
fn line_and_capture_on_the_same_move() {
    let g = Pente::new(9);
    // Black has X X X X . horizontally (a2..d2, place e2 makes five). On a
    // separate arm from e2, set up a capture: e2 also flanks O O X going up.
    // e2 up: e3=O, e4=O, e5=X -> capturing that pair, while a2..e2 makes five.
    let mut s = g.parse_state(
        &[
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . X . . . .", // row 5: e5 = X (far flank of the vertical arm)
            ". . . . O . . . .", // row 4: e4 = O
            ". . . . O . . . .", // row 3: e3 = O
            "X X X X . . . . .", // row 2: a2..d2 = X (place e2 makes five)
            ". . . . . . . . .",
        ],
        0,
        [0, 0],
    );
    place(&g, &mut s, "e2");
    assert!(g.is_terminal(&s), "five completes the line");
    assert_eq!(g.returns(&s, 0), 1.0);
    // The capture resolves on the same move.
    assert_eq!(s.pairs(), [1, 0], "the flanked pair is also captured");
    assert_eq!(at(&g, &s, "e3"), None);
    assert_eq!(at(&g, &s, "e4"), None);
}

// After a capture empties two cells, those cells must be legal moves again (they
// are empty and adjacent to remaining stones).
#[test]
fn captured_cells_reopen_as_legal_moves() {
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
            ". X O O . . . . .", // b2=X, c2=O, d2=O
            ". . . . . . . . .",
        ],
        0,
        [0, 0],
    );
    place(&g, &mut s, "e2"); // captures c2, d2
    assert_eq!(at(&g, &s, "c2"), None);
    assert_eq!(at(&g, &s, "d2"), None);
    // It is now White to move; the freshly-emptied points are near stones and
    // must be legal again.
    assert!(legal_has(&g, &s, "c2"), "captured cell c2 is legal again");
    assert!(legal_has(&g, &s, "d2"), "captured cell d2 is legal again");
}

// Win-by-capture must take precedence even when no line exists; and the loser
// gets -1. Already covered, but assert the WINNER is the capturing side, not the
// side to move after the flip.
#[test]
fn capture_win_credits_the_capturing_side_not_the_next_mover() {
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
            ". X O O . . . . .",
            ". . . . . . . . .",
        ],
        0,
        [4, 0],
    );
    place(&g, &mut s, "e2"); // Black takes the fifth pair
    assert_eq!(s.winner(), Some(0), "Black (the mover) is the winner");
    // After apply, to_move flipped to White, but White did not win.
    assert_eq!(s.to_move(), 1);
    assert_eq!(g.returns(&s, 0), 1.0);
    assert_eq!(g.returns(&s, 1), -1.0);
}

// White can also win (symmetry): not a Black-only thing.
#[test]
fn white_can_win_by_line_and_by_capture() {
    let g = Pente::new(9);
    // White five.
    let mut s = g.parse_state(
        &[
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            "O O O O . . . . .",
            ". . . . . . . . .",
        ],
        1,
        [0, 0],
    );
    place(&g, &mut s, "e2");
    assert!(
        g.is_terminal(&s) && s.winner() == Some(1),
        "white five wins"
    );
    assert_eq!(g.returns(&s, 1), 1.0);

    // White fifth pair.
    let mut s = g.parse_state(
        &[
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". O X X . . . . .",
            ". . . . . . . . .",
        ],
        1,
        [0, 4],
    );
    place(&g, &mut s, "e2");
    assert_eq!(s.pairs(), [0, 5]);
    assert!(
        g.is_terminal(&s) && s.winner() == Some(1),
        "white fifth pair wins"
    );
}

// A self-capture trap that the rules MUST allow as safe: placing your stone to
// complete [opp][you][you][opp] around your OWN pre-existing single stone, where
// you place the second of your two. Verified safe at all four orientations.
#[test]
fn safe_move_into_bracket_all_orientations() {
    let g = Pente::new(11);
    // For each orientation, lay O X . O along the ray, then have Black (color X)
    // fill the inner gap, forming O X X O around its own pair: never a capture.
    let cases: [(&str, &str, &str, &str); 4] = [
        // (opp_low, my_existing, gap_to_fill, opp_high) along the ray
        ("d6", "e6", "f6", "g6"), // horizontal: O X . O at d..g row6
        ("f4", "f5", "f6", "f7"), // vertical
        ("d4", "e5", "f6", "g7"), // diagonal up-right
        ("d8", "e7", "f6", "g5"), // diagonal down-right
    ];
    for (opp_low, mine, gap, opp_high) in cases {
        let mut rows = vec![String::new(); 11];
        for (i, rr) in rows.iter_mut().enumerate() {
            let r = 11 - 1 - i;
            for c in 0..11 {
                let p = (r * 11 + c) as u16;
                let label = if p == g.point(opp_low).unwrap() || p == g.point(opp_high).unwrap() {
                    'O'
                } else if p == g.point(mine).unwrap() {
                    'X'
                } else {
                    '.'
                };
                rr.push(label);
            }
        }
        let row_refs: Vec<&str> = rows.iter().map(|s| s.as_str()).collect();
        let mut s = g.parse_state(&row_refs, 0, [0, 0]);
        place(&g, &mut s, gap);
        assert_eq!(s.pairs(), [0, 0], "{gap}: moving into a bracket is safe");
        assert_eq!(at(&g, &s, mine), Some(0), "{gap}: existing X survives");
        assert_eq!(at(&g, &s, gap), Some(0), "{gap}: placed X survives");
    }
}

// After a capture the removed cells read as empty to the line scanner: there is
// no "ghost" stone that could still complete a line where a captured stone sat.
#[test]
fn capturing_clears_cells_for_the_line_scanner() {
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
            ". X O O . . . . .",
            ". . . . . . . . .",
        ],
        0,
        [0, 0],
    );
    place(&g, &mut s, "e2"); // captures c2,d2
    // No win, board has only b2(X) e2(X) on that row now.
    assert!(!g.is_terminal(&s));
    assert_eq!(at(&g, &s, "c2"), None);
    assert_eq!(at(&g, &s, "d2"), None);
}

// Determinism + zero-sum + termination over many random playouts on several
// sizes, including the smallest legal board.
#[test]
fn random_games_are_zero_sum_and_terminate_on_all_sizes() {
    for size in [5usize, 7, 9, 13] {
        let g = Pente::new(size);
        let mut rng = Rng::new(0xBADC0FFE ^ size as u64);
        for _ in 0..15 {
            let mut s = g.initial_state();
            let mut plies = 0usize;
            // Captures empty cells that must be refilled, so a game can need more
            // than size*size placements. The slack is bounded: each captured pair
            // bumps a monotone counter that ends the game at five, so at most
            // ~9 pairs (18 cells) are ever recaptured before someone wins.
            let bound = size * size + 2 * 9 + 2;
            while !g.is_terminal(&s) {
                assert!(
                    plies <= bound,
                    "size {size}: must terminate within {bound} plies"
                );
                let actions = g.legal_actions(&s);
                assert!(!actions.is_empty(), "size {size}: non-terminal has moves");
                // Every reported legal action must target an empty cell.
                for a in &actions {
                    assert_eq!(s.stone(a.0 as usize), None, "legal action on occupied cell");
                }
                let i = rng.below(actions.len());
                g.apply(&mut s, actions[i]);
                plies += 1;
            }
            let r0 = g.returns(&s, 0);
            let r1 = g.returns(&s, 1);
            assert!(
                (r0 + r1).abs() < 1e-9,
                "size {size}: zero-sum, got {r0}/{r1}"
            );
            assert!(r0 == 1.0 || r0 == -1.0 || r0 == 0.0);
        }
    }
}

// The turn reported must always match to_move and be a Player turn (no chance).
#[test]
fn turn_matches_to_move_and_no_chance_nodes() {
    let g = Pente::new(9);
    let s = g.initial_state();
    assert!(matches!(g.turn(&s), Turn::Player(0)));
    assert!(g.chance_outcomes(&s).is_empty());
    let mut s = s;
    place(&g, &mut s, "e5"); // center on 9x9
    assert!(matches!(g.turn(&s), Turn::Player(1)));
}
