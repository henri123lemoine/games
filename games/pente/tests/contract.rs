//! Game-trait contract + panic hunt: coordinate round-tripping (including the
//! skipped `i` column on a full 19-board), legal-move completeness, the
//! fallback move path, parse_action robustness, and a deep random-self-play
//! fuzz across every legal board size driving the whole trait surface.

use game_core::{Game, GameUi, Rng, Turn};
use pente::{Pente, PenteAction, PenteState};

/// Independent five-in-a-row scan over the whole board for `color` (a u8 stone
/// code), used to re-derive line wins without relying on the engine's own
/// makes_line. Scans all four orientations from every cell.
fn has_five(s: &PenteState, size: usize, color: u8) -> bool {
    let col = color as usize;
    let cell = |r: i32, c: i32| -> bool {
        r >= 0
            && c >= 0
            && (r as usize) < size
            && (c as usize) < size
            && s.stone(r as usize * size + c as usize) == Some(col)
    };
    for r in 0..size as i32 {
        for c in 0..size as i32 {
            for (dr, dc) in [(0, 1), (1, 0), (1, 1), (1, -1)] {
                if (0..5).all(|k| cell(r + dr * k, c + dc * k)) {
                    return true;
                }
            }
        }
    }
    false
}

// Coordinate letters skip `i`; the inverse must round-trip across every column
// on the largest board (19 needs letters a..t with i skipped).
#[test]
fn coordinate_roundtrip_skips_i_on_19_board() {
    let g = Pente::new(19);
    for col in 0..19 {
        for row in 1..=19 {
            // Build the label the UI would print, parse it back, and confirm the
            // index matches.
            let p = (row - 1) * 19 + col;
            let label = g.action_label(&g.initial_state(), PenteAction(p as u16));
            let parsed = g
                .point(&label)
                .unwrap_or_else(|| panic!("failed to parse {label}"));
            assert_eq!(parsed as usize, p, "label {label} did not round-trip");
        }
    }
    // The letter `i` is never a valid column.
    assert!(g.point("i10").is_none(), "column i must be invalid");
    // Column 8 (0-based) prints as `j`, not `i`.
    assert_eq!(g.action_label(&g.initial_state(), PenteAction(8)), "j1");
}

#[test]
fn point_rejects_out_of_range_and_garbage() {
    let g = Pente::new(13);
    assert!(g.point("a0").is_none(), "row 0 is invalid");
    assert!(g.point("a14").is_none(), "row 14 off a 13-board");
    assert!(g.point("z1").is_none(), "column z off a 13-board");
    assert!(g.point("").is_none());
    assert!(g.point("g").is_none(), "no row");
    assert!(g.point("7").is_none(), "no column letter");
    assert!(g.point("gg7").is_none(), "two letters");
    assert!(g.point("g7g").is_none(), "trailing junk");
}

// parse_action must only return *legal* actions and never panic on bad input.
#[test]
fn parse_action_is_legal_only_and_never_panics() {
    let g = Pente::new(13);
    let mut s = g.initial_state();
    g.apply(&mut s, PenteAction(g.center()));
    assert!(
        g.parse_action(&s, "g7").is_none(),
        "occupied center is illegal"
    );
    assert!(
        g.parse_action(&s, "a1").is_none(),
        "far corner is pruned, illegal"
    );
    assert!(g.parse_action(&s, "garbage").is_none());
    assert!(g.parse_action(&s, "").is_none());
    assert!(
        g.parse_action(&s, "i10").is_none(),
        "the i column is unparseable"
    );
    assert!(
        g.parse_action(&s, "h8").is_some(),
        "a nearby empty is legal"
    );
}

// Every legal action targets an empty cell; the union of legal moves and
// occupied cells covers the board with no overlap mid-game.
#[test]
fn legal_actions_are_empty_cells_only() {
    let g = Pente::new(9);
    let mut rng = Rng::new(0x5EED);
    let mut s = g.initial_state();
    for _ in 0..20 {
        if g.is_terminal(&s) {
            break;
        }
        let acts = g.legal_actions(&s);
        for a in &acts {
            assert_eq!(
                s.stone(a.0 as usize),
                None,
                "legal action on an occupied cell"
            );
            // No duplicates.
            assert_eq!(
                acts.iter().filter(|&&b| b == *a).count(),
                1,
                "duplicate action {a:?}"
            );
        }
        let i = rng.below(acts.len());
        g.apply(&mut s, acts[i]);
    }
}

// The all-empty board (post-opening) yields the center-only action; a board
// with only far-flung stones still produces legal moves (fallback path), never
// an empty action set on a non-terminal state.
#[test]
fn never_empty_action_set_on_non_terminal() {
    for size in [5usize, 7, 9, 13, 15, 19] {
        let g = Pente::new(size);
        let mut rng = Rng::new(0xF0F0 ^ size as u64);
        for trial in 0..40 {
            let mut s = g.initial_state();
            let mut steps = 0;
            while !g.is_terminal(&s) {
                let acts = g.legal_actions(&s);
                assert!(
                    !acts.is_empty(),
                    "size {size} trial {trial}: empty action set on non-terminal after {steps} steps"
                );
                let i = rng.below(acts.len());
                g.apply(&mut s, acts[i]);
                steps += 1;
                // Captures refill cells, so allow slack beyond size*size (bounded
                // by the five-pair cap on total captures).
                assert!(
                    steps <= size * size + 2 * 9 + 2,
                    "size {size}: runaway game"
                );
            }
        }
    }
}

// is_terminal/winner/returns are mutually consistent at game end; the loser is
// the negation of the winner; a draw returns 0/0.
#[test]
fn terminal_winner_returns_are_consistent() {
    for size in [5usize, 9, 13] {
        let g = Pente::new(size);
        let mut rng = Rng::new(0xC0DE ^ size as u64);
        let mut decisive = 0;
        let mut draws = 0;
        for _ in 0..60 {
            let mut s = g.initial_state();
            while !g.is_terminal(&s) {
                let acts = g.legal_actions(&s);
                g.apply(&mut s, acts[rng.below(acts.len())]);
            }
            match s.winner() {
                Some(w) => {
                    decisive += 1;
                    assert_eq!(g.returns(&s, w), 1.0);
                    assert_eq!(g.returns(&s, w ^ 1), -1.0);
                    // The recorded winner must actually satisfy a win condition:
                    // five captured pairs OR five-in-a-row on the final board.
                    let won_on_pairs = s.pairs()[w] >= pente::PAIRS_TO_WIN;
                    let won_on_line = has_five(&s, size, w as u8);
                    assert!(
                        won_on_pairs || won_on_line,
                        "size {size}: winner {w} satisfies neither pairs nor a line"
                    );
                }
                None => {
                    draws += 1;
                    assert_eq!(g.returns(&s, 0), 0.0);
                    assert_eq!(g.returns(&s, 1), 0.0);
                    // A draw must be a full board.
                    let filled = (0..size * size).all(|p| s.stone(p).is_some());
                    assert!(filled, "size {size}: a draw must be a full board");
                }
            }
        }
        // Sanity that the fuzz actually reached terminal states of both/either
        // kind (mostly decisive on small boards).
        assert!(decisive + draws == 60);
    }
}

// state_key / infoset_key are deterministic and distinguish meaningful states.
#[test]
fn keys_are_deterministic_and_discriminating() {
    let g = Pente::new(9);
    let s = g.parse_state(&[". . . . . . . . ."; 9], 0, [0, 0]);
    assert_eq!(g.state_key(&s), g.state_key(&s.clone()), "deterministic");
    assert_eq!(
        g.infoset_key(&s, 0),
        g.infoset_key(&s, 1),
        "perfect info: same key"
    );

    let mut moved = s.clone();
    g.apply(&mut moved, PenteAction(g.point("e5").unwrap()));
    assert_ne!(
        g.state_key(&s),
        g.state_key(&moved),
        "a placement changes the key"
    );
}

// Deep fuzz: drive the entire trait surface (turn, legal_actions, apply,
// returns, state_key, num_players, chance_outcomes) under random play on every
// size, asserting no panic and the invariants hold throughout.
#[test]
fn deep_fuzz_no_panics_across_sizes() {
    for size in [5usize, 6, 7, 11, 13, 15, 19] {
        let g = Pente::new(size);
        assert_eq!(g.num_players(), 2);
        let mut rng = Rng::new(0xABCDEF ^ (size as u64).wrapping_mul(2654435761));
        for _ in 0..25 {
            let mut s = g.initial_state();
            let mut prev_moves = 0u32;
            while !g.is_terminal(&s) {
                assert!(g.chance_outcomes(&s).is_empty(), "no chance nodes");
                match g.turn(&s) {
                    Turn::Player(p) => assert!(p < 2),
                    _ => panic!("Pente must only have player turns"),
                }
                let _ = g.state_key(&s);
                let acts = g.legal_actions(&s);
                let a = acts[rng.below(acts.len())];
                g.apply(&mut s, a);
                // The move counter is monotone and a placement happened.
                assert!(s.moves() > prev_moves, "moves() must advance on apply");
                prev_moves = s.moves();
            }
            let r = g.returns(&s, 0);
            assert!(
                r == 1.0 || r == -1.0 || r == 0.0,
                "size {size}: bad return {r}"
            );
        }
    }
}
