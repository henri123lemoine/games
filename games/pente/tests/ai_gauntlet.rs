//! Adversarial AI gauntlet for the Pente alpha-beta engine, run at the lab's
//! DEFAULT depth (4). Three things, all as hard assertions:
//!   * a tactics battery — every position has exactly one class of correct
//!     reply (win now, block the opponent's win, take a free capture, refuse to
//!     hang a pair); the engine must find it,
//!   * mass games vs RandomAgent (seat-swapped) — the engine must dominate,
//!   * a self-play depth ladder — deeper must not lose to shallower.

use game_core::{Agent, Game, GameUi, Rng, Turn};
use pente::{Pente, PenteEval, PenteSpec, PenteState};
use solvers::AlphaBeta;

const DEFAULT_DEPTH: u32 = 4;

fn engine(depth: u32) -> AlphaBeta<Pente, PenteEval, PenteSpec> {
    AlphaBeta::new(depth, PenteEval, PenteSpec)
}

/// The coordinate the engine plays from `s` at `depth`.
fn best(g: &Pente, s: &PenteState, depth: u32) -> String {
    let ab = engine(depth);
    let i = ab.best_action(g, s);
    g.action_label(s, g.legal_actions(s)[i])
}

/// Build a 9x9 position from rows (top first), `to_move` to play, with a
/// non-zero move counter so the center-only opening does not engage.
fn pos(rows: &[&str; 9], to_move: usize, pairs: [u8; 2]) -> (Pente, PenteState) {
    let g = Pente::new(9);
    let s = g.parse_state(rows, to_move, pairs);
    (g, s)
}

// ---- Tactics battery (default depth) ---------------------------------------

#[test]
fn takes_the_fifth_pair_to_win() {
    let (g, s) = pos(
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
    assert_eq!(
        best(&g, &s, DEFAULT_DEPTH),
        "e2",
        "must take the winning fifth pair"
    );
}

#[test]
fn completes_five_in_a_row_to_win() {
    let (g, s) = pos(
        &[
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". X X X X . . . .",
            ". . . . . . . . .",
        ],
        0,
        [0, 0],
    );
    let m = best(&g, &s, DEFAULT_DEPTH);
    assert!(m == "a2" || m == "f2", "must complete the five, played {m}");
}

#[test]
fn blocks_the_opponents_open_four() {
    // White (to move) faces Black's open four; both ends must be considered, the
    // engine must plug one (otherwise Black wins next).
    let (g, s) = pos(
        &[
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". X X X X . . . .",
            ". . . . . . . . .",
        ],
        1,
        [0, 0],
    );
    let m = best(&g, &s, DEFAULT_DEPTH);
    assert!(
        m == "a2" || m == "f2",
        "must block the open four, played {m}"
    );
}

#[test]
fn blocks_a_one_move_capture_win_by_the_opponent() {
    // White sits at 4 pairs and threatens the fifth via X O O .-> flank at e2.
    // Wait: here WHITE captures BLACK. Black (to move) must deny White's winning
    // capture. White's winning move would be to place at e2 making O X X O? No —
    // set up so WHITE's capture of a black pair at e2 wins. Black must pre-empt.
    // Board: O X X . with White at 4 pairs; White plays e2 to capture b2,c2? The
    // capturing flank for White is the empty end. Black must occupy it or break
    // the pair. The denial here is to take e2 (the flank point) itself.
    let (g, s) = pos(
        &[
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". O X X . . . . .", // b2=O, c2=X, d2=X ; White flank at e2 wins
            ". . . . . . . . .",
        ],
        0, // Black to move, must stop White's fifth-pair capture at e2
        [0, 4],
    );
    let m = best(&g, &s, DEFAULT_DEPTH);
    // Defenses: occupy e2 (deny the flank) is the direct one. Extending the pair
    // to c2..d2..e2 (play e2) also removes the capturable two-in-a-row.
    assert_eq!(
        m, "e2",
        "must deny White's winning custodial capture, played {m}"
    );
}

#[test]
fn takes_a_free_capture_when_nothing_better() {
    let (g, s) = pos(
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
    assert_eq!(
        best(&g, &s, DEFAULT_DEPTH),
        "e2",
        "a free capture should be taken"
    );
}

#[test]
fn does_not_walk_a_pair_into_a_standing_capture() {
    // Black has a lone stone d2 with a White stone at c2 (O X . pattern toward
    // the right). Playing e2 makes O X X . — a pair White captures next move by
    // playing f2. With the whole rest of the board available, the engine must
    // not voluntarily form that capturable pair. We assert it does NOT play e2.
    let (g, s) = pos(
        &[
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . O X . . . . .", // c2=O, d2=X
            ". . . . . . . . .",
        ],
        0,
        [0, 0],
    );
    let m = best(&g, &s, DEFAULT_DEPTH);
    assert_ne!(
        m, "e2",
        "must not walk a fresh pair into O X X . capture, played {m}"
    );
}

#[test]
fn prefers_winning_over_a_mere_capture() {
    // A five-completion AND an unrelated free capture are both available; the
    // engine must take the win, not the capture. Black's open four is b2..e2
    // (complete at a2 or f2); a separate free capture sits on row 6 (g6 flanks
    // the white pair e6,f6 against the black anchor d6).
    let (g, s) = pos(
        &[
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . X O O . . .", // d6=X, e6=O, f6=O : g6 would capture the pair
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". . . . . . . . .",
            ". X X X X . . . .", // open four b2..e2
            ". . . . . . . . .",
        ],
        0,
        [0, 0],
    );
    let m = best(&g, &s, DEFAULT_DEPTH);
    assert!(
        m == "a2" || m == "f2",
        "must win the game, not grab a capture; played {m}"
    );
}

// ---- Mass games vs RandomAgent ---------------------------------------------

/// `games` seat-swapped pairs of engine(depth) vs RandomAgent; returns the
/// engine's total score over `games*2` (1 win / 0.5 draw / 0 loss).
fn vs_random(depth: u32, games: u64) -> (f64, u64) {
    let g = Pente::new(9);
    let mut score = 0.0;
    let mut total = 0u64;
    let ab = engine(depth);
    let rnd = game_core::RandomAgent;
    for k in 0..games {
        for engine_is_black in [true, false] {
            let mut s = g.initial_state();
            let mut rng = Rng::new(0xA11CE ^ (k * 2 + engine_is_black as u64));
            while !g.is_terminal(&s) {
                let p = match g.turn(&s) {
                    Turn::Player(p) => p,
                    _ => unreachable!(),
                };
                let engine_to_move = (p == 0) == engine_is_black;
                let i = if engine_to_move {
                    ab.act(&g, &s, p, &mut rng)
                } else {
                    rnd.act(&g, &s, p, &mut rng)
                };
                let a = g.legal_actions(&s)[i];
                g.apply(&mut s, a);
            }
            let seat = if engine_is_black { 0 } else { 1 };
            score += (g.returns(&s, seat) + 1.0) / 2.0;
            total += 1;
        }
    }
    (score, total)
}

#[test]
fn dominates_random_at_default_depth() {
    // 30 pairs = 60 games. A competent threat-aware engine should crush random;
    // demand a very high score (random occasionally stumbles into a fast capture
    // but should essentially never win a game against depth-4 search).
    let (score, total) = vs_random(DEFAULT_DEPTH, 30);
    let frac = score / total as f64;
    assert!(
        frac >= 0.95,
        "engine vs random scored {score}/{total} = {frac:.3}, expected >= 0.95"
    );
}

// ---- Self-play depth ladder ------------------------------------------------

/// Seat-swapped match: deep(depth) vs shallow(depth). Returns deep's score over
/// `games*2`.
fn duel(deep: u32, shallow: u32, games: u64) -> (f64, u64) {
    let g = Pente::new(9);
    let mut score = 0.0;
    let mut total = 0u64;
    let (a, b) = (engine(deep), engine(shallow));
    for k in 0..games {
        for deep_is_black in [true, false] {
            let mut s = g.initial_state();
            let mut rng = Rng::new(0xD00D ^ (k * 2 + deep_is_black as u64));
            while !g.is_terminal(&s) {
                let p = match g.turn(&s) {
                    Turn::Player(p) => p,
                    _ => unreachable!(),
                };
                let deep_to_move = (p == 0) == deep_is_black;
                let i = if deep_to_move {
                    a.act(&g, &s, p, &mut rng)
                } else {
                    b.act(&g, &s, p, &mut rng)
                };
                let a = g.legal_actions(&s)[i];
                g.apply(&mut s, a);
            }
            let seat = if deep_is_black { 0 } else { 1 };
            score += (g.returns(&s, seat) + 1.0) / 2.0;
            total += 1;
        }
    }
    (score, total)
}

#[test]
fn deeper_search_does_not_lose_the_ladder() {
    // Depth 4 must score at least even against depth 2 over a seat-swapped set.
    let (score, total) = duel(4, 2, 6);
    let frac = score / total as f64;
    assert!(
        frac >= 0.5,
        "depth-4 scored {score}/{total} = {frac:.3} vs depth-2, expected >= 0.5"
    );
}
