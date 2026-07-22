//! Pente knowledge for generic search: [`PenteEval`], a threat-aware static
//! evaluation, and [`PenteSpec`], a tactical move-ordering and quiescence
//! policy. Together they turn the lab's generic alpha-beta into a competent
//! Pente engine with no training.

use game_core::{Eval, SearchSpec};

use crate::{
    DIRECTIONS, EMPTY, LINE_TO_WIN, PAIRS_TO_WIN, Pente, PenteAction, PenteState, completes_line,
    for_each_captured_pair,
};

/// Per-pair captured: a pair is a third of the way to half the win-by-line
/// horizon in practice, but its real weight is that five pairs simply win, so
/// each pair carries a large, escalating value.
const PAIR_VALUE: i32 = 180;
/// A stone pair of mine the opponent can capture on the move — they are one
/// tempo from `+PAIR_VALUE` and I am one from `-PAIR_VALUE`, so a live
/// vulnerability is nearly as costly as the capture itself.
const VULNERABLE_PAIR: i32 = 140;
/// A pair of the opponent's I can capture on the move.
const CAPTURE_THREAT: i32 = 90;
const CENTER_BONUS: i32 = 6;

/// Threat-aware static evaluation from `player`'s view, on the unbounded
/// integer scale [`game_core::eval_squash`] maps into `(-1, 1)`.
///
/// The score is `mine − theirs` over four contributions: **line potential**
/// (open/closed runs of two, three, and four, weighted so an open four — an
/// unstoppable five next move — dominates), **capture progress** (pairs already
/// taken, escalating toward the five-pair win), **capture tactics** (pairs I
/// threaten to take minus pairs of mine the opponent can take), and a small
/// **center** term. Pente is won by lines *or* captures, and lost by leaving
/// pairs flanked, so the eval has to see all three at once — a pure
/// line-counting eval walks its own pairs into capture and a pure capture eval
/// ignores five-in-a-row.
pub struct PenteEval;

impl Eval<Pente> for PenteEval {
    fn eval(&self, game: &Pente, state: &PenteState, player: usize) -> f64 {
        let me = player as u8;
        let opp = me ^ 1;
        let score = side_score(game, state, me) - side_score(game, state, opp);
        game_core::eval_squash(f64::from(score), 220.0)
    }
}

/// All of `color`'s contributions to the raw evaluation: line potential plus
/// capture progress plus capture tactics plus center control.
fn side_score(game: &Pente, state: &PenteState, color: u8) -> i32 {
    let size = game.size();
    let cells = &state.cells[..size * size];
    let mut score = line_potential(cells, size, color);

    // Capture progress escalates: the fifth pair wins, so being close is worth
    // far more than the linear sum of pairs would suggest.
    let pairs = state.pairs[color as usize] as i32;
    score += pairs * PAIR_VALUE + pairs * pairs * 12;
    if pairs + 1 >= PAIRS_TO_WIN as i32 {
        score += PAIR_VALUE; // one pair from victory: extra urgency
    }

    // Capture tactics: count the empty intersections from which `color` could,
    // on its move, flank an opponent pair. Each is a standing threat.
    let opp = color ^ 1;
    let mut capture_threats = 0;
    let mut vulnerable = 0;
    for p in 0..cells.len() {
        if cells[p] != EMPTY {
            continue;
        }
        // A move by `color` here that captures an opponent pair.
        if would_capture(cells, size, p, color, opp) {
            capture_threats += 1;
        }
        // A move by `opp` here that captures a pair of `color`'s — a
        // vulnerability `color` is carrying.
        if would_capture(cells, size, p, opp, color) {
            vulnerable += 1;
        }
    }
    score += capture_threats * CAPTURE_THREAT - vulnerable * VULNERABLE_PAIR;

    // Center control: a small pull toward the middle, where lines have room.
    let mid = (size / 2) as i32;
    for (p, &cell) in cells.iter().enumerate() {
        if cell == color {
            let (r, c) = ((p / size) as i32, (p % size) as i32);
            let dist = (r - mid).abs().max((c - mid).abs());
            score += (CENTER_BONUS - dist).max(0);
        }
    }
    score
}

/// Whether `mover` placing at empty `p` would flank an `opp` pair in some
/// direction (the `[mover][opp][opp][mover]` custodial pattern). Pure lookahead
/// on `cells` — does not mutate.
fn would_capture(cells: &[u8], size: usize, p: usize, mover: u8, opp: u8) -> bool {
    let mut hit = false;
    for_each_captured_pair(cells, size, p, mover, opp, |_, _| hit = true);
    hit
}

/// The cell `steps` along `(dr, dc)` from `(row, col)`, or `None` off-board.
fn at(
    cells: &[u8],
    size: usize,
    row: usize,
    col: usize,
    dr: i32,
    dc: i32,
    steps: i32,
) -> Option<u8> {
    let r = row as i32 + dr * steps;
    let c = col as i32 + dc * steps;
    if r >= 0 && c >= 0 && (r as usize) < size && (c as usize) < size {
        Some(cells[r as usize * size + c as usize])
    } else {
        None
    }
}

/// Open-/closed-run line potential for `color`: each maximal contiguous run of
/// `color` stones is scored by its length and how many of its two ends are
/// open (empty, on-board). Open runs are the threats — an open three forces a
/// response, an open four wins next move. Runs are counted once by scanning
/// only from a run's first stone in each orientation.
fn line_potential(cells: &[u8], size: usize, color: u8) -> i32 {
    let mut total = 0;
    for p in 0..cells.len() {
        if cells[p] != color {
            continue;
        }
        let (row, col) = (p / size, p % size);
        for &(dr, dc) in &DIRECTIONS {
            // Only the run's first stone scores it: skip if the previous cell
            // along this orientation is also `color`.
            if at(cells, size, row, col, -dr, -dc, 1) == Some(color) {
                continue;
            }
            let mut len = 1;
            while at(cells, size, row, col, dr, dc, len) == Some(color) {
                len += 1;
            }
            let before = at(cells, size, row, col, -dr, -dc, 1);
            let after = at(cells, size, row, col, dr, dc, len);
            let open_ends = u8::from(before == Some(EMPTY)) + u8::from(after == Some(EMPTY));
            total += run_score(len as usize, open_ends);
        }
    }
    total
}

/// Value of a run of `len` like-colored stones with `open_ends` empty ends. A
/// run with no open end and shorter than five is dead (cannot extend to five)
/// and scores nothing. Five-plus is a win and saturates. The jump from three to
/// four is large because an open four is unstoppable.
fn run_score(len: usize, open_ends: u8) -> i32 {
    if len >= LINE_TO_WIN {
        return 100_000;
    }
    if open_ends == 0 {
        return 0; // a capped sub-five run is inert
    }
    let base = match len {
        4 => 4_000,
        3 => 300,
        2 => 30,
        _ => 3,
    };
    if open_ends == 2 { base * 4 } else { base }
}

// Move-ordering hint magnitudes (higher is searched first).
const WIN_NOW: i64 = 1_000_000;
const CAPTURE: i64 = 5_000;
const BLOCK_FIVE: i64 = 600_000;
const NEAR_STONE: i64 = 100;
const AVOID_SELF_CAPTURE: i64 = -3_000;

/// Move ordering and quiescence for Pente. Ordering puts immediate wins first
/// (completing a five, or the fifth pair), then opponent-five blocks, then
/// captures and moves that build/answer threats, with quiet moves ranked by
/// proximity to existing stones (so the search spends its budget on the live
/// part of a sparse board). Captures and win-completing placements are the
/// "noisy" moves quiescence extends over, so the engine never stops its search
/// one ply before a forced capture or five flips the evaluation.
pub struct PenteSpec;

impl SearchSpec<Pente> for PenteSpec {
    fn order_hint(&self, game: &Pente, s: &PenteState, action: PenteAction) -> i64 {
        let size = game.size();
        let p = action.0 as usize;
        let color = s.to_move as u8;
        let opp = color ^ 1;
        let (row, col) = (p / size, p % size);

        let captures = game.capture_pairs_at(s, p, color);
        if s.pairs[s.to_move] + captures >= PAIRS_TO_WIN {
            return WIN_NOW; // the fifth pair
        }
        if completes_line(&s.cells, size, row, col, color) {
            return WIN_NOW; // five in a row
        }

        let mut hint = 0i64;
        if captures > 0 {
            hint += CAPTURE + 200 * captures as i64;
        }
        // Blocking: would the opponent win by playing here instead? Then taking
        // the point denies a five (captures already cover their fifth pair).
        if completes_line(&s.cells, size, row, col, opp) {
            hint += BLOCK_FIVE;
        }
        // Building/answering line threats near the action's neighborhood.
        hint += NEAR_STONE * neighbor_stones(&s.cells, size, row, col) as i64;
        // Penalize voluntarily forming a capturable pair (walking into
        // [opp][me][me][.] that the opponent flanks next).
        if forms_vulnerable_pair(&s.cells, size, p, color, opp) {
            hint += AVOID_SELF_CAPTURE;
        }
        hint
    }

    fn is_noisy(&self, game: &Pente, s: &PenteState, action: PenteAction) -> bool {
        let size = game.size();
        let p = action.0 as usize;
        let color = s.to_move as u8;
        let (row, col) = (p / size, p % size);
        // Capturing moves and win/loss-deciding placements (a five, or a block
        // of the opponent's five) change the tactical picture and must be read
        // past the horizon.
        game.capture_pairs_at(s, p, color) > 0
            || completes_line(&s.cells, size, row, col, color)
            || completes_line(&s.cells, size, row, col, color ^ 1)
    }
}

/// Occupied intersections within the 8-neighborhood (Chebyshev distance ≤ 2) of
/// `(row, col)` — a cheap "is this near the action" signal for ordering quiet
/// moves on a sparse board.
fn neighbor_stones(cells: &[u8], size: usize, row: usize, col: usize) -> usize {
    let mut n = 0;
    for dr in -2i32..=2 {
        for dc in -2i32..=2 {
            if dr == 0 && dc == 0 {
                continue;
            }
            if let Some(c) = at(cells, size, row, col, dr, dc, 1)
                && c != EMPTY
            {
                n += 1;
            }
        }
    }
    n
}

/// Whether placing `color` at `p` creates a pair `[opp][color][color][empty]`
/// (or the mirror) that `opp` can capture on its next move — the self-capture
/// trap the ordering steers away from.
fn forms_vulnerable_pair(cells: &[u8], size: usize, p: usize, color: u8, opp: u8) -> bool {
    let (row, col) = (p / size, p % size);
    DIRECTIONS.iter().any(|&(dr, dc)| {
        [1, -1].iter().any(|&sign| {
            let (dr, dc) = (dr * sign, dc * sign);
            // Pattern outward from p: friend at +1, then flanking empty at +2,
            // with an opponent at -1 closing the other side.
            at(cells, size, row, col, dr, dc, 1) == Some(color)
                && at(cells, size, row, col, dr, dc, 2) == Some(EMPTY)
                && at(cells, size, row, col, -dr, -dc, 1) == Some(opp)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_core::{Eval, SearchSpec};

    #[test]
    fn open_four_dominates_open_three() {
        let g = Pente::new(9);
        let three = g.parse_state(
            &[
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . X X X . . . .",
                ". . . . . . . . .",
            ],
            1,
            [0, 0],
        );
        let four = g.parse_state(
            &[
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . X X X X . . .",
                ". . . . . . . . .",
            ],
            1,
            [0, 0],
        );
        // From Black's view, the open four is worth far more than the open three.
        assert!(PenteEval.eval(&g, &four, 0) > PenteEval.eval(&g, &three, 0));
    }

    #[test]
    fn vulnerability_is_a_penalty() {
        let g = Pente::new(9);
        // Black pair flanked on one side by White with an open far end:
        // O X X .  — White can play the far end and capture.
        let exposed = g.parse_state(
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
            0,
            [0, 0],
        );
        // The same two black stones with no white flank: safe.
        let safe = g.parse_state(
            &[
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . X X . . . . .",
                ". . . . . . . . .",
            ],
            0,
            [0, 0],
        );
        assert!(
            PenteEval.eval(&g, &exposed, 0) < PenteEval.eval(&g, &safe, 0),
            "a flanked, capturable pair must score worse than a safe one"
        );
    }

    #[test]
    fn winning_capture_is_ordered_first() {
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
        s.moves = 10;
        let flank = PenteAction(g.point("e2").unwrap());
        assert_eq!(
            PenteSpec.order_hint(&g, &s, flank),
            WIN_NOW,
            "the fifth-pair capture is a win-now move"
        );
        assert!(PenteSpec.is_noisy(&g, &s, flank), "and it is noisy");
    }

    #[test]
    fn blocking_opponent_five_is_ordered_high() {
        let g = Pente::new(9);
        // White (to move) faces black's open four X X X X .; the block at the
        // open end must outrank a quiet move far away.
        let mut s = g.parse_state(
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
        s.moves = 8;
        let block = PenteAction(g.point("f2").unwrap());
        let quiet = PenteAction(g.point("j9").unwrap());
        assert!(
            PenteSpec.order_hint(&g, &s, block) > PenteSpec.order_hint(&g, &s, quiet),
            "blocking the five must beat a distant quiet move"
        );
    }
}
