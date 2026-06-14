//! Go knowledge for generic search: [`GoEval`], a cheap area-difference
//! evaluation, and [`GoSpec`], a tactical move-ordering policy.

use game_core::{Eval, SearchSpec};

use crate::{EMPTY, Go, GoAction, GoState, group, neighbors, place};

/// the komi-adjusted area lead, squashed by [`game_core::eval_squash`].
///
/// Area is current Chinese score (stones + exclusively-bordered empty
/// regions), the same quantity [`Go::returns`] thresholds at the end, so the
/// eval and the true outcome agree in sign once territory is settled. Komi is
/// included for the same reason. Dividing by the board size before squashing
/// keeps mid-game leads of a few points informative instead of saturating.
pub struct GoEval;

impl Eval<Go> for GoEval {
    fn eval(&self, game: &Go, state: &GoState, player: usize) -> f64 {
        let black_lead = game.score_margin(state);
        let lead = if player == 0 { black_lead } else { -black_lead };
        game_core::eval_squash(lead, game.size() as f64)
    }
}

const CAPTURE: i64 = 1_000;
const ESCAPE: i64 = 800;
const THREAT: i64 = 600;
const SELF_ATARI: i64 = -500;
const PASS: i64 = -1;
const EYE_FILL: i64 = -10_000;

/// Move ordering for Go: captures first, then escapes from atari, then moves
/// putting an opponent group in atari; quiet moves are neutral, self-ataris
/// rank below passing, and filling one's own true eye ranks last of all (see
/// [`is_eyelike`] for the eye heuristic and its limits).
pub struct GoSpec;

impl SearchSpec<Go> for GoSpec {
    fn order_hint(&self, game: &Go, s: &GoState, action: GoAction) -> i64 {
        let GoAction::Place(p) = action else {
            return PASS;
        };
        let p = p as usize;
        let size = game.size();
        let color = s.to_move as u8;
        if is_eyelike(&s.cells, size, p, color) {
            return EYE_FILL;
        }
        let rescues_atari_group =
            neighbors(size, p).any(|n| s.cells[n] == color && liberties(&s.cells, size, n) == 1);
        let mut cells = s.cells.clone();
        let Some(captured) = place(&mut cells, size, p, color) else {
            return EYE_FILL;
        };
        let own_libs = liberties(&cells, size, p);
        let mut hint = 0;
        if captured > 0 {
            hint += CAPTURE + 10 * captured as i64;
        }
        if rescues_atari_group && own_libs >= 2 {
            hint += ESCAPE;
        }
        if own_libs >= 2 {
            let threatened = stones_put_in_atari(&cells, size, p, color);
            if threatened > 0 {
                hint += THREAT + 5 * threatened as i64;
            }
        }
        if own_libs == 1 && captured == 0 {
            hint += SELF_ATARI;
        }
        hint
    }
}

/// Opponent stones adjacent to the stone just placed at `p` whose groups now
/// have exactly one liberty.
fn stones_put_in_atari(cells: &[u8], size: usize, p: usize, color: u8) -> usize {
    let mut seen = vec![false; cells.len()];
    let mut threatened = 0;
    for n in neighbors(size, p) {
        if cells[n] != (color ^ 1) || seen[n] {
            continue;
        }
        let (stones, _) = group(cells, size, n);
        for &q in &stones {
            seen[q] = true;
        }
        if group_liberties(cells, size, &stones) == 1 {
            threatened += stones.len();
        }
    }
    threatened
}

fn liberties(cells: &[u8], size: usize, start: usize) -> usize {
    let (stones, _) = group(cells, size, start);
    group_liberties(cells, size, &stones)
}

fn group_liberties(cells: &[u8], size: usize, stones: &[usize]) -> usize {
    let mut seen = vec![false; cells.len()];
    let mut libs = 0;
    for &s in stones {
        for n in neighbors(size, s) {
            if cells[n] == EMPTY && !seen[n] {
                seen[n] = true;
                libs += 1;
            }
        }
    }
    libs
}

/// Practical single-point eye test (the playout-policy classic): empty `p` is
/// eye-like for `color` if every orthogonal neighbor is a `color` stone and
/// the diagonals pass the diagonal rule — in the interior the opponent holds
/// at most one of the four diagonals; on the edge or corner (any diagonal
/// off-board) the opponent holds none.
///
/// Limits: this is a local pattern, not life-and-death analysis. Empty
/// diagonals count as safe even if the opponent can later occupy them (some
/// false eyes pass), the surrounding wall is assumed alive (an eye of a dead
/// group still counts), and only single-point eyes are seen — big eyespaces,
/// seki, and positions where filling an eye is correct (e.g. to win an inside
/// capturing race) are all beyond it. Good enough to stop a policy from
/// killing its own groups; not an oracle.
pub(crate) fn is_eyelike(cells: &[u8], size: usize, p: usize, color: u8) -> bool {
    if neighbors(size, p).any(|n| cells[n] != color) {
        return false;
    }
    let (r, c) = ((p / size) as i64, (p % size) as i64);
    let mut opp = 0;
    let mut off_board = 0;
    for (dr, dc) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {
        let (rr, cc) = (r + dr, c + dc);
        if rr < 0 || cc < 0 || rr >= size as i64 || cc >= size as i64 {
            off_board += 1;
        } else if cells[(rr * size as i64 + cc) as usize] == (color ^ 1) {
            opp += 1;
        }
    }
    if off_board > 0 { opp == 0 } else { opp < 2 }
}

/// The empty liberty points of the group given by `stones`.
fn liberty_points(cells: &[u8], size: usize, stones: &[usize]) -> Vec<usize> {
    let mut seen = vec![false; cells.len()];
    let mut libs = Vec::new();
    for &s in stones {
        for n in neighbors(size, s) {
            if cells[n] == EMPTY && !seen[n] {
                seen[n] = true;
                libs.push(n);
            }
        }
    }
    libs
}

/// Whether the group of `cells[start]` is captured in a ladder when the
/// opponent hunts it (opponent to move first). A static tactical feature for
/// the net: a group that looks alive (two liberties) but is dead to the
/// diagonal capture-chase that nets and shallow search read poorly. Total work
/// is capped by a node budget (`~4·size`) so it can never blow up; an
/// unresolved chase is reported as *not* laddered (a safe false negative).
/// Meant for two-liberty groups (one-liberty groups are already the atari
/// planes).
pub(crate) fn laddered(cells: &[u8], size: usize, start: usize) -> bool {
    let color = cells[start];
    if color == EMPTY {
        return false;
    }
    let (stones, _) = group(cells, size, start);
    let mut budget: i32 = 4 * size as i32;
    ladder_hunter(cells, size, &stones, color, &mut budget)
}

/// Hunter (`color ^ 1`) to move — ataris the prey. True if the prey is caught.
fn ladder_hunter(cells: &[u8], size: usize, stones: &[usize], color: u8, budget: &mut i32) -> bool {
    if *budget <= 0 {
        return false;
    }
    let libs = liberty_points(cells, size, stones);
    match libs.len() {
        0 | 1 => return true, // gone, or in atari (hunter just captures)
        2 => {}
        _ => return false, // three+ liberties: not ladderable
    }
    *budget -= 1;
    let hunter = color ^ 1;
    for &lib in &libs {
        let mut b = cells.to_vec();
        if place(&mut b[..], size, lib, hunter).is_none() {
            continue; // illegal atari (suicide)
        }
        match stones.iter().find(|&&s| b[s] == color) {
            None => return true, // prey captured by the atari itself
            Some(&ps) => {
                let (g, _) = group(&b[..], size, ps);
                if ladder_prey(&b[..], size, &g, color, budget) {
                    return true;
                }
            }
        }
    }
    false
}

/// Prey (`color`) to move — must extend out of atari. True if caught anyway.
fn ladder_prey(cells: &[u8], size: usize, stones: &[usize], color: u8, budget: &mut i32) -> bool {
    if *budget <= 0 {
        return false;
    }
    let libs = liberty_points(cells, size, stones);
    match libs.len() {
        0 => return true,
        1 => {}
        _ => return false, // two+ liberties on its own move: escaped
    }
    *budget -= 1;
    let lib = libs[0];
    let mut b = cells.to_vec();
    match place(&mut b[..], size, lib, color) {
        None => true, // extension is suicide → caught
        Some(_) => {
            let (g, _) = group(&b[..], size, lib);
            ladder_hunter(&b[..], size, &g, color, budget)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_core::Game;

    fn cells(g: &Go, rows: &[&str]) -> Vec<u8> {
        g.parse_state(rows, 0).cells
    }

    #[test]
    fn working_ladder_is_caught() {
        let g = Go::new(7);
        // A White stone with two liberties, boxed by Black toward a corner.
        let c = cells(
            &g,
            &[
                ". X . . . . .",
                ". O X . . . .",
                ". . . . . . .",
                ". . . . . . .",
                ". . . . . . .",
                ". . . . . . .",
                ". . . . . . .",
            ],
        );
        let white = c.iter().position(|&x| x == 1).unwrap();
        let (stones, _) = group(&c, 7, white);
        assert_eq!(
            group_liberties(&c, 7, &stones),
            2,
            "setup has two liberties"
        );
        assert!(
            laddered(&c, 7, white),
            "the running stone dies in the corner"
        );
    }

    #[test]
    fn open_stone_is_not_laddered() {
        let g = Go::new(7);
        let c = cells(
            &g,
            &[
                ". . . . . . .",
                ". . . . . . .",
                ". . . . . . .",
                ". . . O . . .",
                ". . . . . . .",
                ". . . . . . .",
                ". . . . . . .",
            ],
        );
        let white = c.iter().position(|&x| x == 1).unwrap();
        assert!(
            !laddered(&c, 7, white),
            "a four-liberty stone is not ladderable"
        );
    }

    #[test]
    fn center_true_eye_with_empty_diagonals() {
        let g = Go::new(5);
        let cells = cells(
            &g,
            &[
                ". . . . .",
                ". . X . .",
                ". X . X .",
                ". . X . .",
                ". . . . .",
            ],
        );
        assert!(is_eyelike(&cells, 5, g.point("c3").unwrap() as usize, 0));
    }

    #[test]
    fn center_eye_with_two_opponent_diagonals_is_false() {
        let g = Go::new(5);
        let cells = cells(
            &g,
            &[
                ". . . . .",
                ". O X O .",
                ". X . X .",
                ". . X . .",
                ". . . . .",
            ],
        );
        assert!(!is_eyelike(&cells, 5, g.point("c3").unwrap() as usize, 0));
    }

    #[test]
    fn center_eye_tolerates_one_opponent_diagonal() {
        let g = Go::new(5);
        let cells = cells(
            &g,
            &[
                ". . . . .",
                ". O X . .",
                ". X . X .",
                ". . X . .",
                ". . . . .",
            ],
        );
        assert!(is_eyelike(&cells, 5, g.point("c3").unwrap() as usize, 0));
    }

    #[test]
    fn corner_eye_requires_clean_diagonal() {
        let g = Go::new(5);
        let clean = cells(
            &g,
            &[
                ". . . . .",
                ". . . . .",
                ". . . . .",
                "X . . . .",
                ". X . . .",
            ],
        );
        let a1 = g.point("a1").unwrap() as usize;
        assert!(is_eyelike(&clean, 5, a1, 0));
        let tainted = cells(
            &g,
            &[
                ". . . . .",
                ". . . . .",
                ". . . . .",
                "X O . . .",
                ". X . . .",
            ],
        );
        assert!(!is_eyelike(&tainted, 5, a1, 0));
    }

    #[test]
    fn productive_move_guard() {
        let g = Go::new(5);
        // Empty board: plenty of productive moves.
        assert!(g.has_productive_move(&g.initial_state()));
        // Black surrounds a single eye at c3 and fills the rest of a 5×5 so
        // its only legal placement is its own eye → no productive move, pass
        // is the right call.
        let s = g.parse_state(
            &[
                "X X X X X",
                "X X X X X",
                "X X . X X",
                "X X X X X",
                "X X X X X",
            ],
            0,
        );
        assert!(!g.has_productive_move(&s), "only its own eye remains");
    }

    #[test]
    fn point_with_non_friendly_neighbor_is_not_an_eye() {
        let g = Go::new(5);
        let cells = cells(
            &g,
            &[
                ". . . . .",
                ". . X . .",
                ". X . O .",
                ". . X . .",
                ". . . . .",
            ],
        );
        assert!(!is_eyelike(&cells, 5, g.point("c3").unwrap() as usize, 0));
    }
}
