//! Go knowledge for generic search: [`GoEval`], a cheap area-difference
//! evaluation, and [`GoSpec`], a tactical move-ordering policy.

use game_core::{Eval, SearchSpec};

use crate::{EMPTY, Go, GoAction, GoState, KOMI, group, neighbors, place};

/// `tanh((player's area − opponent's area, komi-adjusted) / size)`.
///
/// Area is current Chinese score (stones + exclusively-bordered empty
/// regions), the same quantity [`Go::returns`] thresholds at the end, so the
/// eval and the true outcome agree in sign once territory is settled. Komi is
/// included for the same reason. Dividing by the board size before `tanh`
/// keeps mid-game leads of a few points informative instead of saturating.
pub struct GoEval;

impl Eval<Go> for GoEval {
    fn eval(&self, game: &Go, state: &GoState, player: usize) -> f64 {
        let (black, white) = game.area_scores(state);
        let black_lead = black as f64 - white as f64 - KOMI;
        let lead = if player == 0 { black_lead } else { -black_lead };
        (lead / game.size() as f64).tanh()
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
fn is_eyelike(cells: &[u8], size: usize, p: usize, color: u8) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn cells(g: &Go, rows: &[&str]) -> Vec<u8> {
        g.parse_state(rows, 0).cells
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
