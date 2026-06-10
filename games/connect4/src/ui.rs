//! Terminal/serving surface for Connect-4.

use game_core::{Game, GameUi};

use crate::{COLS, Connect4, Connect4State, ROWS};

fn glyph(player: usize) -> char {
    if player == 0 { 'X' } else { 'O' }
}

/// Index into the `view_data` cells string: row-major, top row first.
const fn cell_index(col: usize, row: usize) -> usize {
    (ROWS - 1 - row) * COLS + col
}

/// One winning four as cells-string indices, when the game has been won.
fn win_line(state: &Connect4State) -> Option<[usize; 4]> {
    let winner = state.winner?;
    const DIRS: [(isize, isize); 4] = [(1, 0), (0, 1), (1, 1), (1, -1)];
    for (dc, dr) in DIRS {
        for col in 0..COLS as isize {
            for row in 0..ROWS as isize {
                let cells = [0isize, 1, 2, 3].map(|i| (col + i * dc, row + i * dr));
                let in_bounds = cells.iter().all(|&(c, r)| {
                    (0..COLS as isize).contains(&c) && (0..ROWS as isize).contains(&r)
                });
                if in_bounds
                    && cells
                        .iter()
                        .all(|&(c, r)| state.stone_at(c as usize, r as usize) == Some(winner))
                {
                    return Some(cells.map(|(c, r)| cell_index(c as usize, r as usize)));
                }
            }
        }
    }
    None
}

impl GameUi for Connect4 {
    fn id(&self) -> &'static str {
        "connect4"
    }

    fn render(&self, state: &Connect4State, player: usize) -> String {
        let mut out = String::new();
        for row in (0..ROWS).rev() {
            for col in 0..COLS {
                out.push(match state.stone_at(col, row) {
                    Some(p) => glyph(p),
                    None => '.',
                });
                out.push(' ');
            }
            out.push('\n');
        }
        out.push_str("1 2 3 4 5 6 7\n");
        out.push_str(&format!(
            "You are {}. {} to move.",
            glyph(player),
            glyph(state.mover())
        ));
        out
    }

    fn action_label(&self, _state: &Connect4State, action: u8) -> String {
        format!("col {}", action + 1)
    }

    /// Accepts `"4"`, `"c4"`, `"col 4"` (1-based column). Note the lab client
    /// resolves bare integers as menu indices before calling this, which still
    /// lands on the right column because actions are listed left to right.
    fn parse_action(&self, state: &Connect4State, input: &str) -> Option<u8> {
        let t = input.trim().to_ascii_lowercase();
        let digits = t
            .strip_prefix("col")
            .or_else(|| t.strip_prefix('c'))
            .unwrap_or(&t)
            .trim();
        let col: u8 = digits.parse().ok()?;
        if !(1..=COLS as u8).contains(&col) {
            return None;
        }
        let action = col - 1;
        self.legal_actions(state)
            .contains(&action)
            .then_some(action)
    }

    /// View JSON — the private contract with `web/app/src/frontends/connect4`:
    ///
    /// ```json
    /// {"cells": "<42 chars, row-major, TOP row first: '.' empty,
    ///            'x' player 0, 'o' player 1>",
    ///  "turn": 0|1,
    ///  "winner": 0|1|null,
    ///  "winLine": [i,i,i,i]|null}   // indices into "cells"
    /// ```
    fn view_data(&self, state: &Connect4State, _viewer: usize) -> Option<String> {
        let mut cells = String::with_capacity(COLS * ROWS);
        for row in (0..ROWS).rev() {
            for col in 0..COLS {
                cells.push(match state.stone_at(col, row) {
                    Some(0) => 'x',
                    Some(_) => 'o',
                    None => '.',
                });
            }
        }
        let winner = state.winner.map_or("null".into(), |w: usize| w.to_string());
        let win_line = win_line(state).map_or("null".into(), |c| {
            format!("[{},{},{},{}]", c[0], c[1], c[2], c[3])
        });
        Some(format!(
            r#"{{"cells":"{cells}","turn":{},"winner":{winner},"winLine":{win_line}}}"#,
            state.mover()
        ))
    }

    /// Transition JSON — where the just-dropped disc landed:
    ///
    /// ```json
    /// {"col": 0-6, "row": 0-5, "player": 0|1}   // row 0 = BOTTOM
    /// ```
    fn transition_data(
        &self,
        before: &Connect4State,
        action: u8,
        _after: &Connect4State,
        _viewer: usize,
    ) -> Option<String> {
        let col = action as usize;
        Some(format!(
            r#"{{"col":{col},"row":{},"player":{}}}"#,
            before.heights[col],
            before.mover()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn play_cols(cols: &[u8]) -> (Connect4, Connect4State) {
        let game = Connect4;
        let mut state = game.initial_state();
        for &c in cols {
            game.apply(&mut state, c);
        }
        (game, state)
    }

    #[test]
    fn parse_accepts_bare_c_and_col_forms() {
        let game = Connect4;
        let s = game.initial_state();
        for input in ["4", "c4", "C4", "col 4", "col4", " col  4 "] {
            assert_eq!(game.parse_action(&s, input), Some(3), "input {input:?}");
        }
        assert_eq!(game.parse_action(&s, "0"), None);
        assert_eq!(game.parse_action(&s, "8"), None);
        assert_eq!(game.parse_action(&s, "e4"), None);
    }

    #[test]
    fn parse_rejects_full_column() {
        let game = Connect4;
        let mut s = game.initial_state();
        for _ in 0..ROWS {
            game.apply(&mut s, 2);
        }
        assert_eq!(game.parse_action(&s, "c3"), None);
        assert_eq!(game.parse_action(&s, "c4"), Some(3));
    }

    #[test]
    fn view_data_tracks_cells_and_turn() {
        let game = Connect4;
        let mut s = game.initial_state();
        assert_eq!(
            game.view_data(&s, 0).unwrap(),
            format!(
                r#"{{"cells":"{}","turn":0,"winner":null,"winLine":null}}"#,
                ".".repeat(42)
            )
        );
        game.apply(&mut s, 3);
        let mut cells = vec![b'.'; 42];
        cells[cell_index(3, 0)] = b'x';
        let expected = format!(
            r#"{{"cells":"{}","turn":1,"winner":null,"winLine":null}}"#,
            String::from_utf8(cells).unwrap()
        );
        assert_eq!(game.view_data(&s, 1).unwrap(), expected);
    }

    #[test]
    fn view_data_reports_the_winning_four() {
        let (game, state) = play_cols(&[0, 1, 0, 1, 0, 1, 0]);
        let v = game.view_data(&state, 0).unwrap();
        assert!(v.contains(r#""winner":0"#), "{v}");
        assert!(v.contains(r#""winLine":[35,28,21,14]"#), "{v}");
    }

    #[test]
    fn transition_data_gives_the_landing_cell() {
        let game = Connect4;
        let mut before = game.initial_state();
        game.apply(&mut before, 3);
        let mut after = before.clone();
        game.apply(&mut after, 3);
        assert_eq!(
            game.transition_data(&before, 3, &after, 0).unwrap(),
            r#"{"col":3,"row":1,"player":1}"#
        );
    }

    #[test]
    fn render_hides_nothing_and_marks_mover() {
        let game = Connect4;
        let mut s = game.initial_state();
        game.apply(&mut s, 3);
        let view = game.render(&s, 1);
        assert!(view.contains('X'));
        assert!(view.contains("You are O. O to move."));
        assert_eq!(game.action_label(&s, 3), "col 4");
    }
}
