//! Terminal/serving surface for Connect-4.

use game_core::{Game, GameUi};

use crate::{COLS, Connect4, Connect4State, ROWS};

fn glyph(player: usize) -> char {
    if player == 0 { 'X' } else { 'O' }
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
