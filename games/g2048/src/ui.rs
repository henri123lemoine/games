//! Terminal/serving surface for 2048.

use game_core::{Game, GameUi};

use crate::{Dir, G2048, G2048Action, G2048State, SIZE};

fn dir_name(dir: Dir) -> &'static str {
    match dir {
        Dir::Up => "up",
        Dir::Down => "down",
        Dir::Left => "left",
        Dir::Right => "right",
    }
}

impl GameUi for G2048 {
    fn id(&self) -> &'static str {
        "2048"
    }

    fn render(&self, state: &G2048State, _player: usize) -> String {
        let mut out = String::new();
        for r in 0..SIZE {
            for c in 0..SIZE {
                match state.tile(r, c) {
                    0 => out.push_str(&format!("{:>7}", ".")),
                    v => out.push_str(&format!("{v:>7}")),
                }
            }
            out.push('\n');
        }
        out.push_str(&format!(
            "score {}   best tile {}",
            state.score(),
            state.max_tile()
        ));
        out
    }

    fn action_label(&self, _state: &G2048State, action: G2048Action) -> String {
        match action {
            G2048Action::Shift(dir) => dir_name(dir).into(),
            G2048Action::Spawn { cell, four } => format!(
                "spawn {} at r{}c{}",
                if four { 4 } else { 2 },
                cell as usize / SIZE + 1,
                cell as usize % SIZE + 1
            ),
        }
    }

    /// Accepts `w`/`a`/`s`/`d` and the words `up`/`down`/`left`/`right`
    /// (case-insensitive), only when that shift is legal.
    fn parse_action(&self, state: &G2048State, input: &str) -> Option<G2048Action> {
        let dir = match input.trim().to_ascii_lowercase().as_str() {
            "w" | "up" => Dir::Up,
            "s" | "down" => Dir::Down,
            "a" | "left" => Dir::Left,
            "d" | "right" => Dir::Right,
            _ => return None,
        };
        let action = G2048Action::Shift(dir);
        self.legal_actions(state)
            .contains(&action)
            .then_some(action)
    }

    fn result_text(&self, state: &G2048State, _viewer: usize) -> String {
        debug_assert!(self.is_terminal(state));
        format!(
            "Game over — score {}, best tile {}.",
            state.score(),
            state.max_tile()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_accepts_wasd_and_words() {
        let game = G2048;
        let s = G2048State::from_tiles([[0, 2, 0, 0], [0, 0, 0, 2], [0; 4], [0; 4]], 0);
        for (input, dir) in [
            ("w", Dir::Up),
            ("UP", Dir::Up),
            ("s", Dir::Down),
            ("down", Dir::Down),
            ("a", Dir::Left),
            ("Left", Dir::Left),
            ("d", Dir::Right),
            (" right ", Dir::Right),
        ] {
            assert_eq!(
                game.parse_action(&s, input),
                Some(G2048Action::Shift(dir)),
                "input {input:?}"
            );
        }
        assert_eq!(game.parse_action(&s, "x"), None);
        assert_eq!(game.parse_action(&s, "upp"), None);
    }

    #[test]
    fn parse_rejects_noop_shift() {
        let game = G2048;
        let s = G2048State::from_tiles([[2, 0, 0, 0], [4, 0, 0, 0], [0; 4], [0; 4]], 0);
        assert_eq!(game.parse_action(&s, "a"), None);
        assert_eq!(game.parse_action(&s, "up"), None);
        assert_eq!(
            game.parse_action(&s, "d"),
            Some(G2048Action::Shift(Dir::Right))
        );
    }

    #[test]
    fn render_and_labels() {
        let game = G2048;
        let s = G2048State::from_tiles([[2, 0, 0, 0], [0, 16, 0, 0], [0; 4], [0; 4]], 20);
        let view = game.render(&s, 0);
        assert!(view.contains("2"));
        assert!(view.contains("16"));
        assert!(view.contains("score 20"));
        assert!(view.contains("best tile 16"));
        assert_eq!(game.action_label(&s, G2048Action::Shift(Dir::Left)), "left");
        assert_eq!(
            game.action_label(
                &s,
                G2048Action::Spawn {
                    cell: 6,
                    four: true
                }
            ),
            "spawn 4 at r2c3"
        );
    }
}
