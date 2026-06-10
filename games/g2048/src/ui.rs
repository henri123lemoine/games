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

    /// JSON view for the web frontend (`web/app/src/frontends/2048`):
    ///
    /// ```json
    /// {"cells":[0,2,0,...,16],"score":20,"over":false}
    /// ```
    ///
    /// `cells` is the 16 tile values row-major from the top-left (`0` =
    /// empty, else the tile value, e.g. `2048`); `score` is the cumulative
    /// score; `over` is whether the game has ended.
    fn view_data(&self, state: &G2048State, _viewer: usize) -> Option<String> {
        let cells = (0..SIZE)
            .flat_map(|r| (0..SIZE).map(move |c| (r, c)))
            .map(|(r, c)| state.tile(r, c).to_string())
            .collect::<Vec<_>>()
            .join(",");
        Some(format!(
            "{{\"cells\":[{cells}],\"score\":{},\"over\":{}}}",
            state.score(),
            self.is_terminal(state)
        ))
    }

    /// JSON transition for the web frontend. Shifts emit
    ///
    /// ```json
    /// {"dir":"up","gained":8}
    /// ```
    ///
    /// (`dir` ∈ `up|down|left|right`, `gained` = points the move scored).
    /// Spawn actions emit `{"spawn":{"cell":i,"value":2|4}}` (`cell`
    /// row-major), but in match flow spawns are chance moves resolved without
    /// events — the post-shift view precedes the spawn, so frontends detect
    /// the new tile by diffing consecutive views.
    fn transition_data(
        &self,
        before: &G2048State,
        action: G2048Action,
        after: &G2048State,
        _viewer: usize,
    ) -> Option<String> {
        match action {
            G2048Action::Shift(dir) => Some(format!(
                "{{\"dir\":\"{}\",\"gained\":{}}}",
                dir_name(dir),
                after.score() - before.score()
            )),
            G2048Action::Spawn { cell, four } => Some(format!(
                "{{\"spawn\":{{\"cell\":{cell},\"value\":{}}}}}",
                if four { 4 } else { 2 }
            )),
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
    fn view_data_schema() {
        let game = G2048;
        let s = G2048State::from_tiles([[2, 0, 0, 0], [0, 16, 0, 0], [0; 4], [0; 4]], 20);
        assert_eq!(
            game.view_data(&s, 0).unwrap(),
            "{\"cells\":[2,0,0,0,0,16,0,0,0,0,0,0,0,0,0,0],\"score\":20,\"over\":false}"
        );

        let done = G2048State::from_tiles(
            [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]],
            100,
        );
        assert!(
            game.view_data(&done, 0)
                .unwrap()
                .ends_with("\"over\":true}")
        );
    }

    #[test]
    fn transition_data_schema() {
        let game = G2048;
        let mut s = G2048State::from_tiles([[2, 2, 0, 0], [0; 4], [0; 4], [0; 4]], 0);
        let before = s.clone();
        let action = G2048Action::Shift(Dir::Left);
        game.apply(&mut s, action);
        assert_eq!(
            game.transition_data(&before, action, &s, 0).unwrap(),
            "{\"dir\":\"left\",\"gained\":4}"
        );
        assert_eq!(
            game.transition_data(
                &before,
                G2048Action::Spawn {
                    cell: 6,
                    four: true
                },
                &s,
                0
            )
            .unwrap(),
            "{\"spawn\":{\"cell\":6,\"value\":4}}"
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
