//! Terminal/serving surface for Go.

use game_core::{Game, GameUi};

use crate::{Go, GoAction, GoState, KOMI, col_letter};

fn player_name(p: usize) -> &'static str {
    if p == 0 { "Black (X)" } else { "White (O)" }
}

impl GameUi for Go {
    fn id(&self) -> &'static str {
        "go"
    }

    fn render(&self, state: &GoState, player: usize) -> String {
        let mut out = String::new();
        for r in (0..self.size()).rev() {
            out.push_str(&format!("{:>2}", r + 1));
            for c in 0..self.size() {
                out.push_str(match state.stone(r * self.size() + c) {
                    Some(0) => "  X",
                    Some(_) => "  O",
                    None => "  .",
                });
            }
            out.push('\n');
        }
        out.push_str("  ");
        for c in 0..self.size() {
            out.push_str("  ");
            out.push(col_letter(c));
        }
        out.push('\n');
        let caps = state.captures();
        out.push_str(&format!(
            "Captures: Black {}, White {} | Komi: {KOMI}\n",
            caps[0], caps[1]
        ));
        out.push_str(&format!("You are {}.", player_name(player)));
        if !self.is_terminal(state) {
            out.push_str(&format!(" {} to move.", player_name(state.to_move)));
        }
        out
    }

    fn action_label(&self, _state: &GoState, action: GoAction) -> String {
        match action {
            GoAction::Pass => "pass".into(),
            GoAction::Place(p) => {
                let p = p as usize;
                format!("{}{}", col_letter(p % self.size()), p / self.size() + 1)
            }
        }
    }

    fn parse_action(&self, state: &GoState, input: &str) -> Option<GoAction> {
        let text = input.trim().to_ascii_lowercase();
        let action = if text == "pass" {
            GoAction::Pass
        } else {
            GoAction::Place(self.point(&text)?)
        };
        self.legal_actions(state).into_iter().find(|&a| a == action)
    }

    fn describe_transition(
        &self,
        before: &GoState,
        action: GoAction,
        after: &GoState,
        _viewer: usize,
    ) -> Option<String> {
        match action {
            GoAction::Pass => Some(format!("{} passes.", player_name(before.to_move))),
            GoAction::Place(_) => {
                let n = after.captures()[before.to_move] - before.captures()[before.to_move];
                (n > 0).then(|| format!("captures {n} stone{}", if n == 1 { "" } else { "s" }))
            }
        }
    }

    /// Web view schema: `{"size":N,"cells":"<N*N chars b/w/.>","turn":0|1,
    /// "captures":[b,w],"lastMove":null,"komi":7.5}`. `cells` is indexed
    /// `row * size + col` with row 0 = board row 1; the last move is not part
    /// of the state, so clients track it from [`GameUi::transition_data`].
    fn view_data(&self, state: &GoState, _viewer: usize) -> Option<String> {
        let cells: String = (0..self.size() * self.size())
            .map(|p| match state.stone(p) {
                Some(0) => 'b',
                Some(_) => 'w',
                None => '.',
            })
            .collect();
        let caps = state.captures();
        Some(format!(
            r#"{{"size":{},"cells":"{cells}","turn":{},"captures":[{},{}],"lastMove":null,"komi":{KOMI}}}"#,
            self.size(),
            state.to_move,
            caps[0],
            caps[1],
        ))
    }

    /// Web transition schema: `{"move":"c3"|"pass","seat":0|1}` plus, for
    /// placements, `"point"` (board index) and `"captured"` (board indices of
    /// removed stones).
    fn transition_data(
        &self,
        before: &GoState,
        action: GoAction,
        after: &GoState,
        _viewer: usize,
    ) -> Option<String> {
        let seat = before.to_move;
        match action {
            GoAction::Pass => Some(format!(r#"{{"move":"pass","seat":{seat}}}"#)),
            GoAction::Place(p) => {
                let coord = self.action_label(before, action);
                let captured = (0..self.size() * self.size())
                    .filter(|&q| before.stone(q) == Some(seat ^ 1) && after.stone(q).is_none())
                    .map(|q| q.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                Some(format!(
                    r#"{{"move":"{coord}","seat":{seat},"point":{p},"captured":[{captured}]}}"#
                ))
            }
        }
    }

    fn result_text(&self, state: &GoState, viewer: usize) -> String {
        let (black, white) = self.area_scores(state);
        let verdict = if self.returns(state, viewer) > 0.0 {
            "You win!"
        } else {
            "You lose."
        };
        format!(
            "Black {black} vs White {} ({white} + {KOMI} komi). {verdict}",
            white as f64 + KOMI
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A black play on b3 captures the white stone at b2; the web JSON must
    /// carry the placement point and the captured indices.
    #[test]
    fn view_and_transition_json() {
        let g = Go::new(3);
        let mut s = g.parse_state(&["...", "XOX", ".X."], 0);
        let before = s.clone();
        g.apply(&mut s, GoAction::Place(7));
        assert_eq!(
            g.view_data(&s, 0).unwrap(),
            r#"{"size":3,"cells":".b.b.b.b.","turn":1,"captures":[1,0],"lastMove":null,"komi":7.5}"#
        );
        assert_eq!(
            g.transition_data(&before, GoAction::Place(7), &s, 0)
                .unwrap(),
            r#"{"move":"b3","seat":0,"point":7,"captured":[4]}"#
        );
        assert_eq!(
            g.transition_data(&s, GoAction::Pass, &s, 0).unwrap(),
            r#"{"move":"pass","seat":1}"#
        );
    }
}
