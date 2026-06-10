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
