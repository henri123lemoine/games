//! Terminal/serving surface for Twenty-One.

use game_core::GameUi;

use crate::game::{Action, T21State, TwentyOne};

impl GameUi for TwentyOne {
    fn id(&self) -> &'static str {
        "twentyone"
    }

    fn render(&self, state: &T21State, player: usize) -> String {
        let env = state.env();
        let o = env.observation(player);
        format!(
            "Round {} ({} damage) — hearts you {} / them {}\nYour total: {} (up {}, hidden {})   they show: {}{}",
            env.round(),
            env.round(),
            o.self_hearts,
            o.opp_hearts,
            o.self_total,
            o.self_face_up,
            o.self_face_down,
            o.opp_face_up,
            if o.opp_stood { "  [they stood]" } else { "" }
        )
    }

    fn action_label(&self, _state: &T21State, action: Action) -> String {
        match action {
            Action::Draw => "draw".into(),
            Action::Stand => "stand".into(),
            Action::DrawCard(c) => format!("(card {c})"),
            Action::Deal(a, b, c, d) => format!("(deal {a},{b},{c},{d})"),
        }
    }

    fn parse_action(&self, _state: &T21State, input: &str) -> Option<Action> {
        match input.trim().to_lowercase().as_str() {
            "d" | "draw" => Some(Action::Draw),
            "s" | "stand" => Some(Action::Stand),
            _ => None,
        }
    }

    fn describe_transition(
        &self,
        before: &T21State,
        _action: Action,
        after: &T21State,
        viewer: usize,
    ) -> Option<String> {
        let (b, a) = (before.env(), after.env());
        for p in 0..2 {
            if a.hearts(p) < b.hearts(p) {
                let who = if p == viewer { "you" } else { "they" };
                return Some(format!(
                    "→ round over: {who} lose {} heart(s) (now {} vs {}).",
                    b.hearts(p) - a.hearts(p),
                    a.hearts(viewer),
                    a.hearts(1 - viewer)
                ));
            }
        }
        if a.round() > b.round() {
            Some("→ round over: push, no damage.".into())
        } else {
            None
        }
    }
}
