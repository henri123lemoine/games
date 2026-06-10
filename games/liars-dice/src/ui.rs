//! Terminal/serving surface for Liar's Dice.

use game_core::GameUi;

use crate::{Action, LdState, LiarsDice};

impl GameUi for LiarsDice {
    fn id(&self) -> &'static str {
        "liars-dice"
    }

    fn render(&self, state: &LdState, player: usize) -> String {
        let (q, f) = state.current_bid();
        let mut out = format!(
            "Your hand: {:?}\nDice left per player: {:?}\n",
            state.hand(player),
            &state.dice_left()[..self.players as usize]
        );
        if q == 0 {
            out.push_str("You open the round (type `open QxF`, e.g. `open 2x4`).");
        } else {
            out.push_str(&format!(
                "Current bid: {q} x face {f} (by Player {}).",
                state.last_bidder()
            ));
        }
        out
    }

    fn action_label(&self, _state: &LdState, action: Action) -> String {
        LiarsDice::action_label(self, action)
    }

    fn parse_action(&self, state: &LdState, input: &str) -> Option<Action> {
        let t = input.trim().to_lowercase();
        let (q, _f) = state.current_bid();
        if q > 0 {
            return match t.as_str() {
                "q" | "rq" | "quantity" | "raise quantity" => Some(Action::RaiseQuantity),
                "f" | "rf" | "face" | "raise face" => Some(Action::RaiseFace),
                "l" | "liar" | "call liar" => Some(Action::CallLiar),
                "e" | "exact" | "call exact" => Some(Action::CallExact),
                _ => None,
            };
        }
        let rest = t.strip_prefix("open")?.trim().replace(' ', "");
        let (qs, fs) = rest.split_once('x')?;
        Some(Action::Open(qs.parse().ok()?, fs.parse().ok()?))
    }

    fn describe_transition(
        &self,
        before: &LdState,
        action: Action,
        after: &LdState,
        _viewer: usize,
    ) -> Option<String> {
        if !matches!(action, Action::CallLiar | Action::CallExact) {
            return None;
        }
        let (q, f) = before.current_bid();
        let n = self.players as usize;
        let hands: Vec<Vec<u8>> = (0..n).map(|p| before.hand(p)).collect();
        let actual: usize = hands.iter().flatten().filter(|&&d| d == f).count();
        let mut out = format!(
            "→ called on {q}×{f}. Revealed dice: {hands:?}\n→ actual count of face {f}: {actual}."
        );
        let lost = (0..n).find(|&p| after.dice_left()[p] < before.dice_left()[p]);
        match lost {
            Some(p) => out.push_str(&format!(
                " Player {p} loses a die (now {}).",
                after.dice_left()[p]
            )),
            None => out.push_str(" Exact! Nobody loses a die."),
        }
        Some(out)
    }
}
