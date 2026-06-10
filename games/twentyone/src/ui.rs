//! Terminal/serving surface for Twenty-One.

use game_core::GameUi;

use crate::game::{Action, T21State, TwentyOne};

fn join(cards: &[u8]) -> String {
    cards
        .iter()
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

impl GameUi for TwentyOne {
    fn id(&self) -> &'static str {
        "twentyone"
    }

    fn render(&self, state: &T21State, player: usize) -> String {
        let env = state.env();
        if player >= 2 {
            let (a, b) = (env.observation(0), env.observation(1));
            return format!(
                "Round {} ({} damage) — hearts P0 {} / P1 {}\nP0 total: {} (up {}, hidden {}){}   P1 total: {} (up {}, hidden {}){}",
                env.round(),
                env.round(),
                a.self_hearts,
                b.self_hearts,
                a.self_total,
                a.self_face_up,
                a.self_face_down,
                if a.self_stood { "  [stood]" } else { "" },
                b.self_total,
                b.self_face_up,
                b.self_face_down,
                if b.self_stood { "  [stood]" } else { "" },
            );
        }
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
                let lost = b.hearts(p) - a.hearts(p);
                return Some(if viewer < 2 {
                    let who = if p == viewer { "you" } else { "they" };
                    format!(
                        "→ round over: {who} lose {lost} heart(s) (now {} vs {}).",
                        a.hearts(viewer),
                        a.hearts(1 - viewer)
                    )
                } else {
                    format!(
                        "→ round over: P{p} loses {lost} heart(s) (now {} vs {}).",
                        a.hearts(0),
                        a.hearts(1)
                    )
                });
            }
        }
        if a.round() > b.round() {
            Some("→ round over: push, no damage.".into())
        } else {
            None
        }
    }

    /// Web view schema, viewer-scoped:
    /// `{"round":N,"hearts":[h0,h1],"maxHearts":N,"roundActive":bool,
    /// "deckCount":N,"toAct":0|1|null,"players":[{"up":[..],"down":N|null,
    /// "total":N|null,"stood":bool},..],"lastReveal":{"downs":[d0,d1],
    /// "up":[[..],[..]]}|null}`.
    ///
    /// Public per seat: the opening face-up card and every drawn card (`up`),
    /// hearts, and stood flags. The face-down card and exact total stay
    /// `null` for everyone but the seat itself; a spectator (`viewer >= 2`)
    /// sees both. `lastReveal` is the previous round's showdown, public once
    /// the round ends.
    fn view_data(&self, state: &T21State, viewer: usize) -> Option<String> {
        let env = state.env();
        let active = env.round_active();
        let mut players = String::new();
        for p in 0..2 {
            if p > 0 {
                players.push(',');
            }
            let up = env
                .public_up_cards(p)
                .map(|(cards, len)| join(&cards[..len as usize]))
                .unwrap_or_default();
            let o = env.observation(p);
            let (down, total) = if active && (viewer >= 2 || viewer == p) {
                (o.self_face_down.to_string(), o.self_total.to_string())
            } else {
                ("null".to_string(), "null".to_string())
            };
            players.push_str(&format!(
                r#"{{"up":[{up}],"down":{down},"total":{total},"stood":{}}}"#,
                active && o.self_stood
            ));
        }
        let reveal = match (env.last_reveal(), env.last_public_up()) {
            (Some([d0, d1]), Some((u0, l0, u1, l1))) => format!(
                r#"{{"downs":[{d0},{d1}],"up":[[{}],[{}]]}}"#,
                join(&u0[..l0 as usize]),
                join(&u1[..l1 as usize])
            ),
            _ => "null".to_string(),
        };
        let to_act = if active {
            env.current_player().to_string()
        } else {
            "null".to_string()
        };
        Some(format!(
            r#"{{"round":{},"hearts":[{},{}],"maxHearts":{},"roundActive":{active},"deckCount":{},"toAct":{to_act},"players":[{players}],"lastReveal":{reveal}}}"#,
            env.round(),
            env.hearts(0),
            env.hearts(1),
            self.start_hearts,
            env.deck_mask().count_ones(),
        ))
    }

    /// Web transition schema: `{"kind":"draw"|"stand","seat":0|1}` for
    /// in-round actions (the drawn card resolves at a later chance node and
    /// shows up in the next view), or — when a stand ends the round —
    /// `{"kind":"roundEnd","seat":s,"downs":[d0,d1],"totals":[t0,t1],
    /// "winner":0|1|null,"damage":heartsLost,"hearts":[h0,h1]}`, the public
    /// showdown reveal.
    fn transition_data(
        &self,
        before: &T21State,
        action: Action,
        after: &T21State,
        _viewer: usize,
    ) -> Option<String> {
        let (b, a) = (before.env(), after.env());
        let seat = b.current_player();
        match action {
            Action::Draw => Some(format!(r#"{{"kind":"draw","seat":{seat}}}"#)),
            Action::Stand if a.round() > b.round() => {
                let (o0, o1) = (b.observation(0), b.observation(1));
                let winner = if a.hearts(0) < b.hearts(0) {
                    "1".to_string()
                } else if a.hearts(1) < b.hearts(1) {
                    "0".to_string()
                } else {
                    "null".to_string()
                };
                let damage = (b.hearts(0) - a.hearts(0)).max(b.hearts(1) - a.hearts(1));
                Some(format!(
                    r#"{{"kind":"roundEnd","seat":{seat},"downs":[{},{}],"totals":[{},{}],"winner":{winner},"damage":{damage},"hearts":[{},{}]}}"#,
                    o0.self_face_down,
                    o1.self_face_down,
                    o0.self_total,
                    o1.self_total,
                    a.hearts(0),
                    a.hearts(1),
                ))
            }
            Action::Stand => Some(format!(r#"{{"kind":"stand","seat":{seat}}}"#)),
            Action::DrawCard(_) | Action::Deal(..) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_core::Game;

    fn dealt() -> (TwentyOne, T21State) {
        let g = TwentyOne::new(6);
        let mut s = g.initial_state();
        g.apply(&mut s, Action::Deal(5, 9, 6, 2));
        (g, s)
    }

    #[test]
    fn view_data_hides_opponent_hole_card() {
        let (g, s) = dealt();
        assert_eq!(
            g.view_data(&s, 0).unwrap(),
            r#"{"round":1,"hearts":[6,6],"maxHearts":6,"roundActive":true,"deckCount":7,"toAct":0,"players":[{"up":[5],"down":6,"total":11,"stood":false},{"up":[9],"down":null,"total":null,"stood":false}],"lastReveal":null}"#
        );
    }

    #[test]
    fn spectator_sees_both_hole_cards() {
        let (g, s) = dealt();
        assert_eq!(
            g.view_data(&s, 2).unwrap(),
            r#"{"round":1,"hearts":[6,6],"maxHearts":6,"roundActive":true,"deckCount":7,"toAct":0,"players":[{"up":[5],"down":6,"total":11,"stood":false},{"up":[9],"down":2,"total":11,"stood":false}],"lastReveal":null}"#
        );
    }

    #[test]
    fn round_end_transition_reveals_showdown() {
        let (g, mut s) = dealt();
        g.apply(&mut s, Action::Stand);
        let before = s.clone();
        g.apply(&mut s, Action::Stand);
        assert_eq!(
            g.transition_data(&before, Action::Stand, &s, 0).unwrap(),
            r#"{"kind":"roundEnd","seat":1,"downs":[6,2],"totals":[11,11],"winner":null,"damage":0,"hearts":[6,6]}"#
        );
    }
}
