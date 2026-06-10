//! Terminal/serving surface for Liar's Dice.

use game_core::GameUi;

use crate::{Action, HIST_K, LdState, LiarsDice};

fn dice_json(dice: &[u8]) -> String {
    let items: Vec<String> = dice.iter().map(u8::to_string).collect();
    format!("[{}]", items.join(","))
}

impl LiarsDice {
    fn prev_alive(&self, s: &LdState, from: u8) -> u8 {
        let mut p = (from + self.players - 1) % self.players;
        while s.dice_left[p as usize] == 0 {
            p = (p + self.players - 1) % self.players;
        }
        p
    }

    /// The round's recent bids as `(seat, qty, face)`, oldest first,
    /// reconstructed by walking the raise chain backwards from the current bid
    /// through the `HIST_K`-action history window. The forced 1×1 opener is
    /// nobody's bid and is not included.
    fn bid_trail(&self, s: &LdState) -> Vec<(u8, u8, u8)> {
        if s.done {
            return Vec::new();
        }
        let mut trail = Vec::new();
        let (mut q, mut f) = (s.qty, s.face);
        let mut seat = s.last_bidder;
        for i in (0..HIST_K).rev() {
            let code = s.hist[i];
            if code == 0 || q == 0 {
                break;
            }
            trail.push((seat, q, f));
            match code {
                1 => q -= 1,
                2 if f > 1 => f -= 1,
                2 => {
                    f = self.faces;
                    q -= 1;
                }
                _ => break, // an Open: the round's first bid
            }
            seat = self.prev_alive(s, seat);
        }
        trail.reverse();
        trail
    }
}

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

    fn view_data(&self, s: &LdState, viewer: usize) -> Option<String> {
        let n = self.players as usize;
        let spectator = viewer >= n;
        let phase = if s.done {
            "over"
        } else if (s.rolled as usize) < n {
            "rolling"
        } else {
            "bidding"
        };
        let (q, f) = s.current_bid();
        let bid = if q == 0 {
            "null".to_string()
        } else {
            let forced = s.first_round && s.hist.iter().all(|&c| c == 0);
            format!(
                r#"{{"qty":{q},"face":{f},"by":{},"forced":{forced}}}"#,
                s.last_bidder
            )
        };
        let history = self
            .bid_trail(s)
            .into_iter()
            .map(|(seat, q, f)| format!(r#"{{"seat":{seat},"qty":{q},"face":{f}}}"#))
            .collect::<Vec<_>>()
            .join(",");
        let hands = (0..n)
            .map(|p| {
                let dice = if !(spectator || p == viewer) {
                    "null".to_string()
                } else if (p as u8) < s.rolled {
                    dice_json(&s.hand(p))
                } else {
                    "[]".to_string()
                };
                format!(
                    r#"{{"seat":{p},"alive":{},"count":{},"dice":{dice}}}"#,
                    s.dice_left[p] > 0,
                    s.dice_left[p]
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let winner = if s.done {
            s.winner.to_string()
        } else {
            "null".to_string()
        };
        let viewer_field = if spectator { -1 } else { viewer as i64 };
        Some(format!(
            concat!(
                r#"{{"players":{n},"dice":{dice},"faces":{faces},"viewer":{viewer},"#,
                r#""spectator":{spectator},"phase":"{phase}","round":{round},"#,
                r#""totalDice":{total},"turn":{turn},"winner":{winner},"bid":{bid},"#,
                r#""history":[{history}],"hands":[{hands}]}}"#
            ),
            n = n,
            dice = self.dice,
            faces = self.faces,
            viewer = viewer_field,
            spectator = spectator,
            phase = phase,
            round = s.rounds,
            total = self.total_dice(s),
            turn = s.turn,
            winner = winner,
            bid = bid,
            history = history,
            hands = hands,
        ))
    }

    fn transition_data(
        &self,
        before: &LdState,
        action: Action,
        after: &LdState,
        _viewer: usize,
    ) -> Option<String> {
        let kind = match action {
            Action::CallLiar => "liar",
            Action::CallExact => "exact",
            _ => return None,
        };
        let n = self.players as usize;
        let (q, f) = before.current_bid();
        let actual = self.count_face(before, f);
        let loser = (0..n)
            .find(|&p| after.dice_left[p] < before.dice_left[p])
            .map_or_else(|| "null".to_string(), |p| p.to_string());
        let hands = (0..n)
            .map(|p| dice_json(&before.hand(p)))
            .collect::<Vec<_>>()
            .join(",");
        let winner = if after.done {
            after.winner.to_string()
        } else {
            "null".to_string()
        };
        Some(format!(
            concat!(
                r#"{{"kind":"{kind}","caller":{caller},"bidder":{bidder},"#,
                r#""bid":{{"qty":{q},"face":{f}}},"actual":{actual},"hands":[{hands}],"#,
                r#""loser":{loser},"diceLeft":{dice_left},"gameOver":{over},"#,
                r#""winner":{winner},"nextRound":{next_round}}}"#
            ),
            kind = kind,
            caller = before.turn,
            bidder = before.last_bidder,
            q = q,
            f = f,
            actual = actual,
            hands = hands,
            loser = loser,
            dice_left = dice_json(&after.dice_left[..n]),
            over = after.done,
            winner = winner,
            next_round = after.rounds,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_core::{Game, Turn};

    /// Minimal JSON validity check (no serde in this crate): returns the index
    /// just past one complete value starting at `i`, or `None` if malformed.
    fn skip_value(b: &[u8], mut i: usize) -> Option<usize> {
        fn ws(b: &[u8], mut i: usize) -> usize {
            while i < b.len() && (b[i] as char).is_whitespace() {
                i += 1;
            }
            i
        }
        i = ws(b, i);
        match *b.get(i)? {
            b'{' | b'[' => {
                let (close, sep) = if b[i] == b'{' {
                    (b'}', b':')
                } else {
                    (b']', 0)
                };
                i = ws(b, i + 1);
                if *b.get(i)? == close {
                    return Some(i + 1);
                }
                loop {
                    if sep == b':' {
                        if *b.get(i)? != b'"' {
                            return None;
                        }
                        i = skip_value(b, i)?;
                        i = ws(b, i);
                        if *b.get(i)? != b':' {
                            return None;
                        }
                        i += 1;
                    }
                    i = skip_value(b, i)?;
                    i = ws(b, i);
                    match *b.get(i)? {
                        b',' => i = ws(b, i + 1),
                        c if c == close => return Some(i + 1),
                        _ => return None,
                    }
                }
            }
            b'"' => {
                i += 1;
                while *b.get(i)? != b'"' {
                    i += if b[i] == b'\\' { 2 } else { 1 };
                }
                Some(i + 1)
            }
            b't' => b[i..].starts_with(b"true").then_some(i + 4),
            b'f' => b[i..].starts_with(b"false").then_some(i + 5),
            b'n' => b[i..].starts_with(b"null").then_some(i + 4),
            _ => {
                let start = i;
                while i < b.len() && matches!(b[i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
                {
                    i += 1;
                }
                (i > start).then_some(i)
            }
        }
    }

    fn assert_valid_json(s: &str) {
        let end = skip_value(s.as_bytes(), 0).unwrap_or_else(|| panic!("malformed JSON: {s}"));
        assert_eq!(end, s.len(), "trailing garbage in JSON: {s}");
    }

    /// Roll every hand deterministically (first chance outcome: all dice on
    /// the highest face).
    fn rolled_state(game: &LiarsDice) -> LdState {
        let mut s = game.initial_state();
        while matches!(game.turn(&s), Turn::Chance) {
            let a = game.chance_outcomes(&s)[0].0;
            game.apply(&mut s, a);
        }
        s
    }

    #[test]
    fn view_data_is_viewer_scoped_and_valid_json() {
        let game = LiarsDice::new(3, 2, 6);
        let mut s = rolled_state(&game);
        game.apply(&mut s, Action::RaiseQuantity); // P0: 1x1 -> 2x1
        game.apply(&mut s, Action::RaiseFace); // P1: 2x1 -> 2x2

        let v0 = game.view_data(&s, 0).unwrap();
        assert_valid_json(&v0);
        assert!(v0.contains(r#""seat":0,"alive":true,"count":2,"dice":[6,6]"#));
        assert!(v0.contains(r#""seat":1,"alive":true,"count":2,"dice":null"#));
        assert!(v0.contains(r#""bid":{"qty":2,"face":2,"by":1,"forced":false}"#));
        assert!(
            v0.contains(r#""history":[{"seat":0,"qty":2,"face":1},{"seat":1,"qty":2,"face":2}]"#)
        );
        assert!(v0.contains(r#""phase":"bidding""#) && v0.contains(r#""turn":2"#));
        assert!(v0.contains(r#""viewer":0,"spectator":false"#));

        let spec = game.view_data(&s, 3).unwrap();
        assert_valid_json(&spec);
        assert!(spec.contains(r#""viewer":-1,"spectator":true"#));
        for seat in 0..3 {
            assert!(
                spec.contains(&format!(
                    r#""seat":{seat},"alive":true,"count":2,"dice":[6,6]"#
                )),
                "spectator sees every hand: {spec}"
            );
        }
    }

    #[test]
    fn forced_opening_bid_is_flagged_and_unattributed() {
        let game = LiarsDice::new(3, 2, 6);
        let s = rolled_state(&game);
        let v = game.view_data(&s, 0).unwrap();
        assert_valid_json(&v);
        assert!(v.contains(r#""bid":{"qty":1,"face":1,"by":2,"forced":true}"#));
        assert!(v.contains(r#""history":[]"#));
    }

    #[test]
    fn transition_data_reveals_the_call() {
        let game = LiarsDice::new(3, 2, 6);
        let mut s = rolled_state(&game);
        game.apply(&mut s, Action::RaiseQuantity); // P0: 2x1
        game.apply(&mut s, Action::RaiseFace); // P1: 2x2
        let before = s.clone();
        game.apply(&mut s, Action::CallLiar); // P2 calls; zero 2s -> P1 loses

        let t = game
            .transition_data(&before, Action::CallLiar, &s, 0)
            .unwrap();
        assert_valid_json(&t);
        assert!(t.contains(r#""kind":"liar","caller":2,"bidder":1"#));
        assert!(t.contains(r#""bid":{"qty":2,"face":2},"actual":0"#));
        assert!(t.contains(r#""hands":[[6,6],[6,6],[6,6]]"#));
        assert!(t.contains(r#""loser":1,"diceLeft":[2,1,2]"#));
        assert!(t.contains(r#""gameOver":false,"winner":null"#));

        assert!(
            game.transition_data(&before, Action::RaiseQuantity, &s, 0)
                .is_none()
        );
    }
}
