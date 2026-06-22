//! Terminal and web serving surface for poker. The JSON `view_data` /
//! `transition_data` schema is a private contract with the poker frontend.

use game_core::{Game, GameUi};

use crate::cards::card_str;
use crate::{Action, NO_CARD, Poker, PokerState, Street};

impl Poker {
    /// Human-facing label for an action at `state`, with the chip amount where
    /// a raise/call/all-in carries one.
    fn label(&self, s: &PokerState, a: Action) -> String {
        let p = s.to_act();
        match a {
            Action::Fold => "fold".into(),
            Action::Check => "check".into(),
            Action::Call => format!("call {}", s.to_call(p)),
            Action::Raise(to) => format!("raise to {to}"),
            Action::AllIn => format!("all-in {}", s.stack(p) + s.street_bet(p)),
            Action::Deal(c) => format!("deal {}", card_str(c)),
            Action::NextHand => "next hand".into(),
        }
    }
}

fn cards_json(cards: &[u8]) -> String {
    let items: Vec<String> = cards
        .iter()
        .map(|&c| format!("\"{}\"", card_str(c)))
        .collect();
    format!("[{}]", items.join(","))
}

fn street_str(st: Street) -> &'static str {
    match st {
        Street::Preflop => "preflop",
        Street::Flop => "flop",
        Street::Turn => "turn",
        Street::River => "river",
        Street::Showdown => "showdown",
    }
}

impl GameUi for Poker {
    fn id(&self) -> &'static str {
        "poker"
    }

    fn render(&self, s: &PokerState, player: usize) -> String {
        let n = self.seats();
        let board: Vec<String> = s.board().iter().map(|&c| card_str(c)).collect();
        let mut out = String::new();
        if player < n {
            if let Some(h) = s.hole(player) {
                out.push_str(&format!(
                    "Your hand: {} {}\n",
                    card_str(h[0]),
                    card_str(h[1])
                ));
            }
        } else {
            for p in 0..n {
                if let Some(h) = s.hole(p) {
                    out.push_str(&format!(
                        "Seat {p}: {} {}\n",
                        card_str(h[0]),
                        card_str(h[1])
                    ));
                }
            }
        }
        out.push_str(&format!(
            "Board: [{}]   Pot: {}   Street: {}\n",
            board.join(" "),
            self.pot(s),
            street_str(s.street())
        ));
        for p in 0..n {
            let tag = if s.folded(p) {
                " (folded)"
            } else if s.all_in(p) {
                " (all-in)"
            } else {
                ""
            };
            out.push_str(&format!(
                "  Seat {p}: stack {} bet {}{}\n",
                s.stack(p),
                s.street_bet(p),
                tag
            ));
        }
        if !s.done() && player < n {
            let to_call = s.to_call(player);
            if to_call == 0 {
                out.push_str("You may check, bet, or fold.");
            } else {
                out.push_str(&format!("To call: {to_call}. Fold, call, or raise."));
            }
        }
        out
    }

    fn action_label(&self, s: &PokerState, a: Action) -> String {
        self.label(s, a)
    }

    fn parse_action(&self, s: &PokerState, input: &str) -> Option<Action> {
        let t = input.trim().to_lowercase();
        let p = s.to_act();
        match t.as_str() {
            "f" | "fold" => return Some(Action::Fold),
            "c" | "check" => return Some(Action::Check),
            "call" => return Some(Action::Call),
            "a" | "allin" | "all-in" | "shove" => return Some(Action::AllIn),
            _ => {}
        }
        // "raise <to>" / "bet <to>" / "r <to>" select the matching menu raise,
        // or fall back to the nearest offered size.
        let amount = t
            .strip_prefix("raise to ")
            .or_else(|| t.strip_prefix("raise "))
            .or_else(|| t.strip_prefix("bet "))
            .or_else(|| t.strip_prefix("r "))
            .and_then(|r| r.trim().parse::<u32>().ok());
        if let Some(to) = amount {
            let actions = self.legal_actions(s);
            // Exact match, else the closest raise/all-in.
            if actions.contains(&Action::Raise(to)) {
                return Some(Action::Raise(to));
            }
            let mut best: Option<(Action, u32)> = None;
            for &a in &actions {
                let val = match a {
                    Action::Raise(v) => v,
                    Action::AllIn => s.street_bet(p) + s.stack(p),
                    _ => continue,
                };
                let dist = val.abs_diff(to);
                if best.is_none_or(|(_, d)| dist < d) {
                    best = Some((a, dist));
                }
            }
            return best.map(|(a, _)| a);
        }
        None
    }

    fn describe_transition(
        &self,
        _before: &PokerState,
        _action: Action,
        after: &PokerState,
        viewer: usize,
    ) -> Option<String> {
        if !after.resolved() {
            return None;
        }
        let n = self.seats();
        let mut lines = vec!["→ hand over.".to_string()];
        let showdown = (0..n).filter(|&p| !after.folded(p)).count() > 1;
        if showdown {
            for p in 0..n {
                if !after.folded(p)
                    && let Some(h) = after.hole(p)
                {
                    lines.push(format!(
                        "   Seat {p} shows {} {}",
                        card_str(h[0]),
                        card_str(h[1])
                    ));
                }
            }
        }
        for p in 0..n {
            let net = after.payoff_bb(p);
            if net.abs() > 1e-9 {
                let you = if p == viewer { " (you)" } else { "" };
                lines.push(format!("   Seat {p}{you}: {net:+.1} bb"));
            }
        }
        Some(lines.join("\n"))
    }

    fn view_data(&self, s: &PokerState, viewer: usize) -> Option<String> {
        let n = self.seats();
        let spectator = viewer >= n;
        let phase = if s.resolved() {
            "over"
        } else {
            street_str(s.street())
        };
        let seats = (0..n)
            .map(|p| {
                let hole =
                    if (spectator && s.resolved()) || p == viewer || (s.resolved() && !s.folded(p))
                    {
                        match s.hole(p) {
                            Some(h) if h[0] != NO_CARD => cards_json(&h),
                            _ => "null".to_string(),
                        }
                    } else {
                        "null".to_string()
                    };
                format!(
                    concat!(
                        r#"{{"seat":{p},"stack":{stack},"committed":{committed},"#,
                        r#""streetBet":{bet},"folded":{folded},"allIn":{allin},"#,
                        r#""toAct":{to_act},"net":{net},"hole":{hole}}}"#
                    ),
                    p = p,
                    stack = s.stack(p),
                    committed = s.committed(p),
                    bet = s.street_bet(p),
                    folded = s.folded(p),
                    allin = s.all_in(p),
                    to_act = (!s.resolved() && self.is_to_act(s, p)),
                    net = if s.resolved() {
                        format!("{:.3}", s.payoff_bb(p))
                    } else {
                        "null".to_string()
                    },
                    hole = hole,
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let to_call = if spectator || s.resolved() {
            0
        } else {
            s.to_call(viewer)
        };
        let viewer_field = if spectator { -1 } else { viewer as i64 };
        Some(format!(
            concat!(
                r#"{{"seats":{n},"viewer":{viewer},"spectator":{spectator},"#,
                r#""phase":"{phase}","button":{button},"pot":{pot},"#,
                r#""currentBet":{current_bet},"toCall":{to_call},"bigBlind":{bb},"#,
                r#""board":{board},"players":[{seats}]}}"#
            ),
            n = n,
            viewer = viewer_field,
            spectator = spectator,
            phase = phase,
            button = s.button(),
            pot = self.pot(s),
            current_bet = s.current_bet(),
            to_call = to_call,
            bb = self.big_blind,
            board = cards_json(s.board()),
            seats = seats,
        ))
    }

    fn transition_data(
        &self,
        before: &PokerState,
        action: Action,
        after: &PokerState,
        _viewer: usize,
    ) -> Option<String> {
        // Only player actions and the showdown are worth animating; raw deals
        // are reflected by the next view.
        let n = self.seats();
        let kind = match action {
            Action::Fold => "fold",
            Action::Check => "check",
            Action::Call => "call",
            Action::Raise(_) => "raise",
            Action::AllIn => "allin",
            Action::Deal(_) | Action::NextHand => return None,
        };
        let seat = before.to_act();
        let reveal = if after.resolved() {
            let shows = (0..n)
                .filter(|&p| !after.folded(p))
                .filter_map(|p| {
                    after.hole(p).map(|h| {
                        format!(
                            r#"{{"seat":{p},"hole":{},"net":{:.3}}}"#,
                            cards_json(&h),
                            after.payoff_bb(p)
                        )
                    })
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("[{shows}]")
        } else {
            "null".to_string()
        };
        Some(format!(
            concat!(
                r#"{{"kind":"{kind}","seat":{seat},"amount":{amount},"#,
                r#""pot":{pot},"gameOver":{over},"showdown":{reveal}}}"#
            ),
            kind = kind,
            seat = seat,
            amount = after.committed(seat).saturating_sub(before.committed(seat)),
            pot = self.pot(after),
            over = after.resolved(),
            reveal = reveal,
        ))
    }

    fn result_text(&self, s: &PokerState, viewer: usize) -> String {
        let net = s.payoff_bb(viewer);
        if net > 1e-9 {
            format!("You win {net:+.1} bb.")
        } else if net < -1e-9 {
            format!("You lose {:.1} bb.", net.abs())
        } else {
            "You break even.".into()
        }
    }
}
