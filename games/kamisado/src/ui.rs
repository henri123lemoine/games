//! Terminal/serving surface for Kamisado.

use game_core::{Game, GameUi};

use crate::{
    COLOR_LETTERS, COLOR_NAMES, Kamisado, KamisadoMove, KamisadoState, file, goal_rank, rank,
    resolve_obligation, square_color,
};

fn sq_name(sq: u8) -> String {
    format!("{}{}", (b'a' + file(sq)) as char, rank(sq) + 1)
}

/// Index into the web view's `cells` string: row-major, TOP rank (8) first.
fn cell_index(sq: u8) -> u8 {
    (7 - rank(sq)) * 8 + file(sq)
}

fn parse_sq(s: &[u8]) -> Option<u8> {
    let f = s[0].checked_sub(b'a').filter(|&f| f < 8)?;
    let r = s[1].checked_sub(b'1').filter(|&r| r < 8)?;
    Some(r * 8 + f)
}

fn player_name(p: usize) -> &'static str {
    if p == 0 { "Black" } else { "White" }
}

fn color_name(c: u8) -> &'static str {
    COLOR_NAMES[c as usize]
}

impl GameUi for Kamisado {
    fn id(&self) -> &'static str {
        "kamisado"
    }

    /// Left grid: towers, Black in CAPS, White in lowercase, both named by
    /// their color letter. Right grid: the (static) square colors.
    fn render(&self, state: &KamisadoState, player: usize) -> String {
        let mut out = String::new();
        for r in (0..8u8).rev() {
            out.push_str(&format!("{}  ", r + 1));
            for f in 0..8u8 {
                out.push(match state.tower_at(r * 8 + f) {
                    Some((0, c)) => COLOR_LETTERS[c as usize] as char,
                    Some((_, c)) => COLOR_LETTERS[c as usize].to_ascii_lowercase() as char,
                    None => '.',
                });
                out.push(' ');
            }
            out.push_str("  ");
            for f in 0..8u8 {
                out.push(COLOR_LETTERS[square_color(r * 8 + f) as usize] as char);
                out.push(' ');
            }
            out.push('\n');
        }
        out.push_str("   a b c d e f g h   (square colors)\n");
        out.push_str(
            "Black CAPS, White lower. N=Brown G=Green R=Red Y=Yellow P=Pink U=Purple B=Blue O=Orange\n",
        );
        let you = player_name(player);
        let mover = state.to_move as usize;
        match state.winner {
            Some(w) => out.push_str(&format!(
                "You are {you}. {} wins the round.",
                player_name(w as usize)
            )),
            None => match state.required_color() {
                None => out.push_str(&format!(
                    "You are {you}. {} opens: move any tower.",
                    player_name(mover)
                )),
                Some(c) => out.push_str(&format!(
                    "You are {you}. {} to move the {} tower ({}).",
                    player_name(mover),
                    color_name(c),
                    sq_name(state.towers[mover][c as usize]),
                )),
            },
        }
        out
    }

    fn action_label(&self, _state: &KamisadoState, action: KamisadoMove) -> String {
        format!("{}-{}", sq_name(action.from), sq_name(action.to))
    }

    /// Accepts `"d1-d7"`, `"d1d7"`, `"d1 d7"`, or just a destination (`"d7"`)
    /// when a single tower is obligated to move.
    fn parse_action(&self, state: &KamisadoState, input: &str) -> Option<KamisadoMove> {
        let t: Vec<u8> = input
            .trim()
            .to_ascii_lowercase()
            .bytes()
            .filter(|b| !b" -x".contains(b))
            .collect();
        let mv = match t.len() {
            4 => KamisadoMove {
                from: parse_sq(&t[0..2])?,
                to: parse_sq(&t[2..4])?,
            },
            2 => {
                let to = parse_sq(&t)?;
                let from = state.towers[state.to_move as usize][state.required_color()? as usize];
                KamisadoMove { from, to }
            }
            _ => return None,
        };
        self.legal_actions(state).contains(&mv).then_some(mv)
    }

    /// Narrates the landing color and the obligation it creates — including
    /// blocked towers passing the obligation along, which the post-state
    /// alone no longer shows.
    fn describe_transition(
        &self,
        before: &KamisadoState,
        action: KamisadoMove,
        after: &KamisadoState,
        _viewer: usize,
    ) -> Option<String> {
        let mover = before.to_move as usize;
        let moved = color_name(before.mover_color(action.from));
        let land = color_name(square_color(action.to));
        let head = format!(
            "{} slides {moved} to {} ({land})",
            player_name(mover),
            sq_name(action.to)
        );
        if after.winner == Some(mover as u8) && rank(action.to) == goal_rank(mover) {
            return Some(format!(
                "{head} — the far rank: {} wins.",
                player_name(mover)
            ));
        }
        let mut passes = Vec::new();
        let next = resolve_obligation(&after.towers, after.occ, mover, action.to, |p, c| {
            passes.push(format!("{}'s {} is blocked", player_name(p), color_name(c)))
        });
        match next {
            None => Some(format!(
                "{head}; {} — a deadlock: {} caused it and loses.",
                passes.join(", "),
                player_name(mover)
            )),
            Some((q, c)) if passes.is_empty() => Some(format!(
                "{head}: {} must move {}.",
                player_name(q),
                color_name(c)
            )),
            Some((q, c)) => Some(format!(
                "{head}; {} — {} must move {}.",
                passes.join(", "),
                player_name(q),
                color_name(c)
            )),
        }
    }

    fn result_text(&self, state: &KamisadoState, viewer: usize) -> String {
        debug_assert!(self.is_terminal(state));
        let w = state.winner.expect("terminal state has a winner") as usize;
        let verdict = if w == viewer { "You win!" } else { "You lose." };
        format!("{} takes the round. {verdict}", player_name(w))
    }

    /// View JSON — the private contract with `web/app/src/frontends/kamisado`:
    ///
    /// ```json
    /// {"cells": "<64 chars, row-major, TOP rank (8) first: '.' empty,
    ///            'A'-'H' Black tower of color 0-7, 'a'-'h' White>",
    ///  "turn": 0|1,
    ///  "required": 0-7|null,      // color `turn` is obligated to move
    ///  "requiredCell": 0-63|null, // index of that tower in "cells"
    ///  "winner": 0|1|null,
    ///  "deadlock": true|false}    // winner came from a deadlock, not the far rank
    /// ```
    fn view_data(&self, state: &KamisadoState, _viewer: usize) -> Option<String> {
        let mut cells = vec![b'.'; 64];
        for p in 0..2 {
            for c in 0..8 {
                let base = if p == 0 { b'A' } else { b'a' };
                cells[cell_index(state.towers[p][c]) as usize] = base + c as u8;
            }
        }
        let cells = String::from_utf8(cells).expect("ascii cells");
        let (required, required_cell) = match state.required_color() {
            Some(c) if state.winner.is_none() => (
                c.to_string(),
                cell_index(state.towers[state.to_move as usize][c as usize]).to_string(),
            ),
            _ => ("null".into(), "null".into()),
        };
        let winner = state.winner.map_or("null".into(), |w| w.to_string());
        let deadlock = state.winner.is_some_and(|w| {
            let w = w as usize;
            state.towers[w].iter().all(|&t| rank(t) != goal_rank(w))
        });
        Some(format!(
            r#"{{"cells":"{cells}","turn":{},"required":{required},"requiredCell":{required_cell},"winner":{winner},"deadlock":{deadlock}}}"#,
            state.to_move
        ))
    }

    /// Transition JSON — what moved, the obligation chain it set off, and how
    /// the round ended if it did (cell indices as in `view_data`):
    ///
    /// ```json
    /// {"from": 0-63, "to": 0-63, "player": 0|1, "color": 0-7, "landColor": 0-7,
    ///  "passes": [[player, color], ...],  // blocked towers, in chain order
    ///  "next": [player, color]|null,      // obligation after the move
    ///  "win": bool, "deadlock": bool}
    /// ```
    fn transition_data(
        &self,
        before: &KamisadoState,
        action: KamisadoMove,
        after: &KamisadoState,
        _viewer: usize,
    ) -> Option<String> {
        let mover = before.to_move as usize;
        let color = before.mover_color(action.from);
        let win = after.winner == Some(mover as u8) && rank(action.to) == goal_rank(mover);
        let (mut passes, mut next) = (String::from("["), String::from("null"));
        if !win {
            let mut sep = "";
            let resolved =
                resolve_obligation(&after.towers, after.occ, mover, action.to, |p, c| {
                    passes.push_str(&format!("{sep}[{p},{c}]"));
                    sep = ",";
                });
            if let Some((p, c)) = resolved {
                next = format!("[{p},{c}]");
            }
        }
        passes.push(']');
        Some(format!(
            r#"{{"from":{},"to":{},"player":{mover},"color":{color},"landColor":{},"passes":{passes},"next":{next},"win":{win},"deadlock":{}}}"#,
            cell_index(action.from),
            cell_index(action.to),
            square_color(action.to),
            after.winner.is_some() && !win,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sq(name: &str) -> u8 {
        parse_sq(name.as_bytes()).unwrap()
    }

    fn play(moves: &[(&str, &str)]) -> (Kamisado, KamisadoState) {
        let game = Kamisado;
        let mut state = game.initial_state();
        for &(from, to) in moves {
            let mv = KamisadoMove {
                from: sq(from),
                to: sq(to),
            };
            assert!(game.legal_actions(&state).contains(&mv));
            game.apply(&mut state, mv);
        }
        (game, state)
    }

    #[test]
    fn labels_and_parsing_round_trip() {
        let game = Kamisado;
        let s = game.initial_state();
        for a in game.legal_actions(&s) {
            let label = game.action_label(&s, a);
            assert_eq!(game.parse_action(&s, &label), Some(a));
        }
        assert_eq!(
            game.parse_action(&s, "D1 d7"),
            Some(KamisadoMove {
                from: sq("d1"),
                to: sq("d7")
            })
        );
        assert_eq!(game.parse_action(&s, "d1-d8"), None, "d8 is occupied");
        assert_eq!(
            game.parse_action(&s, "d7"),
            None,
            "opening has no obligation"
        );
    }

    #[test]
    fn bare_destination_uses_the_obligated_tower() {
        let (game, s) = play(&[("d1", "d7")]);
        assert_eq!(
            game.parse_action(&s, "g2"),
            Some(KamisadoMove {
                from: sq("g8"),
                to: sq("g2")
            })
        );
    }

    #[test]
    fn view_data_reports_cells_obligation_and_result() {
        let game = Kamisado;
        let v = game.view_data(&game.initial_state(), 0).unwrap();
        // Rank 8 first: White's towers run Orange..Brown = 'h'..'a'.
        assert!(v.contains(r#""cells":"hgfedcba"#), "{v}");
        assert!(v.ends_with(r#"ABCDEFGH","turn":0,"required":null,"requiredCell":null,"winner":null,"deadlock":false}"#), "{v}");

        let (game, s) = play(&[("d1", "d7")]);
        let v = game.view_data(&s, 1).unwrap();
        assert!(
            v.contains(r#""turn":1,"required":1,"requiredCell":6"#),
            "{v}"
        ); // Green at g8
        let (game, s) = play(&[("d1", "d7"), ("g8", "b3"), ("b3", "d1")]);
        let v = game.view_data(&s, 0).unwrap();
        assert!(v.contains(r#""winner":1,"deadlock":false"#), "{v}");
    }

    #[test]
    fn transition_data_narrates_moves_passes_and_wins() {
        let game = Kamisado;
        let step = |moves: &[(&str, &str)], mv: (&str, &str)| {
            let (_, before) = play(moves);
            let action = KamisadoMove {
                from: sq(mv.0),
                to: sq(mv.1),
            };
            let mut after = before.clone();
            game.apply(&mut after, action);
            game.transition_data(&before, action, &after, 0).unwrap()
        };
        // d1 -> d7: from d1 = cells 59, to d7 = cells 11; lands Green; White must move Green.
        assert_eq!(
            step(&[], ("d1", "d7")),
            r#"{"from":59,"to":11,"player":0,"color":3,"landColor":1,"passes":[],"next":[1,1],"win":false,"deadlock":false}"#
        );
        // g8 -> b3: Black's Yellow is walled in; obligation returns to White's Green.
        assert_eq!(
            step(&[("d1", "d7")], ("g8", "b3")),
            r#"{"from":6,"to":41,"player":1,"color":1,"landColor":3,"passes":[[0,3]],"next":[1,1],"win":false,"deadlock":false}"#
        );
        assert!(
            step(&[("d1", "d7"), ("g8", "b3")], ("b3", "d1"))
                .ends_with(r#""win":true,"deadlock":false}"#)
        );
    }

    #[test]
    fn render_shows_towers_colors_and_obligation() {
        let (game, s) = play(&[("d1", "d7")]);
        let view = game.render(&s, 0);
        assert!(view.contains("You are Black."));
        assert!(view.contains("White to move the Green tower (g8)."));
        assert!(view.contains("a b c d e f g h"));
    }

    #[test]
    fn transitions_narrate_obligations_and_passes() {
        let (game, before) = play(&[]);
        let mv = KamisadoMove {
            from: sq("d1"),
            to: sq("d7"),
        };
        let mut after = before.clone();
        game.apply(&mut after, mv);
        assert_eq!(
            game.describe_transition(&before, mv, &after, 0).unwrap(),
            "Black slides Yellow to d7 (Green): White must move Green."
        );

        let (game, before) = play(&[("d1", "d7")]);
        let mv = KamisadoMove {
            from: sq("g8"),
            to: sq("b3"),
        };
        let mut after = before.clone();
        game.apply(&mut after, mv);
        assert_eq!(
            game.describe_transition(&before, mv, &after, 0).unwrap(),
            "White slides Green to b3 (Yellow); Black's Yellow is blocked — White must move Green."
        );

        let (game, before) = play(&[("d1", "d7"), ("g8", "b3")]);
        let mv = KamisadoMove {
            from: sq("b3"),
            to: sq("d1"),
        };
        let mut after = before.clone();
        game.apply(&mut after, mv);
        assert_eq!(
            game.describe_transition(&before, mv, &after, 0).unwrap(),
            "White slides Green to d1 (Yellow) — the far rank: White wins."
        );
        assert_eq!(
            game.result_text(&after, 1),
            "White takes the round. You win!"
        );
        assert_eq!(
            game.result_text(&after, 0),
            "White takes the round. You lose."
        );
    }
}
