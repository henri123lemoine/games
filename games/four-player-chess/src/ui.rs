use game_core::GameUi;

use crate::FourPlayerChess;
use crate::board::{Color, EndReason, Move, Piece, PieceKind, State, parse_square, square_name};

fn color_code(color: Color) -> char {
    match color {
        Color::Red => 'r',
        Color::Blue => 'b',
        Color::Yellow => 'y',
        Color::Green => 'g',
    }
}

fn piece_token(piece: Piece, active: bool) -> String {
    if piece.is_empty() {
        return "  ".into();
    }
    let color = if active {
        color_code(piece.color())
    } else {
        'x'
    };
    format!("{color}{}", piece.kind().letter())
}

fn end_name(end: EndReason) -> &'static str {
    match end {
        EndReason::Ongoing => "ongoing",
        EndReason::LastArmy => "last-army",
        EndReason::Repetition => "repetition",
        EndReason::FiftyMove => "fifty-move",
        EndReason::InsufficientMaterial => "insufficient-material",
        EndReason::PlyCap => "ply-cap",
    }
}

impl GameUi for FourPlayerChess {
    fn id(&self) -> &'static str {
        "four-player-chess"
    }

    fn render(&self, state: &State, player: usize) -> String {
        let mut out = String::new();
        for y in (0..14).rev() {
            out.push_str(&format!("{:>2} ", y + 1));
            for x in 0..14 {
                let Some(square) = crate::square(x, y) else {
                    out.push_str("   ");
                    continue;
                };
                let piece = state.board[square as usize];
                if piece.is_empty() {
                    out.push_str(" · ");
                } else {
                    out.push_str(&format!(
                        "{} ",
                        piece_token(piece, state.is_active(piece.color()))
                    ));
                }
            }
            out.push('\n');
        }
        out.push_str("    a  b  c  d  e  f  g  h  i  j  k  l  m  n\n");
        out.push_str(&format!(
            "You are {}. {} to move. Scores — Red {} · Blue {} · Yellow {} · Green {}.",
            Color::from_index(player).name(),
            state.to_move.name(),
            state.scores[0],
            state.scores[1],
            state.scores[2],
            state.scores[3],
        ));
        if self.in_check(state, state.to_move) {
            out.push_str(" Check!");
        }
        out
    }

    fn action_label(&self, state: &State, action: Move) -> String {
        let mut label = format!("{}{}", square_name(action.from), square_name(action.to));
        let piece = state.board[action.from as usize];
        if !piece.is_empty()
            && piece.kind() == PieceKind::Pawn
            && crate::pawn_promotes(action.to, piece.color())
        {
            label.push_str("=Q");
        }
        label
    }

    fn parse_action(&self, state: &State, input: &str) -> Option<Move> {
        let clean = input
            .trim()
            .to_ascii_lowercase()
            .replace(['-', 'x'], "")
            .split('=')
            .next()?
            .to_string();
        let split = if clean
            .as_bytes()
            .get(2)
            .is_some_and(|byte| byte.is_ascii_digit())
        {
            3
        } else {
            2
        };
        let action = Move::new(
            parse_square(&clean[..split])?,
            parse_square(&clean[split..])?,
        );
        self.legal_moves(state).contains(&action).then_some(action)
    }

    fn describe_transition(
        &self,
        before: &State,
        action: Move,
        after: &State,
        _viewer: usize,
    ) -> Option<String> {
        let actor = before.to_move;
        let gained = after.scores[actor.index()] - before.scores[actor.index()];
        let eliminated: Vec<_> = Color::ALL
            .into_iter()
            .filter(|&color| before.is_active(color) && !after.is_active(color))
            .map(Color::name)
            .collect();
        let mut parts = Vec::new();
        if gained > 0 {
            parts.push(format!("{} scores +{gained}", actor.name()));
        }
        if !eliminated.is_empty() {
            parts.push(format!("eliminated: {}", eliminated.join(", ")));
        }
        if after.end != EndReason::Ongoing {
            parts.push(format!("game ended ({})", end_name(after.end)));
        } else {
            let checked: Vec<_> = Color::ALL
                .into_iter()
                .filter(|&color| after.is_active(color) && self.in_check(after, color))
                .map(Color::name)
                .collect();
            if !checked.is_empty() {
                parts.push(format!("check: {}", checked.join(", ")));
            }
        }
        let _ = action;
        (!parts.is_empty()).then(|| parts.join("; "))
    }

    fn view_data(&self, state: &State, _viewer: usize) -> Option<String> {
        let mut pieces = Vec::new();
        for (square, &piece) in state.board.iter().enumerate() {
            if piece.is_empty() {
                continue;
            }
            pieces.push(format!(
                concat!(r#"{{"square":"{}","color":"{}","piece":"{}","dead":{},"promoted":{}}}"#),
                square_name(square as u8),
                color_code(piece.color()),
                piece.kind().letter(),
                !state.is_active(piece.color()),
                piece.promoted(),
            ));
        }
        let checks = Color::ALL.map(|color| state.is_active(color) && self.in_check(state, color));
        Some(format!(
            concat!(
                r#"{{"size":14,"pieces":[{}],"turn":"{}","active":[{},{},{},{}],"scores":[{},{},{},{}],"check":[{},{},{},{}],"end":"{}","last":{}}}"#
            ),
            pieces.join(","),
            color_code(state.to_move),
            state.is_active(Color::Red),
            state.is_active(Color::Blue),
            state.is_active(Color::Yellow),
            state.is_active(Color::Green),
            state.scores[0],
            state.scores[1],
            state.scores[2],
            state.scores[3],
            checks[0],
            checks[1],
            checks[2],
            checks[3],
            end_name(state.end),
            game_core::json::string_or_null(state.last_move.map(|action| format!(
                "{}{}",
                square_name(action.from),
                square_name(action.to)
            ))),
        ))
    }

    fn transition_data(
        &self,
        before: &State,
        action: Move,
        after: &State,
        _viewer: usize,
    ) -> Option<String> {
        let piece = before.board[action.from as usize];
        let actor = before.to_move;
        Some(format!(
            concat!(
                r#"{{"from":"{}","to":"{}","color":"{}","piece":"{}","promoted":{},"scoreGain":{},"active":[{},{},{},{}],"scores":[{},{},{},{}],"end":"{}"}}"#
            ),
            square_name(action.from),
            square_name(action.to),
            color_code(actor),
            piece.kind().letter(),
            after.board[action.to as usize].promoted(),
            after.scores[actor.index()] - before.scores[actor.index()],
            after.is_active(Color::Red),
            after.is_active(Color::Blue),
            after.is_active(Color::Yellow),
            after.is_active(Color::Green),
            after.scores[0],
            after.scores[1],
            after.scores[2],
            after.scores[3],
            end_name(after.end),
        ))
    }

    fn result_text(&self, state: &State, viewer: usize) -> String {
        let best = *state.scores.iter().max().expect("scores");
        let winners: Vec<_> = Color::ALL
            .into_iter()
            .filter(|&color| state.scores[color.index()] == best)
            .map(Color::name)
            .collect();
        let placement = if state.scores[viewer] == best {
            if winners.len() == 1 {
                "win"
            } else {
                "tie for first"
            }
        } else {
            "finish behind the leader"
        };
        format!(
            "You {placement}. Winner{}: {} with {best} points. Final scores — Red {}, Blue {}, Yellow {}, Green {}.",
            if winners.len() == 1 { "" } else { "s" },
            winners.join(", "),
            state.scores[0],
            state.scores[1],
            state.scores[2],
            state.scores[3],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_is_structured_and_complete() {
        let game = FourPlayerChess::default();
        let state = State::standard();
        let json = game.view_data(&state, 0).unwrap();
        assert!(json.contains(r#""size":14"#));
        assert_eq!(json.matches(r#""square""#).count(), 64);
        assert!(json.contains(r#""scores":[0,0,0,0]"#));
    }

    #[test]
    fn parses_multi_digit_ranks() {
        let game = FourPlayerChess::default();
        let mut state = State::standard();
        state.to_move = Color::Blue;
        let action = game.parse_action(&state, "b10-d10").unwrap();
        assert_eq!(game.action_label(&state, action), "b10d10");
    }
}
