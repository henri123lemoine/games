//! Terminal/serving surface for chess.

use std::str::FromStr;

use game_core::GameUi;

use crate::board::{Board, Color, Move, Piece};
use crate::{Chess, legal_moves};

fn sq_name(sq: u8) -> String {
    format!("{}{}", (b'a' + sq % 8) as char, (b'1' + sq / 8) as char)
}

fn json_str(v: Option<String>) -> String {
    match v {
        Some(s) => format!("\"{s}\""),
        None => "null".into(),
    }
}

fn piece_char(c: Color, p: Piece) -> char {
    let ch = match p {
        Piece::Pawn => 'p',
        Piece::Knight => 'n',
        Piece::Bishop => 'b',
        Piece::Rook => 'r',
        Piece::Queen => 'q',
        Piece::King => 'k',
    };
    match c {
        Color::White => ch.to_ascii_uppercase(),
        Color::Black => ch,
    }
}

impl GameUi for Chess {
    fn id(&self) -> &'static str {
        "chess"
    }

    fn render(&self, state: &Board, player: usize) -> String {
        let mut out = String::new();
        for rank in (0..8).rev() {
            out.push_str(&format!("{}  ", rank + 1));
            for file in 0..8 {
                match state.squares[rank * 8 + file] {
                    Some((c, p)) => out.push(piece_char(c, p)),
                    None => out.push('.'),
                }
                out.push(' ');
            }
            out.push('\n');
        }
        out.push_str("   a b c d e f g h\n");
        let you = if player == 0 { "White" } else { "Black" };
        let stm = if state.stm == Color::White {
            "White"
        } else {
            "Black"
        };
        out.push_str(&format!(
            "You are {you}. {stm} to move{}.",
            if state.in_check(state.stm) {
                " (in check)"
            } else {
                ""
            }
        ));
        out
    }

    fn action_label(&self, _state: &Board, action: Move) -> String {
        action.to_string()
    }

    fn parse_action(&self, state: &Board, input: &str) -> Option<Move> {
        let mv = Move::from_str(input.trim()).ok()?;
        legal_moves(state)
            .into_iter()
            .find(|m| m.to_string() == mv.to_string())
    }

    fn describe_transition(
        &self,
        _before: &Board,
        _action: Move,
        after: &Board,
        _viewer: usize,
    ) -> Option<String> {
        if after.in_check(after.stm) && !legal_moves(after).is_empty() {
            Some("check!".into())
        } else {
            None
        }
    }

    /// Web view schema — a private contract with
    /// `web/app/src/frontends/chess` (chess is perfect information, so the
    /// viewer is irrelevant):
    ///
    /// ```json
    /// {"board": "<64 chars, rank 8 first, file a first; '.' empty,
    ///            PNBRQK white / pnbrqk black>",
    ///  "turn": "w" | "b",
    ///  "check": bool}
    /// ```
    fn view_data(&self, state: &Board, _viewer: usize) -> Option<String> {
        let mut board = String::with_capacity(64);
        for rank in (0..8).rev() {
            for file in 0..8 {
                match state.squares[rank * 8 + file] {
                    Some((c, p)) => board.push(piece_char(c, p)),
                    None => board.push('.'),
                }
            }
        }
        let turn = if state.stm == Color::White { "w" } else { "b" };
        Some(format!(
            r#"{{"board":"{board}","turn":"{turn}","check":{}}}"#,
            state.in_check(state.stm)
        ))
    }

    /// Transition schema (same contract; squares are coordinates like `"e4"`,
    /// pieces use the `view_data` letter convention, absent values are
    /// `null`):
    ///
    /// ```json
    /// {"from": "e2", "to": "e4", "piece": "P",
    ///  "captured": "p" | null, "capturedSquare": "d5" | null,
    ///  "promo": "Q" | null,
    ///  "castleRookFrom": "h1" | null, "castleRookTo": "f1" | null,
    ///  "check": bool, "mate": bool}
    /// ```
    fn transition_data(
        &self,
        before: &Board,
        action: Move,
        after: &Board,
        _viewer: usize,
    ) -> Option<String> {
        let (color, piece) = before.squares[action.from as usize]?;
        let en_passant = piece == Piece::Pawn
            && before.ep == Some(action.to)
            && action.from % 8 != action.to % 8;
        let captured_sq = if en_passant {
            Some(match color {
                Color::White => action.to - 8,
                Color::Black => action.to + 8,
            })
        } else {
            before.squares[action.to as usize].map(|_| action.to)
        };
        let captured = captured_sq
            .and_then(|sq| before.squares[sq as usize])
            .map(|(c, p)| piece_char(c, p).to_string());
        let (rook_from, rook_to) = if piece == Piece::King && action.to.abs_diff(action.from) == 2 {
            if action.to > action.from {
                (Some(action.from + 3), Some(action.from + 1))
            } else {
                (Some(action.from - 4), Some(action.from - 1))
            }
        } else {
            (None, None)
        };
        let check = after.in_check(after.stm);
        let mate = check && legal_moves(after).is_empty();
        Some(format!(
            concat!(
                r#"{{"from":"{}","to":"{}","piece":"{}","captured":{},"capturedSquare":{},"#,
                r#""promo":{},"castleRookFrom":{},"castleRookTo":{},"check":{check},"mate":{mate}}}"#
            ),
            sq_name(action.from),
            sq_name(action.to),
            piece_char(color, piece),
            json_str(captured),
            json_str(captured_sq.map(sq_name)),
            json_str(action.promo.map(|p| piece_char(color, p).to_string())),
            json_str(rook_from.map(sq_name)),
            json_str(rook_to.map(sq_name)),
            check = check,
            mate = mate,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mv(s: &str) -> Move {
        s.parse().unwrap()
    }

    fn played(before: &Board, m: Move) -> Board {
        let mut after = before.clone();
        after.apply(m);
        after
    }

    #[test]
    fn view_data_start_position() {
        let board = format!("rnbqkbnrpppppppp{}PPPPPPPPRNBQKBNR", ".".repeat(32));
        assert_eq!(
            Chess.view_data(&Board::start(), 0).unwrap(),
            format!(r#"{{"board":"{board}","turn":"w","check":false}}"#)
        );
    }

    #[test]
    fn transition_data_plain_push() {
        let before = Board::start();
        let after = played(&before, mv("e2e4"));
        assert_eq!(
            Chess
                .transition_data(&before, mv("e2e4"), &after, 0)
                .unwrap(),
            r#"{"from":"e2","to":"e4","piece":"P","captured":null,"capturedSquare":null,"promo":null,"castleRookFrom":null,"castleRookTo":null,"check":false,"mate":false}"#
        );
    }

    #[test]
    fn transition_data_castle() {
        let before = Board::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
        let after = played(&before, mv("e1g1"));
        let json = Chess
            .transition_data(&before, mv("e1g1"), &after, 0)
            .unwrap();
        assert!(
            json.contains(r#""castleRookFrom":"h1","castleRookTo":"f1""#),
            "{json}"
        );
    }

    #[test]
    fn transition_data_en_passant() {
        let before = Board::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1").unwrap();
        let after = played(&before, mv("e5d6"));
        let json = Chess
            .transition_data(&before, mv("e5d6"), &after, 0)
            .unwrap();
        assert!(
            json.contains(r#""captured":"p","capturedSquare":"d5""#),
            "{json}"
        );
    }

    #[test]
    fn transition_data_promotion_with_mate() {
        let before = Board::from_fen("7k/4P1pp/8/8/8/8/8/K7 w - - 0 1").unwrap();
        let m = mv("e7e8q");
        let after = played(&before, m);
        let json = Chess.transition_data(&before, m, &after, 0).unwrap();
        assert!(json.contains(r#""promo":"Q""#), "{json}");
        assert!(json.contains(r#""check":true,"mate":true"#), "{json}");
    }
}
