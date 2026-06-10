//! Terminal/serving surface for chess.

use std::str::FromStr;

use game_core::GameUi;

use crate::board::{Board, Color, Move, Piece};
use crate::{Chess, legal_moves};

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
}
