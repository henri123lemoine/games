//! Terminal/serving surface for Othello. Black is `X`, White is `O`; squares
//! are named file+row (`"d3"`), with row 1 at the top as in game transcripts.

use game_core::GameUi;

use crate::{Board, Move, Othello};

fn square_name(sq: u8) -> String {
    let file = (b'a' + sq % 8) as char;
    let row = sq / 8 + 1;
    format!("{file}{row}")
}

fn parse_square(input: &str) -> Option<u8> {
    let mut chars = input.chars();
    let file = chars.next()?.to_ascii_lowercase();
    let row = chars.next()?;
    if chars.next().is_some() || !('a'..='h').contains(&file) || !('1'..='8').contains(&row) {
        return None;
    }
    Some((row as u8 - b'1') * 8 + (file as u8 - b'a'))
}

fn disc_char(player: usize) -> char {
    if player == 0 { 'X' } else { 'O' }
}

impl GameUi for Othello {
    fn id(&self) -> &'static str {
        "othello"
    }

    fn render(&self, state: &Board, player: usize) -> String {
        let legal = state.placements();
        let mut out = String::from("   a b c d e f g h\n");
        for row in 0..8u8 {
            out.push_str(&format!(" {} ", row + 1));
            for col in 0..8u8 {
                let sq = row * 8 + col;
                match state.disc_at(sq) {
                    Some(p) => out.push(disc_char(p)),
                    None if legal & (1 << sq) != 0 => out.push('*'),
                    None => out.push('.'),
                }
                out.push(' ');
            }
            out.push('\n');
        }
        out.push_str(&format!(
            "You are {} ({}). {} to move — X {}, O {}.",
            disc_char(player),
            if player == 0 { "Black" } else { "White" },
            disc_char(state.side_to_move()),
            state.discs(0),
            state.discs(1),
        ));
        out
    }

    fn action_label(&self, _state: &Board, action: Move) -> String {
        match action {
            Move::Place(sq) => square_name(sq),
            Move::Pass => "pass".into(),
        }
    }

    fn parse_action(&self, state: &Board, input: &str) -> Option<Move> {
        let input = input.trim();
        if input.eq_ignore_ascii_case("pass") {
            return (state.placements() == 0).then_some(Move::Pass);
        }
        let sq = parse_square(input)?;
        (state.placements() & (1 << sq) != 0).then_some(Move::Place(sq))
    }

    fn describe_transition(
        &self,
        before: &Board,
        action: Move,
        _after: &Board,
        _viewer: usize,
    ) -> Option<String> {
        match action {
            Move::Pass => Some(format!(
                "{} has no legal move and passes.",
                disc_char(before.side_to_move())
            )),
            Move::Place(_) => None,
        }
    }

    fn result_text(&self, state: &Board, viewer: usize) -> String {
        let (mine, theirs) = (state.discs(viewer), state.discs(viewer ^ 1));
        match mine.cmp(&theirs) {
            std::cmp::Ordering::Greater => format!("You win, {mine}-{theirs}."),
            std::cmp::Ordering::Less => format!("You lose, {mine}-{theirs}."),
            std::cmp::Ordering::Equal => format!("Draw, {mine}-{theirs}."),
        }
    }
}
