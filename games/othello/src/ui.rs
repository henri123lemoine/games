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

fn squares(mut bits: u64) -> impl Iterator<Item = u8> {
    std::iter::from_fn(move || {
        if bits == 0 {
            return None;
        }
        let sq = bits.trailing_zeros() as u8;
        bits &= bits - 1;
        Some(sq)
    })
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

    /// View JSON — the private contract with `web/app/src/frontends/othello`:
    ///
    /// ```json
    /// {"cells": "<64 chars, square 0 (a1) .. 63 (h8), row 1 first:
    ///            '.' empty, 'b' Black (player 0), 'w' White>",
    ///  "turn": 0|1,
    ///  "counts": [black, white],
    ///  "legal": ["c4", ...]}   // placements for the side to move
    /// ```
    fn view_data(&self, state: &Board, _viewer: usize) -> Option<String> {
        let mut cells = String::with_capacity(64);
        for sq in 0..64u8 {
            cells.push(match state.disc_at(sq) {
                Some(0) => 'b',
                Some(_) => 'w',
                None => '.',
            });
        }
        let legal = squares(state.placements())
            .map(|sq| format!(r#""{}""#, square_name(sq)))
            .collect::<Vec<_>>()
            .join(",");
        Some(format!(
            r#"{{"cells":"{cells}","turn":{},"counts":[{},{}],"legal":[{legal}]}}"#,
            state.side_to_move(),
            state.discs(0),
            state.discs(1),
        ))
    }

    /// Transition JSON — what the move changed, for flip animation:
    ///
    /// ```json
    /// {"move": "<square>"|"pass",
    ///  "player": 0|1,
    ///  "placed": <square index>|null,
    ///  "flipped": [<square indices>]}
    /// ```
    fn transition_data(
        &self,
        before: &Board,
        action: Move,
        after: &Board,
        _viewer: usize,
    ) -> Option<String> {
        let player = before.side_to_move();
        match action {
            Move::Pass => Some(format!(
                r#"{{"move":"pass","player":{player},"placed":null,"flipped":[]}}"#
            )),
            Move::Place(sq) => {
                let flipped = squares(before.bb(player ^ 1) & after.bb(player))
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                Some(format!(
                    r#"{{"move":"{}","player":{player},"placed":{sq},"flipped":[{flipped}]}}"#,
                    square_name(sq)
                ))
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use game_core::Game;

    #[test]
    fn view_data_describes_the_start_position() {
        let game = Othello;
        let state = game.initial_state();
        let mut cells = vec![b'.'; 64];
        cells[27] = b'w';
        cells[28] = b'b';
        cells[35] = b'b';
        cells[36] = b'w';
        let expected = format!(
            r#"{{"cells":"{}","turn":0,"counts":[2,2],"legal":["d3","c4","f5","e6"]}}"#,
            String::from_utf8(cells).unwrap()
        );
        assert_eq!(game.view_data(&state, 0).unwrap(), expected);
    }

    #[test]
    fn transition_data_lists_placed_and_flipped() {
        let game = Othello;
        let before = game.initial_state();
        let mut after = before;
        game.apply(&mut after, Move::Place(19));
        assert_eq!(
            game.transition_data(&before, Move::Place(19), &after, 0)
                .unwrap(),
            r#"{"move":"d3","player":0,"placed":19,"flipped":[27]}"#
        );
        assert_eq!(
            game.transition_data(&after, Move::Pass, &before, 0)
                .unwrap(),
            r#"{"move":"pass","player":1,"placed":null,"flipped":[]}"#
        );
    }
}
