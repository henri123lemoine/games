//! Terminal/serving surface for Pente.

use game_core::{Game, GameUi};

use crate::{Pente, PenteAction, PenteState, col_letter};

fn player_name(p: usize) -> &'static str {
    if p == 0 { "Black (X)" } else { "White (O)" }
}

impl GameUi for Pente {
    fn id(&self) -> &'static str {
        "pente"
    }

    fn render(&self, state: &PenteState, player: usize) -> String {
        let mut out = String::new();
        for r in (0..self.size()).rev() {
            out.push_str(&format!("{:>2}", r + 1));
            for c in 0..self.size() {
                out.push_str(match state.stone(r * self.size() + c) {
                    Some(0) => "  X",
                    Some(_) => "  O",
                    None => "  .",
                });
            }
            out.push('\n');
        }
        out.push_str("  ");
        for c in 0..self.size() {
            out.push_str("  ");
            out.push(col_letter(c));
        }
        out.push('\n');
        let pairs = state.pairs();
        out.push_str(&format!(
            "Captured pairs: Black {}/5, White {}/5\n",
            pairs[0], pairs[1]
        ));
        out.push_str(&format!("You are {}.", player_name(player)));
        if !self.is_terminal(state) {
            out.push_str(&format!(" {} to move.", player_name(state.to_move())));
            if state.moves() == 0 {
                out.push_str(" (first move must be the center)");
            }
        }
        out
    }

    fn action_label(&self, _state: &PenteState, action: PenteAction) -> String {
        let p = action.0 as usize;
        format!("{}{}", col_letter(p % self.size()), p / self.size() + 1)
    }

    fn parse_action(&self, state: &PenteState, input: &str) -> Option<PenteAction> {
        let text = input.trim().to_ascii_lowercase();
        let action = PenteAction(self.point(&text)?);
        self.legal_actions(state).into_iter().find(|&a| a == action)
    }

    fn describe_transition(
        &self,
        before: &PenteState,
        _action: PenteAction,
        after: &PenteState,
        _viewer: usize,
    ) -> Option<String> {
        let mover = before.to_move();
        let took = after.pairs()[mover] - before.pairs()[mover];
        (took > 0).then(|| format!("captures {took} pair{}", if took == 1 { "" } else { "s" }))
    }

    /// Web view schema: `{"size":N,"cells":"<N*N chars b/w/.>","turn":0|1,
    /// "pairs":[b,w],"last":idx|null,"winner":0|1|null}`. `cells` is indexed
    /// `row * size + col` with row 0 = board row 1.
    fn view_data(&self, state: &PenteState, _viewer: usize) -> Option<String> {
        let cells: String = (0..self.size() * self.size())
            .map(|p| match state.stone(p) {
                Some(0) => 'b',
                Some(_) => 'w',
                None => '.',
            })
            .collect();
        let pairs = state.pairs();
        let last = state.last_move().map_or("null".into(), |p| p.to_string());
        let winner = if self.is_terminal(state) {
            match state.winner() {
                Some(w) => w.to_string(),
                None => "null".into(),
            }
        } else {
            "null".into()
        };
        Some(format!(
            r#"{{"size":{},"cells":"{cells}","turn":{},"pairs":[{},{}],"last":{last},"winner":{winner}}}"#,
            self.size(),
            state.to_move(),
            pairs[0],
            pairs[1],
        ))
    }

    /// Web transition schema: `{"move":"g7","seat":0|1,"point":idx}` plus
    /// `"captured"`: the board indices of stones removed by the custodial
    /// capture (empty when the move captured nothing).
    fn transition_data(
        &self,
        before: &PenteState,
        action: PenteAction,
        after: &PenteState,
        _viewer: usize,
    ) -> Option<String> {
        let seat = before.to_move();
        let coord = self.action_label(before, action);
        let captured = (0..self.size() * self.size())
            .filter(|&q| before.stone(q) == Some(seat ^ 1) && after.stone(q).is_none())
            .map(|q| q.to_string())
            .collect::<Vec<_>>()
            .join(",");
        Some(format!(
            r#"{{"move":"{coord}","seat":{seat},"point":{},"captured":[{captured}]}}"#,
            action.0
        ))
    }

    fn result_text(&self, state: &PenteState, viewer: usize) -> String {
        let pairs = state.pairs();
        let verdict = match state.winner() {
            Some(w) if w == viewer => "You win!",
            Some(_) => "You lose.",
            None => "A draw.",
        };
        let how = match state.winner() {
            Some(w) if pairs[w] >= crate::PAIRS_TO_WIN => format!(" ({} captured pairs)", pairs[w]),
            Some(_) => " (five in a row)".to_string(),
            None => String::new(),
        };
        format!("{verdict}{how}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_and_parse_roundtrip() {
        let g = Pente::new(13);
        let mut s = g.initial_state();
        g.apply(&mut s, PenteAction(g.center()));
        // g7 is the center on 13x13; after it, a nearby legal point parses back.
        let a = g.parse_action(&s, "h8").unwrap();
        assert_eq!(g.action_label(&s, a), "h8");
        assert!(g.parse_action(&s, "g7").is_none(), "center is occupied");
    }

    #[test]
    fn view_and_transition_json_report_capture() {
        let g = Pente::new(5);
        // X O O . with Black to flank at d3 (index for row3 col d).
        let mut s = g.parse_state(
            &[
                ". . . . .",
                ". . . . .",
                "X O O . .",
                ". . . . .",
                ". . . . .",
            ],
            0,
            [0, 0],
        );
        s.moves = 10;
        let before = s.clone();
        let flank = PenteAction(g.point("d3").unwrap());
        g.apply(&mut s, flank);
        let p = g.point("d3").unwrap();
        let b = g.point("b3").unwrap();
        let c = g.point("c3").unwrap();
        assert_eq!(
            g.view_data(&s, 0).unwrap(),
            format!(
                r#"{{"size":5,"cells":"{}","turn":1,"pairs":[1,0],"last":{p},"winner":null}}"#,
                cells_string(&g, &s)
            )
        );
        assert_eq!(
            g.transition_data(&before, flank, &s, 0).unwrap(),
            format!(r#"{{"move":"d3","seat":0,"point":{p},"captured":[{b},{c}]}}"#)
        );
    }

    fn cells_string(g: &Pente, s: &PenteState) -> String {
        (0..g.size() * g.size())
            .map(|p| match s.stone(p) {
                Some(0) => 'b',
                Some(_) => 'w',
                None => '.',
            })
            .collect()
    }
}
