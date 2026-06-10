//! Terminal view: a framed grid with directional head glyph, body, and food.

use game_core::GameUi;

use crate::{Dir, Snake, SnakeAction, SnakeState, Status};

fn dir_letter(dir: Dir) -> &'static str {
    match dir {
        Dir::Up => "n",
        Dir::Right => "e",
        Dir::Down => "s",
        Dir::Left => "w",
    }
}

fn status_word(status: Status) -> &'static str {
    match status {
        Status::Alive => "alive",
        Status::Crashed => "crashed",
        Status::Starved => "starved",
        Status::Won => "won",
    }
}

impl GameUi for Snake {
    fn id(&self) -> &'static str {
        "snake"
    }

    fn render(&self, state: &SnakeState, _player: usize) -> String {
        let mut grid = vec![vec!['.'; self.w]; self.h];
        if let Some((fx, fy)) = state.food {
            grid[fy as usize][fx as usize] = '*';
        }
        for &(x, y) in state.body.iter().skip(1) {
            grid[y as usize][x as usize] = 'o';
        }
        let (hx, hy) = state.body[0];
        grid[hy as usize][hx as usize] = match state.heading {
            Dir::Up => '^',
            Dir::Right => '>',
            Dir::Down => 'v',
            Dir::Left => '<',
        };
        let mut out = format!(
            "length {}/{}   hunger {}/{}\n",
            state.len(),
            self.area(),
            state.hunger(),
            self.starvation_cap()
        );
        let frame = "#".repeat(self.w + 2);
        out.push_str(&frame);
        out.push('\n');
        for row in grid {
            out.push('#');
            out.extend(row);
            out.push_str("#\n");
        }
        out.push_str(&frame);
        out
    }

    fn action_label(&self, _state: &SnakeState, action: SnakeAction) -> String {
        match action {
            SnakeAction::TurnLeft => "left".into(),
            SnakeAction::Straight => "straight".into(),
            SnakeAction::TurnRight => "right".into(),
            SnakeAction::Food(c) => {
                let (x, y) = self.cell_xy(c);
                format!("food at ({x}, {y})")
            }
        }
    }

    /// JSON view for the web frontend (`web/app/src/frontends/snake`):
    ///
    /// ```json
    /// {"width":10,"height":10,"snake":[[5,5],[4,5],[3,5]],"food":[7,2],
    ///  "dir":"e","score":3,"status":"alive"}
    /// ```
    ///
    /// `snake` lists `[x, y]` cells head first (`x` rightward, `y` downward
    /// from the top-left); `food` is `null` while a spawn is pending; `dir` is
    /// the current heading (`n|e|s|w`); `score` is the snake length; `status`
    /// ∈ `alive|crashed|starved|won`.
    fn view_data(&self, state: &SnakeState, _viewer: usize) -> Option<String> {
        let snake = state
            .body
            .iter()
            .map(|&(x, y)| format!("[{x},{y}]"))
            .collect::<Vec<_>>()
            .join(",");
        let food = match state.food {
            Some((x, y)) => format!("[{x},{y}]"),
            None => "null".into(),
        };
        Some(format!(
            "{{\"width\":{},\"height\":{},\"snake\":[{snake}],\"food\":{food},\
             \"dir\":\"{}\",\"score\":{},\"status\":\"{}\"}}",
            self.w,
            self.h,
            dir_letter(state.heading),
            state.len(),
            status_word(state.status)
        ))
    }

    fn parse_action(&self, _state: &SnakeState, input: &str) -> Option<SnakeAction> {
        match input.trim().to_ascii_lowercase().as_str() {
            "l" | "left" => Some(SnakeAction::TurnLeft),
            "s" | "straight" => Some(SnakeAction::Straight),
            "r" | "right" => Some(SnakeAction::TurnRight),
            _ => None,
        }
    }

    fn result_text(&self, state: &SnakeState, _viewer: usize) -> String {
        let (len, area) = (state.len(), self.area());
        match state.status {
            Status::Won => format!("Board full — you win! Final length {len}/{area}."),
            Status::Starved => format!(
                "Starved: {} moves without eating. Final length {len}/{area}.",
                self.starvation_cap()
            ),
            Status::Crashed => format!("Crashed! Final length {len}/{area}."),
            Status::Alive => {
                debug_assert!(false, "result_text on a non-terminal state");
                String::new()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_core::Game;

    #[test]
    fn view_data_schema() {
        let game = Snake::new(10, 10);
        let mut s = game.initial_state();
        assert_eq!(
            game.view_data(&s, 0).unwrap(),
            "{\"width\":10,\"height\":10,\"snake\":[[5,5],[4,5],[3,5]],\
             \"food\":null,\"dir\":\"e\",\"score\":3,\"status\":\"alive\"}"
        );

        game.apply(&mut s, SnakeAction::Food(27));
        game.apply(&mut s, SnakeAction::TurnLeft);
        assert_eq!(
            game.view_data(&s, 0).unwrap(),
            "{\"width\":10,\"height\":10,\"snake\":[[5,4],[5,5],[4,5]],\
             \"food\":[7,2],\"dir\":\"n\",\"score\":3,\"status\":\"alive\"}"
        );
    }

    #[test]
    fn view_data_reports_terminal_status() {
        let game = Snake::new(10, 10);

        let mut circling = game.initial_state();
        game.apply(&mut circling, SnakeAction::Food(0));
        while !game.is_terminal(&circling) {
            game.apply(&mut circling, SnakeAction::TurnRight);
        }
        let json = game.view_data(&circling, 0).unwrap();
        assert!(json.contains("\"status\":\"starved\""), "{json}");

        let mut beeline = game.initial_state();
        game.apply(&mut beeline, SnakeAction::Food(0));
        while !game.is_terminal(&beeline) {
            game.apply(&mut beeline, SnakeAction::Straight);
        }
        let json = game.view_data(&beeline, 0).unwrap();
        assert!(json.contains("\"status\":\"crashed\""), "{json}");
    }
}
