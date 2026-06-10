//! Terminal view: a framed grid with directional head glyph, body, and food.

use game_core::GameUi;

use crate::{Dir, Snake, SnakeAction, SnakeState, Status};

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
