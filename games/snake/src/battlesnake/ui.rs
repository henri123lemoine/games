//! Terminal and structured UI for canonical simultaneous Battlesnake.

use game_core::SimultaneousGameUi;

use super::{BattleSnake, Battlesnake, BoardState, Direction, Elimination, SIDE, bits, xy};

fn direction_word(direction: Direction) -> &'static str {
    match direction {
        Direction::Up => "up",
        Direction::Right => "right",
        Direction::Down => "down",
        Direction::Left => "left",
    }
}

fn direction_letter(direction: Direction) -> &'static str {
    match direction {
        Direction::Up => "n",
        Direction::Right => "e",
        Direction::Down => "s",
        Direction::Left => "w",
    }
}

fn elimination_word(elimination: Elimination) -> &'static str {
    match elimination {
        Elimination::Alive => "alive",
        Elimination::OutOfHealth => "out-of-health",
        Elimination::Hazard => "hazard",
        Elimination::OutOfBounds => "out-of-bounds",
        Elimination::SelfCollision => "self-collision",
        Elimination::BodyCollision => "body-collision",
        Elimination::HeadToHead => "head-to-head",
    }
}

fn cells_json(snake: &BattleSnake) -> String {
    snake
        .cells()
        .map(|cell| {
            let (x, y) = xy(cell);
            format!("[{x},{y}]")
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn mask_json(mask: u128) -> String {
    bits(mask)
        .map(|cell| {
            let (x, y) = xy(cell);
            format!("[{x},{y}]")
        })
        .collect::<Vec<_>>()
        .join(",")
}

impl<const N: usize> SimultaneousGameUi for Battlesnake<N> {
    fn id(&self) -> &'static str {
        "snake"
    }

    fn render(&self, state: &BoardState<N>, viewer: usize) -> String {
        let mut grid = [['.'; SIDE]; SIDE];
        for cell in bits(state.hazards()) {
            let (x, y) = xy(cell);
            grid[y as usize][x as usize] = '~';
        }
        for cell in bits(state.food()) {
            let (x, y) = xy(cell);
            grid[y as usize][x as usize] = '*';
        }
        for (player, snake) in state.snakes().iter().enumerate() {
            if !snake.is_alive() {
                continue;
            }
            let body = char::from(b'0' + player as u8);
            for cell in snake.cells().skip(1) {
                let (x, y) = xy(cell);
                grid[y as usize][x as usize] = body;
            }
            let (x, y) = xy(snake.head());
            grid[y as usize][x as usize] = char::from(b'A' + player as u8);
        }

        let role = if viewer < N {
            format!("you are Snake {}", char::from(b'A' + viewer as u8))
        } else {
            "spectating".to_string()
        };
        let status = state
            .snakes()
            .iter()
            .enumerate()
            .map(|(player, snake)| {
                format!(
                    "{}:len {} hp {} {}",
                    char::from(b'A' + player as u8),
                    snake.len(),
                    snake.health(),
                    elimination_word(snake.elimination())
                )
            })
            .collect::<Vec<_>>()
            .join("   ");
        let mut out = format!("turn {}  {role}\n{status}\n", state.turn_number());
        let frame = "#".repeat(SIDE + 2);
        out.push_str(&frame);
        out.push('\n');
        for y in (0..SIDE).rev() {
            out.push('#');
            out.extend(grid[y]);
            out.push_str("#\n");
        }
        out.push_str(&frame);
        out
    }

    fn action_label(&self, _state: &BoardState<N>, _player: usize, action: Direction) -> String {
        direction_word(action).into()
    }

    fn parse_action(
        &self,
        _state: &BoardState<N>,
        _player: usize,
        input: &str,
    ) -> Option<Direction> {
        match input.trim().to_ascii_lowercase().as_str() {
            "n" | "up" | "u" | "w" => Some(Direction::Up),
            "e" | "right" | "r" | "d" => Some(Direction::Right),
            "s" | "down" => Some(Direction::Down),
            "left" | "l" | "a" => Some(Direction::Left),
            _ => None,
        }
    }

    fn describe_joint_transition(
        &self,
        before: &BoardState<N>,
        _actions: &[Direction],
        after: &BoardState<N>,
        _viewer: usize,
    ) -> Option<String> {
        let deaths = (0..N)
            .filter(|&player| before.snake(player).is_alive() && !after.snake(player).is_alive())
            .map(|player| {
                format!(
                    "Snake {}: {}",
                    char::from(b'A' + player as u8),
                    elimination_word(after.snake(player).elimination())
                )
            })
            .collect::<Vec<_>>();
        (!deaths.is_empty()).then(|| deaths.join("; "))
    }

    fn view_data(&self, state: &BoardState<N>, _viewer: usize) -> Option<String> {
        let snakes = state
            .snakes()
            .iter()
            .map(|snake| {
                format!(
                    "{{\"cells\":[{}],\"dir\":\"{}\",\"alive\":{},\"score\":{},\"health\":{},\"elimination\":\"{}\"}}",
                    cells_json(snake),
                    direction_letter(snake.heading()),
                    snake.is_alive(),
                    snake.len(),
                    snake.health().max(0),
                    elimination_word(snake.elimination())
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let outcome = if state.alive_count() > 1 {
            "ongoing".to_string()
        } else if let Some(winner) = state.snakes().iter().position(|snake| snake.is_alive()) {
            format!("win{winner}")
        } else {
            "draw".to_string()
        };
        Some(format!(
            "{{\"side\":{SIDE},\"coordinateSystem\":\"battlesnake\",\"simultaneous\":true,\"snakes\":[{snakes}],\"food\":[{}],\"hazards\":[{}],\"turn\":{},\"outcome\":\"{outcome}\"}}",
            mask_json(state.food()),
            mask_json(state.hazards()),
            state.turn_number()
        ))
    }

    fn transition_data(
        &self,
        _before: &BoardState<N>,
        actions: &[Direction],
        after: &BoardState<N>,
        _viewer: usize,
    ) -> Option<String> {
        let moves = actions
            .iter()
            .map(|action| format!("\"{}\"", direction_word(*action)))
            .collect::<Vec<_>>()
            .join(",");
        let eliminations = after
            .snakes()
            .iter()
            .map(|snake| format!("\"{}\"", elimination_word(snake.elimination())))
            .collect::<Vec<_>>()
            .join(",");
        Some(format!(
            "{{\"moves\":[{moves}],\"eliminations\":[{eliminations}],\"turn\":{}}}",
            after.turn_number()
        ))
    }

    fn result_text(&self, state: &BoardState<N>, viewer: usize) -> String {
        let winner = state.snakes().iter().position(|snake| snake.is_alive());
        match (viewer < N, winner) {
            (_, None) => "Draw — all snakes were eliminated.".into(),
            (true, Some(winner)) if winner == viewer => "You win!".into(),
            (true, Some(winner)) => {
                format!("You lose — Snake {} wins.", char::from(b'A' + winner as u8))
            }
            (false, Some(winner)) => {
                format!("Snake {} wins.", char::from(b'A' + winner as u8))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use game_core::{SimultaneousGame, SimultaneousGameUi};

    use super::*;

    #[test]
    fn structured_view_declares_simultaneous_canonical_coordinates() {
        let game = Battlesnake::<2>::standard();
        let data = game
            .view_data(&game.initial_state(), 0)
            .expect("structured view");
        assert!(data.contains("\"side\":11"));
        assert!(data.contains("\"coordinateSystem\":\"battlesnake\""));
        assert!(data.contains("\"simultaneous\":true"));
        assert!(!data.contains("pending"));
    }

    #[test]
    fn food_mask_json_is_not_confused_by_body_cells() {
        let game = Battlesnake::<2>::standard();
        let state = game.initial_state();
        for snake in state.snakes() {
            assert_eq!(state.food() & super::super::bit(snake.head()), 0);
        }
    }
}
