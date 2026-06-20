//! Terminal and web surface for the 1v1 [`Duel`].

use game_core::GameUi;

use crate::duel::{Dir, Duel, DuelAction, DuelState, Outcome};

fn dir_letter(dir: Dir) -> &'static str {
    match dir {
        Dir::Up => "n",
        Dir::Right => "e",
        Dir::Down => "s",
        Dir::Left => "w",
    }
}

fn dir_word(dir: Dir) -> &'static str {
    match dir {
        Dir::Up => "up",
        Dir::Right => "right",
        Dir::Down => "down",
        Dir::Left => "left",
    }
}

fn outcome_word(outcome: Outcome) -> &'static str {
    match outcome {
        Outcome::Ongoing => "ongoing",
        Outcome::Win(0) => "win0",
        Outcome::Win(_) => "win1",
        Outcome::Draw => "draw",
    }
}

fn snake_cells(state: &DuelState, seat: usize) -> String {
    state
        .worm(seat)
        .cells()
        .map(|(x, y)| format!("[{x},{y}]"))
        .collect::<Vec<_>>()
        .join(",")
}

impl GameUi for Duel {
    fn id(&self) -> &'static str {
        "snake"
    }

    fn render(&self, state: &DuelState, viewer: usize) -> String {
        let side = self.side();
        let mut grid = vec![vec!['.'; side]; side];
        if let Some((fx, fy)) = state.food() {
            grid[fy][fx] = '*';
        }
        for (seat, body, head) in [(0usize, 'a', 'A'), (1usize, 'b', 'B')] {
            let worm = state.worm(seat);
            for (x, y) in worm.cells().skip(1) {
                grid[y][x] = body;
            }
            let (hx, hy) = worm.head();
            grid[hy][hx] = head;
        }
        let mut out = format!(
            "step {}/{}\nyou are Snake {}   A:len {} hp {}{}   B:len {} hp {}{}\n",
            state.steps(),
            self.step_cap(),
            if viewer == 1 { "B" } else { "A" },
            state.score(0),
            state.health(0),
            if state.worm(0).alive() { "" } else { " (dead)" },
            state.score(1),
            state.health(1),
            if state.worm(1).alive() { "" } else { " (dead)" },
        );
        let frame = "#".repeat(side + 2);
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

    fn action_label(&self, _state: &DuelState, action: DuelAction) -> String {
        match action {
            DuelAction::Move(d) => dir_word(d).into(),
            DuelAction::Food(c) => {
                let (x, y) = (c as usize % self.side(), c as usize / self.side());
                format!("food at ({x}, {y})")
            }
        }
    }

    fn parse_action(&self, _state: &DuelState, input: &str) -> Option<DuelAction> {
        let dir = match input.trim().to_ascii_lowercase().as_str() {
            "n" | "up" | "u" | "w" => Dir::Up,
            "e" | "right" | "r" | "d" => Dir::Right,
            "s" | "down" => Dir::Down,
            "left" | "l" | "a" => Dir::Left,
            _ => return None,
        };
        Some(DuelAction::Move(dir))
    }

    /// View JSON — the private contract with `web/app/src/frontends/snake`:
    ///
    /// ```json
    /// {"side":20,
    ///  "snakes":[ {"cells":[[x,y],...head first],"dir":"n|e|s|w",
    ///              "alive":true,"score":3,"health":100}, {...} ],
    ///  "food":[x,y]|null,
    ///  "step":0,"cap":400,
    ///  "outcome":"ongoing|win0|win1|draw"}
    /// ```
    ///
    /// `x` grows rightward, `y` downward. `snakes[0]` is seat 0 (Snake A),
    /// `snakes[1]` seat 1 (Snake B). `health` is `0..=100`; a snake reaching
    /// `0` starves to death.
    fn view_data(&self, state: &DuelState, _viewer: usize) -> Option<String> {
        let snake = |seat: usize| {
            let worm = state.worm(seat);
            format!(
                "{{\"cells\":[{}],\"dir\":\"{}\",\"alive\":{},\"score\":{},\"health\":{}}}",
                snake_cells(state, seat),
                dir_letter(worm.heading()),
                worm.alive(),
                state.score(seat),
                worm.health(),
            )
        };
        let food = match state.food() {
            Some((x, y)) => format!("[{x},{y}]"),
            None => "null".into(),
        };
        Some(format!(
            "{{\"side\":{},\"snakes\":[{},{}],\"food\":{food},\"step\":{},\"cap\":{},\
             \"outcome\":\"{}\"}}",
            self.side(),
            snake(0),
            snake(1),
            state.steps(),
            self.step_cap(),
            outcome_word(state.outcome()),
        ))
    }

    fn result_text(&self, state: &DuelState, viewer: usize) -> String {
        let names = ["Snake A", "Snake B"];
        let (a, b) = (state.score(0), state.score(1));
        match state.outcome() {
            Outcome::Win(w) => {
                let by = if state.worm(1 - w).alive() {
                    format!("on score ({a} vs {b})")
                } else {
                    "— the opponent crashed".into()
                };
                if viewer == w {
                    format!("You win {by}.")
                } else if viewer == 1 - w {
                    format!("You lose — {} wins {by}.", names[w])
                } else {
                    format!("{} wins {by}.", names[w])
                }
            }
            Outcome::Draw => format!("Draw ({a} vs {b})."),
            Outcome::Ongoing => {
                debug_assert!(false, "result_text on a non-terminal state");
                String::new()
            }
        }
    }
}
