//! Type-erased matches: one human seat against bot seats in any [`GameUi`]
//! game. `AnyMatch` is the uniform surface the terminal client (and, later, a
//! web service) drives — it never knows which game it is running.

use game_core::{Agent, GameUi, Rng, Turn};

pub trait AnyMatch {
    /// Apply chance and bot moves until it is the human's turn or the game
    /// ends; returns narration lines (bot actions, revealed transitions).
    fn advance(&mut self) -> Vec<String>;
    fn is_over(&self) -> bool;
    /// The human's current view of the state.
    fn view(&self) -> String;
    /// Labels of the human's legal actions, menu-ordered.
    fn legal_labels(&self) -> Vec<String>;
    /// Apply human input — a menu index or game-specific text (e.g. `e2e4`).
    /// Returns narration for the applied move, or an error to re-prompt.
    fn apply_human(&mut self, input: &str) -> Result<Vec<String>, String>;
    /// Result line for the human once `is_over`.
    fn result_text(&self) -> String;
}

/// An `AnyMatch` over a concrete game: the human at `human` seat, a bot
/// everywhere else.
pub struct TypedMatch<G: GameUi> {
    game: G,
    state: G::State,
    bots: Vec<Option<Box<dyn Agent<G>>>>,
    human: usize,
    rng: Rng,
}

impl<G: GameUi + 'static> TypedMatch<G> {
    pub fn new(game: G, bots: Vec<Option<Box<dyn Agent<G>>>>, human: usize, seed: u64) -> Self {
        assert_eq!(bots.len(), game.num_players());
        assert!(bots[human].is_none(), "human seat must have no bot");
        let state = game.initial_state();
        Self {
            game,
            state,
            bots,
            human,
            rng: Rng::new(seed),
        }
    }

    pub fn boxed(self) -> Box<dyn AnyMatch> {
        Box::new(self)
    }

    fn apply_with_narration(&mut self, actor: usize, index: usize) -> Vec<String> {
        let actions = self.game.legal_actions(&self.state);
        let action = actions[index];
        let before = self.state.clone();
        self.game.apply(&mut self.state, action);
        let mut lines = Vec::new();
        let who = if actor == self.human {
            "You".to_string()
        } else {
            format!("Player {actor}")
        };
        lines.push(format!(
            "{who}: {}",
            self.game.action_label(&before, action)
        ));
        if let Some(t) = self
            .game
            .describe_transition(&before, action, &self.state, self.human)
        {
            lines.push(t);
        }
        lines
    }
}

impl<G: GameUi + 'static> AnyMatch for TypedMatch<G> {
    fn advance(&mut self) -> Vec<String> {
        let mut lines = Vec::new();
        loop {
            if self.game.is_terminal(&self.state) {
                return lines;
            }
            match self.game.turn(&self.state) {
                Turn::Chance => {
                    let outs = self.game.chance_outcomes(&self.state);
                    let r = self.rng.unit();
                    let mut acc = 0.0;
                    let mut chosen = outs[outs.len() - 1].0;
                    for (a, p) in &outs {
                        acc += *p;
                        if r < acc {
                            chosen = *a;
                            break;
                        }
                    }
                    self.game.apply(&mut self.state, chosen);
                }
                Turn::Player(p) if p == self.human => return lines,
                Turn::Player(p) => {
                    let r = self.rng.unit();
                    let i = self.bots[p]
                        .as_ref()
                        .expect("non-human seat has a bot")
                        .act(&self.game, &self.state, p, r);
                    lines.extend(self.apply_with_narration(p, i));
                }
            }
        }
    }

    fn is_over(&self) -> bool {
        self.game.is_terminal(&self.state)
    }

    fn view(&self) -> String {
        self.game.render(&self.state, self.human)
    }

    fn legal_labels(&self) -> Vec<String> {
        self.game
            .legal_actions(&self.state)
            .into_iter()
            .map(|a| self.game.action_label(&self.state, a))
            .collect()
    }

    fn apply_human(&mut self, input: &str) -> Result<Vec<String>, String> {
        let actions = self.game.legal_actions(&self.state);
        let index = if let Ok(i) = input.trim().parse::<usize>() {
            if i >= actions.len() {
                return Err(format!("{i} is out of range (0-{})", actions.len() - 1));
            }
            i
        } else if let Some(parsed) = self.game.parse_action(&self.state, input) {
            let label = self.game.action_label(&self.state, parsed);
            actions
                .iter()
                .position(|&a| self.game.action_label(&self.state, a) == label)
                .ok_or_else(|| format!("'{}' is not legal here", input.trim()))?
        } else {
            return Err(format!("could not understand '{}'", input.trim()));
        };
        Ok(self.apply_with_narration(self.human, index))
    }

    fn result_text(&self) -> String {
        self.game.result_text(&self.state, self.human)
    }
}
