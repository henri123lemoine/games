//! Type-erased matches: driven seats (the human, plus any seat whose moves
//! the client computes — e.g. the browser's WebGPU bot) against bot seats in
//! any [`GameUi`] game. `AnyMatch` is the uniform surface every client
//! drives — the terminal binary and the wasm web engine alike — and it never
//! knows which game it is running.

use game_core::{Agent, GameUi, Rng, Turn};

/// One applied action, narrated for the match's human viewer, with optional
/// game-private JSON for rich clients to animate from.
pub struct MatchEvent {
    /// Seat that acted.
    pub seat: usize,
    /// The action's bare label (e.g. `"e2e4"`).
    pub label: String,
    /// Narration line as the terminal prints it (e.g. `"Player 2: e2e4"`).
    pub text: String,
    /// Extra transition narration the post-state no longer shows (reveals).
    pub detail: Option<String>,
    /// Game-private transition JSON from [`GameUi::transition_data`].
    pub data: Option<String>,
}

pub trait AnyMatch {
    /// Apply chance moves and then a single bot move; `None` once it is a
    /// driven seat's turn or the game is over. One event per call lets
    /// clients animate move by move.
    fn step(&mut self) -> Option<MatchEvent>;
    /// Apply chance and bot moves until it is a driven seat's turn or the
    /// game ends.
    fn advance(&mut self) -> Vec<MatchEvent> {
        let mut events = Vec::new();
        while let Some(e) = self.step() {
            events.push(e);
        }
        events
    }
    fn is_over(&self) -> bool;
    /// The human's current view of the state, as terminal text.
    fn view(&self) -> String;
    /// The human's view as game-private JSON, when the game provides one.
    fn view_data(&self) -> Option<String>;
    /// Labels of the human's legal actions, menu-ordered.
    fn legal_labels(&self) -> Vec<String>;
    /// Apply input at the driven seat to act — a menu index or game-specific
    /// text (e.g. `e2e4`). Returns the applied move's event, or an error to
    /// re-prompt.
    fn apply_human(&mut self, input: &str) -> Result<MatchEvent, String>;
    /// Result line for the human once `is_over`.
    fn result_text(&self) -> String;
    /// Seat to act, when it is a player's turn (not chance, not terminal).
    fn to_act(&self) -> Option<usize>;
    fn num_seats(&self) -> usize;
    /// The human's seat; `None` when spectating (every seat a bot).
    fn human_seat(&self) -> Option<usize>;
}

/// An `AnyMatch` over a concrete game: the human at `human` seat, and a bot
/// at every seat that has one — a `None` bot elsewhere marks an externally
/// driven seat, whose moves arrive through [`AnyMatch::apply_human`]. A
/// `human` seat beyond the seat count means no seat is the human's; with all
/// seats botted that is spectator mode.
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
        assert!(
            human >= bots.len() || bots[human].is_none(),
            "human seat must have no bot"
        );
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

    fn apply_event(&mut self, actor: usize, index: usize) -> MatchEvent {
        let actions = self.game.legal_actions(&self.state);
        let action = actions[index];
        let before = self.state.clone();
        self.game.apply(&mut self.state, action);
        let label = self.game.action_label(&before, action);
        let who = if actor == self.human {
            "You".to_string()
        } else {
            format!("Player {actor}")
        };
        MatchEvent {
            seat: actor,
            text: format!("{who}: {label}"),
            detail: self
                .game
                .describe_transition(&before, action, &self.state, self.human),
            data: self
                .game
                .transition_data(&before, action, &self.state, self.human),
            label,
        }
    }
}

impl<G: GameUi + 'static> AnyMatch for TypedMatch<G> {
    fn step(&mut self) -> Option<MatchEvent> {
        loop {
            if self.game.is_terminal(&self.state) {
                return None;
            }
            match self.game.turn(&self.state) {
                Turn::Chance => {
                    let outs = self.game.chance_outcomes(&self.state);
                    let i = game_core::rand::sample_outcome(&outs, &mut self.rng);
                    self.game.apply(&mut self.state, outs[i].0);
                }
                Turn::Player(p) if self.bots[p].is_none() => return None,
                Turn::Player(p) => {
                    let bot = self.bots[p].take().expect("checked above");
                    let i = bot.act(&self.game, &self.state, p, &mut self.rng);
                    self.bots[p] = Some(bot);
                    return Some(self.apply_event(p, i));
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

    fn view_data(&self) -> Option<String> {
        self.game.view_data(&self.state, self.human)
    }

    fn legal_labels(&self) -> Vec<String> {
        self.game
            .legal_actions(&self.state)
            .into_iter()
            .map(|a| self.game.action_label(&self.state, a))
            .collect()
    }

    fn apply_human(&mut self, input: &str) -> Result<MatchEvent, String> {
        let actor = match self.to_act() {
            Some(p) if self.bots[p].is_none() => p,
            _ => return Err("no driven seat to act".to_string()),
        };
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
        Ok(self.apply_event(actor, index))
    }

    fn result_text(&self) -> String {
        let n = self.bots.len();
        if self.human < n {
            return self.game.result_text(&self.state, self.human);
        }
        if n == 1 {
            return self.game.result_text(&self.state, 0);
        }
        let winners: Vec<usize> = (0..n)
            .filter(|&p| self.game.returns(&self.state, p) > 0.0)
            .collect();
        match winners.as_slice() {
            [] => "Draw.".to_string(),
            [w] => format!("Player {w} wins."),
            many => format!(
                "Winners: {}.",
                many.iter()
                    .map(|w| format!("Player {w}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }

    fn to_act(&self) -> Option<usize> {
        if self.game.is_terminal(&self.state) {
            return None;
        }
        match self.game.turn(&self.state) {
            Turn::Player(p) => Some(p),
            Turn::Chance => None,
        }
    }

    fn num_seats(&self) -> usize {
        self.bots.len()
    }

    fn human_seat(&self) -> Option<usize> {
        (self.human < self.bots.len()).then_some(self.human)
    }
}
