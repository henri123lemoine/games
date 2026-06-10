//! The user-facing surface of a game: per-player views and action labels.
//!
//! Anything that implements [`GameUi`] gets the lab's generic terminal client
//! (and, later, the same web service) for free — games never write their own
//! play loop.

use crate::{Game, Turn};

pub trait GameUi: Game {
    /// Stable identifier used by registries/CLIs (e.g. `"chess"`).
    fn id(&self) -> &'static str;

    /// What `player` can see of `state`, rendered as terminal text (must not
    /// leak other players' hidden information).
    fn render(&self, state: &Self::State, player: usize) -> String;

    /// Short human-readable label for an action at `state` (e.g. `"e2e4"`,
    /// `"call LIAR"`, `"draw"`).
    fn action_label(&self, state: &Self::State, action: Self::Action) -> String;

    /// Parse free-form user input into an action, if the game supports textual
    /// moves (e.g. `"e2e4"`). Numeric menu selection always works regardless.
    fn parse_action(&self, _state: &Self::State, _input: &str) -> Option<Self::Action> {
        None
    }

    /// Describe a just-played transition for `viewer` — used to narrate events
    /// the post-state no longer shows (e.g. a Liar's Dice call revealing all
    /// hands before the next round is rolled). Default: nothing extra.
    fn describe_transition(
        &self,
        _before: &Self::State,
        _action: Self::Action,
        _after: &Self::State,
        _viewer: usize,
    ) -> Option<String> {
        None
    }

    /// What `viewer` can see of `state`, as game-private JSON for rich (web)
    /// clients. The schema is a contract between this game and its own
    /// frontend; the shared layers never interpret it. Default: no structured
    /// view (clients fall back to [`GameUi::render`]).
    fn view_data(&self, _state: &Self::State, _viewer: usize) -> Option<String> {
        None
    }

    /// Structured counterpart of [`GameUi::describe_transition`]: game-private
    /// JSON describing a just-played transition for rich clients to animate
    /// from. Default: nothing extra.
    fn transition_data(
        &self,
        _before: &Self::State,
        _action: Self::Action,
        _after: &Self::State,
        _viewer: usize,
    ) -> Option<String> {
        None
    }

    /// One line announcing the result at a terminal state, from `viewer`'s seat.
    fn result_text(&self, state: &Self::State, viewer: usize) -> String {
        debug_assert!(self.is_terminal(state));
        let r = self.returns(state, viewer);
        if r > 0.0 {
            "You win!".into()
        } else if r < 0.0 {
            "You lose.".into()
        } else {
            "Draw.".into()
        }
    }

    /// Whether it is `player`'s turn to act.
    fn is_to_act(&self, state: &Self::State, player: usize) -> bool {
        matches!(self.turn(state), Turn::Player(p) if p == player)
    }
}
