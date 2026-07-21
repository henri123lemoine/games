//! Type-erased matches: driven seats (the human, plus any seat whose moves
//! the client computes — e.g. the browser's WebGPU bot) against bot seats in
//! any [`GameUi`] game. `AnyMatch` is the uniform surface every client
//! drives — the terminal binary and the wasm web engine alike — and it never
//! knows which game it is running.

use game_core::{
    Agent, GameUi, Rng, SimultaneousAgent, SimultaneousGameUi, SimultaneousTurn, Turn,
};

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
    /// Prepare any bot work that is independent of the driven player's next
    /// action. Turn-based games inherit the no-op; simultaneous games can use
    /// this to think from the shared pre-action state before human input lands.
    fn prepare(&mut self) {}
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

/// An `AnyMatch` over a concrete game: the human at `human` seat (`None` to
/// spectate), and a bot at every seat that has one — a `None` bot elsewhere
/// marks an externally driven seat, whose moves arrive through
/// [`AnyMatch::apply_human`].
pub struct TypedMatch<G: GameUi> {
    game: G,
    state: G::State,
    bots: Vec<Option<Box<dyn Agent<G>>>>,
    human: Option<usize>,
    rng: Rng,
}

impl<G: GameUi + 'static> TypedMatch<G> {
    pub fn new(
        game: G,
        bots: Vec<Option<Box<dyn Agent<G>>>>,
        human: Option<usize>,
        seed: u64,
    ) -> Self {
        let state = game.initial_state();
        Self::from_state(game, state, bots, human, seed)
    }

    /// Like [`TypedMatch::new`] but starting from an explicit state instead of
    /// the game's initial one — for games that offer a skip-the-setup start
    /// (e.g. Stratego's `setup=random`, which begins past deployment).
    pub fn from_state(
        game: G,
        state: G::State,
        bots: Vec<Option<Box<dyn Agent<G>>>>,
        human: Option<usize>,
        seed: u64,
    ) -> Self {
        assert_eq!(bots.len(), game.num_players());
        assert!(
            human.is_none_or(|h| bots[h].is_none()),
            "human seat must have no bot"
        );
        Self {
            game,
            state,
            bots,
            human,
            rng: Rng::new(seed),
        }
    }

    /// The [`GameUi`] viewer index: games treat an out-of-range viewer as a
    /// spectator (no seat's hidden information is theirs to see).
    fn viewer(&self) -> usize {
        self.human.unwrap_or(usize::MAX)
    }

    pub fn boxed(self) -> Box<dyn AnyMatch> {
        Box::new(self)
    }

    fn apply_event(&mut self, actor: usize, index: usize) -> MatchEvent {
        let viewer = self.viewer();
        let action = self.game.action_at(&self.state, index);
        let before = self.state.clone();
        self.game.apply(&mut self.state, action);
        let label = self.game.action_label_for(&before, action, viewer);
        let who = if Some(actor) == self.human {
            "You".to_string()
        } else {
            format!("Player {actor}")
        };
        MatchEvent {
            seat: actor,
            text: format!("{who}: {label}"),
            detail: self
                .game
                .describe_transition(&before, action, &self.state, viewer),
            data: self
                .game
                .transition_data(&before, action, &self.state, viewer),
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
                    let action = self.game.sample_chance_action(&self.state, &mut self.rng);
                    self.game.apply(&mut self.state, action);
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
        self.game.render(&self.state, self.viewer())
    }

    fn view_data(&self) -> Option<String> {
        self.game.view_data(&self.state, self.viewer())
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
            let n = self.game.num_actions(&self.state);
            if i >= n {
                return Err(format!("{i} is out of range (0-{})", n - 1));
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
        if let Some(h) = self.human {
            return self.game.result_text(&self.state, h);
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
        self.human
    }
}

/// Type-erased lab match for a genuinely simultaneous game. A complete joint
/// action is collected from one immutable pre-state and applied atomically;
/// one [`MatchEvent`] therefore represents one whole turn, not one seat's
/// partially revealed choice.
pub struct SimultaneousTypedMatch<G: SimultaneousGameUi> {
    game: G,
    state: G::State,
    bots: Vec<Option<Box<dyn SimultaneousAgent<G>>>>,
    human: Option<usize>,
    rng: Rng,
    prepared: Option<Vec<Option<usize>>>,
}

impl<G: SimultaneousGameUi + 'static> SimultaneousTypedMatch<G> {
    pub fn new(
        game: G,
        bots: Vec<Option<Box<dyn SimultaneousAgent<G>>>>,
        human: Option<usize>,
        seed: u64,
    ) -> Self {
        assert_eq!(bots.len(), game.num_players());
        assert!(human.is_none_or(|h| bots[h].is_none()));
        assert!(
            bots.iter()
                .enumerate()
                .all(|(p, bot)| bot.is_some() || Some(p) == human),
            "only the human seat may be externally driven"
        );
        let mut rng = Rng::new(seed);
        let state = game.initial_state_with_rng(&mut rng);
        Self {
            game,
            state,
            bots,
            human,
            rng,
            prepared: None,
        }
    }

    pub fn from_state(
        game: G,
        state: G::State,
        bots: Vec<Option<Box<dyn SimultaneousAgent<G>>>>,
        human: Option<usize>,
        seed: u64,
    ) -> Self {
        assert_eq!(bots.len(), game.num_players());
        assert!(human.is_none_or(|h| bots[h].is_none()));
        assert!(
            bots.iter()
                .enumerate()
                .all(|(p, bot)| bot.is_some() || Some(p) == human),
            "only the human seat may be externally driven"
        );
        Self {
            game,
            state,
            bots,
            human,
            rng: Rng::new(seed),
            prepared: None,
        }
    }

    fn viewer(&self) -> usize {
        self.human.unwrap_or(usize::MAX)
    }

    pub fn boxed(self) -> Box<dyn AnyMatch> {
        Box::new(self)
    }

    fn resolve_chance(&mut self) {
        while !self.game.is_terminal(&self.state)
            && self.game.turn(&self.state) == SimultaneousTurn::Chance
        {
            self.prepared = None;
            let action = self.game.sample_chance_action(&self.state, &mut self.rng);
            self.game.apply_chance(&mut self.state, action);
        }
    }

    /// Cache every non-human choice from the current shared state. This is
    /// legal specifically because simultaneous agents may not observe the
    /// human's action before choosing their own.
    fn prepare_round(&mut self) {
        self.resolve_chance();
        if self.prepared.is_some()
            || self.game.is_terminal(&self.state)
            || self.game.turn(&self.state) != SimultaneousTurn::Players
        {
            return;
        }
        let mut prepared = vec![None; self.bots.len()];
        for (player, slot) in prepared.iter_mut().enumerate() {
            if Some(player) == self.human || !self.game.is_active(&self.state, player) {
                continue;
            }
            *slot = Some(
                self.bots[player]
                    .as_ref()
                    .expect("every active non-human seat has a bot")
                    .act(&self.game, &self.state, player, &mut self.rng),
            );
        }
        self.prepared = Some(prepared);
    }

    fn apply_round(&mut self, driven: Option<(usize, usize)>) -> MatchEvent {
        let before = self.state.clone();
        let prepared = self.prepared.take();
        let mut actions = Vec::with_capacity(self.bots.len());
        for player in 0..self.bots.len() {
            let index = if !self.game.is_active(&before, player) {
                0
            } else if driven.is_some_and(|(seat, _)| seat == player) {
                driven.expect("checked").1
            } else if let Some(index) = prepared.as_ref().and_then(|moves| moves[player]) {
                index
            } else {
                self.bots[player]
                    .as_ref()
                    .expect("every active non-human seat has a bot")
                    .act(&self.game, &before, player, &mut self.rng)
            };
            assert!(index < self.game.num_actions(&before, player));
            actions.push(self.game.action_at(&before, player, index));
        }

        let active_labels: Vec<_> = (0..self.bots.len())
            .filter(|&player| self.game.is_active(&before, player))
            .map(|player| {
                let who = if Some(player) == self.human {
                    "You".to_string()
                } else {
                    format!("Player {player}")
                };
                format!(
                    "{who}: {}",
                    self.game.action_label(&before, player, actions[player])
                )
            })
            .collect();
        let actor = driven.map_or_else(
            || {
                (0..self.bots.len())
                    .find(|&p| self.game.is_active(&before, p))
                    .expect("a non-terminal position has an active player")
            },
            |(player, _)| player,
        );
        let label = self.game.action_label(&before, actor, actions[actor]);
        let viewer = self.viewer();
        self.game.apply_joint(&mut self.state, &actions);
        MatchEvent {
            seat: actor,
            label,
            text: active_labels.join("; "),
            detail: self
                .game
                .describe_joint_transition(&before, &actions, &self.state, viewer),
            data: self
                .game
                .transition_data(&before, &actions, &self.state, viewer),
        }
    }
}

impl<G: SimultaneousGameUi + 'static> AnyMatch for SimultaneousTypedMatch<G> {
    fn step(&mut self) -> Option<MatchEvent> {
        self.resolve_chance();
        if self.game.is_terminal(&self.state) {
            return None;
        }
        if self
            .human
            .is_some_and(|p| self.game.is_active(&self.state, p))
        {
            return None;
        }
        Some(self.apply_round(None))
    }

    fn prepare(&mut self) {
        self.prepare_round();
    }

    fn is_over(&self) -> bool {
        self.game.is_terminal(&self.state)
    }

    fn view(&self) -> String {
        self.game.render(&self.state, self.viewer())
    }

    fn view_data(&self) -> Option<String> {
        self.game.view_data(&self.state, self.viewer())
    }

    fn legal_labels(&self) -> Vec<String> {
        let Some(player) = self.human else {
            return Vec::new();
        };
        if self.game.is_terminal(&self.state)
            || self.game.turn(&self.state) != SimultaneousTurn::Players
            || !self.game.is_active(&self.state, player)
        {
            return Vec::new();
        }
        self.game
            .legal_actions(&self.state, player)
            .into_iter()
            .map(|action| self.game.action_label(&self.state, player, action))
            .collect()
    }

    fn apply_human(&mut self, input: &str) -> Result<MatchEvent, String> {
        let player = self
            .human
            .filter(|&p| {
                !self.game.is_terminal(&self.state)
                    && self.game.turn(&self.state) == SimultaneousTurn::Players
                    && self.game.is_active(&self.state, p)
            })
            .ok_or_else(|| "no driven seat to act".to_string())?;
        let actions = self.game.legal_actions(&self.state, player);
        let index = if let Ok(index) = input.trim().parse::<usize>() {
            if index >= actions.len() {
                return Err(format!("{index} is out of range (0-{})", actions.len() - 1));
            }
            index
        } else if let Some(action) = self.game.parse_action(&self.state, player, input) {
            actions
                .iter()
                .position(|candidate| *candidate == action)
                .ok_or_else(|| format!("'{}' is not legal here", input.trim()))?
        } else {
            return Err(format!("could not understand '{}'", input.trim()));
        };
        Ok(self.apply_round(Some((player, index))))
    }

    fn result_text(&self) -> String {
        if let Some(human) = self.human {
            return self.game.result_text(&self.state, human);
        }
        let winners: Vec<_> = (0..self.bots.len())
            .filter(|&p| self.game.returns(&self.state, p) > 0.0)
            .collect();
        match winners.as_slice() {
            [] => "Draw.".into(),
            [winner] => format!("Player {winner} wins."),
            many => format!(
                "Winners: {}.",
                many.iter()
                    .map(|winner| format!("Player {winner}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }

    fn to_act(&self) -> Option<usize> {
        self.human.filter(|&player| {
            !self.game.is_terminal(&self.state)
                && self.game.turn(&self.state) == SimultaneousTurn::Players
                && self.game.is_active(&self.state, player)
        })
    }

    fn num_seats(&self) -> usize {
        self.bots.len()
    }

    fn human_seat(&self) -> Option<usize> {
        self.human
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use game_core::{SimultaneousGame, SimultaneousGameUi};

    use super::*;

    #[derive(Clone)]
    struct State(Option<[u8; 2]>);

    struct Matching;

    impl SimultaneousGame for Matching {
        type State = State;
        type Action = u8;
        type ChanceAction = ();

        fn num_players(&self) -> usize {
            2
        }

        fn initial_state(&self) -> State {
            State(None)
        }

        fn turn(&self, _state: &State) -> SimultaneousTurn {
            SimultaneousTurn::Players
        }

        fn is_terminal(&self, state: &State) -> bool {
            state.0.is_some()
        }

        fn is_active(&self, state: &State, _player: usize) -> bool {
            !self.is_terminal(state)
        }

        fn returns(&self, state: &State, player: usize) -> f64 {
            let actions = state.0.expect("terminal state");
            let first = if actions[0] == actions[1] { 1.0 } else { -1.0 };
            if player == 0 { first } else { -first }
        }

        fn legal_actions(&self, _state: &State, _player: usize) -> Vec<u8> {
            vec![0, 1]
        }

        fn apply_joint(&self, state: &mut State, actions: &[u8]) {
            state.0 = Some([actions[0], actions[1]]);
        }

        fn chance_outcomes(&self, _state: &State) -> Vec<((), f64)> {
            Vec::new()
        }

        fn apply_chance(&self, _state: &mut State, _action: ()) {
            unreachable!()
        }
    }

    impl SimultaneousGameUi for Matching {
        fn id(&self) -> &'static str {
            "matching"
        }

        fn render(&self, state: &State, _player: usize) -> String {
            format!("{:?}", state.0)
        }

        fn action_label(&self, _state: &State, _player: usize, action: u8) -> String {
            action.to_string()
        }
    }

    #[test]
    fn prepare_caches_hidden_bot_choice_until_human_action() {
        let calls = Arc::new(AtomicUsize::new(0));
        let bot_calls = Arc::clone(&calls);
        let bot = move |_: &Matching, state: &State, player: usize, _: &mut Rng| {
            assert!(
                state.0.is_none(),
                "bot must see the shared pre-action state"
            );
            assert_eq!(player, 1);
            bot_calls.fetch_add(1, Ordering::Relaxed);
            1
        };
        let bots: Vec<Option<Box<dyn SimultaneousAgent<Matching>>>> =
            vec![None, Some(Box::new(bot))];
        let mut game = SimultaneousTypedMatch::new(Matching, bots, Some(0), 7);

        game.prepare();
        game.prepare();
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        let event = game.apply_human("0").expect("legal human action");
        assert_eq!(event.text, "You: 0; Player 1: 1");
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert!(game.is_over());
    }
}
