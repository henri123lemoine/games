//! Rules and arena contract for games whose players choose from the same
//! public state before any choice is revealed.
//!
//! A simultaneous game must not be encoded as a sequence of ordinary
//! [`Game`](crate::Game) nodes: doing so lets later players condition on moves
//! that are hidden in the real game. This contract collects one action per
//! living player and gives the complete joint action to the rules in one call.

use crate::Rng;

/// Which kind of transition is waiting at a simultaneous-game state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimultaneousTurn {
    /// A stochastic environment transition.
    Chance,
    /// Every active player chooses from this exact same state.
    Players,
}

/// An N-player simultaneous-action game with optional chance transitions.
pub trait SimultaneousGame: Sync {
    type State: Clone + Send + Sync;
    type Action: Copy + std::fmt::Debug + Send + Sync + PartialEq;
    type ChanceAction: Copy + std::fmt::Debug + Send + Sync + PartialEq;

    fn num_players(&self) -> usize;
    fn initial_state(&self) -> Self::State;

    /// Seed-aware initialization for games with randomized starting layouts.
    /// Deterministic games inherit the ordinary initial state.
    fn initial_state_with_rng(&self, _rng: &mut Rng) -> Self::State {
        self.initial_state()
    }

    fn turn(&self, state: &Self::State) -> SimultaneousTurn;
    fn is_terminal(&self, state: &Self::State) -> bool;
    fn is_active(&self, state: &Self::State, player: usize) -> bool;
    fn returns(&self, state: &Self::State, player: usize) -> f64;

    /// Stable action order for `player` at a joint decision.
    fn legal_actions(&self, state: &Self::State, player: usize) -> Vec<Self::Action>;

    fn num_actions(&self, state: &Self::State, player: usize) -> usize {
        self.legal_actions(state, player).len()
    }

    fn action_at(&self, state: &Self::State, player: usize, index: usize) -> Self::Action {
        self.legal_actions(state, player)[index]
    }

    /// Apply one action for every player. Inactive players still occupy their
    /// index in `actions`; their value is ignored by the rules.
    fn apply_joint(&self, state: &mut Self::State, actions: &[Self::Action]);

    fn chance_outcomes(&self, state: &Self::State) -> Vec<(Self::ChanceAction, f64)>;

    fn sample_chance(&self, state: &Self::State, rng: &mut Rng) -> (Self::ChanceAction, f64) {
        let outcomes = self.chance_outcomes(state);
        let index = crate::rand::sample_outcome(&outcomes, rng);
        outcomes[index]
    }

    fn sample_chance_action(&self, state: &Self::State, rng: &mut Rng) -> Self::ChanceAction {
        self.sample_chance(state, rng).0
    }

    fn apply_chance(&self, state: &mut Self::State, action: Self::ChanceAction);

    fn state_key(&self, _state: &Self::State) -> Option<u64> {
        None
    }

    fn action_id(&self, action: &Self::Action) -> u64 {
        crate::hash::fnv1a(format!("{action:?}").as_bytes())
    }
}

/// Chooses one member of a simultaneous joint action without observing any
/// other player's choice for that transition.
pub trait SimultaneousAgent<G: SimultaneousGame> {
    fn act(&self, game: &G, state: &G::State, player: usize, rng: &mut Rng) -> usize;
}

/// Policy/value features for one player's perspective at a simultaneous
/// decision. Unlike [`crate::PolicyValueEncoder`], the perspective is explicit
/// because every active player acts from the same public state.
pub trait SimultaneousPolicyValueEncoder<G: SimultaneousGame>: Sync {
    fn input_len(&self) -> usize;
    fn policy_len(&self) -> usize;
    fn encode_state(&self, game: &G, state: &G::State, player: usize) -> Vec<f32>;
    fn action_index(&self, game: &G, state: &G::State, player: usize, action: G::Action) -> usize;
}

impl<G, F> SimultaneousAgent<G> for F
where
    G: SimultaneousGame,
    F: Fn(&G, &G::State, usize, &mut Rng) -> usize,
{
    fn act(&self, game: &G, state: &G::State, player: usize, rng: &mut Rng) -> usize {
        self(game, state, player, rng)
    }
}

pub struct RandomSimultaneousAgent;

impl<G: SimultaneousGame> SimultaneousAgent<G> for RandomSimultaneousAgent {
    fn act(&self, game: &G, state: &G::State, player: usize, rng: &mut Rng) -> usize {
        rng.below(game.num_actions(state, player))
    }
}

/// Play a complete simultaneous game with one agent per player.
pub fn play_simultaneous_n<G: SimultaneousGame>(
    game: &G,
    agents: &[&dyn SimultaneousAgent<G>],
    rng: &mut Rng,
) -> G::State {
    assert_eq!(agents.len(), game.num_players(), "one agent per player");
    let mut state = game.initial_state_with_rng(rng);
    while !game.is_terminal(&state) {
        match game.turn(&state) {
            SimultaneousTurn::Chance => {
                let action = game.sample_chance_action(&state, rng);
                game.apply_chance(&mut state, action);
            }
            SimultaneousTurn::Players => {
                // Every call sees the same immutable state. Choices are not
                // applied until the complete joint action has been collected.
                let actions: Vec<_> = (0..game.num_players())
                    .map(|player| {
                        let index = if game.is_active(&state, player) {
                            agents[player].act(game, &state, player, rng)
                        } else {
                            0
                        };
                        game.action_at(&state, player, index)
                    })
                    .collect();
                game.apply_joint(&mut state, &actions);
            }
        }
    }
    state
}

/// Two-player convenience wrapper returning player zero's terminal utility.
pub fn play_simultaneous<G: SimultaneousGame>(
    game: &G,
    first: &impl SimultaneousAgent<G>,
    second: &impl SimultaneousAgent<G>,
    rng: &mut Rng,
) -> f64 {
    let agents: [&dyn SimultaneousAgent<G>; 2] = [first, second];
    let terminal = play_simultaneous_n(game, &agents, rng);
    game.returns(&terminal, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct MatrixState {
        chosen: Option<[u8; 2]>,
    }

    struct MatrixGame;

    impl SimultaneousGame for MatrixGame {
        type State = MatrixState;
        type Action = u8;
        type ChanceAction = ();

        fn num_players(&self) -> usize {
            2
        }
        fn initial_state(&self) -> MatrixState {
            MatrixState { chosen: None }
        }
        fn turn(&self, _state: &MatrixState) -> SimultaneousTurn {
            SimultaneousTurn::Players
        }
        fn is_terminal(&self, state: &MatrixState) -> bool {
            state.chosen.is_some()
        }
        fn is_active(&self, _state: &MatrixState, _player: usize) -> bool {
            true
        }
        fn returns(&self, state: &MatrixState, player: usize) -> f64 {
            let choices = state.chosen.expect("terminal");
            if choices[0] == choices[1] {
                0.0
            } else if player == 0 {
                1.0
            } else {
                -1.0
            }
        }
        fn legal_actions(&self, _state: &MatrixState, _player: usize) -> Vec<u8> {
            vec![0, 1]
        }
        fn apply_joint(&self, state: &mut MatrixState, actions: &[u8]) {
            state.chosen = Some([actions[0], actions[1]]);
        }
        fn chance_outcomes(&self, _state: &MatrixState) -> Vec<((), f64)> {
            Vec::new()
        }
        fn apply_chance(&self, _state: &mut MatrixState, _action: ()) {
            unreachable!()
        }
    }

    #[test]
    fn joint_choices_are_collected_before_transition() {
        let never_moved = |_: &MatrixGame, state: &MatrixState, _: usize, _: &mut Rng| {
            assert!(state.chosen.is_none());
            1
        };
        let agents: [&dyn SimultaneousAgent<MatrixGame>; 2] = [&never_moved, &never_moved];
        let end = play_simultaneous_n(&MatrixGame, &agents, &mut Rng::new(1));
        assert_eq!(end.chosen, Some([1, 1]));
    }
}
