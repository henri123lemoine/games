//! Multiplayer PUCT regression: leaf values stay in absolute seat order while
//! each node maximizes the component belonging to its own mover.

use game_core::{Game, PolicyValueEncoder, Rng, Turn};
use solvers::azero::{EvalResult, Gather, PuctConfig, Search, Value};

#[derive(Clone)]
struct State {
    branch: u8,
    depth: u16,
    to_move: usize,
}

struct FourSeatTree;

impl Game for FourSeatTree {
    type State = State;
    type Action = u8;

    fn num_players(&self) -> usize {
        4
    }

    fn initial_state(&self) -> State {
        State {
            branch: 2,
            depth: 0,
            to_move: 0,
        }
    }

    fn turn(&self, state: &State) -> Turn {
        Turn::Player(state.to_move)
    }

    fn is_terminal(&self, _state: &State) -> bool {
        false
    }

    fn returns(&self, _state: &State, _player: usize) -> f64 {
        unreachable!("the synthetic tree has no terminal nodes")
    }

    fn legal_actions(&self, state: &State) -> Vec<u8> {
        if state.depth == 0 {
            vec![0, 1]
        } else {
            vec![0]
        }
    }

    fn chance_outcomes(&self, _state: &State) -> Vec<(u8, f64)> {
        Vec::new()
    }

    fn apply(&self, state: &mut State, action: u8) {
        if state.depth == 0 {
            state.branch = action;
        }
        state.depth += 1;
        state.to_move = (state.to_move + 1) % 4;
    }

    fn infoset_key(&self, state: &State, _player: usize) -> u64 {
        u64::from(state.branch) | (u64::from(state.depth) << 8)
    }
}

struct Encoder;

impl PolicyValueEncoder<FourSeatTree> for Encoder {
    fn input_len(&self) -> usize {
        3
    }

    fn policy_len(&self) -> usize {
        2
    }

    fn encode_state(&self, _game: &FourSeatTree, state: &State) -> Vec<f32> {
        vec![
            f32::from(state.branch),
            f32::from(state.depth),
            state.to_move as f32,
        ]
    }

    fn action_index(&self, _game: &FourSeatTree, _state: &State, action: u8) -> usize {
        usize::from(action)
    }
}

#[test]
fn root_uses_its_absolute_seat_value_across_other_players_turns() {
    let game = FourSeatTree;
    let state = game.initial_state();
    let cfg = PuctConfig {
        sims: 64,
        max_leaves: 1,
        root_noise: 0.0,
        cycle_draws: false,
        ..PuctConfig::default()
    };
    let mut search = Search::new(None);
    let mut rng = Rng::new(7);
    let mut results = Vec::new();
    while let Gather::Requests(requests) = search.advance(
        &game,
        &Encoder,
        &state,
        &cfg,
        &mut rng,
        std::mem::take(&mut results),
        &|_| false,
        None,
    ) {
        results = requests
            .into_iter()
            .map(|request| {
                let branch = request.features[0] as u8;
                let value = match branch {
                    0 => Value::seats(&[1.0, -1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0]),
                    1 => Value::seats(&[-1.0 / 3.0, 1.0, -1.0 / 3.0, -1.0 / 3.0]),
                    _ => Value::seats(&[0.0; 4]),
                };
                EvalResult {
                    priors: vec![1.0 / request.support.len() as f32; request.support.len()],
                    value,
                }
            })
            .collect();
    }

    let visits = search.root_visits();
    assert_eq!(visits.len(), 2);
    assert!(visits[0] > visits[1] * 2, "root visits were {visits:?}");
}
