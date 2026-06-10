use game_core::{Game, GameUi, Rng, Turn};
use othello::{Move, Othello};

#[test]
fn initial_position_has_four_black_moves() {
    let game = Othello;
    let state = game.initial_state();
    assert_eq!(game.turn(&state), Turn::Player(0));
    assert_eq!(state.discs(0), 2);
    assert_eq!(state.discs(1), 2);

    let actions = game.legal_actions(&state);
    let labels: Vec<String> = actions
        .iter()
        .map(|&a| game.action_label(&state, a))
        .collect();
    assert_eq!(labels, ["d3", "c4", "f5", "e6"]);
}

#[test]
fn known_flip_sequence_produces_right_counts() {
    let game = Othello;
    let mut state = game.initial_state();

    let d3 = game.parse_action(&state, "d3").expect("d3 is legal");
    game.apply(&mut state, d3);
    assert_eq!((state.discs(0), state.discs(1)), (4, 1));
    assert_eq!(game.turn(&state), Turn::Player(1));

    let c5 = game.parse_action(&state, "c5").expect("c5 is legal");
    game.apply(&mut state, c5);
    assert_eq!((state.discs(0), state.discs(1)), (3, 3));
    assert_eq!(game.turn(&state), Turn::Player(0));
}

#[test]
fn shortest_perfect_game_wipes_white_out() {
    let game = Othello;
    let mut state = game.initial_state();
    for mv in ["e6", "f4", "e3", "f6", "g5", "d6", "e7", "f5", "c5"] {
        assert!(!game.is_terminal(&state));
        let action = game.parse_action(&state, mv).unwrap_or_else(|| {
            panic!("{mv} should be legal");
        });
        game.apply(&mut state, action);
    }
    assert!(game.is_terminal(&state));
    assert_eq!((state.discs(0), state.discs(1)), (13, 0));
    assert_eq!(game.returns(&state, 0), 1.0);
    assert_eq!(game.returns(&state, 1), -1.0);
}

#[test]
fn pass_is_the_only_action_when_stuck() {
    let game = Othello;
    let mut state = game.initial_state();
    let mut rng = Rng::new(7);
    loop {
        let actions = game.legal_actions(&state);
        if actions == [Move::Pass] {
            assert_eq!(game.action_label(&state, actions[0]), "pass");
            assert_eq!(game.parse_action(&state, "pass"), Some(Move::Pass));
            let Turn::Player(passer) = game.turn(&state) else {
                panic!("othello has no chance nodes");
            };
            game.apply(&mut state, actions[0]);
            assert_eq!(game.turn(&state), Turn::Player(passer ^ 1));
            return;
        }
        assert_eq!(game.parse_action(&state, "pass"), None);
        let i = ((rng.unit() * actions.len() as f64) as usize).min(actions.len() - 1);
        game.apply(&mut state, actions[i]);
        if game.is_terminal(&state) {
            state = game.initial_state();
        }
    }
}

#[test]
fn random_playthroughs_terminate_with_disc_count_winner() {
    let game = Othello;
    let mut rng = Rng::new(42);
    for _ in 0..200 {
        let mut state = game.initial_state();
        let mut plies = 0;
        while !game.is_terminal(&state) {
            plies += 1;
            assert!(plies <= 130, "game did not terminate");
            let actions = game.legal_actions(&state);
            assert!(!actions.is_empty());
            let i = ((rng.unit() * actions.len() as f64) as usize).min(actions.len() - 1);
            game.apply(&mut state, actions[i]);
        }
        let (black, white) = (state.discs(0), state.discs(1));
        assert!(black + white >= 5 && black + white <= 64);
        let r0 = game.returns(&state, 0);
        assert_eq!(r0, -game.returns(&state, 1));
        match black.cmp(&white) {
            std::cmp::Ordering::Greater => assert_eq!(r0, 1.0),
            std::cmp::Ordering::Less => assert_eq!(r0, -1.0),
            std::cmp::Ordering::Equal => assert_eq!(r0, 0.0),
        }
    }
}
