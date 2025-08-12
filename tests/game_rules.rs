use twentyone::env::{Action, Env, Observation};

fn deck_from(slice: &[u8]) -> [u8; 11] {
    let mut arr = [0u8; 11];
    arr.copy_from_slice(slice);
    arr
}

#[test]
fn starts_with_six_hearts_round_one() {
    let env = Env::new_with_preset_decks(vec![deck_from(&[11, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9])]);
    assert_eq!(env.round(), 1);
    assert_eq!(env.hearts(0), 6);
    assert_eq!(env.hearts(1), 6);
}

#[test]
fn initial_deal_and_tie_causes_no_damage() {
    let mut env = Env::new_with_preset_decks(vec![deck_from(&[11, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9])]);
    env.start_new_round().unwrap();
    // P0 and P1 both have 12 (11+1 and 10+2)
    let o0: Observation = env.observation(0);
    let o1: Observation = env.observation(1);
    assert_eq!(o0.self_total, 12);
    assert_eq!(o1.self_total, 12);

    // Both stand
    let r = env.step(Action::Stand).unwrap();
    assert!(!r.round_over);
    let r = env.step(Action::Stand).unwrap();
    assert!(r.round_over);
    assert!(!r.game_over);
    assert!(r.outcome.unwrap().winner.is_none());
    assert_eq!(env.hearts(0), 6);
    assert_eq!(env.hearts(1), 6);
}

#[test]
fn bust_then_opponent_stands_results_in_loss() {
    // Deal: P0 up=11, P1 up=10, P0 down=9 (20), P1 down=1 (11), Next draw is 2 (P0 busts)
    let mut env = Env::new_with_preset_decks(vec![deck_from(&[11, 10, 9, 1, 2, 3, 4, 5, 6, 7, 8])]);
    env.start_new_round().unwrap();
    assert_eq!(env.current_player(), 0);
    // P0 draws and busts to 22
    let r = env.step(Action::Draw).unwrap();
    assert!(!r.round_over);
    // P1 stands; round ends with P0 losing 1 heart
    let r = env.step(Action::Stand).unwrap();
    assert!(r.round_over);
    let outcome = r.outcome.unwrap();
    assert_eq!(outcome.winner, Some(1));
    assert_eq!(outcome.damage, 1);
    assert_eq!(env.hearts(0), 5);
    assert_eq!(env.hearts(1), 6);
}

#[test]
fn both_over_21_closest_wins() {
    // Deal: P0 up=10, P1 up=9, P0 down=5 (15), P1 down=6 (15), P0 draw=7 -> 22, P1 draw=8 -> 23
    let mut env = Env::new_with_preset_decks(vec![deck_from(&[10, 9, 5, 6, 7, 8, 1, 2, 3, 4, 11])]);
    env.start_new_round().unwrap();
    // P0 draws to 22
    let _ = env.step(Action::Draw).unwrap();
    // P1 draws to 23; both players busted and auto-stood -> round ends
    let r = env.step(Action::Draw).unwrap();
    assert!(r.round_over);
    let outcome = r.outcome.unwrap();
    assert_eq!(outcome.winner, Some(0));
}

#[test]
fn damage_increases_with_round_number() {
    // Two rounds: P1 wins both. Round 1 damage 1, round 2 damage 2.
    let decks = vec![
        deck_from(&[11, 10, 9, 1, 2, 3, 4, 5, 6, 7, 8]),
        deck_from(&[11, 10, 9, 1, 2, 3, 4, 5, 6, 7, 8]),
    ];
    let mut env = Env::new_with_preset_decks(decks);
    env.start_new_round().unwrap();
    // P0 draws bust, P1 stands
    let _ = env.step(Action::Draw).unwrap();
    let r = env.step(Action::Stand).unwrap();
    assert!(r.round_over);
    assert_eq!(env.hearts(0), 5);
    assert_eq!(env.hearts(1), 6);

    env.start_new_round().unwrap();
    // P0 draws bust, P1 stands again
    let _ = env.step(Action::Draw).unwrap();
    let r = env.step(Action::Stand).unwrap();
    assert!(r.round_over);
    assert_eq!(env.hearts(0), 3); // lost 1, then 2
    assert_eq!(env.hearts(1), 6);
}

#[test]
fn game_ends_when_hearts_zero() {
    // Configure starting hearts = 2. Make P1 win two rounds (1 + 2 damage).
    let decks = vec![
        deck_from(&[11, 10, 9, 1, 2, 3, 4, 5, 6, 7, 8]),
        deck_from(&[11, 10, 9, 1, 2, 3, 4, 5, 6, 7, 8]),
    ];
    let mut env = Env::new_with_preset_decks_and_hearts(decks, 2);
    env.start_new_round().unwrap();
    let _ = env.step(Action::Draw).unwrap();
    let r = env.step(Action::Stand).unwrap();
    assert!(r.round_over);
    assert!(!r.game_over);

    env.start_new_round().unwrap();
    let _ = env.step(Action::Draw).unwrap();
    let r = env.step(Action::Stand).unwrap();
    assert!(r.round_over);
    assert!(r.game_over);
}
