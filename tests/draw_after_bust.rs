use twentyone::env::{Action, Env};

// Ensure drawing after bust is allowed and does not error.
#[test]
fn drawing_after_bust_is_allowed() {
    // Deck order:
    // P0 up=10, P1 up=1, P0 down=11 (P0 total 21), P1 down=1 (P1 total 2)
    // Next draw is 2 -> P0 draws to 23 (bust), then P1 stands, then P0 draws again
    let order = [10, 1, 11, 1, 2, 3, 4, 5, 6, 7, 8];
    let mut env = Env::new_with_preset_decks(vec![order]);
    env.start_new_round().unwrap();
    assert_eq!(env.current_player(), 0);

    // P0 draws: goes from 21 to 23 (bust)
    let r = env.step(Action::Draw).unwrap();
    assert!(!r.round_over);
    assert_eq!(env.current_player(), 1);

    // P1 stands
    let r = env.step(Action::Stand).unwrap();
    assert!(!r.round_over);
    assert_eq!(env.current_player(), 0);

    // Previously invalid: P0 draws after bust; now it must be allowed
    let r = env.step(Action::Draw).unwrap();
    assert!(!r.round_over);
}

