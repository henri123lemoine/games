use twentyone::env::{Action, Env};

fn deck_from(slice: &[u8]) -> [u8; 11] {
    let mut arr = [0u8; 11];
    arr.copy_from_slice(slice);
    arr
}

#[test]
fn debug_first_stand() {
    let mut env = Env::new_with_preset_decks(vec![deck_from(&[11, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9])]);
    env.start_new_round().unwrap();
    let r = env.step(Action::Stand).unwrap();
    println!("round_over={} game_over={}", r.round_over, r.game_over);
}

#[test]
fn debug_bust_then_stands() {
    let mut env = Env::new_with_preset_decks(vec![[11, 10, 9, 1, 2, 3, 4, 5, 6, 7, 8]]);
    env.start_new_round().unwrap();
    // P0 draw -> bust
    let r = env.step(Action::Draw).unwrap();
    println!("after p0 draw: round_over={}", r.round_over);
    // P1 stand
    let r = env.step(Action::Stand).unwrap();
    println!("after p1 stand: round_over={}", r.round_over);
    // P0 stand
    let r = env.step(Action::Stand).unwrap();
    println!("after p0 stand: round_over={}", r.round_over);
}
