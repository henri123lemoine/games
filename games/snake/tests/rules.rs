use game_core::{Game, GameUi, Rng, Turn};
use snake::{Snake, SnakeAction, SnakeState, Status};

fn food_at(game: &Snake, state: &mut SnakeState, x: usize, y: usize) {
    assert_eq!(game.turn(state), Turn::Chance);
    game.apply(state, SnakeAction::Food((y * game.width() + x) as u16));
}

#[test]
fn initial_layout_is_centered_length_three_chance_node() {
    let g = Snake::default();
    let s = g.initial_state();
    assert_eq!(s.len(), 3);
    assert_eq!(s.head(), (5, 5));
    assert_eq!(s.status(), Status::Alive);
    assert_eq!(g.turn(&s), Turn::Chance);
    assert_eq!(g.returns(&s, 0), 0.03);
    assert_eq!(g.score(&s), 0.03);
}

#[test]
fn food_chance_is_uniform_over_empty_cells_and_sums_to_one() {
    let g = Snake::default();
    let s = g.initial_state();
    let outs = g.chance_outcomes(&s);
    assert_eq!(outs.len(), g.area() - 3);
    let total: f64 = outs.iter().map(|(_, p)| p).sum();
    assert!((total - 1.0).abs() < 1e-9);
    for &(a, p) in &outs {
        assert!((p - 1.0 / 97.0).abs() < 1e-12);
        let SnakeAction::Food(c) = a else {
            panic!("chance outcome must be Food, got {a:?}")
        };
        let (x, y) = (c as usize % g.width(), c as usize / g.width());
        assert!(!(y == 5 && (3..=5).contains(&x)), "food on the snake");
    }
}

#[test]
fn eating_grows_resets_hunger_and_raises_returns() {
    let g = Snake::default();
    let mut s = g.initial_state();
    food_at(&g, &mut s, 6, 5);
    assert_eq!(g.turn(&s), Turn::Player(0));
    let before = g.returns(&s, 0);
    g.apply(&mut s, SnakeAction::Straight);
    assert_eq!(s.len(), 4);
    assert_eq!(s.head(), (6, 5));
    assert_eq!(s.hunger(), 0);
    assert!(g.returns(&s, 0) > before);
    assert_eq!(g.returns(&s, 0), 0.04);
    assert_eq!(g.turn(&s), Turn::Chance, "new food after a meal");
    assert!(!g.is_terminal(&s));
}

#[test]
fn wall_collision_terminates_with_current_score() {
    let g = Snake::default();
    let mut s = g.initial_state();
    food_at(&g, &mut s, 0, 0);
    for _ in 0..4 {
        g.apply(&mut s, SnakeAction::Straight);
        assert!(!g.is_terminal(&s));
    }
    assert_eq!(s.head(), (9, 5));
    assert_eq!(g.legal_actions(&s).len(), 3, "fatal moves stay legal");
    g.apply(&mut s, SnakeAction::Straight);
    assert!(g.is_terminal(&s));
    assert_eq!(s.status(), Status::Crashed);
    assert_eq!(s.len(), 3);
    assert_eq!(g.returns(&s, 0), 0.03);
}

#[test]
fn self_collision_terminates() {
    let g = Snake::default();
    let mut s = g.initial_state();
    for x in [6, 7] {
        food_at(&g, &mut s, x, 5);
        g.apply(&mut s, SnakeAction::Straight);
    }
    assert_eq!(s.len(), 5);
    food_at(&g, &mut s, 0, 0);
    g.apply(&mut s, SnakeAction::TurnLeft);
    g.apply(&mut s, SnakeAction::TurnLeft);
    assert!(!g.is_terminal(&s));
    g.apply(&mut s, SnakeAction::TurnLeft);
    assert!(g.is_terminal(&s));
    assert_eq!(s.status(), Status::Crashed);
    assert_eq!(g.returns(&s, 0), 0.05);
}

#[test]
fn moving_into_the_vacating_tail_is_safe() {
    let g = Snake::default();
    let mut s = g.initial_state();
    food_at(&g, &mut s, 0, 0);
    for _ in 0..4 {
        g.apply(&mut s, SnakeAction::TurnRight);
    }
    assert!(!g.is_terminal(&s));
    assert_eq!(s.head(), (5, 5), "a 2x2 right circle returns home");
}

#[test]
fn starvation_cap_triggers_at_exactly_area_moves() {
    let g = Snake::default();
    let mut s = g.initial_state();
    food_at(&g, &mut s, 0, 0);
    for i in 0..99 {
        g.apply(&mut s, SnakeAction::TurnRight);
        assert!(!g.is_terminal(&s), "alive after move {}", i + 1);
    }
    assert_eq!(s.hunger(), 99);
    g.apply(&mut s, SnakeAction::TurnRight);
    assert!(g.is_terminal(&s));
    assert_eq!(s.status(), Status::Starved);
    assert_eq!(g.returns(&s, 0), 0.03);
}

#[test]
fn filling_the_board_wins_with_returns_one() {
    let g = Snake::new(4, 1);
    let mut s = g.initial_state();
    let outs = g.chance_outcomes(&s);
    assert_eq!(outs.len(), 1);
    assert_eq!(outs[0].1, 1.0);
    food_at(&g, &mut s, 3, 0);
    g.apply(&mut s, SnakeAction::Straight);
    assert!(g.is_terminal(&s));
    assert_eq!(s.status(), Status::Won);
    assert_eq!(g.returns(&s, 0), 1.0);
}

#[test]
fn snake_filling_the_board_at_start_is_an_immediate_win() {
    let g = Snake::new(3, 1);
    let s = g.initial_state();
    assert!(g.is_terminal(&s));
    assert_eq!(s.status(), Status::Won);
    assert_eq!(g.returns(&s, 0), 1.0);
}

#[test]
fn random_playthroughs_terminate_within_bound() {
    for (w, h) in [(10, 10), (6, 6), (4, 3)] {
        let g = Snake::new(w, h);
        let cap = g.area() * (g.area() + 2);
        let mut rng = Rng::new(7 + w as u64);
        for _ in 0..200 {
            let mut s = g.initial_state();
            let mut steps = 0;
            while !g.is_terminal(&s) {
                steps += 1;
                assert!(steps <= cap, "{w}x{h} game exceeded {cap} steps");
                match g.turn(&s) {
                    Turn::Chance => {
                        let outs = g.chance_outcomes(&s);
                        let r = rng.unit();
                        let mut acc = 0.0;
                        let mut chosen = outs[outs.len() - 1].0;
                        for &(a, p) in &outs {
                            acc += p;
                            if r < acc {
                                chosen = a;
                                break;
                            }
                        }
                        g.apply(&mut s, chosen);
                    }
                    Turn::Player(_) => {
                        let acts = g.legal_actions(&s);
                        let i = ((rng.unit() * acts.len() as f64) as usize).min(acts.len() - 1);
                        g.apply(&mut s, acts[i]);
                    }
                }
            }
            let r = g.returns(&s, 0);
            assert!((0.0..=1.0).contains(&r));
        }
    }
}

#[test]
fn eval_rises_as_head_approaches_food_and_stays_under_one_meal() {
    use game_core::Eval;
    let g = Snake::default();
    let eval = snake::SnakeEval;
    let mut s = g.initial_state();
    food_at(&g, &mut s, 9, 5);
    let far = eval.eval(&g, &s, 0);
    g.apply(&mut s, SnakeAction::Straight);
    let near = eval.eval(&g, &s, 0);
    assert!(near > far);
    let one_meal = 1.0 / g.area() as f64;
    assert!(far > g.score(&s) && near < g.score(&s) + one_meal);
}

#[test]
fn ui_renders_glyphs_labels_and_parses_lsr() {
    let g = Snake::default();
    let mut s = g.initial_state();
    food_at(&g, &mut s, 0, 0);
    let view = g.render(&s, 0);
    assert!(view.contains('>'), "head glyph shows heading:\n{view}");
    assert_eq!(view.matches('o').count(), 2, "two body segments");
    assert!(view.contains('*'), "food glyph");
    assert!(view.contains("length 3/100"));
    let labels: Vec<String> = g
        .legal_actions(&s)
        .iter()
        .map(|&a| g.action_label(&s, a))
        .collect();
    assert_eq!(labels, ["left", "straight", "right"]);
    assert_eq!(g.parse_action(&s, " L "), Some(SnakeAction::TurnLeft));
    assert_eq!(g.parse_action(&s, "straight"), Some(SnakeAction::Straight));
    assert_eq!(g.parse_action(&s, "r"), Some(SnakeAction::TurnRight));
    assert_eq!(g.parse_action(&s, "Right"), Some(SnakeAction::TurnRight));
    assert_eq!(g.parse_action(&s, "x"), None);
    assert_eq!(g.id(), "snake");
}

#[test]
fn infoset_keys_distinguish_states() {
    let g = Snake::default();
    let mut a = g.initial_state();
    food_at(&g, &mut a, 0, 0);
    let mut b = a.clone();
    g.apply(&mut b, SnakeAction::Straight);
    assert_ne!(g.infoset_key(&a, 0), g.infoset_key(&b, 0));
    assert_eq!(g.state_key(&a), Some(g.infoset_key(&a, 0)));
}
