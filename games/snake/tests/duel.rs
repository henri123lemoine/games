use game_core::{Eval, Game, GameUi, Rng, Turn};
use snake::DuelEval;
use snake::duel::{Dir, Duel, DuelAction, DuelState, Outcome, SIDE};

/// Place food on `(x, y)` (a chance node), asserting the cell is free.
fn food_at(g: &Duel, s: &mut DuelState, x: usize, y: usize) {
    assert_eq!(g.turn(s), Turn::Chance, "food only spawns at a chance node");
    g.apply(s, DuelAction::Food((y * SIDE + x) as u16));
}

/// Run one full tick: seat 0 then seat 1 commit headings (both advance on
/// seat 1's commit). Spawns food on a fixed far corner first if none is set,
/// so the tick is never blocked on a chance node.
fn tick(g: &Duel, s: &mut DuelState, d0: Dir, d1: Dir) {
    if matches!(g.turn(s), Turn::Chance) {
        // A corner cell free of both snakes for the early game.
        food_at(g, s, 0, 0);
    }
    assert_eq!(g.turn(s), Turn::Player(0));
    g.apply(s, DuelAction::Move(d0));
    assert_eq!(g.turn(s), Turn::Player(1));
    g.apply(s, DuelAction::Move(d1));
}

#[test]
fn initial_layout_is_two_length_three_snakes_facing_inward() {
    let g = Duel::new();
    let s = g.initial_state();
    assert_eq!(g.num_players(), 2);
    assert_eq!(s.worm(0).len(), 3);
    assert_eq!(s.worm(1).len(), 3);
    assert!(s.worm(0).alive() && s.worm(1).alive());
    assert_eq!(s.worm(0).heading(), Dir::Right);
    assert_eq!(s.worm(1).heading(), Dir::Left);
    assert_eq!(s.worm(0).head(), (4, SIDE / 2));
    assert_eq!(s.worm(1).head(), (SIDE - 5, SIDE / 2));
    assert_eq!(g.turn(&s), Turn::Chance, "food spawns first");
    assert_eq!(s.outcome(), Outcome::Ongoing);
    assert!(!g.is_terminal(&s));
}

#[test]
fn food_chance_is_uniform_over_empty_cells_and_sums_to_one() {
    let g = Duel::new();
    let s = g.initial_state();
    let outs = g.chance_outcomes(&s);
    assert_eq!(outs.len(), g.area() - 6, "six occupied cells excluded");
    let total: f64 = outs.iter().map(|(_, p)| p).sum();
    assert!((total - 1.0).abs() < 1e-9);
    for &(a, p) in &outs {
        assert!((p - 1.0 / (g.area() as f64 - 6.0)).abs() < 1e-12);
        let DuelAction::Food(c) = a else {
            panic!("chance outcome must be Food, got {a:?}");
        };
        let (x, y) = (c as usize % SIDE, c as usize / SIDE);
        let on_snake = s
            .worm(0)
            .cells()
            .chain(s.worm(1).cells())
            .any(|p| p == (x, y));
        assert!(!on_snake, "food never spawns on a snake");
    }
}

#[test]
fn turn_order_is_chance_then_seat0_then_seat1() {
    let g = Duel::new();
    let mut s = g.initial_state();
    assert_eq!(g.turn(&s), Turn::Chance);
    food_at(&g, &mut s, 0, 0);
    assert_eq!(g.turn(&s), Turn::Player(0));
    g.apply(&mut s, DuelAction::Move(Dir::Right));
    assert_eq!(g.turn(&s), Turn::Player(1), "seat 1 moves seeing seat 0");
    assert_eq!(s.pending(), Some(Dir::Right));
    g.apply(&mut s, DuelAction::Move(Dir::Left));
    assert_eq!(s.steps(), 1, "the tick advanced on seat 1's commit");
    assert_eq!(s.pending(), None);
}

#[test]
fn both_snakes_advance_one_cell_per_tick() {
    let g = Duel::new();
    let mut s = g.initial_state();
    let (h0, h1) = (s.worm(0).head(), s.worm(1).head());
    tick(&g, &mut s, Dir::Right, Dir::Left);
    assert_eq!(s.worm(0).head(), (h0.0 + 1, h0.1));
    assert_eq!(s.worm(1).head(), (h1.0 - 1, h1.1));
    assert_eq!(s.worm(0).len(), 3, "no growth without food");
    assert_eq!(s.worm(1).len(), 3);
}

#[test]
fn a_snake_into_a_wall_dies_and_the_other_wins() {
    let g = Duel::new();
    let mut s = g.initial_state();
    // Seat 0 drives up toward the top wall (y=10 → 0 over ten ticks, then off);
    // seat 1 coasts left along its row, staying well clear and alive.
    for _ in 0..(SIDE / 2) {
        tick(&g, &mut s, Dir::Up, Dir::Left);
        assert!(!g.is_terminal(&s));
    }
    assert_eq!(s.worm(0).head().1, 0);
    tick(&g, &mut s, Dir::Up, Dir::Left);
    assert!(g.is_terminal(&s));
    assert_eq!(s.outcome(), Outcome::Win(1));
    assert_eq!(g.returns(&s, 1), 1.0);
    assert_eq!(g.returns(&s, 0), -1.0);
    assert!(!s.worm(0).alive() && s.worm(1).alive());
}

#[test]
fn reversing_into_own_neck_is_ignored() {
    let g = Duel::new();
    let mut s = g.initial_state();
    // Seat 0 heads Right; asking for Left (a 180° reverse) keeps it Right.
    let before = s.worm(0).head();
    tick(&g, &mut s, Dir::Left, Dir::Up);
    assert!(s.worm(0).alive(), "reverse is ignored, not fatal");
    assert_eq!(s.worm(0).head(), (before.0 + 1, before.1));
    assert_eq!(s.worm(0).heading(), Dir::Right);
}

#[test]
fn eating_food_grows_and_keeps_the_tail() {
    let g = Duel::new();
    let mut s = g.initial_state();
    // Drop food directly ahead of seat 0's head (x=5, mid row).
    food_at(&g, &mut s, 5, SIDE / 2);
    assert_eq!(g.turn(&s), Turn::Player(0));
    g.apply(&mut s, DuelAction::Move(Dir::Right));
    g.apply(&mut s, DuelAction::Move(Dir::Up));
    assert_eq!(s.worm(0).len(), 4, "seat 0 ate and grew");
    assert_eq!(s.worm(1).len(), 3);
    assert_eq!(s.food(), None, "food consumed");
    assert_eq!(g.turn(&s), Turn::Chance, "a new food spawns");
}

#[test]
fn head_on_collision_kills_the_shorter_snake() {
    let g = Duel::new();
    let mut s = g.initial_state();
    // Feed seat 0 once so it is longer, then steer both heads to the same cell.
    food_at(&g, &mut s, 5, SIDE / 2);
    g.apply(&mut s, DuelAction::Move(Dir::Right));
    g.apply(&mut s, DuelAction::Move(Dir::Left));
    assert_eq!(s.worm(0).len(), 4);
    assert_eq!(s.worm(1).len(), 3);
    // Heads at (5,m) and (14,m): march toward each other until adjacent, then
    // both into the shared middle cell.
    while s.worm(1).head().0 - s.worm(0).head().0 > 1 {
        tick(&g, &mut s, Dir::Right, Dir::Left);
        assert!(!g.is_terminal(&s));
    }
    tick(&g, &mut s, Dir::Right, Dir::Left);
    assert!(g.is_terminal(&s));
    assert_eq!(s.outcome(), Outcome::Win(0), "the longer snake survives");
    assert!(s.worm(0).alive() && !s.worm(1).alive());
}

#[test]
fn equal_length_head_on_collision_is_a_draw() {
    let g = Duel::new();
    let mut s = g.initial_state();
    // Both length 3, heads at (4,m) and (15,m): an odd gap (11) lets them meet
    // on a shared cell. March them together.
    while s.worm(1).head().0 - s.worm(0).head().0 > 1 {
        tick(&g, &mut s, Dir::Right, Dir::Left);
        assert!(!g.is_terminal(&s));
    }
    assert_eq!(s.worm(0).len(), s.worm(1).len());
    tick(&g, &mut s, Dir::Right, Dir::Left);
    assert!(g.is_terminal(&s));
    assert_eq!(s.outcome(), Outcome::Draw);
    assert_eq!(g.returns(&s, 0), 0.0);
    assert_eq!(g.returns(&s, 1), 0.0);
}

#[test]
fn running_into_the_opponent_body_is_fatal() {
    let g = Duel::new();
    let mut s = g.initial_state();
    let m = SIDE / 2;
    // Seat 1 turns up and climbs, laying a vertical body at x = SIDE-5 along
    // rows m, m-1, m-2 (its tail vacates the start row). Seat 0 climbs to that
    // row from the left and turns into seat 1's neck cell at (SIDE-5, m-2).
    //
    // Build seat 1's wall while seat 0 marches up the left side, then steer
    // seat 0 across the top toward seat 1's column.
    tick(&g, &mut s, Dir::Up, Dir::Up); // s1 head (SIDE-5, m-1)
    tick(&g, &mut s, Dir::Up, Dir::Up); // s1 head (SIDE-5, m-2), body climbs
    // Seat 1 now occupies x=SIDE-5 at rows m-2..=m? Tail vacated; body is the
    // last three head cells: (SIDE-5, m-2), (SIDE-5, m-1), (SIDE-5, m).
    let neck = (SIDE - 5, m - 1);
    assert!(
        s.worm(1).cells().any(|c| c == neck),
        "seat 1 laid a vertical wall"
    );
    // Seat 0's head is at (4, m-2). Run it right toward seat 1's column, then
    // it must crash on the wall (a non-tail body cell). Hold seat 1 still by
    // looping it tightly so its wall persists around `neck`.
    let s1_loop = [Dir::Left, Dir::Down, Dir::Right, Dir::Up];
    let mut i = 0;
    while !g.is_terminal(&s) {
        let before_s0_alive = s.worm(0).alive();
        tick(&g, &mut s, Dir::Right, s1_loop[i % 4]);
        i += 1;
        if g.is_terminal(&s) {
            assert!(before_s0_alive);
            break;
        }
        assert!(s.steps() < g.step_cap(), "should crash before the cap");
    }
    assert!(g.is_terminal(&s));
    assert!(!s.worm(0).alive(), "seat 0 crashed");
    assert_ne!(s.outcome(), Outcome::Win(0));
}

#[test]
fn step_cap_decides_on_length() {
    let g = Duel::new();
    let mut s = g.initial_state();
    // Feed seat 0 once so it leads on length, then have both circle safely
    // around their own corners until the cap. A 2x2 clockwise loop never
    // crashes and never eats (we keep food in a corner the loops avoid).
    food_at(&g, &mut s, 5, SIDE / 2);
    g.apply(&mut s, DuelAction::Move(Dir::Right));
    g.apply(&mut s, DuelAction::Move(Dir::Left));
    assert_eq!(s.worm(0).len(), 4);

    let cap = g.step_cap();
    let loop_dirs = [Dir::Up, Dir::Right, Dir::Down, Dir::Left];
    let mut i = 0;
    while !g.is_terminal(&s) {
        // Keep food parked in a corner neither circling snake reaches.
        if matches!(g.turn(&s), Turn::Chance) {
            food_at(&g, &mut s, 0, 0);
        }
        let d = loop_dirs[i % 4];
        g.apply(&mut s, DuelAction::Move(d));
        g.apply(&mut s, DuelAction::Move(d));
        i += 1;
        assert!(s.steps() <= cap, "did not exceed the cap");
    }
    assert_eq!(s.steps(), cap);
    assert_eq!(s.outcome(), Outcome::Win(0), "longer snake wins on the cap");
}

#[test]
fn random_playthroughs_terminate_within_bound() {
    let g = Duel::new();
    let bound = g.step_cap() + 4;
    let mut rng = Rng::new(99);
    for _ in 0..400 {
        let mut s = g.initial_state();
        let mut steps = 0;
        while !g.is_terminal(&s) {
            steps += 1;
            assert!(steps <= 3 * bound, "exceeded the step bound");
            match g.turn(&s) {
                Turn::Chance => {
                    let outs = g.chance_outcomes(&s);
                    let i = game_core::rand::sample_outcome(&outs, &mut rng);
                    g.apply(&mut s, outs[i].0);
                }
                Turn::Player(_) => {
                    let acts = g.legal_actions(&s);
                    let i = rng.below(acts.len());
                    g.apply(&mut s, acts[i]);
                }
            }
        }
        assert!(s.steps() <= g.step_cap());
        let r = g.returns(&s, 0);
        assert!(r == 1.0 || r == -1.0 || r == 0.0);
        assert_eq!(g.returns(&s, 0) + g.returns(&s, 1), 0.0, "zero-sum");
    }
}

#[test]
fn eval_prefers_more_territory_and_length() {
    let g = Duel::new();
    let eval = DuelEval;
    let mut s = g.initial_state();
    food_at(&g, &mut s, 5, SIDE / 2);
    let base0 = eval.eval(&g, &s, 0);
    // Eval is antisymmetric between the two seats in a symmetric-ish position.
    let base1 = eval.eval(&g, &s, 1);
    assert!(
        (base0 + base1).abs() < 0.5,
        "roughly zero-sum: {base0} {base1}"
    );

    // After seat 0 eats, its evaluation should rise relative to before.
    g.apply(&mut s, DuelAction::Move(Dir::Right));
    g.apply(&mut s, DuelAction::Move(Dir::Up));
    assert_eq!(s.worm(0).len(), 4);
    food_at(&g, &mut s, 0, 0);
    assert!(eval.eval(&g, &s, 0) > 0.0, "the longer snake is favoured");
    assert!(eval.eval(&g, &s, 1) < eval.eval(&g, &s, 0));
}

#[test]
fn view_data_schema_round_trips() {
    let g = Duel::new();
    let mut s = g.initial_state();
    food_at(&g, &mut s, 7, 2);
    let json = g.view_data(&s, 0).unwrap();
    assert!(json.contains("\"side\":20"), "{json}");
    assert!(json.contains("\"food\":[7,2]"), "{json}");
    assert!(json.contains("\"outcome\":\"ongoing\""), "{json}");
    assert!(json.contains("\"dir\":\"e\""), "seat 0 faces east: {json}");
    assert!(json.contains("\"dir\":\"w\""), "seat 1 faces west: {json}");
    assert!(json.contains("\"alive\":true"), "{json}");
    // Head-first cells: seat 0's head is at (4, 10).
    assert!(json.contains("[[4,10],[3,10],[2,10]]"), "{json}");
}

#[test]
fn render_marks_both_snakes_and_the_viewer() {
    let g = Duel::new();
    let mut s = g.initial_state();
    food_at(&g, &mut s, 7, 2);
    let view = g.render(&s, 1);
    assert!(
        view.contains('A') && view.contains('B'),
        "two heads:\n{view}"
    );
    assert!(view.contains('*'), "food glyph");
    assert!(view.contains("you are Snake B"));
    assert_eq!(g.action_label(&s, DuelAction::Move(Dir::Up)), "up");
    assert_eq!(g.parse_action(&s, "up"), Some(DuelAction::Move(Dir::Up)));
    assert_eq!(
        g.parse_action(&s, "ArrowLeft"),
        None,
        "raw key names are the frontend's job"
    );
    assert_eq!(g.id(), "snake");
}

#[test]
fn state_key_distinguishes_pending_and_advanced_states() {
    let g = Duel::new();
    let mut s = g.initial_state();
    food_at(&g, &mut s, 0, 0);
    let pre = g.state_key(&s);
    let mut committed = s.clone();
    g.apply(&mut committed, DuelAction::Move(Dir::Right));
    assert_ne!(
        pre,
        g.state_key(&committed),
        "seat 0's commit changes the key"
    );
    let mut advanced = committed.clone();
    g.apply(&mut advanced, DuelAction::Move(Dir::Left));
    assert_ne!(g.state_key(&committed), g.state_key(&advanced));
}
