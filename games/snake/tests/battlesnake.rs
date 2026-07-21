use game_core::{Rng, SimultaneousGame, SimultaneousTurn};
use snake::battlesnake::{
    BattleSnake, Battlesnake, ChanceAction, Direction, Elimination, InitialFood, Mode, Rules, bit,
    cell,
};

fn no_food_rules() -> Rules {
    Rules {
        food_spawn_chance: 0,
        minimum_food: 0,
        ..Rules::default()
    }
}

fn body(points: &[(u8, u8)], health: i16, heading: Direction) -> BattleSnake {
    let cells: Vec<_> = points.iter().map(|&(x, y)| cell(x, y)).collect();
    BattleSnake::from_cells(&cells, health, heading)
}

#[test]
fn standard_start_is_official_stacked_layout_with_fixed_food() {
    let game = Battlesnake::<4>::standard();
    let state = game.initial_state();
    assert_eq!(game.num_players(), 4);
    assert_eq!(state.turn_number(), 0);
    assert_eq!(state.alive_count(), 4);
    assert_eq!(game.turn(&state), SimultaneousTurn::Players);
    assert_eq!(
        state.food().count_ones(),
        5,
        "one nearby food per snake plus center"
    );
    for snake in state.snakes() {
        assert_eq!(snake.len(), 3);
        assert_eq!(snake.health(), 100);
        assert!(snake.cells().all(|part| part == snake.head()));
    }
}

#[test]
fn seeded_initialization_varies_the_official_random_layout() {
    let game = Battlesnake::<4>::standard();
    let first = SimultaneousGame::initial_state_with_rng(&game, &mut Rng::new(1));
    let varied = (2..=8).any(|seed| {
        let state = SimultaneousGame::initial_state_with_rng(&game, &mut Rng::new(seed));
        state.food() != first.food()
            || state
                .snakes()
                .iter()
                .zip(first.snakes())
                .any(|(snake, baseline)| snake.head() != baseline.head())
    });

    assert!(
        varied,
        "different match seeds must explore different starts"
    );
}

#[test]
fn one_food_opening_starts_and_stays_at_exactly_one() {
    let game = Battlesnake::<2>::new(Rules {
        initial_food: InitialFood::One,
        food_spawn_chance: 0,
        minimum_food: 1,
        ..Rules::default()
    });
    let initial = game.initial_state();
    assert_eq!(initial.food().count_ones(), 1);

    let snakes = [
        body(&[(4, 5), (3, 5), (2, 5)], 100, Direction::Right),
        body(&[(9, 9), (9, 8), (9, 7)], 100, Direction::Up),
    ];
    let mut after_eat = snake::battlesnake::BoardState::from_parts(snakes, bit(cell(5, 5)), 0, 0);
    game.apply_joint(&mut after_eat, &[Direction::Right, Direction::Up]);
    assert_eq!(after_eat.food().count_ones(), 0);
    assert_eq!(game.turn(&after_eat), SimultaneousTurn::Chance);
    let outcomes = game.chance_outcomes(&after_eat);
    assert!(
        outcomes
            .iter()
            .all(|(action, _)| matches!(action, ChanceAction::PlaceFood(_)))
    );
    game.apply_chance(&mut after_eat, outcomes[0].0);
    assert_eq!(after_eat.food().count_ones(), 1);
    assert_eq!(game.turn(&after_eat), SimultaneousTurn::Players);
}

#[test]
fn moves_are_resolved_as_one_joint_action() {
    let game = Battlesnake::<2>::new(no_food_rules());
    let snakes = [
        body(&[(4, 5), (3, 5), (2, 5)], 100, Direction::Right),
        body(&[(6, 5), (7, 5), (8, 5)], 100, Direction::Left),
    ];
    let mut state = snake::battlesnake::BoardState::from_parts(snakes, 0, 0, 8);
    game.apply_joint(&mut state, &[Direction::Right, Direction::Left]);
    assert!(game.is_terminal(&state));
    assert_eq!(state.snake(0).head(), cell(5, 5));
    assert_eq!(state.snake(1).head(), cell(5, 5));
    assert_eq!(state.snake(0).elimination(), Elimination::HeadToHead);
    assert_eq!(state.snake(1).elimination(), Elimination::HeadToHead);
    assert_eq!(game.returns(&state, 0), 0.0);
    assert_eq!(game.returns(&state, 1), 0.0);
}

#[test]
fn swapping_heads_is_two_body_collisions_not_a_length_contest() {
    let game = Battlesnake::<2>::new(no_food_rules());
    let snakes = [
        body(&[(4, 5), (3, 5), (2, 5), (1, 5)], 100, Direction::Right),
        body(&[(5, 5), (6, 5), (7, 5)], 100, Direction::Left),
    ];
    let mut state = snake::battlesnake::BoardState::from_parts(snakes, 0, 0, 0);
    game.apply_joint(&mut state, &[Direction::Right, Direction::Left]);
    assert_eq!(state.snake(0).head(), cell(5, 5));
    assert_eq!(state.snake(1).head(), cell(4, 5));
    assert_eq!(state.snake(0).elimination(), Elimination::BodyCollision);
    assert_eq!(state.snake(1).elimination(), Elimination::BodyCollision);
}

#[test]
fn feeding_happens_before_head_to_head_length_comparison() {
    let game = Battlesnake::<2>::new(no_food_rules());
    let snakes = [
        body(&[(4, 5), (3, 5), (2, 5)], 1, Direction::Right),
        body(&[(6, 5), (7, 5)], 100, Direction::Left),
    ];
    let mut state =
        snake::battlesnake::BoardState::from_parts(snakes, bit(cell(5, 5)), bit(cell(5, 5)), 0);
    game.apply_joint(&mut state, &[Direction::Right, Direction::Left]);
    assert!(state.snake(0).is_alive());
    assert_eq!(state.snake(0).len(), 4);
    assert_eq!(
        state.snake(0).health(),
        100,
        "food beats starvation and hazard damage"
    );
    assert_eq!(state.snake(1).elimination(), Elimination::HeadToHead);
    assert_eq!(state.food(), 0);
}

#[test]
fn old_tail_vacates_when_eating_but_a_doubled_tail_stays() {
    let game = Battlesnake::<2>::new(no_food_rules());
    let pursuer = body(&[(1, 1), (1, 0), (0, 0)], 100, Direction::Right);
    let leader = body(&[(4, 1), (3, 1), (2, 1)], 100, Direction::Right);

    let mut safe = snake::battlesnake::BoardState::from_parts([pursuer, leader], 0, 0, 0);
    game.apply_joint(&mut safe, &[Direction::Right, Direction::Right]);
    assert!(
        safe.snake(0).is_alive(),
        "the old tail at (2,1) was vacated"
    );

    let pursuer = body(&[(1, 1), (1, 0), (0, 0)], 100, Direction::Right);
    let leader = body(&[(4, 1), (3, 1), (2, 1)], 100, Direction::Right);
    let mut eating =
        snake::battlesnake::BoardState::from_parts([pursuer, leader], bit(cell(5, 1)), 0, 0);
    game.apply_joint(&mut eating, &[Direction::Right, Direction::Right]);
    assert_eq!(eating.snake(0).head(), cell(2, 1));
    assert!(
        eating.snake(0).is_alive(),
        "eating still drops the pre-move tail"
    );
    assert_eq!(eating.snake(1).len(), 4);
    assert_eq!(
        eating.snake(1).cells().collect::<Vec<_>>(),
        [cell(5, 1), cell(4, 1), cell(3, 1), cell(3, 1)]
    );

    let pursuer = body(&[(1, 1), (1, 0), (0, 0)], 100, Direction::Right);
    let doubled = body(&[(4, 1), (3, 1), (2, 1), (2, 1)], 100, Direction::Right);
    let mut next_turn = snake::battlesnake::BoardState::from_parts([pursuer, doubled], 0, 0, 1);
    game.apply_joint(&mut next_turn, &[Direction::Right, Direction::Right]);
    assert_eq!(
        next_turn.snake(0).elimination(),
        Elimination::BodyCollision,
        "one copy of a doubled tail remains occupied"
    );
}

#[test]
fn longer_snake_is_only_survivor_of_multiway_head_collision() {
    let game = Battlesnake::<4>::new(no_food_rules());
    let snakes = [
        body(&[(5, 4), (5, 3)], 100, Direction::Up),
        body(&[(6, 5), (7, 5)], 100, Direction::Left),
        body(&[(5, 6), (5, 7), (5, 8)], 100, Direction::Down),
        body(&[(4, 5), (3, 5)], 100, Direction::Right),
    ];
    let mut state = snake::battlesnake::BoardState::from_parts(snakes, 0, 0, 0);
    game.apply_joint(
        &mut state,
        &[
            Direction::Up,
            Direction::Left,
            Direction::Down,
            Direction::Right,
        ],
    );
    assert_eq!(state.alive_count(), 1);
    assert!(state.snake(2).is_alive());
    assert_eq!(game.returns(&state, 2), 1.0);
}

#[test]
fn food_chance_matches_minimum_then_spawn_probability() {
    let rules = Rules {
        food_spawn_chance: 25,
        minimum_food: 1,
        ..Rules::default()
    };
    let game = Battlesnake::<2>::new(rules);
    let snakes = [
        body(&[(1, 1), (1, 0), (0, 0)], 100, Direction::Up),
        body(&[(9, 9), (9, 10), (10, 10)], 100, Direction::Down),
    ];
    let mut missing = snake::battlesnake::BoardState::from_parts(snakes, 0, 0, 0);
    game.apply_joint(&mut missing, &[Direction::Up, Direction::Down]);
    assert_eq!(game.turn(&missing), SimultaneousTurn::Chance);
    let required = game.chance_outcomes(&missing);
    assert!(
        required
            .iter()
            .all(|(action, _)| matches!(action, ChanceAction::PlaceFood(_)))
    );
    assert!((required.iter().map(|(_, p)| p).sum::<f64>() - 1.0).abs() < 1e-12);
    game.apply_chance(&mut missing, required[0].0);
    assert_eq!(missing.food().count_ones(), 1);
    assert_eq!(game.turn(&missing), SimultaneousTurn::Players);

    game.apply_joint(&mut missing, &[Direction::Up, Direction::Down]);
    let roll = game.chance_outcomes(&missing);
    let no_food = roll
        .iter()
        .find(|(action, _)| *action == ChanceAction::NoFood)
        .expect("no-spawn outcome");
    assert!(
        (no_food.1 - 0.76).abs() < 1e-12,
        "the live standard map's strict inequality makes setting 25 an effective 24%"
    );
    assert!((roll.iter().map(|(_, p)| p).sum::<f64>() - 1.0).abs() < 1e-12);
}

#[test]
fn random_food_never_spawns_on_a_possible_next_head_cell() {
    let game = Battlesnake::<2>::new(Rules {
        food_spawn_chance: 100,
        minimum_food: 0,
        ..Rules::default()
    });
    let snakes = [
        body(&[(5, 5), (5, 4), (5, 3)], 100, Direction::Up),
        body(&[(9, 9), (9, 8), (9, 7)], 100, Direction::Up),
    ];
    let mut state = snake::battlesnake::BoardState::from_parts(snakes, 0, 0, 0);
    game.apply_joint(&mut state, &[Direction::Up, Direction::Up]);
    let forbidden = [cell(5, 7), cell(6, 6), cell(5, 5), cell(4, 6)];
    for (action, _) in game.chance_outcomes(&state) {
        if let ChanceAction::PlaceFood(cell) = action {
            assert!(!forbidden.contains(&cell));
        }
    }
}

#[test]
fn wrapped_and_constrictor_modes_apply_their_canonical_modifiers() {
    let wrapped = Battlesnake::<2>::new(Rules {
        mode: Mode::Wrapped,
        food_spawn_chance: 0,
        minimum_food: 0,
        ..Rules::default()
    });
    let snakes = [
        body(&[(0, 5), (1, 5), (2, 5)], 100, Direction::Left),
        body(&[(9, 9), (9, 8), (9, 7)], 100, Direction::Up),
    ];
    let mut state = snake::battlesnake::BoardState::from_parts(snakes, 0, 0, 0);
    wrapped.apply_joint(&mut state, &[Direction::Left, Direction::Up]);
    assert_eq!(state.snake(0).head(), cell(10, 5));
    assert!(state.snake(0).is_alive());

    let constrictor = Battlesnake::<2>::new(Rules {
        mode: Mode::Constrictor,
        ..no_food_rules()
    });
    let snakes = [
        body(&[(2, 2), (2, 1), (2, 0)], 7, Direction::Up),
        body(&[(8, 8), (8, 7), (8, 6)], 4, Direction::Up),
    ];
    let mut state = snake::battlesnake::BoardState::from_parts(snakes, 0, 0, 0);
    constrictor.apply_joint(&mut state, &[Direction::Up, Direction::Up]);
    assert_eq!(state.snake(0).len(), 4);
    assert_eq!(state.snake(1).len(), 4);
    assert_eq!(state.snake(0).health(), 100);
    assert_eq!(state.food(), 0);
}

#[test]
fn wrapped_edge_crossing_still_collides_with_a_body() {
    let game = Battlesnake::<2>::new(Rules {
        mode: Mode::Wrapped,
        food_spawn_chance: 0,
        minimum_food: 0,
        ..Rules::default()
    });
    let snakes = [
        body(&[(0, 5)], 10, Direction::Left),
        body(
            &[(10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6)],
            10,
            Direction::Down,
        ),
    ];
    let mut state = snake::battlesnake::BoardState::from_parts(snakes, 0, 0, 0);
    game.apply_joint(&mut state, &[Direction::Left, Direction::Down]);
    assert_eq!(state.snake(0).head(), cell(10, 5));
    assert_eq!(state.snake(0).elimination(), Elimination::BodyCollision);
    assert!(state.snake(1).is_alive());
}

#[test]
fn eating_across_a_wrapped_edge_duplicates_the_post_move_tail() {
    let game = Battlesnake::<2>::new(Rules {
        mode: Mode::Wrapped,
        food_spawn_chance: 0,
        minimum_food: 0,
        ..Rules::default()
    });
    let snakes = [
        body(&[(0, 5), (1, 5)], 10, Direction::Left),
        body(&[(5, 5)], 10, Direction::Left),
    ];
    let mut state = snake::battlesnake::BoardState::from_parts(snakes, bit(cell(10, 5)), 0, 0);
    game.apply_joint(&mut state, &[Direction::Left, Direction::Left]);
    assert_eq!(state.snake(0).health(), 100);
    assert_eq!(
        state.snake(0).cells().collect::<Vec<_>>(),
        [cell(10, 5), cell(0, 5), cell(0, 5)]
    );
}

#[test]
fn royale_shrinks_after_the_configured_turn() {
    let game = Battlesnake::<2>::new(Rules {
        mode: Mode::Royale,
        food_spawn_chance: 0,
        minimum_food: 0,
        shrink_every_n_turns: 1,
        seed: 9,
        ..Rules::default()
    });
    let snakes = [
        body(&[(2, 2), (2, 1), (2, 0)], 100, Direction::Up),
        body(&[(8, 8), (8, 7), (8, 6)], 100, Direction::Up),
    ];
    let mut state = snake::battlesnake::BoardState::from_parts(snakes, 0, 0, 0);
    game.apply_joint(&mut state, &[Direction::Up, Direction::Up]);
    assert_ne!(state.hazards(), 0);
}
