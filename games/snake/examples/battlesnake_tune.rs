//! Reproducible evaluator tuning and feature ablation for Battlesnake BNS.
//!
//! SPSA estimates a gradient with two evaluations regardless of parameter
//! count, making it a good fit for a noisy win/loss objective. Every estimate
//! uses common seeds and seat-swapped games. The `ablate` mode zeroes one
//! feature in both phases at a time and measures the full evaluator against it.

use std::time::Duration;

use game_core::{Rng, SimultaneousGame, SimultaneousTurn};
use snake::battlesnake::search::{
    EvaluationWeights, FEATURE_NAMES, OpponentModel, SearchConfig, Searcher,
};
use snake::battlesnake::{Battlesnake, Direction, Rules, SIDE, bit, cell, xy};

fn arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    let args: Vec<_> = std::env::args().collect();
    args.windows(2)
        .find(|window| window[0] == name)
        .and_then(|window| window[1].parse().ok())
        .unwrap_or(default)
}

fn searcher(weights: EvaluationWeights, millis: u64, depth: u8) -> Searcher<2> {
    Searcher::new(SearchConfig {
        time_limit: Duration::from_millis(millis),
        max_depth: depth,
        quiescence_depth: 1,
        opponent_model: OpponentModel::Full,
        tt_bits: 15,
        weights,
    })
}

fn random_nonfatal_opening_action(
    game: &Battlesnake<2>,
    state: &snake::battlesnake::BoardState<2>,
    seat: usize,
    rng: &mut Rng,
) -> Direction {
    let mut obstacles = 0u128;
    for snake in state.snakes() {
        for part in snake.cells().take(snake.len().saturating_sub(1)) {
            obstacles |= bit(part);
        }
    }
    let snake = state.snake(seat);
    let (x, y) = xy(snake.head());
    let safe: Vec<_> = Direction::ALL
        .into_iter()
        .filter(|direction| {
            let (dx, dy) = match direction {
                Direction::Up => (0, 1),
                Direction::Right => (1, 0),
                Direction::Down => (0, -1),
                Direction::Left => (-1, 0),
            };
            let (nx, ny) = (i16::from(x) + dx, i16::from(y) + dy);
            if !(0..SIDE as i16).contains(&nx) || !(0..SIDE as i16).contains(&ny) {
                return false;
            }
            let destination = cell(nx as u8, ny as u8);
            obstacles & bit(destination) == 0
                && !(state.hazards() & bit(destination) != 0
                    && state.food() & bit(destination) == 0
                    && snake.health() <= game.rules().hazard_damage + 1)
        })
        .collect();
    if safe.is_empty() {
        Direction::ALL[rng.below(4)]
    } else {
        safe[rng.below(safe.len())]
    }
}

#[allow(clippy::too_many_arguments)]
fn game(
    candidate: EvaluationWeights,
    reference: EvaluationWeights,
    candidate_seat: usize,
    seed: u64,
    millis: u64,
    depth: u8,
    opening_turns: u16,
    turn_cap: u16,
) -> f64 {
    let game = Battlesnake::<2>::new(Rules {
        seed,
        ..Rules::default()
    });
    let mut state = game.initial_state();
    let mut candidate_search = searcher(candidate, millis, depth);
    let mut reference_search = searcher(reference, millis, depth);
    let mut rng = Rng::new(seed ^ 0xA81A_7100);
    while !game.is_terminal(&state) && state.turn_number() < turn_cap {
        while !game.is_terminal(&state) && game.turn(&state) == SimultaneousTurn::Chance {
            let chance = game.sample_chance_action(&state, &mut rng);
            game.apply_chance(&mut state, chance);
        }
        if game.is_terminal(&state) {
            break;
        }
        let mut joint = [Direction::Up; 2];
        for (seat, action) in joint.iter_mut().enumerate() {
            *action = if state.turn_number() < opening_turns {
                random_nonfatal_opening_action(&game, &state, seat, &mut rng)
            } else if seat == candidate_seat {
                candidate_search.search(&game, &state, seat).action
            } else {
                reference_search.search(&game, &state, seat).action
            };
        }
        game.apply_joint(&mut state, &joint);
    }
    if game.is_terminal(&state) {
        game.returns(&state, candidate_seat)
    } else {
        0.0
    }
}

#[allow(clippy::too_many_arguments)]
fn paired_score(
    candidate: EvaluationWeights,
    reference: EvaluationWeights,
    pairs: u64,
    seed: u64,
    millis: u64,
    depth: u8,
    opening_turns: u16,
    turn_cap: u16,
) -> f64 {
    paired_results(
        candidate,
        reference,
        pairs,
        seed,
        millis,
        depth,
        opening_turns,
        turn_cap,
    )
    .0
}

#[allow(clippy::too_many_arguments)]
fn paired_results(
    candidate: EvaluationWeights,
    reference: EvaluationWeights,
    pairs: u64,
    seed: u64,
    millis: u64,
    depth: u8,
    opening_turns: u16,
    turn_cap: u16,
) -> (f64, u64, u64, u64) {
    let outcomes: Vec<_> = (0..pairs)
        .flat_map(|pair| {
            let game_seed = game_core::hash::combine(seed, pair);
            [
                game(
                    candidate,
                    reference,
                    0,
                    game_seed,
                    millis,
                    depth,
                    opening_turns,
                    turn_cap,
                ),
                game(
                    candidate,
                    reference,
                    1,
                    game_seed,
                    millis,
                    depth,
                    opening_turns,
                    turn_cap,
                ),
            ]
        })
        .collect();
    let wins = outcomes.iter().filter(|&&outcome| outcome > 0.0).count() as u64;
    let losses = outcomes.iter().filter(|&&outcome| outcome < 0.0).count() as u64;
    let draws = outcomes.len() as u64 - wins - losses;
    let average_return = outcomes.iter().sum::<f64>() / outcomes.len().max(1) as f64;
    (average_return, wins, draws, losses)
}

fn perturb(weights: EvaluationWeights, signs: &[[i16; 12]; 2], amount: i16) -> EvaluationWeights {
    let mut out = weights;
    for (index, (&early_sign, &late_sign)) in signs[0].iter().zip(&signs[1]).enumerate() {
        out.early[index] = (out.early[index] + amount * early_sign).clamp(-100, 100);
        out.late[index] = (out.late[index] + amount * late_sign).clamp(-100, 100);
    }
    out
}

fn print_weights(weights: EvaluationWeights) {
    println!("early={:?}", weights.early);
    println!("late ={:?}", weights.late);
}

fn tune() {
    let iterations: u64 = arg("--iters", 8);
    let pairs: u64 = arg("--pairs", 12);
    let millis: u64 = arg("--millis", 10);
    let depth: u8 = arg("--depth", 5);
    // The canonical stacked opening makes uniformly random moves overwhelmingly
    // suicidal. Initial-layout and food seeds already provide paired diversity;
    // opt into random opening noise explicitly when studying robustness.
    let opening_turns: u16 = arg("--opening", 0);
    let turn_cap: u16 = arg("--turn-cap", 300);
    let delta: i16 = arg("--delta", 2);
    let step: i16 = arg("--step", 1);
    let seed: u64 = arg("--seed", 0x5A5A_2026);
    let gate_pairs: u64 = arg("--gate-pairs", 64);
    let original = EvaluationWeights::shapeshifter_standard();
    let mut weights = original;
    let mut rng = Rng::new(seed);
    for iteration in 0..iterations {
        let signs = std::array::from_fn(|_| {
            std::array::from_fn(|_| if rng.below(2) == 0 { -1 } else { 1 })
        });
        let plus = perturb(weights, &signs, delta);
        let minus = perturb(weights, &signs, -delta);
        let comparison_seed = game_core::hash::combine(seed, iteration);
        let plus_score = paired_score(
            plus,
            weights,
            pairs,
            comparison_seed,
            millis,
            depth,
            opening_turns,
            turn_cap,
        );
        let minus_score = paired_score(
            minus,
            weights,
            pairs,
            comparison_seed,
            millis,
            depth,
            opening_turns,
            turn_cap,
        );
        let direction = (plus_score - minus_score).total_cmp(&0.0);
        if direction != std::cmp::Ordering::Equal {
            let signed_step = if direction == std::cmp::Ordering::Greater {
                step
            } else {
                -step
            };
            weights = perturb(weights, &signs, signed_step);
        }
        let gate = paired_score(
            weights,
            original,
            pairs,
            game_core::hash::combine(seed ^ 0x6A7E, iteration),
            millis,
            depth,
            opening_turns,
            turn_cap,
        );
        println!(
            "iter {}: plus {plus_score:+.3}, minus {minus_score:+.3}, tuned-vs-original {gate:+.3}",
            iteration + 1
        );
        print_weights(weights);
    }
    let held_out = paired_results(
        weights,
        original,
        gate_pairs,
        seed ^ 0xF1A1_6A7E,
        millis,
        depth,
        opening_turns,
        turn_cap,
    );
    println!(
        "held-out final-vs-original: return {:+.3}, score {:.3}, {}-{}-{} over {} games",
        held_out.0,
        (held_out.0 + 1.0) / 2.0,
        held_out.1,
        held_out.2,
        held_out.3,
        gate_pairs * 2,
    );
}

fn ablate() {
    let pairs: u64 = arg("--pairs", 16);
    let millis: u64 = arg("--millis", 10);
    let depth: u8 = arg("--depth", 5);
    let opening_turns: u16 = arg("--opening", 0);
    let turn_cap: u16 = arg("--turn-cap", 300);
    let seed: u64 = arg("--seed", 0xAB1A_7E00);
    let full = EvaluationWeights::shapeshifter_standard();
    println!("feature ablation: positive means the full evaluator beat the ablation");
    for (index, name) in FEATURE_NAMES.iter().enumerate() {
        let mut ablated = full;
        ablated.early[index] = 0;
        ablated.late[index] = 0;
        let score = paired_score(
            full,
            ablated,
            pairs,
            game_core::hash::combine(seed, index as u64),
            millis,
            depth,
            opening_turns,
            turn_cap,
        );
        println!("{name:>24}: {score:+.3}");
    }
}

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("tune") => tune(),
        Some("ablate") => ablate(),
        _ => {
            eprintln!("usage: battlesnake_tune <tune|ablate> [--pairs N --millis N --depth N ...]");
            std::process::exit(2);
        }
    }
}
