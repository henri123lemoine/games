//! Canonical simultaneous evaluation against fixed, non-leaking baselines.

use std::time::Duration;

use game_core::{Rng, SimultaneousGame, SimultaneousTurn};
use snake::battlesnake::search::{OpponentModel, SearchConfig, Searcher};
use snake::battlesnake::{Battlesnake, BoardState, Direction, Rules};

use super::sim_selfplay::{SolveConfig, evaluate_states, mix};
use crate::net::Infer;

const TURN_CAP: u16 = 1_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Opponent {
    Random,
    Search { millis: u64, depth: u8 },
}

impl Opponent {
    pub fn name(self) -> String {
        match self {
            Self::Random => "random".into(),
            Self::Search { millis, depth } => format!("bns-{millis}ms-d{depth}"),
        }
    }
}

pub struct LadderEntry {
    pub name: String,
    pub score: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub wins: u32,
    pub draws: u32,
    pub losses: u32,
}

struct EvalGame {
    game: Battlesnake<2>,
    state: BoardState<2>,
    opponent: Opponent,
    net_seat: usize,
    searcher: Option<Searcher<2>>,
    chance_rng: Rng,
    opponent_rng: Rng,
    net_rng: Rng,
    outcome: Option<f64>,
}

impl EvalGame {
    fn new(seed: u64, opponent: Opponent, net_seat: usize) -> Self {
        let game = Battlesnake::new(Rules {
            seed,
            ..Rules::default()
        });
        let state = game.initial_state();
        let searcher = match opponent {
            Opponent::Random => None,
            Opponent::Search { millis, depth } => Some(Searcher::new(SearchConfig {
                time_limit: Duration::from_millis(millis),
                max_depth: depth,
                quiescence_depth: 2,
                opponent_model: OpponentModel::Full,
                tt_bits: 12,
                ..SearchConfig::default()
            })),
        };
        Self {
            game,
            state,
            opponent,
            net_seat,
            searcher,
            // Seat-swapped games use identical chance and opponent streams.
            // Separate streams keep random-opponent choices from perturbing
            // food placement when game trajectories remain paired.
            chance_rng: Rng::new(mix(seed, 0xC11A_CE00)),
            opponent_rng: Rng::new(mix(seed, 0x0FF0_5E70)),
            net_rng: Rng::new(mix(seed, 0x0E71_0000)),
            outcome: None,
        }
    }

    fn opponent_action(&mut self) -> Direction {
        let seat = 1 - self.net_seat;
        match self.opponent {
            Opponent::Random => Direction::ALL[self.opponent_rng.below(4)],
            Opponent::Search { .. } => {
                self.searcher
                    .as_mut()
                    .expect("search opponent")
                    .search(&self.game, &self.state, seat)
                    .action
            }
        }
    }

    fn finish_if_needed(&mut self) {
        if self.game.is_terminal(&self.state) {
            self.outcome = Some(self.game.returns(&self.state, self.net_seat));
        } else if self.state.turn_number() >= TURN_CAP {
            self.outcome = Some(0.0);
        }
    }
}

pub fn ladder(
    infer: &Infer,
    solve: SolveConfig,
    opponents: &[Opponent],
    pairs: u32,
    seed: u64,
) -> Vec<LadderEntry> {
    let mut games = Vec::new();
    for (opponent_index, &opponent) in opponents.iter().enumerate() {
        for pair in 0..pairs {
            let game_seed = mix(seed, (opponent_index as u64) << 32 | u64::from(pair));
            games.push(EvalGame::new(game_seed, opponent, 0));
            games.push(EvalGame::new(game_seed, opponent, 1));
        }
    }

    while games.iter().any(|game| game.outcome.is_none()) {
        for game in &mut games {
            if game.outcome.is_none() {
                resolve_chance(&game.game, &mut game.state, &mut game.chance_rng);
                game.finish_if_needed();
            }
        }
        let active: Vec<_> = games
            .iter()
            .enumerate()
            .filter(|(_, game)| game.outcome.is_none())
            .map(|(index, _)| index)
            .collect();
        if active.is_empty() {
            break;
        }
        let active_games: Vec<_> = active.iter().map(|&index| games[index].game).collect();
        let active_states: Vec<_> = active.iter().map(|&index| games[index].state).collect();
        let neural = evaluate_states(infer, &active_games, &active_states, solve);
        let opponents: Vec<_> = active
            .iter()
            .map(|&index| games[index].opponent_action())
            .collect();
        for ((&index, equilibrium), opponent_action) in active.iter().zip(neural).zip(opponents) {
            let game = &mut games[index];
            debug_assert!(equilibrium.values[game.net_seat].is_finite());
            let mut joint = [Direction::Up; 2];
            joint[game.net_seat] = Direction::ALL
                [sample_strategy(&equilibrium.strategies[game.net_seat], &mut game.net_rng)];
            joint[1 - game.net_seat] = opponent_action;
            game.game.apply_joint(&mut game.state, &joint);
        }
    }

    opponents
        .iter()
        .map(|&opponent| {
            let outcomes: Vec<_> = games
                .iter()
                .filter(|game| game.opponent == opponent)
                .map(|game| game.outcome.unwrap_or(0.0))
                .collect();
            let wins = outcomes.iter().filter(|&&value| value > 0.0).count() as u32;
            let losses = outcomes.iter().filter(|&&value| value < 0.0).count() as u32;
            let draws = outcomes.len() as u32 - wins - losses;
            let (score, ci_low, ci_high) = score_interval(wins, draws, losses, 0.5);
            LadderEntry {
                name: opponent.name(),
                score,
                ci_low,
                ci_high,
                wins,
                draws,
                losses,
            }
        })
        .collect()
}

struct FieldGame {
    game: Battlesnake<4>,
    state: BoardState<4>,
    opponent: Opponent,
    hero: usize,
    searchers: [Option<Searcher<4>>; 4],
    chance_rng: Rng,
    opponent_rngs: [Rng; 4],
    net_rng: Rng,
    outcome: Option<f64>,
}

impl FieldGame {
    fn new(seed: u64, opponent: Opponent, hero: usize) -> Self {
        let game = Battlesnake::new(Rules {
            seed,
            ..Rules::default()
        });
        let searchers = std::array::from_fn(|seat| match opponent {
            Opponent::Random => None,
            Opponent::Search { millis, depth } if seat != hero => {
                Some(Searcher::new(SearchConfig {
                    time_limit: Duration::from_millis(millis),
                    max_depth: depth,
                    quiescence_depth: 2,
                    opponent_model: OpponentModel::MoveCombination,
                    tt_bits: 12,
                    ..SearchConfig::default()
                }))
            }
            Opponent::Search { .. } => None,
        });
        Self {
            state: game.initial_state(),
            game,
            opponent,
            hero,
            searchers,
            chance_rng: Rng::new(mix(seed, 0xC11A_CE00)),
            opponent_rngs: std::array::from_fn(|seat| {
                Rng::new(mix(seed, 0x0FF0_5E70 + seat as u64))
            }),
            net_rng: Rng::new(mix(seed, 0x0E71_0000)),
            outcome: None,
        }
    }

    fn opponent_actions(&mut self) -> [Direction; 4] {
        std::array::from_fn(|seat| {
            if seat == self.hero || !self.game.is_active(&self.state, seat) {
                return Direction::Up;
            }
            match self.opponent {
                Opponent::Random => Direction::ALL[self.opponent_rngs[seat].below(4)],
                Opponent::Search { .. } => {
                    self.searchers[seat]
                        .as_mut()
                        .expect("field search opponent")
                        .search(&self.game, &self.state, seat)
                        .action
                }
            }
        })
    }

    fn finish_if_needed(&mut self) {
        if self.game.is_terminal(&self.state) {
            self.outcome = Some(self.game.returns(&self.state, self.hero));
        } else if self.state.turn_number() >= TURN_CAP {
            self.outcome = Some(0.0);
        }
    }
}

/// Four-player field evaluation with the neural hero rotated through every
/// seat. `score` is raw win share; fair play against three equal opponents is
/// therefore 0.25 rather than the duel convention of counting draws as half.
pub fn field_ladder(
    infer: &Infer,
    solve: SolveConfig,
    opponents: &[Opponent],
    sets: u32,
    seed: u64,
) -> Vec<LadderEntry> {
    let mut games = Vec::new();
    for (opponent_index, &opponent) in opponents.iter().enumerate() {
        for set in 0..sets {
            let game_seed = mix(seed, (opponent_index as u64) << 32 | u64::from(set));
            for hero in 0..4 {
                games.push(FieldGame::new(game_seed, opponent, hero));
            }
        }
    }

    while games.iter().any(|game| game.outcome.is_none()) {
        for game in &mut games {
            if game.outcome.is_none() {
                resolve_chance(&game.game, &mut game.state, &mut game.chance_rng);
                game.finish_if_needed();
            }
        }
        let active: Vec<_> = games
            .iter()
            .enumerate()
            .filter(|(_, game)| game.outcome.is_none())
            .map(|(index, _)| index)
            .collect();
        if active.is_empty() {
            break;
        }
        let active_games: Vec<_> = active.iter().map(|&index| games[index].game).collect();
        let active_states: Vec<_> = active.iter().map(|&index| games[index].state).collect();
        let neural = evaluate_states(infer, &active_games, &active_states, solve);
        let opponents: Vec<_> = active
            .iter()
            .map(|&index| games[index].opponent_actions())
            .collect();
        for ((&index, equilibrium), mut joint) in active.iter().zip(neural).zip(opponents) {
            let game = &mut games[index];
            debug_assert!(equilibrium.values[game.hero].is_finite());
            joint[game.hero] = Direction::ALL
                [sample_strategy(&equilibrium.strategies[game.hero], &mut game.net_rng)];
            game.game.apply_joint(&mut game.state, &joint);
        }
    }

    opponents
        .iter()
        .map(|&opponent| {
            let outcomes: Vec<_> = games
                .iter()
                .filter(|game| game.opponent == opponent)
                .map(|game| game.outcome.unwrap_or(0.0))
                .collect();
            let wins = outcomes.iter().filter(|&&value| value > 0.0).count() as u32;
            let losses = outcomes.iter().filter(|&&value| value < 0.0).count() as u32;
            let draws = outcomes.len() as u32 - wins - losses;
            let (score, ci_low, ci_high) = score_interval(wins, draws, losses, 0.0);
            LadderEntry {
                name: opponent.name(),
                score,
                ci_low,
                ci_high,
                wins,
                draws,
                losses,
            }
        })
        .collect()
}

pub fn net_vs_net(
    first: &Infer,
    first_solve: SolveConfig,
    second: &Infer,
    second_solve: SolveConfig,
    pairs: u32,
    seed: u64,
) -> (u32, u32, u32) {
    struct Game {
        game: Battlesnake<2>,
        state: BoardState<2>,
        first_seat: usize,
        rng: Rng,
        first_rng: Rng,
        second_rng: Rng,
        outcome: Option<f64>,
    }
    let mut games = Vec::new();
    for pair in 0..pairs {
        let game_seed = mix(seed, u64::from(pair));
        for first_seat in 0..2 {
            let game = Battlesnake::new(Rules {
                seed: game_seed,
                ..Rules::default()
            });
            games.push(Game {
                state: game.initial_state(),
                game,
                first_seat,
                rng: Rng::new(mix(game_seed, 0xC11A_CE00)),
                first_rng: Rng::new(mix(game_seed, 0xF1A5_7000)),
                second_rng: Rng::new(mix(game_seed, 0x5EC0_0000)),
                outcome: None,
            });
        }
    }
    while games.iter().any(|game| game.outcome.is_none()) {
        for game in &mut games {
            if game.outcome.is_some() {
                continue;
            }
            resolve_chance(&game.game, &mut game.state, &mut game.rng);
            if game.game.is_terminal(&game.state) {
                game.outcome = Some(game.game.returns(&game.state, game.first_seat));
            } else if game.state.turn_number() >= TURN_CAP {
                game.outcome = Some(0.0);
            }
        }
        let active: Vec<_> = games
            .iter()
            .enumerate()
            .filter(|(_, game)| game.outcome.is_none())
            .map(|(index, _)| index)
            .collect();
        if active.is_empty() {
            break;
        }
        let active_games: Vec<_> = active.iter().map(|&index| games[index].game).collect();
        let active_states: Vec<_> = active.iter().map(|&index| games[index].state).collect();
        let first_roots = evaluate_states(first, &active_games, &active_states, first_solve);
        let second_roots = evaluate_states(second, &active_games, &active_states, second_solve);
        for ((&index, first_root), second_root) in active.iter().zip(first_roots).zip(second_roots)
        {
            let game = &mut games[index];
            debug_assert!(first_root.values[game.first_seat].is_finite());
            let other = 1 - game.first_seat;
            let mut joint = [Direction::Up; 2];
            joint[game.first_seat] = Direction::ALL
                [sample_strategy(&first_root.strategies[game.first_seat], &mut game.first_rng)];
            joint[other] = Direction::ALL
                [sample_strategy(&second_root.strategies[other], &mut game.second_rng)];
            game.game.apply_joint(&mut game.state, &joint);
        }
    }
    let wins = games
        .iter()
        .filter(|game| game.outcome.unwrap_or(0.0) > 0.0)
        .count() as u32;
    let losses = games
        .iter()
        .filter(|game| game.outcome.unwrap_or(0.0) < 0.0)
        .count() as u32;
    (wins, games.len() as u32 - wins - losses, losses)
}

/// Four-player asymmetric cross-play: `hero_net` controls one snake, rotated
/// through every seat, while `field_net` controls the other three. Run the
/// reverse composition separately because multiplayer strength is not a
/// symmetric head-to-head statistic.
pub fn net_vs_net_field(
    hero_net: &Infer,
    hero_solve: SolveConfig,
    field_net: &Infer,
    field_solve: SolveConfig,
    sets: u32,
    seed: u64,
) -> (u32, u32, u32) {
    struct Game {
        game: Battlesnake<4>,
        state: BoardState<4>,
        hero: usize,
        chance_rng: Rng,
        hero_rng: Rng,
        field_rngs: [Rng; 4],
        outcome: Option<f64>,
    }

    let mut games = Vec::new();
    for set in 0..sets {
        let game_seed = mix(seed, u64::from(set));
        for hero in 0..4 {
            let game = Battlesnake::new(Rules {
                seed: game_seed,
                ..Rules::default()
            });
            games.push(Game {
                state: game.initial_state(),
                game,
                hero,
                chance_rng: Rng::new(mix(game_seed, 0xC11A_CE00)),
                hero_rng: Rng::new(mix(game_seed, 0x4E70_0000)),
                field_rngs: std::array::from_fn(|seat| {
                    Rng::new(mix(game_seed, 0xF1E1_D000 + seat as u64))
                }),
                outcome: None,
            });
        }
    }

    while games.iter().any(|game| game.outcome.is_none()) {
        for game in &mut games {
            if game.outcome.is_some() {
                continue;
            }
            resolve_chance(&game.game, &mut game.state, &mut game.chance_rng);
            if game.game.is_terminal(&game.state) {
                game.outcome = Some(game.game.returns(&game.state, game.hero));
            } else if game.state.turn_number() >= TURN_CAP {
                game.outcome = Some(0.0);
            }
        }
        let active: Vec<_> = games
            .iter()
            .enumerate()
            .filter(|(_, game)| game.outcome.is_none())
            .map(|(index, _)| index)
            .collect();
        if active.is_empty() {
            break;
        }
        let active_games: Vec<_> = active.iter().map(|&index| games[index].game).collect();
        let active_states: Vec<_> = active.iter().map(|&index| games[index].state).collect();
        let hero_roots = evaluate_states(hero_net, &active_games, &active_states, hero_solve);
        let field_roots = evaluate_states(field_net, &active_games, &active_states, field_solve);
        for ((&index, hero_root), field_root) in active.iter().zip(hero_roots).zip(field_roots) {
            let game = &mut games[index];
            let joint: [Direction; 4] = std::array::from_fn(|seat| {
                if seat == game.hero {
                    Direction::ALL[sample_strategy(&hero_root.strategies[seat], &mut game.hero_rng)]
                } else {
                    Direction::ALL
                        [sample_strategy(&field_root.strategies[seat], &mut game.field_rngs[seat])]
                }
            });
            game.game.apply_joint(&mut game.state, &joint);
        }
    }

    let wins = games
        .iter()
        .filter(|game| game.outcome.unwrap_or(0.0) > 0.0)
        .count() as u32;
    let losses = games
        .iter()
        .filter(|game| game.outcome.unwrap_or(0.0) < 0.0)
        .count() as u32;
    (wins, games.len() as u32 - wins - losses, losses)
}

/// Balanced four-player cross-play. Each net controls two snakes, with all
/// three distinct seat partitions and their complements evaluated for every
/// seed. A win belongs to the net controlling the sole surviving snake.
pub fn net_vs_net_split(
    first_net: &Infer,
    first_solve: SolveConfig,
    second_net: &Infer,
    second_solve: SolveConfig,
    sets: u32,
    seed: u64,
) -> (u32, u32, u32) {
    struct Game {
        game: Battlesnake<4>,
        state: BoardState<4>,
        first_seats: [bool; 4],
        chance_rng: Rng,
        action_rngs: [Rng; 4],
        outcome: Option<f64>,
    }

    // Every possible 2-vs-2 allocation appears exactly once: the three
    // partitions below plus their complements.
    const PARTITIONS: [[bool; 4]; 3] = [
        [true, true, false, false],
        [true, false, true, false],
        [true, false, false, true],
    ];
    let mut games = Vec::new();
    for set in 0..sets {
        let game_seed = mix(seed, u64::from(set));
        for base in PARTITIONS {
            for complement in [false, true] {
                let first_seats = base.map(|seat| seat ^ complement);
                let game = Battlesnake::new(Rules {
                    seed: game_seed,
                    ..Rules::default()
                });
                games.push(Game {
                    state: game.initial_state(),
                    game,
                    first_seats,
                    chance_rng: Rng::new(mix(game_seed, 0xC11A_CE00)),
                    action_rngs: std::array::from_fn(|seat| {
                        Rng::new(mix(game_seed, 0xAC71_0000 + seat as u64))
                    }),
                    outcome: None,
                });
            }
        }
    }

    while games.iter().any(|game| game.outcome.is_none()) {
        for game in &mut games {
            if game.outcome.is_some() {
                continue;
            }
            resolve_chance(&game.game, &mut game.state, &mut game.chance_rng);
            if game.game.is_terminal(&game.state) {
                let first_won = (0..4).any(|seat| {
                    game.first_seats[seat] && game.game.returns(&game.state, seat) > 0.0
                });
                let second_won = (0..4).any(|seat| {
                    !game.first_seats[seat] && game.game.returns(&game.state, seat) > 0.0
                });
                game.outcome = Some(match (first_won, second_won) {
                    (true, false) => 1.0,
                    (false, true) => -1.0,
                    _ => 0.0,
                });
            } else if game.state.turn_number() >= TURN_CAP {
                game.outcome = Some(0.0);
            }
        }
        let active: Vec<_> = games
            .iter()
            .enumerate()
            .filter(|(_, game)| game.outcome.is_none())
            .map(|(index, _)| index)
            .collect();
        if active.is_empty() {
            break;
        }
        let active_games: Vec<_> = active.iter().map(|&index| games[index].game).collect();
        let active_states: Vec<_> = active.iter().map(|&index| games[index].state).collect();
        let first_roots = evaluate_states(first_net, &active_games, &active_states, first_solve);
        let second_roots = evaluate_states(second_net, &active_games, &active_states, second_solve);
        for ((&index, first_root), second_root) in active.iter().zip(first_roots).zip(second_roots)
        {
            let game = &mut games[index];
            let joint: [Direction; 4] = std::array::from_fn(|seat| {
                let strategy = if game.first_seats[seat] {
                    &first_root.strategies[seat]
                } else {
                    &second_root.strategies[seat]
                };
                Direction::ALL[sample_strategy(strategy, &mut game.action_rngs[seat])]
            });
            game.game.apply_joint(&mut game.state, &joint);
        }
    }

    let wins = games
        .iter()
        .filter(|game| game.outcome.unwrap_or(0.0) > 0.0)
        .count() as u32;
    let losses = games
        .iter()
        .filter(|game| game.outcome.unwrap_or(0.0) < 0.0)
        .count() as u32;
    (wins, games.len() as u32 - wins - losses, losses)
}

/// Wilson interval around a match score. `draw_weight` is 0.5 for duels and
/// zero for multiplayer win share, where every non-win is a miss.
pub fn score_interval(wins: u32, draws: u32, losses: u32, draw_weight: f64) -> (f64, f64, f64) {
    let n = f64::from(wins + draws + losses);
    if n == 0.0 {
        return (0.0, 0.0, 1.0);
    }
    let score = (f64::from(wins) + draw_weight * f64::from(draws)) / n;
    let z = 1.959_963_984_540_054;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (score + z2 / (2.0 * n)) / denominator;
    let half = z * (score * (1.0 - score) / n + z2 / (4.0 * n * n)).sqrt() / denominator;
    (score, (center - half).max(0.0), (center + half).min(1.0))
}

fn resolve_chance<const N: usize>(game: &Battlesnake<N>, state: &mut BoardState<N>, rng: &mut Rng) {
    while !game.is_terminal(state) && game.turn(state) == SimultaneousTurn::Chance {
        let action = game.sample_chance_action(state, rng);
        game.apply_chance(state, action);
    }
}

fn sample_strategy(strategy: &[f32; 4], rng: &mut Rng) -> usize {
    rng.pick(&strategy.map(f64::from))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_interval_contains_observed_score() {
        let (score, low, high) = score_interval(12, 4, 8, 0.5);
        assert!((score - 14.0 / 24.0).abs() < 1e-12);
        assert!(low < score && score < high);
    }

    #[test]
    fn equilibrium_action_sampling_respects_pure_support() {
        let mut rng = Rng::new(7);
        for _ in 0..32 {
            assert_eq!(sample_strategy(&[0.0, 0.0, 1.0, 0.0], &mut rng), 2);
        }
    }
}
