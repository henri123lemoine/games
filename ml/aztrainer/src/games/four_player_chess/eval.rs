//! Seat-rotated four-player evaluation. Reports strict win share against the
//! fair 25% baseline and raw point-score share. The current net can face fixed
//! baselines or a field of one past checkpoint in all three opposing seats.

use four_player_chess::encode::{FourPlayerChessEncoder, shares_to_returns};
use four_player_chess::{FourPlayerChess, GreedyAgent, MobilityAgent, State};
use game_core::{Agent, Game, Rng, ScoreShare, winner};
use solvers::azero::{self, EvalResult, Gather, PuctConfig, Value, argmax};

use crate::net::Infer;

#[derive(Clone, Copy)]
pub enum Opponent {
    Random,
    Greedy,
    Mobility,
}

impl Opponent {
    pub fn name(self) -> &'static str {
        match self {
            Opponent::Random => "random",
            Opponent::Greedy => "greedy",
            Opponent::Mobility => "mobility",
        }
    }

    fn agent(self) -> Box<dyn Agent<FourPlayerChess>> {
        match self {
            Opponent::Random => Box::new(game_core::RandomAgent),
            Opponent::Greedy => Box::new(GreedyAgent),
            Opponent::Mobility => Box::new(MobilityAgent),
        }
    }
}

#[derive(Clone)]
pub struct Entry {
    pub name: String,
    pub games: u32,
    pub strict_wins: u32,
    pub score_share: f64,
}

impl Entry {
    pub fn win_share(&self) -> f64 {
        f64::from(self.strict_wins) / f64::from(self.games.max(1))
    }
}

fn map_results(results: &mut [EvalResult]) {
    for result in results {
        let Value::Seats(values) = result.value else {
            panic!("four-player net returned a scalar value")
        };
        result.value = Value::Seats(shares_to_returns(&values[..4]));
    }
}

fn net_move(
    infer: &Infer,
    game: &FourPlayerChess,
    state: &State,
    sims: u32,
    rng: &mut Rng,
) -> usize {
    let cfg = PuctConfig {
        sims,
        root_noise: 0.0,
        cycle_draws: true,
        ..PuctConfig::default()
    };
    let mut search = azero::Search::new(None);
    let mut results = Vec::new();
    while let Gather::Requests(requests) = search.advance(
        game,
        &FourPlayerChessEncoder,
        state,
        &cfg,
        rng,
        std::mem::take(&mut results),
        &|_| false,
        None,
    ) {
        results = infer.forward_batch(&requests);
        map_results(&mut results);
    }
    argmax(search.root_visits())
}

fn summarize(name: String, game: &FourPlayerChess, results: &[(State, usize)]) -> Entry {
    let strict_wins = results
        .iter()
        .filter(|(state, hero)| winner(game, state) == Some(*hero))
        .count() as u32;
    let score_share = results
        .iter()
        .map(|(state, hero)| game.score_share(state, *hero))
        .sum::<f64>()
        / results.len().max(1) as f64;
    Entry {
        name,
        games: results.len() as u32,
        strict_wins,
        score_share,
    }
}

pub fn vs_baseline(
    infer: &Infer,
    opponent: Opponent,
    games: u32,
    sims: u32,
    ply_cap: u16,
    seed: u64,
) -> Entry {
    let game = FourPlayerChess::with_ply_cap(ply_cap);
    let mut results = Vec::with_capacity(games as usize);
    for index in 0..games {
        let hero = index as usize % 4;
        let mut state = game.initial_state();
        let mut rng = Rng::new(super::selfplay::mix(seed, u64::from(index)));
        let agents: Vec<_> = (0..4).map(|_| opponent.agent()).collect();
        while !game.is_terminal(&state) {
            let actor = state.to_move.index();
            let action_index = if actor == hero {
                net_move(infer, &game, &state, sims, &mut rng)
            } else {
                agents[actor].act(&game, &state, actor, &mut rng)
            };
            let action = game.action_at(&state, action_index);
            game.apply(&mut state, action);
        }
        results.push((state, hero));
    }
    summarize(opponent.name().into(), &game, &results)
}

pub fn vs_past(
    current: &Infer,
    past: &Infer,
    games: u32,
    sims: u32,
    ply_cap: u16,
    seed: u64,
) -> Entry {
    let game = FourPlayerChess::with_ply_cap(ply_cap);
    let mut results = Vec::with_capacity(games as usize);
    for index in 0..games {
        let hero = index as usize % 4;
        let mut state = game.initial_state();
        let mut rng = Rng::new(super::selfplay::mix(seed, u64::from(index)));
        while !game.is_terminal(&state) {
            let actor = state.to_move.index();
            let infer = if actor == hero { current } else { past };
            let action_index = net_move(infer, &game, &state, sims, &mut rng);
            let action = game.action_at(&state, action_index);
            game.apply(&mut state, action);
        }
        results.push((state, hero));
    }
    summarize("past-checkpoint".into(), &game, &results)
}
