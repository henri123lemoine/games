//! Reproducible search microbenchmark on fixed canonical positions.

use std::time::Duration;

use snake::battlesnake::search::{OpponentModel, SearchConfig, Searcher};
use snake::battlesnake::{BattleSnake, Battlesnake, BoardState, Direction, Rules, bit, cell};

fn snake(cells: &[(u8, u8)], direction: Direction) -> BattleSnake {
    let cells: Vec<_> = cells.iter().map(|&(x, y)| cell(x, y)).collect();
    BattleSnake::from_cells(&cells, 82, direction)
}

fn config(model: OpponentModel) -> SearchConfig {
    SearchConfig {
        time_limit: Duration::from_millis(200),
        max_depth: u8::MAX,
        quiescence_depth: 3,
        opponent_model: model,
        ..SearchConfig::default()
    }
}

fn main() {
    let duel = Battlesnake::<2>::new(Rules::default());
    let duel_state = BoardState::from_parts(
        [
            snake(&[(3, 5), (2, 5), (1, 5), (0, 5)], Direction::Right),
            snake(&[(7, 5), (8, 5), (9, 5), (10, 5)], Direction::Left),
        ],
        bit(cell(5, 5)) | bit(cell(3, 8)) | bit(cell(7, 2)),
        0,
        34,
    );
    run("duel/full", &duel, &duel_state, OpponentModel::Full);

    let four = Battlesnake::<4>::new(Rules::default());
    let four_state = BoardState::from_parts(
        [
            snake(&[(2, 5), (1, 5), (0, 5)], Direction::Right),
            snake(&[(8, 5), (9, 5), (10, 5)], Direction::Left),
            snake(&[(5, 2), (5, 1), (5, 0)], Direction::Up),
            snake(&[(5, 8), (5, 9), (5, 10)], Direction::Down),
        ],
        bit(cell(5, 5)) | bit(cell(2, 8)) | bit(cell(8, 2)),
        0,
        18,
    );
    run(
        "four/mcs",
        &four,
        &four_state,
        OpponentModel::MoveCombination,
    );
    run(
        "four/brs+",
        &four,
        &four_state,
        OpponentModel::BestReplyPlus,
    );
    run("four/full", &four, &four_state, OpponentModel::Full);
}

fn run<const N: usize>(
    name: &str,
    game: &Battlesnake<N>,
    state: &BoardState<N>,
    model: OpponentModel,
) {
    let result = Searcher::new(config(model)).search(game, state, 0);
    let seconds = result.stats.elapsed.as_secs_f64();
    let nps = if seconds == 0.0 {
        0
    } else {
        (result.stats.nodes as f64 / seconds) as u64
    };
    println!(
        "{name:10} move={:?} depth={} score={} nodes={} nps={} tt_hits={} probes={} elapsed_ms={}",
        result.action,
        result.stats.depth,
        result.stats.score,
        result.stats.nodes,
        nps,
        result.stats.tt_hits,
        result.stats.bns_probes,
        result.stats.elapsed.as_millis(),
    );
}
