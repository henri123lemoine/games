//! Prints deterministic search traces (root visits, value, Q) for a fixed
//! set of positions, seeds and configs, evaluating with the reference model.
//! Diffing two runs of this probe across a refactor proves the search
//! unchanged on a real network — including noise, repetition handling and
//! tree reuse.
//!
//! ```text
//! cargo run --release -p azinfer --example search_probe -- <export.azweb>
//! ```

use std::collections::HashMap;

use azinfer::argmax;
use azinfer::mcts::{Gather, MctsConfig, Search};
use azinfer::model::Model;
use chess::{Board, legal_moves};
use game_core::Rng;

fn run_to_done(
    search: &mut Search,
    board: &Board,
    history: &HashMap<u64, u8>,
    cfg: &MctsConfig,
    rng: &mut Rng,
    model: &Model,
) {
    let mut results = Vec::new();
    while let Gather::Requests(reqs) =
        search.advance(board, history, cfg, rng, std::mem::take(&mut results))
    {
        results = model.eval(&reqs);
    }
}

fn trace(label: &str, search: &Search) {
    let visits: Vec<String> = search
        .root_moves()
        .iter()
        .zip(search.root_visits())
        .map(|(m, n)| format!("{m:?}:{n}"))
        .collect();
    println!(
        "{label} value={:.9} q={:.9} visits={}",
        search.root_value(),
        search.root_q(),
        visits.join(",")
    );
}

fn main() {
    let export = std::env::args()
        .nth(1)
        .expect("usage: search_probe <export.azweb>");
    let data = std::fs::read(export).expect("read export");
    let model = Model::parse(&data).expect("parse export");

    let fens = [
        ("startpos", chess::START_FEN),
        (
            "midgame",
            "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 4 9",
        ),
        ("backrank", "6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1"),
        ("near50", "8/8/4k3/8/8/3NK3/8/8 w - - 96 120"),
        ("krk", "8/8/8/4k3/8/8/4K3/4R3 w - - 0 1"),
    ];
    let configs = [
        (
            "noised",
            MctsConfig {
                sims: 96,
                root_noise: 0.25,
                max_leaves: 8,
                ..MctsConfig::default()
            },
            42u64,
        ),
        (
            "quiet",
            MctsConfig {
                sims: 64,
                root_noise: 0.0,
                max_leaves: 1,
                ..MctsConfig::default()
            },
            7u64,
        ),
    ];

    for (name, fen) in fens {
        let board = Board::from_fen(fen).expect("valid fen");
        for (cname, cfg, seed) in &configs {
            let mut rng = Rng::new(*seed);
            let mut history = HashMap::new();
            history.insert(board.key(), 1);
            let mut search = Search::new(None);
            run_to_done(&mut search, &board, &history, cfg, &mut rng, &model);
            trace(&format!("{name}/{cname}"), &search);

            // Tree reuse: play the most-visited move, keep the subtree, search
            // the reply position.
            let choice = argmax(search.root_visits());
            let mv = search.root_moves()[choice];
            let mut next = board.clone();
            next.apply(mv);
            if !legal_moves(&next).is_empty()
                && next.halfmove < 100
                && !next.insufficient_material()
            {
                *history.entry(next.key()).or_insert(0) += 1;
                let mut reused = Search::new(search.extract_child(choice));
                run_to_done(&mut reused, &next, &history, cfg, &mut rng, &model);
                trace(&format!("{name}/{cname}/reuse[{mv:?}]"), &reused);
            }
        }
    }

    // Repetition pressure: every successor of the back-rank position except
    // the mate is marked as already seen; search must funnel into the mate.
    let board = Board::from_fen("6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1").unwrap();
    let mut history = HashMap::new();
    history.insert(board.key(), 1);
    let mate: chess::Move = "e1e8".parse().unwrap();
    for m in legal_moves(&board) {
        if m != mate {
            let mut nb = board.clone();
            nb.apply(m);
            history.insert(nb.key(), 1);
        }
    }
    let cfg = MctsConfig {
        sims: 128,
        root_noise: 0.0,
        max_leaves: 4,
        ..MctsConfig::default()
    };
    let mut rng = Rng::new(3);
    let mut search = Search::new(None);
    run_to_done(&mut search, &board, &history, &cfg, &mut rng, &model);
    trace("repetition", &search);
}
