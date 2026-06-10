//! Tests for the AlphaBeta upgrades: transposition table + killer/history
//! ordering must shrink the tree without changing move quality, and the soft
//! time budget must be respected.
//!
//! The `measure_*` tests are `#[ignore]`d benchmarks; run them with
//! `cargo test --release -p solvers -- --ignored --nocapture`.

use std::time::{Duration, Instant};

use chess::{Board, Chess, ChessSpec, MaterialEval, evaluate};
use game_core::{Game, Rng, Turn};
use solvers::AlphaBeta;

const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

fn engine(depth: u32, features: bool) -> AlphaBeta<Chess, MaterialEval, ChessSpec> {
    let mut e = AlphaBeta::new(depth, MaterialEval, ChessSpec);
    e.use_tt = features;
    e.use_killers = features;
    e.use_aspiration = features;
    e
}

fn best_move(e: &AlphaBeta<Chess, MaterialEval, ChessSpec>, fen: &str) -> String {
    let board = Board::from_fen(fen).expect("valid FEN");
    let actions = Chess.legal_actions(&board);
    actions[e.best_action(&Chess, &board)].to_string()
}

/// Positions with a unique clearly-best move: features-on must find the same
/// move as features-off while visiting strictly fewer nodes.
#[test]
fn tt_and_ordering_shrink_tree_same_move() {
    let cases = [
        ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "a1a8"),
        (
            "rnb1kbnr/ppp1pppp/8/3q4/8/2N5/PPPP1PPP/R1BQKBNR w KQkq - 0 1",
            "c3d5",
        ),
        ("r3k3/8/8/1N6/8/8/8/4K3 w - - 0 1", "b5c7"),
    ];
    for (fen, expected) in cases {
        let off = engine(5, false);
        let on = engine(5, true);
        assert_eq!(best_move(&off, fen), expected, "features-off on {fen}");
        assert_eq!(best_move(&on, fen), expected, "features-on on {fen}");
        let (n_off, n_on) = (off.node_count(), on.node_count());
        assert!(
            n_on < n_off,
            "features-on should visit fewer nodes on {fen}: on={n_on} off={n_off}"
        );
    }
}

/// Every combination of the three toggles still finds the forced mate.
#[test]
fn all_toggle_combinations_find_mate_in_one() {
    for mask in 0..8u8 {
        let mut e = AlphaBeta::new(4, MaterialEval, ChessSpec);
        e.use_tt = mask & 1 != 0;
        e.use_killers = mask & 2 != 0;
        e.use_aspiration = mask & 4 != 0;
        assert_eq!(
            best_move(&e, "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"),
            "a1a8",
            "toggle mask {mask:03b}"
        );
    }
}

/// `with_time` is a soft budget: the search must return well within 2x.
#[test]
fn with_time_returns_within_double_budget() {
    let budget_ms = 250;
    let e = AlphaBeta::new(1, MaterialEval, ChessSpec).with_time(budget_ms);
    let board = Board::from_fen(KIWIPETE).expect("valid FEN");
    let start = Instant::now();
    let idx = e.best_action(&Chess, &board);
    let elapsed = start.elapsed();
    assert!(idx < Chess.legal_actions(&board).len());
    assert!(
        elapsed < Duration::from_millis(2 * budget_ms),
        "took {elapsed:?} on a {budget_ms}ms budget"
    );
    assert!(e.node_count() > 0);
}

/// Depth-6 node counts, features off vs on, from the startpos and Kiwipete.
#[test]
#[ignore]
fn measure_depth6_nodes() {
    let startpos = Board::start();
    let kiwipete = Board::from_fen(KIWIPETE).expect("valid FEN");
    for (name, board) in [("startpos", &startpos), ("kiwipete", &kiwipete)] {
        for features in [false, true] {
            let e = engine(6, features);
            let t = Instant::now();
            let idx = e.best_action(&Chess, board);
            let mv = Chess.legal_actions(board)[idx].to_string();
            println!(
                "{name} depth=6 features={features}: nodes={} move={mv} time={:?}",
                e.node_count(),
                t.elapsed()
            );
        }
    }
}

fn random_opening(seed: u64, plies: u32) -> Board {
    let game = Chess;
    let mut rng = Rng::new(seed);
    'outer: loop {
        let mut s = game.initial_state();
        for _ in 0..plies {
            let actions = game.legal_actions(&s);
            let i = ((rng.unit() * actions.len() as f64) as usize).min(actions.len() - 1);
            game.apply(&mut s, actions[i]);
            if game.is_terminal(&s) {
                continue 'outer;
            }
        }
        return s;
    }
}

/// +1 if the features-on engine wins, 0 draw, -1 loss. Unfinished games after
/// 160 plies are adjudicated on the static eval at +-200cp.
fn play_game(depth: u32, on_white: bool, opening: &Board) -> f64 {
    let game = Chess;
    let on = engine(depth, true);
    let off = engine(depth, false);
    let on_seat = if on_white { 0 } else { 1 };
    let mut s = opening.clone();
    let mut plies = 0;
    while !game.is_terminal(&s) && plies < 160 {
        let mover = match game.turn(&s) {
            Turn::Player(p) => p,
            Turn::Chance => unreachable!(),
        };
        let idx = if mover == on_seat {
            on.best_action(&game, &s)
        } else {
            off.best_action(&game, &s)
        };
        let actions = game.legal_actions(&s);
        game.apply(&mut s, actions[idx]);
        plies += 1;
    }
    if game.is_terminal(&s) {
        return game.returns(&s, on_seat);
    }
    let stm_cp = evaluate(&s) as f64;
    let on_cp = if matches!(game.turn(&s), Turn::Player(p) if p == on_seat) {
        stm_cp
    } else {
        -stm_cp
    };
    if on_cp > 200.0 {
        1.0
    } else if on_cp < -200.0 {
        -1.0
    } else {
        0.0
    }
}

/// Fixed-depth-5 self-match, features-on vs features-off: 30 random openings,
/// each played with both color assignments (60 games).
#[test]
#[ignore]
fn measure_selfmatch_depth5() {
    use rayon::prelude::*;
    let depth = 5;
    let openings: Vec<Board> = (0..30).map(|i| random_opening(0xA5EED + i, 4)).collect();
    let t = Instant::now();
    let results: Vec<f64> = openings
        .par_iter()
        .flat_map(|o| [(true, o), (false, o)])
        .map(|(on_white, o)| play_game(depth, on_white, o))
        .collect();
    let wins = results.iter().filter(|&&r| r > 0.0).count();
    let losses = results.iter().filter(|&&r| r < 0.0).count();
    let draws = results.len() - wins - losses;
    println!(
        "depth-5 self-match (features-on perspective): {wins}-{draws}-{losses} over {} games, {:?}",
        results.len(),
        t.elapsed()
    );
}
