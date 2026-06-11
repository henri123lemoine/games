//! Unit tests for [`rich_evaluate`] terms, plus the honest RichEval vs
//! MaterialEval A/B (behind `#[ignore]`: `cargo test --release -p chess
//! --test eval_ab -- --ignored --nocapture`).

use chess::{Board, Chess, ChessSpec, MaterialEval, RichEval, evaluate, rich_evaluate};
use game_core::{Agent, Game, Rng};
use solvers::AlphaBeta;

fn board(fen: &str) -> Board {
    Board::from_fen(fen).expect("valid FEN")
}

/// Evaluation from White's point of view regardless of side to move.
fn white_pov(b: &Board) -> i32 {
    match b.stm {
        chess::Color::White => rich_evaluate(b),
        chess::Color::Black => -rich_evaluate(b),
    }
}

/// Color-mirror a FEN: flip ranks, swap piece colors, flip side to move,
/// swap castling rights, mirror the en-passant square.
fn mirrored_fen(fen: &str) -> String {
    let parts: Vec<&str> = fen.split_whitespace().collect();
    let swap = |c: char| -> char {
        if c.is_ascii_uppercase() {
            c.to_ascii_lowercase()
        } else {
            c.to_ascii_uppercase()
        }
    };
    let placement: Vec<String> = parts[0]
        .split('/')
        .rev()
        .map(|rank| rank.chars().map(swap).collect())
        .collect();
    let stm = if parts[1] == "w" { "b" } else { "w" };
    let castling: String = if parts[2] == "-" {
        "-".to_string()
    } else {
        parts[2].chars().map(swap).collect()
    };
    let ep: String = if parts[3] == "-" {
        "-".to_string()
    } else {
        let bytes = parts[3].as_bytes();
        let rank = (b'1' + b'8' - bytes[1]) as char;
        format!("{}{}", bytes[0] as char, rank)
    };
    format!("{} {stm} {castling} {ep} 0 1", placement.join("/"))
}

#[test]
fn mirror_negates() {
    for fen in [
        chess::START_FEN,
        "rnbqkbnr/pp2pppp/8/2ppP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
        "8/3k4/8/2P5/8/5p2/3K4/8 b - - 0 1",
    ] {
        let b = board(fen);
        let m = board(&mirrored_fen(fen));
        assert_eq!(white_pov(&b), -white_pov(&m), "asymmetric eval for {fen}");
    }
}

#[test]
fn doubled_pawns_penalized() {
    // Same material, equal PST squares (a4 and c4 are both 0), all pawns
    // isolated and passed at the same ranks in both; only doubling differs.
    let doubled = board("k7/8/8/8/P7/8/P7/4K3 w - - 0 1");
    let split = board("k7/8/8/8/2P5/8/P7/4K3 w - - 0 1");
    assert!(
        rich_evaluate(&split) > rich_evaluate(&doubled),
        "doubled {} !< split {}",
        rich_evaluate(&doubled),
        rich_evaluate(&split)
    );
}

#[test]
fn isolated_pawns_penalized() {
    // b4 and c4 share a PST value; a2+b4 are connected, a2+c4 both isolated.
    let connected = board("k7/8/8/8/1P6/8/P7/4K3 w - - 0 1");
    let isolated = board("k7/8/8/8/2P5/8/P7/4K3 w - - 0 1");
    assert!(
        rich_evaluate(&connected) > rich_evaluate(&isolated),
        "connected {} !> isolated {}",
        rich_evaluate(&connected),
        rich_evaluate(&isolated)
    );
}

#[test]
fn passed_pawn_bonus() {
    // a5 and h5 share a PST value. Against a black b7 pawn, a5 is blocked
    // (not passed) while h5 is passed; black's b7 becomes passed too but at a
    // far lower relative rank, so White nets the difference.
    let blocked = board("k7/1p6/8/P7/8/8/8/4K3 w - - 0 1");
    let passed = board("k7/1p6/8/7P/8/8/8/4K3 w - - 0 1");
    assert!(
        rich_evaluate(&passed) > rich_evaluate(&blocked),
        "passed {} !> blocked {}",
        rich_evaluate(&passed),
        rich_evaluate(&blocked)
    );
}

#[test]
fn passed_pawn_rank_scaling() {
    // PST alone gives e6 only +5 over e5; the passed-pawn bonus must scale
    // with rank well beyond that.
    let on_e5 = board("8/8/8/4P3/8/8/8/K2k4 w - - 0 1");
    let on_e6 = board("8/8/4P3/8/8/8/8/K2k4 w - - 0 1");
    let gain = rich_evaluate(&on_e6) - rich_evaluate(&on_e5);
    assert!(gain > 5, "advancing the passer only gained {gain}");
}

const MAX_PLIES: usize = 600;

/// Returns White's score (1 / 0.5 / 0); over-long games are adjudicated draws.
fn play_game(game: &Chess, white: &dyn Agent<Chess>, black: &dyn Agent<Chess>, op: &Board) -> f64 {
    let mut s = op.clone();
    for _ in 0..MAX_PLIES {
        if game.is_terminal(&s) {
            let r = game.returns(&s, 0);
            return if r > 0.0 {
                1.0
            } else if r < 0.0 {
                0.0
            } else {
                0.5
            };
        }
        let p = s.stm.index();
        let mover = if p == 0 { white } else { black };
        let i = mover.act(game, &s, p, &mut Rng::new(1));
        let a = game.legal_actions(&s)[i];
        game.apply(&mut s, a);
    }
    0.5
}

/// A roughly balanced position after `plies` uniformly random moves.
fn random_opening(game: &Chess, rng: &mut Rng, plies: usize) -> Board {
    loop {
        let mut s = game.initial_state();
        for _ in 0..plies {
            if game.is_terminal(&s) {
                break;
            }
            let acts = game.legal_actions(&s);
            let i = rng.below(acts.len());
            game.apply(&mut s, acts[i]);
        }
        if !game.is_terminal(&s) && evaluate(&s).abs() <= 150 {
            return s;
        }
    }
}

/// Paired seat-swapped games from random openings; returns Rich's
/// (wins, draws, losses) over `2 * n_openings` games.
fn run_ab(depth: u32, n_openings: usize, seed: u64) -> (u32, u32, u32) {
    let game = Chess;
    let mut rng = Rng::new(seed);
    let openings: Vec<Board> = (0..n_openings)
        .map(|_| random_opening(&game, &mut rng, 6))
        .collect();

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(n_openings);
    let chunk = openings.len().div_ceil(threads);

    let mut totals = (0u32, 0u32, 0u32);
    std::thread::scope(|scope| {
        let handles: Vec<_> = openings
            .chunks(chunk)
            .map(|ops| {
                scope.spawn(move || {
                    let game = Chess;
                    let rich = AlphaBeta::new(depth, RichEval, ChessSpec);
                    let mat = AlphaBeta::new(depth, MaterialEval, ChessSpec);
                    let mut wdl = (0u32, 0u32, 0u32);
                    for op in ops {
                        let as_white = play_game(&game, &rich, &mat, op);
                        let as_black = 1.0 - play_game(&game, &mat, &rich, op);
                        for s in [as_white, as_black] {
                            if s > 0.75 {
                                wdl.0 += 1;
                            } else if s < 0.25 {
                                wdl.2 += 1;
                            } else {
                                wdl.1 += 1;
                            }
                        }
                    }
                    wdl
                })
            })
            .collect();
        for h in handles {
            let (w, d, l) = h.join().expect("A/B worker panicked");
            totals.0 += w;
            totals.1 += d;
            totals.2 += l;
        }
    });
    totals
}

fn report(label: &str, (w, d, l): (u32, u32, u32)) -> f64 {
    let games = f64::from(w + d + l);
    let score = (f64::from(w) + 0.5 * f64::from(d)) / games;
    println!("{label}: +{w} ={d} -{l}  score {:.1}%", 100.0 * score);
    score
}

#[test]
#[ignore = "minutes-long A/B: cargo test --release -p chess --test eval_ab -- --ignored --nocapture"]
fn rich_vs_material_depth4() {
    let score = report("depth 4 (80 games)", run_ab(4, 40, 0xAB4));
    assert!(score >= 0.55, "RichEval scored only {score:.3} at depth 4");
}

#[test]
#[ignore = "minutes-long A/B: cargo test --release -p chess --test eval_ab -- --ignored --nocapture"]
fn rich_vs_material_depth5() {
    let score = report("depth 5 (40 games)", run_ab(5, 20, 0xAB5));
    assert!(score >= 0.50, "RichEval scored only {score:.3} at depth 5");
}
