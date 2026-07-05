//! Replays the vendored reference game logs (`tests/fixtures/chase/`, see
//! `ATTRIBUTION.md` there) through our engine, certifying `chase.rs`'s port
//! of the real `chase_state.cu` kernel against real, played-out games.
//!
//! The `continuous_chase_games_new/` set is validated ground truth: the
//! reference's own `test_continuous_chase_new.py` asserts every recorded
//! move but the last is legal, and the last is illegal, directly against the
//! real CUDA env's `current_legal_action_mask` — not against any Python
//! oracle. We require 120/120 clean here.
//!
//! The `continuous_chase_games/` (old-format) set predates that direct-env
//! test; its own reference test (`test_continuous_chase.py`) validates
//! against a Python oracle (`MinimalGameStateMachine`) with a fallback
//! tiebreak, not a clean direct env assertion. We still replay all 99 and
//! report the outcome, but don't require 99/99 — a divergence there means
//! the *old* oracle disagreed with the real kernel, not that our port is
//! wrong (this is exactly the class of bug the fixture-differential
//! methodology exists to catch; see `chase.rs`'s module docs).

use std::fs;
use std::path::Path;

use stratego::action::{Action, NUM_ACTIONS};
use stratego::board::{Board, Color, Piece, PieceType};
use stratego::chase::ChaseState;
use stratego::rules;

const RED_LETTERS: &str = "cdefghijklmb";
const BLUE_LETTERS: &str = "opqrstuvwxyn";

fn data_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/chase")
}

fn parse_board(s: &str) -> Board {
    assert_eq!(s.len(), 100, "board string must be 100 cells");
    let mut board = Board::blank();
    let mut num_hidden = [[0u8; 12]; 2];
    let mut unmoved = [0u8; 2];
    let mut next_id = [0u8; 2];
    for (cell, ch) in s.chars().enumerate() {
        if ch == 'a' || ch == '_' {
            continue;
        }
        let lc = ch.to_ascii_lowercase();
        let (color, player, idx) = if let Some(i) = RED_LETTERS.find(lc) {
            (Color::Red, 0usize, i)
        } else if let Some(i) = BLUE_LETTERS.find(lc) {
            (Color::Blue, 1usize, i)
        } else {
            panic!("unrecognized board char {ch:?} at cell {cell}");
        };
        let kind = PieceType::from_u8(idx as u8);
        let id = next_id[player];
        next_id[player] += 1;
        board.pieces[cell] = Piece::new(kind, color, id);
        num_hidden[player][idx] += 1;
        unmoved[player] += 1;
    }
    board.num_hidden = num_hidden;
    board.num_hidden_unmoved = unmoved;
    // Board::blank() seeds `chase` from the pieces-less blank board; this is
    // a hand-placed position (not board_from_arrangements), so reseed with
    // the true starting layout.
    board.chase = ChaseState::new_from_board(&board);
    board
}

fn load_game(path: &Path) -> (Board, Vec<u16>) {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let mut lines = text.lines();
    let board_line = lines.next().expect("board line").trim();
    let board = parse_board(board_line);
    let actions: Vec<u16> = lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.trim()
                .parse::<u16>()
                .unwrap_or_else(|e| panic!("parse action {l:?}: {e}"))
        })
        .collect();
    (board, actions)
}

fn chase_off_mask(board: &Board, player: usize) -> Box<[bool; NUM_ACTIONS]> {
    let mut mask = rules::raw_legal_mask(board, player);
    board.twosquare[player].remove_actions(&mut mask);
    mask
}

enum Verdict {
    Ok,
    Mismatch { ply: usize, detail: String },
}

/// Replays `actions` from `board`. If `final_is_chase_violation`, the last
/// action must be illegal specifically because of the chase rule (illegal
/// under `legal_mask`, legal under `chase_off_mask`); every earlier action
/// must be legal at the time it was played.
fn replay(mut board: Board, actions: &[u16], final_is_chase_violation: bool) -> Verdict {
    let n = actions.len();
    for (ply, &a) in actions.iter().enumerate() {
        let player = ply % 2;
        let is_last = ply + 1 == n;
        let mask = rules::legal_mask(&board, player);
        let legal = mask[a as usize];

        if is_last && final_is_chase_violation {
            if legal {
                return Verdict::Mismatch {
                    ply,
                    detail: format!(
                        "expected final action {a} (player {player}) to be a chase violation, \
                         our engine says legal"
                    ),
                };
            }
            let off = chase_off_mask(&board, player);
            if !off[a as usize] {
                return Verdict::Mismatch {
                    ply,
                    detail: format!(
                        "final action {a} (player {player}) illegal even with chase removed \
                         (raw+twosquare) -- not isolated to chase"
                    ),
                };
            }
            return Verdict::Ok;
        }

        if !legal {
            return Verdict::Mismatch {
                ply,
                detail: format!("action {a} (player {player}) recorded as played, but illegal"),
            };
        }
        rules::apply(&mut board, Action(a), player);
    }
    Verdict::Ok
}

fn game_files(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut entries: Vec<_> = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir {dir:?}: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "txt").unwrap_or(false))
        .collect();
    entries.sort();
    entries
}

#[test]
fn continuous_chase_games_new_all_120_clean() {
    let dir = data_dir().join("continuous_chase_games_new");
    let files = game_files(&dir);
    assert_eq!(files.len(), 120, "expected all 120 vendored files present");

    let mut mismatches = Vec::new();
    for path in &files {
        let (board, actions) = load_game(path);
        if actions.is_empty() {
            continue;
        }
        if let Verdict::Mismatch { ply, detail } = replay(board, &actions, true) {
            mismatches.push(format!("{}: ply={ply}: {detail}", path.display()));
        }
    }
    assert!(
        mismatches.is_empty(),
        "{}/120 files mismatched:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}

#[test]
fn continuous_chase_games_old_format_report() {
    // Not asserted clean (see module docs): the old-format files predate the
    // direct-env validation the _new set has. We still replay all 99 and
    // print a summary so a genuine engine regression here is still visible,
    // without making CI red over a stale-oracle disagreement.
    let dir = data_dir().join("continuous_chase_games");
    let files = game_files(&dir);
    assert_eq!(files.len(), 99, "expected all 99 vendored files present");

    let mut ok = 0;
    let mut mismatches = Vec::new();
    for path in &files {
        let (board, actions) = load_game(path);
        if actions.is_empty() {
            continue;
        }
        match replay(board, &actions, true) {
            Verdict::Ok => ok += 1,
            Verdict::Mismatch { ply, detail } => {
                mismatches.push(format!("{}: ply={ply}: {detail}", path.display()));
            }
        }
    }
    println!(
        "continuous_chase_games (old format): {ok}/{} clean under the new (kernel-faithful) port",
        files.len()
    );
    for m in &mismatches {
        println!("  DIVERGES: {m}");
    }
}

#[test]
fn strategus_attack_chase_replays_clean() {
    let path = data_dir().join("strategus_games/attack_chase.txt");
    let (board, actions) = load_game(&path);
    let actions = &actions[..376.min(actions.len())];
    let n = actions.len();
    let mut b = board;
    for (ply, &a) in actions.iter().enumerate() {
        let player = ply % 2;
        let mask = rules::legal_mask(&b, player);
        assert!(
            mask[a as usize],
            "action {a} (player {player}) at ply {ply} recorded as played, but illegal"
        );
        rules::apply(&mut b, Action(a), player);
    }
    let to_play = n % 2;
    let illegal_action =
        Action::from_abs(64, 65, to_play).expect("64->65 must be a straight orthogonal move");
    let mask = rules::legal_mask(&b, to_play);
    assert!(
        !mask[illegal_action.0 as usize],
        "expected 64->65 illegal for player {to_play} after {n} plies"
    );
}
