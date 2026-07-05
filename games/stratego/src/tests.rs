//! Unit tests: rule-mechanic oracles ported from the reference `tests/` plus a
//! random-playout invariant through `game_core`'s arena.

use crate::action::{Action, NUM_ACTIONS};
use crate::arrangement::Arrangement;
use crate::board::{Board, Color, PieceType};
use crate::game::{Move, State, Stratego};
use crate::rules::{self, Battle};
use game_core::{Game, GameUi, RandomAgent, Rng, Turn, play_n};

// --- Battle table (test_battle_planes.py battle oracle) ---------------------

/// Reference attacker-death predicate (`test_battle_planes.py:40-49`).
fn ref_attacker_dies(atk: u8, def: u8) -> bool {
    if def == 10 {
        return false; // flag
    }
    if atk == 0 && def == 9 {
        return false; // spy takes marshal
    }
    if atk == 2 && def == 11 {
        return false; // miner defuses bomb
    }
    atk <= def
}

/// Reference defender-death predicate (`test_battle_planes.py:52-61`).
fn ref_defender_dies(atk: u8, def: u8) -> bool {
    if def == 10 {
        return true; // flag
    }
    if atk == 0 && def == 9 {
        return true;
    }
    if atk == 2 && def == 11 {
        return true;
    }
    def <= atk
}

fn ref_is_tie(atk: u8, def: u8) -> bool {
    atk == def
}

#[test]
fn battle_table_matches_reference_oracle() {
    // Attackers are movable (rank < FLAG); defenders span every occupiable type
    // (movables, flag=10, bomb=11).
    for atk in 0..=9u8 {
        for def in [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] {
            let from = PieceType::from_u8(atk);
            let to = PieceType::from_u8(def);
            let outcome = rules::resolve(from, to);

            let attacker_dies = ref_attacker_dies(atk, def);
            let defender_dies = ref_defender_dies(atk, def);
            let tie = ref_is_tie(atk, def);

            match outcome {
                Battle::Tie => assert!(
                    tie && attacker_dies && defender_dies,
                    "tie mismatch atk={atk} def={def}"
                ),
                Battle::DefenderWins => assert!(
                    attacker_dies && !defender_dies && !tie,
                    "defender-wins mismatch atk={atk} def={def}"
                ),
                Battle::AttackerWins => assert!(
                    defender_dies && !attacker_dies && !tie,
                    "attacker-wins mismatch atk={atk} def={def}"
                ),
            }
        }
    }
}

#[test]
fn battle_specials() {
    use PieceType::*;
    assert_eq!(rules::resolve(Spy, Marshal), Battle::AttackerWins);
    assert_eq!(rules::resolve(Marshal, Spy), Battle::AttackerWins);
    assert_eq!(rules::resolve(Miner, Bomb), Battle::AttackerWins);
    assert_eq!(rules::resolve(General, Bomb), Battle::DefenderWins);
    assert_eq!(rules::resolve(Marshal, Bomb), Battle::DefenderWins);
    assert_eq!(rules::resolve(Scout, Flag), Battle::AttackerWins);
    assert_eq!(rules::resolve(Captain, Captain), Battle::Tie);
    assert_eq!(rules::resolve(General, Captain), Battle::AttackerWins);
    assert_eq!(rules::resolve(Captain, General), Battle::DefenderWins);
}

// --- Action encode/decode ---------------------------------------------------

#[test]
fn action_roundtrip_all_legal_slides() {
    for player in 0..2 {
        for idx in 0..NUM_ACTIONS {
            let a = Action(idx as u16);
            let (src, dst) = a.to_abs(player);
            assert!(src < 100 && dst < 100, "decode out of range idx={idx}");
            let same_row = src / 10 == dst / 10;
            let same_col = src % 10 == dst % 10;
            assert!(
                same_row ^ same_col,
                "not single-axis idx={idx} {src}->{dst}"
            );
            let re = Action::from_abs(src, dst, player).expect("re-encodes");
            assert_eq!(re, a, "roundtrip idx={idx} player={player} {src}->{dst}");
        }
    }
}

#[test]
fn abs_encode_matches_known_cases() {
    assert_eq!(Action::from_abs(0, 10, 0).unwrap().0, 0);
    assert_eq!(Action::from_abs(99, 89, 1).unwrap().0, 0);
    assert!(Action::from_abs(0, 11, 0).is_none());
}

// --- Arrangement / setup (test_is_terminal_arrangement.py) ------------------

#[test]
fn arrangement_char_bijection_roundtrip() {
    let s = "CDEFGHIJKLMBAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
    let arr = Arrangement::from_chars(s).expect("parse");
    assert_eq!(arr.to_chars(), s);
    for t in 0..14u8 {
        let kind = PieceType::from_u8(t);
        if kind == PieceType::Lake {
            continue;
        }
        let ch = crate::arrangement::type_to_char(kind);
        assert_eq!(crate::arrangement::char_to_type(ch), Some(kind));
    }
}

/// Reference terminal check (`test_is_terminal_arrangement.py:68-70`): terminal
/// iff all six corridor cells hold immovable pieces.
fn ref_is_terminal(arr: &Arrangement) -> bool {
    crate::arrangement::CORRIDOR_CELLS
        .iter()
        .all(|&loc| matches!(arr.0[loc], PieceType::Bomb | PieceType::Flag))
}

#[test]
fn is_terminal_arrangement_matches_reference() {
    let mut rng = Rng::new(7);
    let letters = "CDDDDDDDDEEEEEFFFFGGGGHHHHIIIJJKLMBBBBBB";
    let base: Vec<char> = letters.chars().collect();
    assert_eq!(base.len(), 40);
    for _ in 0..3000 {
        let mut shuffled = base.clone();
        for i in (1..shuffled.len()).rev() {
            let j = rng.below(i + 1);
            shuffled.swap(i, j);
        }
        let s: String = shuffled.into_iter().collect();
        let arr = Arrangement::from_chars(&s).unwrap();
        assert_eq!(
            arr.is_terminal(),
            ref_is_terminal(&arr),
            "terminal mismatch for {s}"
        );
    }
}

// --- Helpers: drive a move-phase board with absolute moves ------------------

const FIXED_ARR: &str = "KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD";

fn play_state(red: &str, blue: &str) -> State {
    Stratego::from_arrangement_strings(red, blue).expect("valid arrangements")
}

fn legal_abs(board: &Board, player: usize, src: usize, dst: usize) -> bool {
    match Action::from_abs(src, dst, player) {
        Some(a) => rules::legal_mask(board, player)[a.0 as usize],
        None => false,
    }
}

fn apply_abs(board: &mut Board, player: usize, src: usize, dst: usize) {
    let act = Action::from_abs(src, dst, player).expect("encodable move");
    rules::apply(board, act, player);
}

fn board_mut(state: &mut State) -> &mut Board {
    match state {
        State::Play { board, .. } => board,
        _ => panic!("not in play phase"),
    }
}

// --- Two-square rule (test_twosquare_rule.py) -------------------------------

#[test]
fn twosquare_basic_oscillation_is_pruned() {
    let mut st = play_state(FIXED_ARR, FIXED_ARR);
    let b = board_mut(&mut st);
    for (p, s, d) in [
        (0, 39, 49),
        (1, 60, 50),
        (0, 49, 39),
        (1, 50, 60),
        (0, 39, 49),
        (1, 60, 50),
    ] {
        apply_abs(b, p, s, d);
    }
    assert!(!legal_abs(b, 0, 49, 39), "down should be pruned");
    assert!(legal_abs(b, 0, 49, 59), "up should be legal");
    assert!(legal_abs(b, 0, 49, 48), "left should be legal");
}

#[test]
fn twosquare_removes_only_the_reverse_destination() {
    let mut st = play_state(FIXED_ARR, FIXED_ARR);
    let b = board_mut(&mut st);
    for (p, s, d) in [
        (0, 39, 49),
        (1, 60, 50),
        (0, 49, 39),
        (1, 50, 60),
        (0, 39, 49),
        (1, 60, 50),
    ] {
        apply_abs(b, p, s, d);
    }
    let raw = rules::raw_legal_mask(b, 0);
    let down = Action::from_abs(49, 39, 0).unwrap().0 as usize;
    assert!(raw[down], "physically the down move exists");
    assert!(
        !rules::legal_mask(b, 0)[down],
        "but is masked by two-square"
    );
}

#[test]
fn twosquare_different_direction_resets() {
    let mut st = play_state(FIXED_ARR, FIXED_ARR);
    let b = board_mut(&mut st);
    for (p, s, d) in [
        (0, 20, 30),
        (1, 99, 98),
        (0, 30, 40),
        (1, 98, 97),
        (0, 40, 50),
        (1, 97, 96),
        (0, 50, 51),
        (1, 96, 95),
        (0, 51, 50),
        (1, 95, 94),
        (0, 50, 40),
        (1, 94, 93),
    ] {
        apply_abs(b, p, s, d);
    }
    for dest in [50, 41, 30] {
        assert!(legal_abs(b, 0, 40, dest), "dest {dest} should be legal");
    }
}

#[test]
fn twosquare_scout_range_variant() {
    let mut st = play_state(FIXED_ARR, FIXED_ARR);
    let b = board_mut(&mut st);
    for (p, s, d) in [
        (0, 20, 50),
        (1, 99, 98),
        (0, 50, 30),
        (1, 98, 97),
        (0, 30, 50),
        (1, 97, 96),
    ] {
        apply_abs(b, p, s, d);
    }
    for dest in [20, 30, 40] {
        assert!(
            !legal_abs(b, 0, 50, dest),
            "scout dest {dest} should be barred"
        );
    }
}

// --- Continuous chase rule (chase_state.cu kernel port) --------------------
//
// The reference ships a Python test-generation oracle (`continuous_chase.py`
// / `continuous_chase_new.py`) that is *not* faithful to the real kernel
// (`chase_state.cu`) — see `chase.rs`'s module docs. The saved reference game
// logs below are still ground truth (validated against the real env's
// `current_legal_action_mask`), so they're what drives our confidence in the
// port; see `chase_replay.rs` for the full vendored-file suite. The unit
// tests here are lightweight, hand-constructed sanity checks against
// `crate::chase::ChaseState`'s own module tests plus one end-to-end check
// through `rules::apply`/`legal_mask`.

#[test]
fn chase_state_requires_two_threatening_plies_before_any_violation() {
    use crate::chase::ChaseState;
    let mut board = Board::blank();
    board.pieces[20] = crate::board::Piece::new(PieceType::Scout, Color::Red, 0);
    board.pieces[22] = crate::board::Piece::new(PieceType::Scout, Color::Blue, 0);
    let mut sm = ChaseState::new_from_board(&board);

    // A single threatening move only brings chase_length to 1 — no move can
    // be a violation yet, matching the kernel's `delta >= chase_length` guard
    // (needs chase_length >= 2 for delta=1 to ever be checked).
    board.pieces[21] = board.pieces[20];
    board.pieces[20] = crate::board::Piece::EMPTY;
    sm.commit(&board, 0, 20, 21, false);
    assert!(!sm.would_violate(&board, 0, 21, 20));
}

/// Parses a 100-char absolute board string (`BoardString` stripped of `@.`
/// markers) into a move-phase board. Red occupies home rows 0-3, blue rows 6-9;
/// `piece_id` is the home POV cell. Uppercase letters are hidden pieces.
fn board_from_abs_string(s: &str) -> Board {
    use crate::board::{Color, Piece, PieceType};
    let mut board = Board::blank();
    let mut num_hidden = [[0u8; 12]; 2];
    let mut unmoved = [0u8; 2];
    for (cell, ch) in s.chars().enumerate() {
        let (color, kind) = decode_board_char(ch);
        match color {
            Color::Empty | Color::Lake => continue,
            Color::Red => {
                board.pieces[cell] = Piece::new(kind, Color::Red, cell as u8);
                num_hidden[0][kind as usize] += 1;
                unmoved[0] += 1;
            }
            Color::Blue => {
                board.pieces[cell] = Piece::new(kind, Color::Blue, (99 - cell) as u8);
                num_hidden[1][kind as usize] += 1;
                unmoved[1] += 1;
            }
        }
        let _ = PieceType::Empty;
    }
    board.num_hidden = num_hidden;
    board.num_hidden_unmoved = unmoved;
    // Board::blank() seeds `chase` from the pieces-less blank board; this
    // helper places pieces directly (bypassing board_from_arrangements's own
    // reseed), so redo it here with the real starting position.
    board.chase = crate::chase::ChaseState::new_from_board(&board);
    board
}

fn decode_board_char(ch: char) -> (Color, PieceType) {
    use PieceType::*;
    const RED: [(char, PieceType); 12] = [
        ('C', Spy),
        ('D', Scout),
        ('E', Miner),
        ('F', Sergeant),
        ('G', Lieutenant),
        ('H', Captain),
        ('I', Major),
        ('J', Colonel),
        ('K', General),
        ('L', Marshal),
        ('M', Flag),
        ('B', Bomb),
    ];
    const BLUE: [(char, PieceType); 12] = [
        ('O', Spy),
        ('P', Scout),
        ('Q', Miner),
        ('R', Sergeant),
        ('S', Lieutenant),
        ('T', Captain),
        ('U', Major),
        ('V', Colonel),
        ('W', General),
        ('X', Marshal),
        ('Y', Flag),
        ('N', Bomb),
    ];
    let up = ch.to_ascii_uppercase();
    if up == '_' {
        return (Color::Lake, PieceType::Lake);
    }
    if let Some(&(_, t)) = RED.iter().find(|&&(c, _)| c == up) {
        return (Color::Red, t);
    }
    if let Some(&(_, t)) = BLUE.iter().find(|&&(c, _)| c == up) {
        return (Color::Blue, t);
    }
    (Color::Empty, PieceType::Empty)
}

/// Reference saved chase games (`continuous_chase_games/`): a board string and
/// an action sequence whose final action reproduces an earlier threatening
/// position. `test_continuous_chase.py` asserts that, after replaying every
/// move but the last, the continuous-chasing rule removes that final action
/// from the to-move player's legal mask.
const CHASE_GAMES: &[(&str, &[u16])] = &[
    (
        "aaaaaLaDDMaaCaaaaaaBaaaaEaaaaaaaaaaaaaKaaa__aa__aaaa__aa__aaaaaaaaaPaaaaaaaaaaaaaaaWaaaaPOQaaaaaXaNY",
        &[
            1405, 3, 6, 9, 7, 113, 1324, 1123, 116, 119, 112, 1022, 338, 1416, 1526, 115, 227, 221,
            1748, 1729, 222, 931, 1425, 225, 1637, 1032, 1132, 228, 338, 335, 1233, 338, 1334, 448,
            1526, 558, 235, 1668, 1527, 667, 225, 1577, 1426, 1345, 1325, 776, 124, 886, 14, 896,
            4, 786, 114,
        ],
    ),
    (
        "MBaaaaaaaaaaaaaaaaaaCaaLKaaaaDDaaaaaaaEaaa__aa__aaaa__aa__aaaaaaPPaaaaaaaaQaaaaaXWOaaaaaaaYNaaaaaaaa",
        &[
            1123, 117, 120, 1627, 1022, 228, 338, 334, 448, 1325, 110, 118, 558, 1738, 120, 1224,
            221, 335, 330, 1123, 910, 445, 1768, 1355, 331, 554, 341, 228, 331, 222, 340, 339, 669,
            338, 341, 232, 779, 122, 789, 1012, 230, 449, 911, 448, 231, 1344, 220, 459, 430, 458,
            350, 1748, 1130, 449, 921, 1759, 920, 1758, 221, 1264, 1133, 663, 1232, 111, 779, 121,
            889, 1345, 124, 11, 899, 1, 789, 111, 779, 121, 889, 11,
        ],
    ),
];

#[test]
fn chase_saved_reference_games_mask_the_violating_action() {
    for (game_idx, &(board_str, actions)) in CHASE_GAMES.iter().enumerate() {
        let mut board = board_from_abs_string(board_str);
        let mut player = 0usize;
        for (ply, &a) in actions[..actions.len() - 1].iter().enumerate() {
            assert!(
                rules::legal_mask(&board, player)[a as usize],
                "game {game_idx} ply {ply}: replay action {a} illegal for player {player}"
            );
            rules::apply(&mut board, Action(a), player);
            player = 1 - player;
        }
        let violating = *actions.last().unwrap();
        assert!(
            !rules::legal_mask(&board, player)[violating as usize],
            "continuous-chasing rule must mask the reproducing action {violating}"
        );
        assert!(
            rules::raw_legal_mask(&board, player)[violating as usize],
            "the action is physically legal absent the chase rule"
        );
    }
}

#[test]
fn chase_move_not_adjacent_to_any_opponent_never_violates() {
    use crate::chase::ChaseState;
    let board = Board::blank();
    let sm = ChaseState::new_from_board(&board);
    // No pieces on the board at all, so no move can ever be "a threat";
    // would_violate must short-circuit false regardless of chase_length.
    assert!(!sm.would_violate(&board, 0, 11, 12));
}

#[test]
fn chase_revert_own_last_move_allowed_end_to_end() {
    // End-to-end through rules::apply/legal_mask (not the isolated
    // ChaseState unit, which chase.rs's own tests already cover): a player
    // oscillating back to the square they just vacated must never be masked,
    // regardless of how active the chase is.
    let mut board = Board::blank();
    board.pieces[20] = crate::board::Piece::new(PieceType::Scout, Color::Red, 0);
    board.pieces[22] = crate::board::Piece::new(PieceType::Scout, Color::Blue, 0);
    board.num_hidden[0][PieceType::Scout as usize] = 1;
    board.num_hidden[1][PieceType::Scout as usize] = 1;
    board.num_hidden_unmoved = [1, 1];

    rules::apply(&mut board, Action::from_abs(20, 21, 0).unwrap(), 0);
    assert!(
        rules::legal_mask(&board, 0)[Action::from_abs(21, 20, 0).unwrap().0 as usize],
        "undoing your own immediately-previous move must always be legal"
    );
}

// --- Termination & reward ---------------------------------------------------

fn lone_piece(board: &mut Board, cell: usize, kind: PieceType, color: Color, id: u8) {
    board.pieces[cell] = crate::board::Piece::new(kind, color, id);
    let p = color.player().unwrap();
    if (kind as usize) < 12 {
        board.num_hidden[p][kind as usize] += 1;
    }
    board.num_hidden_unmoved[p] += 1;
}

#[test]
fn flag_capture_wins_for_capturing_player() {
    let mut board = Board::blank();
    lone_piece(&mut board, 88, PieceType::Scout, Color::Red, 0);
    lone_piece(&mut board, 0, PieceType::Flag, Color::Red, 1);
    lone_piece(&mut board, 98, PieceType::Flag, Color::Blue, 0);

    let act = Action::from_abs(88, 98, 0).unwrap();
    let applied = rules::apply(&mut board, act, 0);
    assert!(applied.flag_captured, "red captures blue flag");
    let flag_captured = Some(0);
    assert!(rules::is_terminal(&board, 1, flag_captured));
    assert_eq!(rules::reward_pl0(&board, 1, flag_captured), 1.0);
}

#[test]
fn stuck_player_loses() {
    let mut board = Board::blank();
    lone_piece(&mut board, 0, PieceType::Scout, Color::Red, 0);
    lone_piece(&mut board, 99, PieceType::Flag, Color::Blue, 0);
    lone_piece(&mut board, 98, PieceType::Bomb, Color::Blue, 1);
    lone_piece(&mut board, 89, PieceType::Bomb, Color::Blue, 2);

    assert!(!rules::has_any_legal(&board, 1), "blue is stuck");
    assert!(rules::is_terminal(&board, 1, None));
    assert_eq!(rules::has_legal_movement(&board, 1), 1);
    assert_eq!(rules::reward_pl0(&board, 1, None), 1.0);
}

#[test]
fn timeout_gives_zero_reward() {
    let mut board = Board::blank();
    lone_piece(&mut board, 0, PieceType::Scout, Color::Red, 0);
    lone_piece(&mut board, 99, PieceType::Scout, Color::Blue, 0);
    board.num_moves_since_last_attack = rules::MAX_NUM_MOVES_BETWEEN_ATTACKS + 1;
    assert!(rules::is_terminal(&board, 0, None));
    assert_eq!(rules::reward_pl0(&board, 0, None), 0.0);
}

// --- Game-trait flow & deployment -------------------------------------------

#[test]
fn deployment_produces_a_full_legal_board() {
    let game = Stratego;
    let mut state = game.initial_state();
    let mut rng = Rng::new(3);
    let mut placements = 0;
    while let State::Deploy { .. } = state {
        let actions = game.legal_actions(&state);
        assert!(!actions.is_empty(), "deployment always has a legal type");
        let i = rng.below(actions.len());
        game.apply(&mut state, actions[i]);
        placements += 1;
        assert!(
            placements <= 80,
            "deployment terminates within 80 placements"
        );
    }
    assert_eq!(placements, 80, "exactly 80 placements (40 each)");
    match &state {
        State::Play { board, to_play, .. } => {
            assert_eq!(*to_play, 0);
            assert_eq!(board.num_hidden_unmoved, [40, 40]);
        }
        _ => panic!("deployment should yield a play state"),
    }
}

/// Whether the deployment offers the flag once placement reaches `target`,
/// driving the real `Game` and never voluntarily placing the flag (so its
/// supply survives to `target`).
fn flag_offered_at(target: usize) -> bool {
    let game = Stratego;
    let mut state = game.initial_state();
    loop {
        let State::Deploy { current, .. } = &state else {
            panic!("deployment ended before reaching square {target}");
        };
        if current.next_square() == target {
            return game
                .legal_actions(&state)
                .contains(&Move::Place(PieceType::Flag));
        }
        let actions = game.legal_actions(&state);
        let pick = actions
            .iter()
            .position(|a| !matches!(a, Move::Place(PieceType::Flag)))
            .expect("a non-flag placement is always available");
        game.apply(&mut state, actions[pick]);
    }
}

#[test]
fn deployment_flag_handedness_is_right_half_any_row() {
    // Forced handedness is a *column* constraint: the flag may sit anywhere in
    // the right half (columns 5-9) of the home grid, on ANY home row — exactly
    // the reference `right_side` mask. It must NOT be pinned to a single row.

    // Right half (cols 5-9) offers the flag on every home row (0..=3).
    for sq in [5, 9, 15, 19, 25, 29, 35, 39] {
        assert!(
            flag_offered_at(sq),
            "flag must be legal on right-half square {sq} (row {}, col {})",
            sq / 10,
            sq % 10
        );
    }
    // Left half (cols 0-4) never offers the flag, on any row.
    for sq in [0, 4, 10, 14, 24, 30, 34] {
        assert!(
            !flag_offered_at(sq),
            "flag must be illegal on left-half square {sq} (row {}, col {})",
            sq / 10,
            sq % 10
        );
    }
    // Explicitly not row-locked: legal on the front row (row 0) and the back row
    // (row 3) alike — the old `row == 3` pin would have failed square 5.
    assert!(flag_offered_at(5), "flag legal on front row 0, col 5");
    assert!(flag_offered_at(35), "flag legal on back row 3, col 5");
}

#[test]
fn random_arrangement_flag_side_is_symmetric() {
    // Forced handedness alone would put every random self-play flag on the right;
    // the ~50% mirror flip restores symmetry, so the flag lands left about as
    // often as right (no systematic, exploitable side bias).
    let mut rng = Rng::new(7);
    let (mut left, mut right) = (0u32, 0u32);
    for _ in 0..2000 {
        let arr = crate::game::random_arrangement(&mut rng);
        let flag = arr.0.iter().position(|&t| t == PieceType::Flag).unwrap();
        if flag % 10 >= 5 {
            right += 1;
        } else {
            left += 1;
        }
    }
    assert!(
        left > 800 && right > 800,
        "flag side should be ~balanced after the mirror flip, got left={left} right={right}"
    );
}

#[test]
fn infoset_key_hides_hidden_opponent_ranks() {
    let game = Stratego;
    let mut a = Board::blank();
    let mut b = Board::blank();
    for board in [&mut a, &mut b] {
        lone_piece(board, 0, PieceType::Marshal, Color::Red, 0);
    }
    a.pieces[99] = crate::board::Piece::new(PieceType::Spy, Color::Blue, 0);
    b.pieces[99] = crate::board::Piece::new(PieceType::General, Color::Blue, 0);
    a.num_hidden[1][PieceType::Spy as usize] = 1;
    b.num_hidden[1][PieceType::General as usize] = 1;
    a.num_hidden_unmoved[1] = 1;
    b.num_hidden_unmoved[1] = 1;

    let sa = State::Play {
        board: Box::new(a),
        to_play: 0,
        flag_captured: None,
    };
    let sb = State::Play {
        board: Box::new(b),
        to_play: 0,
        flag_captured: None,
    };
    assert_eq!(
        game.infoset_key(&sa, 0),
        game.infoset_key(&sb, 0),
        "red cannot distinguish hidden blue ranks"
    );
    assert_ne!(
        game.infoset_key(&sa, 1),
        game.infoset_key(&sb, 1),
        "blue knows its own ranks"
    );
}

// --- Hidden-information UI ---------------------------------------------------

/// Two play states identical except for the *hidden* rank of a blue piece, used
/// to probe what each viewer's render reveals.
fn hidden_rank_pair() -> (State, State) {
    let build = |blue_kind: PieceType| {
        let mut board = Board::blank();
        lone_piece(&mut board, 50, PieceType::Marshal, Color::Red, 0);
        lone_piece(&mut board, 55, blue_kind, Color::Blue, 0);
        State::Play {
            board: Box::new(board),
            to_play: 0,
            flag_captured: None,
        }
    };
    (build(PieceType::Spy), build(PieceType::General))
}

#[test]
fn render_never_leaks_a_hidden_opponent_rank() {
    let game = Stratego;
    let (sa, sb) = hidden_rank_pair();

    // Player 0 (red) sees blue's piece only as a face-down `?`, so the two
    // states — which differ solely in blue's hidden rank — render identically.
    // The two states differ *only* in blue's hidden rank, so a byte-identical
    // render for red is proof the rank cannot leak — there is nothing else for
    // it to differ on. The hidden blue piece shows as a face-down `?`.
    let red_a = game.render(&sa, 0);
    let red_b = game.render(&sb, 0);
    assert_eq!(
        red_a, red_b,
        "red must not be able to read blue's hidden rank"
    );
    assert!(red_a.contains(" ? "), "the hidden blue piece shows as `?`");

    // Blue owns the piece, so blue's own view distinguishes the two states.
    assert_ne!(
        game.render(&sa, 1),
        game.render(&sb, 1),
        "blue sees its own piece's true rank"
    );

    // A spectator (out-of-range viewer, e.g. `seat=watch`) has nothing to hide
    // from, so it sees every rank and the two states render differently.
    assert_ne!(
        game.render(&sa, usize::MAX),
        game.render(&sb, usize::MAX),
        "a spectator sees both sides' true ranks"
    );
}

#[test]
fn combat_narration_reveals_the_loser_to_the_right_seat() {
    // Red Marshal attacks a hidden Blue General and wins; the reveal (the
    // General the post-state's vacated square no longer shows) must be narrated
    // to red, who just learned what it destroyed.
    let game = Stratego;
    let mut board = Board::blank();
    lone_piece(&mut board, 50, PieceType::Marshal, Color::Red, 0);
    lone_piece(&mut board, 51, PieceType::General, Color::Blue, 0);
    let before = State::Play {
        board: Box::new(board),
        to_play: 0,
        flag_captured: None,
    };
    let action = Move::Step(Action::from_abs(50, 51, 0).unwrap());
    let mut after = before.clone();
    game.apply(&mut after, action);

    let to_red = game
        .describe_transition(&before, action, &after, 0)
        .expect("an attack is narrated");
    assert!(
        to_red.contains("Gen"),
        "red learns the defender was a General: {to_red}"
    );
    assert!(
        to_red.contains("your"),
        "the narration is framed from red's seat: {to_red}"
    );
    // A quiet (non-attacking) slide reveals nothing and is not narrated.
    let mut quiet_board = Board::blank();
    lone_piece(&mut quiet_board, 50, PieceType::Scout, Color::Red, 0);
    lone_piece(&mut quiet_board, 99, PieceType::Flag, Color::Blue, 0);
    let quiet_before = State::Play {
        board: Box::new(quiet_board),
        to_play: 0,
        flag_captured: None,
    };
    let quiet = Move::Step(Action::from_abs(50, 51, 0).unwrap());
    let mut quiet_after = quiet_before.clone();
    game.apply(&mut quiet_after, quiet);
    assert!(
        game.describe_transition(&quiet_before, quiet, &quiet_after, 0)
            .is_none(),
        "a non-attacking slide has no combat to narrate"
    );
}

// --- Random-playout invariant (through the arena) ---------------------------

#[test]
fn random_playouts_are_well_formed() {
    let game = Stratego;
    let agents: [&dyn game_core::Agent<Stratego>; 2] = [&RandomAgent, &RandomAgent];
    for seed in 0..4000u64 {
        let mut rng = Rng::new(seed);
        let mut state = game.initial_state();
        let mut steps = 0u64;
        while !game.is_terminal(&state) {
            let actions = game.legal_actions(&state);
            assert!(
                !actions.is_empty(),
                "seed {seed}: empty legal actions at a non-terminal node"
            );
            match game.turn(&state) {
                Turn::Player(p) => {
                    let i = agents[p].act(&game, &state, p, &mut rng);
                    game.apply(&mut state, actions[i]);
                }
                Turn::Chance => panic!("stratego has no chance nodes"),
            }
            steps += 1;
            assert!(
                steps < 80 + rules::MAX_NUM_MOVES as u64 + 10,
                "seed {seed}: game exceeded the move cap"
            );
        }
        let r0 = game.returns(&state, 0);
        let r1 = game.returns(&state, 1);
        assert!(
            [-1.0, 0.0, 1.0].contains(&r0),
            "seed {seed}: r0={r0} out of range"
        );
        assert!((r0 + r1).abs() < 1e-9, "seed {seed}: not zero-sum");
    }
}

#[test]
fn arena_play_n_runs_a_full_game() {
    let game = Stratego;
    let agents: [&dyn game_core::Agent<Stratego>; 2] = [&RandomAgent, &RandomAgent];
    let mut rng = Rng::new(123);
    let terminal = play_n(&game, &agents, &mut rng);
    assert!(game.is_terminal(&terminal));
    assert!(!game.result_text(&terminal, 0).is_empty());
}
