use game_core::{Game, ScoreShare};

use super::*;

fn sq(name: &str) -> u8 {
    parse_square(name).unwrap()
}

fn put(state: &mut State, at: &str, color: Color, kind: PieceKind) {
    state.board[sq(at) as usize] = Piece::new(color, kind);
}

fn position(to_move: Color) -> State {
    let mut state = State::empty(to_move);
    put(&mut state, "h1", Color::Red, PieceKind::King);
    put(&mut state, "a7", Color::Blue, PieceKind::King);
    put(&mut state, "h14", Color::Yellow, PieceKind::King);
    put(&mut state, "n7", Color::Green, PieceKind::King);
    state
}

fn play(game: &FourPlayerChess, state: &mut State, from: &str, to: &str) -> i16 {
    let action = Move::new(sq(from), sq(to));
    assert!(
        game.legal_moves(state).contains(&action),
        "expected legal move {from}{to}; legal={:?}",
        game.legal_moves(state)
            .into_iter()
            .map(|mv| format!("{}{}", square_name(mv.from), square_name(mv.to)))
            .collect::<Vec<_>>()
    );
    game.apply_move(state, action)
}

#[test]
fn standard_board_is_the_160_square_cross_with_modern_king_orientation() {
    let state = State::standard();
    assert_eq!(
        (0..14)
            .flat_map(|y| (0..14).map(move |x| (x, y)))
            .filter(|&(x, y)| is_valid_xy(x, y))
            .count(),
        160
    );
    assert_eq!(
        state.board.iter().filter(|piece| !piece.is_empty()).count(),
        64
    );

    assert_eq!(
        state.piece_at(sq("d1")),
        Some(Piece::new(Color::Red, PieceKind::Rook))
    );
    assert_eq!(
        state.piece_at(sq("h1")),
        Some(Piece::new(Color::Red, PieceKind::King))
    );
    assert_eq!(
        state.piece_at(sq("a8")),
        Some(Piece::new(Color::Blue, PieceKind::King))
    );
    assert_eq!(
        state.piece_at(sq("g14")),
        Some(Piece::new(Color::Yellow, PieceKind::King))
    );
    assert_eq!(
        state.piece_at(sq("n7")),
        Some(Piece::new(Color::Green, PieceKind::King))
    );
    assert_eq!(
        state.piece_at(sq("a7")),
        Some(Piece::new(Color::Blue, PieceKind::Queen))
    );
    assert_eq!(
        state.piece_at(sq("n8")),
        Some(Piece::new(Color::Green, PieceKind::Queen))
    );
    assert_eq!(state.to_move, Color::Red);
}

#[test]
fn initial_turns_follow_red_blue_yellow_green() {
    let game = FourPlayerChess::with_ply_cap(8);
    let mut state = game.initial_state();
    assert_eq!(state.to_move, Color::Red);
    play(&game, &mut state, "d2", "d3");
    assert_eq!(state.to_move, Color::Blue);
    play(&game, &mut state, "b11", "c11");
    assert_eq!(state.to_move, Color::Yellow);
    play(&game, &mut state, "k13", "k12");
    assert_eq!(state.to_move, Color::Green);
    play(&game, &mut state, "m4", "l4");
    assert_eq!(state.to_move, Color::Red);
}

#[test]
fn a_move_may_not_leave_its_own_king_in_check() {
    let game = FourPlayerChess::default();
    let mut state = position(Color::Red);
    state.board[sq("h1") as usize] = Piece::EMPTY;
    put(&mut state, "h4", Color::Red, PieceKind::King);
    put(&mut state, "h5", Color::Red, PieceKind::Rook);
    put(&mut state, "h10", Color::Yellow, PieceKind::Rook);

    assert!(
        !game
            .legal_moves(&state)
            .contains(&Move::new(sq("h5"), sq("g5")))
    );
    assert!(
        game.legal_moves(&state)
            .contains(&Move::new(sq("h5"), sq("h6")))
    );
}

#[test]
fn active_captures_score_and_dead_armies_are_inert_and_worthless() {
    let game = FourPlayerChess::default();
    let mut state = position(Color::Red);
    put(&mut state, "d4", Color::Red, PieceKind::Rook);
    put(&mut state, "d6", Color::Blue, PieceKind::Knight);
    assert_eq!(play(&game, &mut state, "d4", "d6"), 3);
    assert_eq!(state.scores[Color::Red.index()], 3);

    state.to_move = Color::Red;
    state.active &= !(1 << Color::Yellow.index());
    put(&mut state, "d8", Color::Yellow, PieceKind::Queen);
    assert_eq!(play(&game, &mut state, "d6", "d8"), 0);
    assert_eq!(state.scores[Color::Red.index()], 3);
}

#[test]
fn promotion_is_automatic_and_promoted_queen_is_worth_one() {
    let game = FourPlayerChess::default();
    let mut state = position(Color::Red);
    put(&mut state, "d7", Color::Red, PieceKind::Pawn);
    play(&game, &mut state, "d7", "d8");
    assert_eq!(
        state.piece_at(sq("d8")),
        Some(Piece::promoted_queen(Color::Red))
    );

    state.to_move = Color::Blue;
    put(&mut state, "a8", Color::Blue, PieceKind::Rook);
    assert_eq!(play(&game, &mut state, "a8", "d8"), 1);
}

#[test]
fn promotion_still_resets_the_no_pawn_move_counter() {
    let game = FourPlayerChess::default();
    let mut state = position(Color::Red);
    put(&mut state, "d7", Color::Red, PieceKind::Pawn);
    state.halfmove = 199;

    play(&game, &mut state, "d7", "d8");

    assert_eq!(state.halfmove, 0);
    assert_eq!(state.end, EndReason::Ongoing);
}

#[test]
fn capturing_a_king_scores_twenty_and_eliminates_that_army() {
    let game = FourPlayerChess::default();
    let mut state = position(Color::Red);
    state.board[sq("a7") as usize] = Piece::EMPTY;
    put(&mut state, "d6", Color::Blue, PieceKind::King);
    put(&mut state, "d4", Color::Red, PieceKind::Rook);

    assert_eq!(play(&game, &mut state, "d4", "d6"), 20);
    assert!(!state.is_active(Color::Blue));
    assert_eq!(state.to_move, Color::Yellow);
}

#[test]
fn checkmate_awards_twenty_to_the_mating_player() {
    let game = FourPlayerChess::default();
    let mut state = position(Color::Red);
    state.board[sq("a7") as usize] = Piece::EMPTY;
    put(&mut state, "a4", Color::Blue, PieceKind::King);
    put(&mut state, "d3", Color::Red, PieceKind::Rook);
    put(&mut state, "d5", Color::Red, PieceKind::Queen);

    assert_eq!(play(&game, &mut state, "d3", "d4"), 20);
    assert!(!state.is_active(Color::Blue));
}

#[test]
fn delayed_checkmate_credit_survives_an_intervening_army_move() {
    let game = FourPlayerChess::default();
    let mut state = position(Color::Red);
    state.board[sq("h14") as usize] = Piece::EMPTY;
    put(&mut state, "a4", Color::Yellow, PieceKind::King);
    put(&mut state, "d3", Color::Red, PieceKind::Rook);
    put(&mut state, "d5", Color::Red, PieceKind::Queen);

    play(&game, &mut state, "d3", "d4");
    assert_eq!(state.to_move, Color::Blue);
    assert_eq!(state.scores, [0, 0, 0, 0]);

    play(&game, &mut state, "a7", "a6");
    assert!(!state.is_active(Color::Yellow));
    assert_eq!(state.scores, [20, 0, 0, 0]);
}

#[test]
fn self_stalemate_awards_twenty_to_the_stalemated_player() {
    let game = FourPlayerChess::default();
    let mut state = position(Color::Red);
    state.board[sq("a7") as usize] = Piece::EMPTY;
    put(&mut state, "a4", Color::Blue, PieceKind::King);
    put(&mut state, "b6", Color::Red, PieceKind::Queen);
    put(&mut state, "e3", Color::Red, PieceKind::Rook);

    assert_eq!(play(&game, &mut state, "e3", "e2"), 0);
    assert!(!state.is_active(Color::Blue));
    assert_eq!(state.scores[Color::Blue.index()], 20);
}

#[test]
fn a_non_queen_double_check_scores_five() {
    let game = FourPlayerChess::default();
    let mut state = position(Color::Red);
    state.board[sq("a7") as usize] = Piece::EMPTY;
    state.board[sq("n7") as usize] = Piece::EMPTY;
    put(&mut state, "a5", Color::Blue, PieceKind::King);
    put(&mut state, "n5", Color::Green, PieceKind::King);
    put(&mut state, "g4", Color::Red, PieceKind::Rook);

    assert_eq!(play(&game, &mut state, "g4", "g5"), 5);
    assert!(game.in_check(&state, Color::Blue));
    assert!(game.in_check(&state, Color::Green));
}

#[test]
fn a_discovered_check_counts_toward_the_simultaneous_check_bonus() {
    let game = FourPlayerChess::default();
    let mut state = position(Color::Red);
    state.board[sq("a7") as usize] = Piece::EMPTY;
    state.board[sq("n7") as usize] = Piece::EMPTY;
    put(&mut state, "a5", Color::Blue, PieceKind::King);
    put(&mut state, "j11", Color::Green, PieceKind::King);
    put(&mut state, "g5", Color::Red, PieceKind::Rook);
    put(&mut state, "d5", Color::Red, PieceKind::Bishop);

    assert_eq!(play(&game, &mut state, "d5", "e6"), 5);
    assert!(game.in_check(&state, Color::Blue));
    assert!(game.in_check(&state, Color::Green));
}

#[test]
fn four_player_en_passant_can_capture_on_the_transit_square_too() {
    let game = FourPlayerChess::default();
    let mut state = position(Color::Red);
    put(&mut state, "d5", Color::Red, PieceKind::Pawn);
    put(&mut state, "d6", Color::Blue, PieceKind::Pawn);
    put(&mut state, "c6", Color::Green, PieceKind::Knight);
    state.en_passant[Color::Blue.index()] = sq("d6");

    assert_eq!(play(&game, &mut state, "d5", "c6"), 4);
    assert!(state.piece_at(sq("d6")).is_none());
    assert_eq!(
        state.piece_at(sq("c6")),
        Some(Piece::new(Color::Red, PieceKind::Pawn))
    );
}

#[test]
fn castling_moves_the_king_two_and_rook_one() {
    let game = FourPlayerChess::default();
    let mut state = position(Color::Red);
    put(&mut state, "k1", Color::Red, PieceKind::Rook);
    state.castling = castle_bit(Color::Red, true);

    play(&game, &mut state, "h1", "j1");
    assert_eq!(
        state.piece_at(sq("j1")),
        Some(Piece::new(Color::Red, PieceKind::King))
    );
    assert_eq!(
        state.piece_at(sq("i1")),
        Some(Piece::new(Color::Red, PieceKind::Rook))
    );
    assert!(state.piece_at(sq("k1")).is_none());
}

#[test]
fn modern_blue_castling_uses_the_rook_above_the_king() {
    let game = FourPlayerChess::default();
    let mut state = position(Color::Blue);
    state.board[sq("a7") as usize] = Piece::EMPTY;
    state.board[sq("a8") as usize] = Piece::new(Color::Blue, PieceKind::King);
    put(&mut state, "a11", Color::Blue, PieceKind::Rook);
    state.castling = castle_bit(Color::Blue, true);

    play(&game, &mut state, "a8", "a10");
    assert_eq!(
        state.piece_at(sq("a10")),
        Some(Piece::new(Color::Blue, PieceKind::King))
    );
    assert_eq!(
        state.piece_at(sq("a9")),
        Some(Piece::new(Color::Blue, PieceKind::Rook))
    );
    assert!(state.piece_at(sq("a11")).is_none());
}

#[test]
fn draw_points_go_only_to_active_armies() {
    let game = FourPlayerChess::default();
    let mut state = position(Color::Red);
    state.active &= !(1 << Color::Green.index());
    put(&mut state, "d4", Color::Red, PieceKind::Knight);
    state.halfmove = 199;

    play(&game, &mut state, "d4", "e6");
    assert_eq!(state.end, EndReason::FiftyMove);
    assert_eq!(state.scores, [10, 10, 10, 0]);
}

#[test]
fn terminal_returns_match_the_value_heads_centered_win_shares() {
    let game = FourPlayerChess::default();
    let mut state = position(Color::Red);
    state.scores = [12, 12, 3, 1];
    state.end = EndReason::Repetition;
    let returns: Vec<_> = (0..4).map(|seat| game.returns(&state, seat)).collect();
    assert_eq!(returns, vec![1.0 / 3.0, 1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0]);
    assert_eq!(returns.iter().sum::<f64>(), 0.0);
}

#[test]
fn terminal_score_share_uses_the_raw_points() {
    let game = FourPlayerChess::default();
    let mut state = position(Color::Red);
    state.scores = [30, 20, 10, 0];
    state.end = EndReason::LastArmy;
    assert_eq!(game.score_share(&state, Color::Red.index()), 0.5);
    assert_eq!(game.score_share(&state, Color::Blue.index()), 1.0 / 3.0);
}

#[test]
fn deterministic_random_play_preserves_core_invariants() {
    let game = FourPlayerChess::with_ply_cap(240);
    for seed in 1usize..=12 {
        let mut state = game.initial_state();
        let mut cursor = seed * 7_919;
        while !game.is_terminal(&state) {
            let legal = game.legal_moves(&state);
            assert!(!legal.is_empty());
            let action = legal[cursor % legal.len()];
            cursor = cursor.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            game.apply_move(&mut state, action);

            assert!(state.board.iter().enumerate().all(|(index, piece)| {
                piece.is_empty() || {
                    let (x, y) = xy(index as u8);
                    is_valid_xy(x, y)
                }
            }));
            for color in Color::ALL {
                let kings = state
                    .board
                    .iter()
                    .filter(|piece| {
                        !piece.is_empty()
                            && piece.color() == color
                            && piece.kind() == PieceKind::King
                    })
                    .count();
                assert!(kings <= 1);
                assert!(!state.is_active(color) || kings == 1);
            }
        }
        assert_eq!(state.end, EndReason::PlyCap);
    }
}
