//! The [`Game`] and [`GameUi`] implementation: a serialized per-square
//! deployment phase followed by the move phase.
//!
//! Deployment is modelled as the setup net sees it — one piece-type placement
//! decision per empty home square in row-major order, red's 40 squares then
//! blue's 40. After both arrangements complete, the board is materialised and
//! play alternates red (player 0) then blue.

use game_core::{Game, GameUi, Turn};

use crate::action::{Action, NUM_ACTIONS};
use crate::arrangement::{Arrangement, DeploymentState, board_from_arrangements, type_to_char};
use crate::board::{Board, Color, HOME_CELLS, PieceType};
use crate::rules;

/// A single decision: a piece-type placement during deployment, or a move
/// during play.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Move {
    Place(PieceType),
    Step(Action),
}

/// Full game state across both phases.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum State {
    /// Deploying: red's arrangement first, then blue's. `red` holds the
    /// finished red arrangement once blue starts.
    Deploy {
        red: Option<Arrangement>,
        current: DeploymentState,
    },
    /// Move phase.
    Play {
        board: Box<Board>,
        to_play: usize,
        flag_captured: Option<usize>,
    },
}

/// The Stratego game (classic variant, forced flag handedness during setup).
#[derive(Debug, Clone, Copy, Default)]
pub struct Stratego;

impl Stratego {
    /// Builds a move-phase state directly from two arrangement strings — for
    /// tests and scripted setups.
    pub fn from_arrangement_strings(red: &str, blue: &str) -> Option<State> {
        let red = Arrangement::from_chars(red)?;
        let blue = Arrangement::from_chars(blue)?;
        Some(State::Play {
            board: Box::new(board_from_arrangements(&red, &blue)),
            to_play: 0,
            flag_captured: None,
        })
    }
}

impl Game for Stratego {
    type State = State;
    type Action = Move;

    fn num_players(&self) -> usize {
        2
    }

    fn initial_state(&self) -> State {
        State::Deploy {
            red: None,
            current: DeploymentState::classic(0, true),
        }
    }

    fn turn(&self, state: &State) -> Turn {
        match state {
            State::Deploy { current, .. } => Turn::Player(current.player),
            State::Play { to_play, .. } => Turn::Player(*to_play),
        }
    }

    fn is_terminal(&self, state: &State) -> bool {
        match state {
            State::Deploy { .. } => false,
            State::Play {
                board,
                to_play,
                flag_captured,
            } => rules::is_terminal(board, *to_play, *flag_captured),
        }
    }

    fn returns(&self, state: &State, player: usize) -> f64 {
        match state {
            State::Deploy { .. } => 0.0,
            State::Play {
                board,
                to_play,
                flag_captured,
            } => {
                let r0 = rules::reward_pl0(board, *to_play, *flag_captured);
                if player == 0 { r0 } else { -r0 }
            }
        }
    }

    fn legal_actions(&self, state: &State) -> Vec<Move> {
        match state {
            State::Deploy { current, .. } => {
                current.legal_types().into_iter().map(Move::Place).collect()
            }
            State::Play { board, to_play, .. } => {
                let mask = rules::legal_mask(board, *to_play);
                (0..NUM_ACTIONS)
                    .filter(|&i| mask[i])
                    .map(|i| Move::Step(Action(i as u16)))
                    .collect()
            }
        }
    }

    fn chance_outcomes(&self, _state: &State) -> Vec<(Move, f64)> {
        Vec::new()
    }

    fn apply(&self, state: &mut State, action: Move) {
        match (&mut *state, action) {
            (State::Deploy { red, current }, Move::Place(kind)) => {
                current.place(kind);
                if !current.is_complete() {
                    return;
                }
                let arrangement = current.arrangement();
                if current.player == 0 {
                    *red = Some(arrangement);
                    *current = DeploymentState::classic(1, true);
                } else {
                    let red = red.take().expect("red arrangement complete");
                    let board = board_from_arrangements(&red, &arrangement);
                    *state = State::Play {
                        board: Box::new(board),
                        to_play: 0,
                        flag_captured: None,
                    };
                }
            }
            (
                State::Play {
                    board,
                    to_play,
                    flag_captured,
                },
                Move::Step(act),
            ) => {
                let applied = rules::apply(board, act, *to_play);
                if applied.flag_captured {
                    *flag_captured = Some(*to_play);
                }
                *to_play = 1 - *to_play;
            }
            _ => panic!("action does not match game phase"),
        }
    }

    fn infoset_key(&self, state: &State, player: usize) -> u64 {
        infoset_key(state, player)
    }

    fn action_id(&self, action: &Move) -> u64 {
        match action {
            Move::Place(t) => 1 << 16 | *t as u64,
            Move::Step(a) => a.0 as u64,
        }
    }
}

/// Information-set key for `player`: own piece identities + types at their
/// cells, the public board (visible opponent pieces, empties), and the public
/// counters. Hidden opponent ranks are never mixed in.
fn infoset_key(state: &State, player: usize) -> u64 {
    let mut bytes: Vec<u8> = Vec::with_capacity(320);
    match state {
        State::Deploy { red, current } => {
            bytes.push(0);
            bytes.push(current.player as u8);
            // Only the deploying player sees their own in-progress placements.
            if current.player == player {
                bytes.extend(current.placed.iter().map(|&t| t as u8));
            }
            if player == 0
                && current.player == 1
                && let Some(red) = red
            {
                bytes.extend(red.0.iter().map(|&t| t as u8));
            }
        }
        State::Play {
            board,
            to_play,
            flag_captured,
        } => {
            bytes.extend_from_slice(&[1, *to_play as u8]);
            bytes.extend_from_slice(&board.num_moves.to_le_bytes());
            bytes.extend_from_slice(&board.num_moves_since_last_attack.to_le_bytes());
            bytes.push(flag_captured.map_or(0xff, |p| p as u8));
            let own = Color::of_player(player);
            for cell in 0..100usize {
                let p = &board.pieces[cell];
                let descriptor: [u8; 3] = match p.color {
                    Color::Empty => [cell as u8, 0, 0],
                    Color::Lake => [cell as u8, 1, 0],
                    c if c == own => {
                        // Our piece: identity + true type, always visible to us.
                        [cell as u8, 2 + p.has_moved as u8, p.kind as u8]
                    }
                    _ => {
                        // Opponent: reveal type only when visible, else opaque.
                        let t = if p.visible { p.kind as u8 } else { 0xfe };
                        [cell as u8, 4 + p.has_moved as u8, t]
                    }
                };
                bytes.extend_from_slice(&descriptor);
            }
        }
    }
    game_core::hash::fnv1a(&bytes)
}

/// Compact text for a [`PieceType`] when shown to its owner.
fn type_label(t: PieceType) -> &'static str {
    match t {
        PieceType::Spy => "Spy",
        PieceType::Scout => "Scout",
        PieceType::Miner => "Miner",
        PieceType::Sergeant => "Sgt",
        PieceType::Lieutenant => "Lt",
        PieceType::Captain => "Cpt",
        PieceType::Major => "Maj",
        PieceType::Colonel => "Col",
        PieceType::General => "Gen",
        PieceType::Marshal => "Marshal",
        PieceType::Flag => "Flag",
        PieceType::Bomb => "Bomb",
        PieceType::Lake => "Lake",
        PieceType::Empty => "Empty",
    }
}

impl GameUi for Stratego {
    fn id(&self) -> &'static str {
        "stratego"
    }

    fn render(&self, state: &State, player: usize) -> String {
        match state {
            State::Deploy { current, .. } => {
                let sq = current.next_square();
                format!(
                    "Deployment: player {} places on home square {} (row {}, col {}). \
                     Placed {}/{}.",
                    current.player,
                    sq,
                    sq / 10,
                    sq % 10,
                    current.placed.len(),
                    HOME_CELLS,
                )
            }
            State::Play { board, to_play, .. } => render_board(board, player, *to_play),
        }
    }

    fn action_label(&self, state: &State, action: Move) -> String {
        match (state, action) {
            (_, Move::Place(t)) => format!("{} ({})", type_to_char(t), type_label(t)),
            (State::Play { to_play, .. }, Move::Step(a)) => {
                let (src, dst) = a.to_abs(*to_play);
                format!("{}->{}", src, dst)
            }
            (_, Move::Step(a)) => format!("{:?}", a),
        }
    }

    fn parse_action(&self, state: &State, input: &str) -> Option<Move> {
        let input = input.trim();
        match state {
            State::Deploy { .. } => {
                let ch = input.chars().next()?;
                crate::arrangement::char_to_type(ch).map(Move::Place)
            }
            State::Play { to_play, board, .. } => {
                let (a, b) = input.split_once("->").or_else(|| input.split_once(' '))?;
                let src = a.trim().parse::<usize>().ok()?;
                let dst = b.trim().parse::<usize>().ok()?;
                let act = Action::from_abs(src, dst, *to_play)?;
                let mask = rules::legal_mask(board, *to_play);
                mask[act.0 as usize].then_some(Move::Step(act))
            }
        }
    }
}

fn render_board(board: &Board, player: usize, to_play: usize) -> String {
    let own = Color::of_player(player);
    let mut out = String::from("    0  1  2  3  4  5  6  7  8  9\n");
    for row in (0..10).rev() {
        out.push_str(&format!("{:>2}  ", row));
        for col in 0..10 {
            let cell = row * 10 + col;
            let p = &board.pieces[cell];
            let tok = match p.color {
                Color::Empty => " . ".to_string(),
                Color::Lake => " ~ ".to_string(),
                c if c == own => format!("{:>2} ", piece_glyph(p.kind)),
                _ => {
                    if p.visible {
                        format!("{:>2}*", piece_glyph(p.kind))
                    } else {
                        " ? ".to_string()
                    }
                }
            };
            out.push_str(&tok);
        }
        out.push('\n');
    }
    out.push_str(&format!(
        "You are player {} ({}). Player {} to move.",
        player,
        if player == 0 { "red" } else { "blue" },
        to_play,
    ));
    out
}

fn piece_glyph(t: PieceType) -> String {
    match t {
        PieceType::Flag => "F".into(),
        PieceType::Bomb => "B".into(),
        PieceType::Spy => "S".into(),
        PieceType::Scout => "2".into(),
        PieceType::Miner => "3".into(),
        _ => ((t as u8) + 1).to_string(),
    }
}
