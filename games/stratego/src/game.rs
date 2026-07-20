//! The [`Game`] and [`GameUi`] implementation: a serialized per-square
//! deployment phase followed by the move phase.
//!
//! Deployment is modelled as the setup net sees it — one piece-type placement
//! decision per empty home square in row-major order, red's 40 squares then
//! blue's 40. After both arrangements complete, the board is materialised and
//! play alternates red (player 0) then blue.

use game_core::{Game, GameUi, Rng, Turn};

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

    /// A move-phase state whose two sides are independent random *legal*
    /// arrangements (classic supply, forced flag handedness, never a
    /// trivially-stuck setup) — the `setup=random` start that skips the 80-square
    /// deployment grind. The same per-square `DeploymentState` the deploy phase
    /// drives generates each side, so a random start is reachable by the normal
    /// deployment moves.
    pub fn random_play_state(rng: &mut Rng) -> State {
        let red = random_arrangement(rng);
        let blue = random_arrangement(rng);
        State::Play {
            board: Box::new(board_from_arrangements(&red, &blue)),
            to_play: 0,
            flag_captured: None,
        }
    }
}

/// Draws a uniformly-random legal classic arrangement by stepping the
/// per-square [`DeploymentState`] machine, rejecting the rare trivially-stuck
/// layout so the resulting game is not over before it starts.
///
/// Forced handedness pins every generated flag to the right half, so the raw
/// distribution is left-right asymmetric. We restore symmetry the way the
/// reference does (`generate_arrangements`, `arrangement/sampling.py`): flip the
/// finished setup across the centre column with probability 1/2. The mirror flip
/// preserves legality and non-terminality, so a self-play setup is never
/// systematically biased to one side.
pub fn random_arrangement(rng: &mut Rng) -> Arrangement {
    loop {
        let mut deploy = DeploymentState::classic(0, true);
        while !deploy.is_complete() {
            let types = deploy.legal_types();
            deploy.place(types[rng.below(types.len())]);
        }
        let mut arrangement = deploy.arrangement();
        if rng.below(2) == 1 {
            arrangement = arrangement.flipped();
        }
        if !arrangement.is_terminal() {
            return arrangement;
        }
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

    fn describe_transition(
        &self,
        before: &State,
        action: Move,
        _after: &State,
        viewer: usize,
    ) -> Option<String> {
        let (State::Play { board, to_play, .. }, Move::Step(act)) = (before, action) else {
            return None;
        };
        combat_narration(board, act, *to_play, viewer)
    }

    /// View schema (game-private contract with `web/app/src/frontends/stratego`;
    /// hidden information is scoped to `viewer` — a seat sees its own ranks and
    /// only the revealed enemy ranks, a spectator sees everything):
    ///
    /// ```json
    /// {"phase": "deploy" | "play", "viewer": 0, "toAct": 0,
    ///  "cells": [null | "~" | {"o": 0|1, "r": "10".."2"|"S"|"B"|"F"|null,
    ///                          "v": bool, "m": bool}, ... 100, cell 0 first],
    ///  "nextSquare": 17 | null,       // deploy: the viewer's next home square
    ///  "supply": [12 ints] | null,    // deploy: the viewer's remaining types
    ///  "deployed": [40, 12] | null,   // deploy: pieces placed per seat
    ///  "lastMove": {"from": 30, "to": 40} | null,   // play only
    ///  "captured": [["4","B"], ["10"]] | null}      // play: ranks lost per seat
    /// ```
    fn view_data(&self, state: &State, viewer: usize) -> Option<String> {
        Some(match state {
            State::Deploy { red, current } => deploy_view_json(red.as_ref(), current, viewer),
            State::Play { board, to_play, .. } => play_view_json(board, *to_play, viewer),
        })
    }

    /// Transition schema (same contract; a battle's ranks are public — combat
    /// reveals both sides — while a quiet mover's rank stays viewer-scoped):
    ///
    /// ```json
    /// {"from": 30, "to": 40, "mover": {"o": 0, "r": "7" | null},
    ///  "battle": null | {"attacker": "7", "defender": "B",
    ///                    "outcome": "win" | "loss" | "tie", "flag": bool}}
    /// ```
    fn transition_data(
        &self,
        before: &State,
        action: Move,
        _after: &State,
        viewer: usize,
    ) -> Option<String> {
        let (State::Play { board, to_play, .. }, Move::Step(act)) = (before, action) else {
            return None;
        };
        let (from, to) = act.to_abs(*to_play);
        let mover = board.pieces[from];
        let target = board.pieces[to];
        let battle = if target.color == Color::of_player(1 - *to_play) {
            let outcome = match crate::rules::resolve(mover.kind, target.kind) {
                crate::rules::Battle::AttackerWins => "win",
                crate::rules::Battle::DefenderWins => "loss",
                crate::rules::Battle::Tie => "tie",
            };
            format!(
                r#"{{"attacker":"{}","defender":"{}","outcome":"{outcome}","flag":{}}}"#,
                piece_glyph(mover.kind),
                piece_glyph(target.kind),
                target.kind == PieceType::Flag,
            )
        } else {
            "null".to_string()
        };
        let spectator = viewer >= 2;
        let mover_rank =
            (spectator || viewer == *to_play || mover.visible).then(|| piece_glyph(mover.kind));
        Some(format!(
            r#"{{"from":{from},"to":{to},"mover":{{"o":{to_play},"r":{}}},"battle":{battle}}}"#,
            game_core::json::string_or_null(mover_rank),
        ))
    }

    /// A deployment placement is itself hidden information: any other seat's
    /// log sees only that a piece was placed, never which.
    fn action_label_for(&self, state: &State, action: Move, viewer: usize) -> String {
        match (state, action) {
            (State::Deploy { current, .. }, Move::Place(_))
                if viewer < 2 && viewer != current.player =>
            {
                "places a hidden piece".to_string()
            }
            _ => self.action_label(state, action),
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

/// One `cells` entry: `null`, `"~"`, or the piece object with its rank scoped
/// to what `viewer` may know.
fn cell_json(p: &crate::board::Piece, viewer: usize, spectator: bool) -> String {
    match p.color {
        Color::Empty => "null".to_string(),
        Color::Lake => "\"~\"".to_string(),
        color => {
            let owner = usize::from(color == Color::Blue);
            let known = spectator || viewer == owner || p.visible;
            format!(
                r#"{{"o":{owner},"r":{},"v":{},"m":{}}}"#,
                game_core::json::string_or_null(known.then(|| piece_glyph(p.kind))),
                p.visible,
                p.has_moved,
            )
        }
    }
}

fn deploy_view_json(red: Option<&Arrangement>, current: &DeploymentState, viewer: usize) -> String {
    let spectator = viewer >= 2;
    let mut cells: Vec<String> = vec!["null".to_string(); 100];
    for cell in crate::board::LAKES {
        cells[cell] = "\"~\"".to_string();
    }
    let mut place = |cell: usize, kind: PieceType, owner: usize| {
        let known = spectator || viewer == owner;
        cells[cell] = format!(
            r#"{{"o":{owner},"r":{},"v":false,"m":false}}"#,
            game_core::json::string_or_null(known.then(|| piece_glyph(kind))),
        );
    };
    if let Some(red) = red {
        for (slot, &kind) in red.0.iter().enumerate() {
            place(slot, kind, 0);
        }
    }
    for (slot, &kind) in current.placed.iter().enumerate() {
        let cell = if current.player == 0 { slot } else { 99 - slot };
        place(cell, kind, current.player);
    }

    let own_view = spectator || viewer == current.player;
    let next_square = own_view.then(|| {
        let slot = current.next_square();
        if current.player == 0 { slot } else { 99 - slot }
    });
    let supply = own_view.then(|| {
        let counts: Vec<String> = current.remaining[..12].iter().map(u8::to_string).collect();
        format!("[{}]", counts.join(","))
    });
    let deployed = [
        if current.player == 0 {
            current.placed.len()
        } else {
            HOME_CELLS
        },
        if current.player == 1 {
            current.placed.len()
        } else {
            0
        },
    ];
    format!(
        r#"{{"phase":"deploy","viewer":{viewer},"toAct":{},"cells":[{}],"nextSquare":{},"supply":{},"deployed":[{},{}],"lastMove":null,"captured":null}}"#,
        current.player,
        cells.join(","),
        next_square.map_or("null".to_string(), |s| s.to_string()),
        supply.unwrap_or_else(|| "null".to_string()),
        deployed[0],
        deployed[1],
    )
}

fn play_view_json(board: &Board, to_play: usize, viewer: usize) -> String {
    let spectator = viewer >= 2;
    let cells: Vec<String> = board
        .pieces
        .iter()
        .map(|p| cell_json(p, viewer, spectator))
        .collect();
    let last_move = board.action_history.last().map(|&a| {
        let mover = (board.action_history.len() - 1) % 2;
        let (from, to) = Action(a).to_abs(mover);
        format!(r#"{{"from":{from},"to":{to}}}"#)
    });
    let captured: Vec<String> = (0..2)
        .map(|side| {
            let ranks: Vec<String> = board.death_status[side]
                .iter()
                .filter(|d| d.is_dead)
                .map(|d| format!("\"{}\"", piece_glyph(PieceType::from_u8(d.piece_type))))
                .collect();
            format!("[{}]", ranks.join(","))
        })
        .collect();
    format!(
        r#"{{"phase":"play","viewer":{viewer},"toAct":{to_play},"cells":[{}],"nextSquare":null,"supply":null,"deployed":null,"lastMove":{},"captured":[{}]}}"#,
        cells.join(","),
        last_move.unwrap_or_else(|| "null".to_string()),
        captured.join(","),
    )
}

fn render_board(board: &Board, player: usize, to_play: usize) -> String {
    // A spectator (no own seat, e.g. `seat=watch`) has no hidden information to
    // protect, so it sees every rank. A seated viewer sees its own pieces and
    // only the opponent pieces the rules have revealed; everything else is `?`.
    let spectator = player >= 2;
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
                Color::Red | Color::Blue => {
                    let mine = !spectator && p.color == own;
                    let side = if p.color == Color::Red { 'r' } else { 'b' };
                    if spectator {
                        format!("{:>2}{}", piece_glyph(p.kind), side)
                    } else if mine {
                        format!("{:>2} ", piece_glyph(p.kind))
                    } else if p.visible {
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
    if spectator {
        out.push_str(&format!(
            "Spectating (red lowercase r, blue b). Player {to_play} ({}) to move.",
            if to_play == 0 { "red" } else { "blue" },
        ));
    } else {
        out.push_str(&format!(
            "You are player {} ({}); `?` = hidden enemy, `*` = revealed. Player {} to move.",
            player,
            if player == 0 { "red" } else { "blue" },
            to_play,
        ));
    }
    out
}

/// Narrates a battle to `viewer` after the fact: a Stratego attack reveals the
/// loser's (and any tie's) rank, which the post-state's vacated square no longer
/// shows. Returns `None` for non-attacking slides. A spectator sees every
/// reveal; a seated viewer is told what it now knows.
fn combat_narration(board: &Board, act: Action, attacker: usize, viewer: usize) -> Option<String> {
    let (from_abs, to_abs) = act.to_abs(attacker);
    let atk = board.pieces[from_abs];
    let def = board.pieces[to_abs];
    if def.color == Color::Empty || def.color == Color::Lake {
        return None;
    }
    let defender = 1 - attacker;
    let outcome = crate::rules::resolve(atk.kind, def.kind);
    let who = |seat: usize| {
        if seat == viewer {
            "your".to_string()
        } else {
            format!("the {}", if seat == 0 { "red" } else { "blue" })
        }
    };
    let atk_name = format!("{} {}", who(attacker), type_label(atk.kind));
    let def_name = format!("{} {}", who(defender), type_label(def.kind));
    Some(match outcome {
        crate::rules::Battle::AttackerWins => {
            if def.kind == PieceType::Flag {
                format!("Combat: {atk_name} captured {def_name} — the flag falls!")
            } else {
                format!("Combat: {atk_name} struck and removed {def_name}.")
            }
        }
        crate::rules::Battle::DefenderWins => {
            format!("Combat: {atk_name} attacked {def_name} and was lost.")
        }
        crate::rules::Battle::Tie => {
            format!("Combat: {atk_name} and {def_name} traded — both removed.")
        }
    })
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
