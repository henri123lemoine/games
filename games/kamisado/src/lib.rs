//! Kamisado (single round) as a [`game_core::Game`].
//!
//! Two players, eight "dragon towers" each on the official 8x8 board of
//! colored squares. Towers move any distance straight or diagonally *forward*
//! (never sideways or back, no jumping, no capturing). Black moves first and
//! may open with any tower; afterwards each player must move their tower whose
//! color matches the square the opponent's tower just landed on. First tower
//! to reach the opponent's home rank wins the round.
//!
//! When the required tower has no legal move it "moves zero squares": the
//! obligation passes to the opponent's tower matching the square the blocked
//! tower stands on, and so on. Because the position is unchanged while
//! obligations bounce, that chain is deterministic — it either reaches a tower
//! that can move or revisits a blocked tower, which means the passing would
//! recur forever: a *deadlock*, lost by the player who caused it (the one who
//! made the last actual move). [`Game::apply`] resolves the whole chain, so
//! every action in this encoding moves a tower and the side to move may take
//! several actions in a row (search handles non-alternating turns). Every move
//! advances a tower at least one rank, so a round lasts at most 112 actions
//! and no position ever repeats; there are no draws.
//!
//! The single-round game is weakly solved — a first-player win (see
//! `examples/solve.rs`, which reproduces the proof). The board colors and move
//! rules here are cross-validated against the hamisado analysis project's
//! published perft counts in the tests below.

mod ui;

use game_core::{Eval, Game, SearchSpec, Turn, hash};

/// Color indices follow the file order of Black's home rank (a1..h1):
/// Brown, Green, Red, Yellow, Pink, Purple, Blue, Orange.
pub const COLOR_NAMES: [&str; 8] = [
    "Brown", "Green", "Red", "Yellow", "Pink", "Purple", "Blue", "Orange",
];

/// One letter per color for compact rendering (`N` = Brown, `U` = Purple —
/// the two that lose the initial-letter race to Blue and Pink/Purple).
pub(crate) const COLOR_LETTERS: [u8; 8] = *b"NGRYPUBO";

/// The official board, `BOARD_COLOR[rank * 8 + file]` with rank 0 = Black's
/// home rank and file 0 = a. Every rank and file holds each color once, the
/// a1–h8 diagonal is all Brown and h1–a8 all Orange, and the whole grid is
/// symmetric under 180° rotation — the signature structure of the real board.
#[rustfmt::skip]
const BOARD_COLOR: [u8; 64] = [
    0, 1, 2, 3, 4, 5, 6, 7, // 1  Brown Green Red Yellow Pink Purple Blue Orange
    5, 0, 3, 6, 1, 4, 7, 2, // 2
    6, 3, 0, 5, 2, 7, 4, 1, // 3
    3, 2, 1, 0, 7, 6, 5, 4, // 4
    4, 5, 6, 7, 0, 1, 2, 3, // 5
    1, 4, 7, 2, 5, 0, 3, 6, // 6
    2, 7, 4, 1, 6, 3, 0, 5, // 7
    7, 6, 5, 4, 3, 2, 1, 0, // 8  Orange Blue Purple Pink Yellow Red Green Brown
];

/// Sentinel for [`KamisadoState::required`]: the opening move, any tower.
const FREE: u8 = 8;

const fn bit(sq: u8) -> u64 {
    1u64 << sq
}

pub(crate) const fn rank(sq: u8) -> u8 {
    sq / 8
}

pub(crate) const fn file(sq: u8) -> u8 {
    sq % 8
}

/// The rank a player is racing toward (the opponent's home rank).
pub(crate) const fn goal_rank(p: usize) -> u8 {
    if p == 0 { 7 } else { 0 }
}

const fn forward(p: usize) -> i8 {
    if p == 0 { 1 } else { -1 }
}

pub(crate) fn square_color(sq: u8) -> u8 {
    BOARD_COLOR[sq as usize]
}

/// A tower slide, `from`/`to` as square indices (`rank * 8 + file`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KamisadoMove {
    pub from: u8,
    pub to: u8,
}

#[derive(Clone, Debug)]
pub struct KamisadoState {
    /// `towers[p][c]` = square of player `p`'s tower of color `c`.
    pub(crate) towers: [[u8; 8]; 2],
    /// Occupancy of all 16 towers.
    pub(crate) occ: u64,
    pub(crate) to_move: u8,
    /// Color obligated to move, or [`FREE`] on the opening move.
    pub(crate) required: u8,
    pub(crate) winner: Option<u8>,
}

impl KamisadoState {
    /// The color of the tower `to_move` must play (`None` on the opening move).
    pub fn required_color(&self) -> Option<u8> {
        (self.required != FREE).then_some(self.required)
    }

    pub fn tower_at(&self, sq: u8) -> Option<(usize, u8)> {
        for p in 0..2 {
            for c in 0..8 {
                if self.towers[p][c] == sq {
                    return Some((p, c as u8));
                }
            }
        }
        None
    }

    /// The color of `to_move`'s tower standing on `from`.
    pub(crate) fn mover_color(&self, from: u8) -> u8 {
        if self.required != FREE {
            debug_assert_eq!(
                self.towers[self.to_move as usize][self.required as usize],
                from
            );
            return self.required;
        }
        let c = self.towers[self.to_move as usize]
            .iter()
            .position(|&sq| sq == from)
            .expect("move from a square without an own tower");
        c as u8
    }

    /// Test/analysis constructor from explicit tower placements.
    #[cfg(test)]
    pub(crate) fn from_towers(towers: [[u8; 8]; 2], to_move: usize, required: u8) -> Self {
        let mut occ = 0u64;
        for sq in towers.iter().flatten() {
            assert_eq!(occ & bit(*sq), 0, "two towers on square {sq}");
            occ |= bit(*sq);
        }
        Self {
            towers,
            occ,
            to_move: to_move as u8,
            required,
            winner: None,
        }
    }

    fn key(&self) -> u64 {
        let mut a = 0u64;
        let mut b = 0u64;
        for c in 0..8 {
            a = a << 6 | self.towers[0][c] as u64;
            b = b << 6 | self.towers[1][c] as u64;
        }
        b = b << 5 | (self.to_move as u64) << 4 | self.required as u64;
        hash::combine(hash::combine(0, a), b)
    }
}

/// Whether `p`'s tower on `sq` has any move (all three forward steps blocked
/// or off-board means it must pass its obligation on).
pub(crate) fn has_move(p: usize, sq: u8, occ: u64) -> bool {
    let (x, y) = (file(sq) as i8, rank(sq) as i8);
    let ny = y + forward(p);
    if !(0..8).contains(&ny) {
        return false;
    }
    (-1i8..=1).any(|dx| {
        let nx = x + dx;
        (0..8).contains(&nx) && occ & bit((ny * 8 + nx) as u8) == 0
    })
}

/// Whether `p`'s tower on `sq` has an unobstructed slide to the goal rank —
/// an immediate winning move if the tower ever gets the obligation.
pub(crate) fn can_reach_goal(p: usize, sq: u8, occ: u64) -> bool {
    let (x, y) = (file(sq) as i8, rank(sq) as i8);
    let dy = forward(p);
    'dir: for dx in [-1i8, 0, 1] {
        let (mut nx, mut ny) = (x + dx, y + dy);
        while (0..8).contains(&nx) && (0..8).contains(&ny) {
            if occ & bit((ny * 8 + nx) as u8) != 0 {
                continue 'dir;
            }
            if ny == goal_rank(p) as i8 {
                return true;
            }
            nx += dx;
            ny += dy;
        }
    }
    false
}

pub(crate) fn push_tower_moves(p: usize, from: u8, occ: u64, out: &mut Vec<KamisadoMove>) {
    let (x, y) = (file(from) as i8, rank(from) as i8);
    let dy = forward(p);
    for dx in [-1i8, 0, 1] {
        let (mut nx, mut ny) = (x + dx, y + dy);
        while (0..8).contains(&nx) && (0..8).contains(&ny) {
            let to = (ny * 8 + nx) as u8;
            if occ & bit(to) != 0 {
                break;
            }
            out.push(KamisadoMove { from, to });
            nx += dx;
            ny += dy;
        }
    }
}

/// Follow the obligation chain after `mover` lands on `land`: who moves next
/// with which tower, or `None` for a deadlock. The chain visits each tower at
/// most once — revisiting one means the passing recurs forever.
/// `on_pass` sees each blocked tower as `(player, color)` in chain order.
pub(crate) fn resolve_obligation(
    towers: &[[u8; 8]; 2],
    occ: u64,
    mover: usize,
    land: u8,
    mut on_pass: impl FnMut(usize, u8),
) -> Option<(usize, u8)> {
    let mut q = 1 - mover;
    let mut c = BOARD_COLOR[land as usize];
    let mut visited = 0u16;
    loop {
        let sq = towers[q][c as usize];
        if has_move(q, sq, occ) {
            return Some((q, c));
        }
        on_pass(q, c);
        visited |= 1 << (q as u16 * 8 + c as u16);
        c = BOARD_COLOR[sq as usize];
        q = 1 - q;
        if visited & (1 << (q as u16 * 8 + c as u16)) != 0 {
            return None;
        }
    }
}

/// Black (player 0, home rank 1) moves first, per the official rules. The
/// sides are otherwise symmetric: the board is 180°-symmetric and each tower
/// starts on its own color.
pub struct Kamisado;

impl Game for Kamisado {
    type State = KamisadoState;
    type Action = KamisadoMove;

    fn initial_state(&self) -> KamisadoState {
        // Rank 1's colors run Brown..Orange (color c at file c), rank 8's
        // run Orange..Brown — every tower starts on its own color.
        let towers = [
            std::array::from_fn(|c| c as u8),
            std::array::from_fn(|c| 56 + (7 - c as u8)),
        ];
        KamisadoState {
            towers,
            occ: 0xFF | 0xFF << 56,
            to_move: 0,
            required: FREE,
            winner: None,
        }
    }

    fn turn(&self, state: &KamisadoState) -> Turn {
        Turn::Player(state.to_move as usize)
    }

    fn is_terminal(&self, state: &KamisadoState) -> bool {
        state.winner.is_some()
    }

    fn returns(&self, state: &KamisadoState, player: usize) -> f64 {
        match state.winner {
            Some(w) if w as usize == player => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        }
    }

    fn legal_actions(&self, state: &KamisadoState) -> Vec<KamisadoMove> {
        let p = state.to_move as usize;
        let mut out = Vec::new();
        if state.required == FREE {
            for c in 0..8 {
                push_tower_moves(p, state.towers[p][c], state.occ, &mut out);
            }
        } else {
            push_tower_moves(
                p,
                state.towers[p][state.required as usize],
                state.occ,
                &mut out,
            );
        }
        out
    }

    fn chance_outcomes(&self, _state: &KamisadoState) -> Vec<(KamisadoMove, f64)> {
        vec![]
    }

    fn apply(&self, state: &mut KamisadoState, action: KamisadoMove) {
        let p = state.to_move as usize;
        let c = state.mover_color(action.from);
        debug_assert_eq!(state.occ & bit(action.to), 0, "destination occupied");
        state.towers[p][c as usize] = action.to;
        state.occ ^= bit(action.from) | bit(action.to);
        if rank(action.to) == goal_rank(p) {
            state.winner = Some(p as u8);
            return;
        }
        match resolve_obligation(&state.towers, state.occ, p, action.to, |_, _| {}) {
            Some((q, col)) => {
                state.to_move = q as u8;
                state.required = col;
            }
            // Deadlock: the mover caused it and loses the round.
            None => state.winner = Some(1 - p as u8),
        }
    }

    fn infoset_key(&self, state: &KamisadoState, _player: usize) -> u64 {
        state.key()
    }

    fn state_key(&self, state: &KamisadoState) -> Option<u64> {
        Some(state.key())
    }

    fn action_id(&self, action: &KamisadoMove) -> u64 {
        (action.from as u64) << 6 | action.to as u64
    }
}

const THREAT_WEIGHT: i32 = 6;
const BLOCKED_WEIGHT: i32 = 2;

/// Static evaluation: net rank progress, plus a bonus per tower with an
/// unobstructed slide to the goal rank (a standing one-move win whenever the
/// obligation lands on it) and a penalty per fully blocked tower. Squashed
/// into `(-1, 1)` to stay on the returns scale.
pub struct KamisadoEval;

impl Eval<Kamisado> for KamisadoEval {
    fn eval(&self, _game: &Kamisado, state: &KamisadoState, player: usize) -> f64 {
        let mut net = 0i32;
        for p in 0..2 {
            let sign = if p == player { 1 } else { -1 };
            for c in 0..8 {
                let sq = state.towers[p][c];
                let advance = if p == 0 { rank(sq) } else { 7 - rank(sq) };
                net += sign * advance as i32;
                if can_reach_goal(p, sq, state.occ) {
                    net += sign * THREAT_WEIGHT;
                }
                if !has_move(p, sq, state.occ) {
                    net -= sign * BLOCKED_WEIGHT;
                }
            }
        }
        game_core::eval_squash(f64::from(net), 24.0)
    }
}

/// Move ordering: immediate wins first, then longer advances (more forcing),
/// demoting moves whose landing color hands the opponent a tower that already
/// has a clear slide to its goal rank — usually an instant loss.
pub struct KamisadoSpec;

impl SearchSpec<Kamisado> for KamisadoSpec {
    fn order_hint(&self, _game: &Kamisado, state: &KamisadoState, action: KamisadoMove) -> i64 {
        let p = state.to_move as usize;
        if rank(action.to) == goal_rank(p) {
            return 1_000;
        }
        let advance = i64::from(rank(action.to).abs_diff(rank(action.from)));
        let occ = state.occ ^ bit(action.from) ^ bit(action.to);
        let reply = state.towers[1 - p][BOARD_COLOR[action.to as usize] as usize];
        if can_reach_goal(1 - p, reply, occ) {
            advance - 100
        } else {
            advance
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sq(name: &str) -> u8 {
        let b = name.as_bytes();
        (b[1] - b'1') * 8 + (b[0] - b'a')
    }

    fn play(moves: &[(&str, &str)]) -> (Kamisado, KamisadoState) {
        let game = Kamisado;
        let mut state = game.initial_state();
        for &(from, to) in moves {
            let mv = KamisadoMove {
                from: sq(from),
                to: sq(to),
            };
            assert!(
                game.legal_actions(&state).contains(&mv),
                "{from}-{to} not legal"
            );
            game.apply(&mut state, mv);
        }
        (game, state)
    }

    #[test]
    fn board_is_the_official_grid() {
        // Latin square in ranks and files, Brown a1–h8 diagonal, Orange
        // anti-diagonal, 180°-rotation symmetry.
        for i in 0..8u8 {
            let rank_colors: u32 = (0..8).map(|f| 1 << BOARD_COLOR[(i * 8 + f) as usize]).sum();
            let file_colors: u32 = (0..8).map(|r| 1 << BOARD_COLOR[(r * 8 + i) as usize]).sum();
            assert_eq!(rank_colors, 0xFF);
            assert_eq!(file_colors, 0xFF);
            assert_eq!(BOARD_COLOR[(i * 8 + i) as usize], 0);
            assert_eq!(BOARD_COLOR[(i * 8 + (7 - i)) as usize], 7);
        }
        for s in 0..64 {
            assert_eq!(BOARD_COLOR[s], BOARD_COLOR[63 - s]);
        }
    }

    #[test]
    fn towers_start_on_their_own_colors() {
        let game = Kamisado;
        let s = game.initial_state();
        for p in 0..2 {
            for c in 0..8 {
                assert_eq!(BOARD_COLOR[s.towers[p][c] as usize], c as u8);
            }
        }
    }

    #[test]
    fn opening_has_102_moves() {
        let game = Kamisado;
        assert_eq!(game.legal_actions(&game.initial_state()).len(), 102);
    }

    /// The hamisado test sequence: after `d1-d7`, White must answer with the
    /// Green tower (13 moves); after `g8-b3`, Black's obligated Yellow tower
    /// at d7 is walled in, so the obligation passes straight back to White's
    /// Green tower (4 moves); `b3-d1` then reaches Black's home rank.
    #[test]
    fn obligation_passes_collapse_and_win_triggers() {
        let (game, s) = play(&[("d1", "d7")]);
        assert_eq!(game.turn(&s), Turn::Player(1));
        assert_eq!(s.required_color(), Some(1)); // Green
        assert_eq!(game.legal_actions(&s).len(), 13);

        let (game, s) = play(&[("d1", "d7"), ("g8", "b3")]);
        assert_eq!(game.turn(&s), Turn::Player(1), "pass hands the turn back");
        assert_eq!(s.required_color(), Some(1));
        let acts = game.legal_actions(&s);
        assert_eq!(acts.len(), 4);
        assert!(acts.iter().all(|m| m.from == sq("b3")));

        let (game, s) = play(&[("d1", "d7"), ("g8", "b3"), ("b3", "d1")]);
        assert!(game.is_terminal(&s));
        assert_eq!(game.returns(&s, 1), 1.0);
        assert_eq!(game.returns(&s, 0), -1.0);
    }

    /// A cyclic obligation chain is a deadlock, lost by the player who made
    /// the last actual move. White's Purple lands on Brown (e5); Black's
    /// Brown tower (b1) is walled in and stands on Green, White's Green (c3)
    /// is walled in and stands on Brown — the chain revisits Black's Brown,
    /// so White caused a deadlock and loses.
    #[test]
    fn cyclic_obligation_chain_is_a_deadlock_loss_for_the_mover() {
        let mut black = [0u8; 8];
        black[0] = sq("b1"); // Brown
        black[1] = sq("a1"); // Green
        black[2] = sq("c1");
        black[3] = sq("b2"); // Yellow: part of the wall
        black[4] = sq("c2"); // Pink: part of the wall
        black[5] = sq("e1");
        black[6] = sq("g1");
        black[7] = sq("h1");
        let mut white = [0u8; 8];
        white[2] = sq("a2"); // Red: part of the wall
        white[6] = sq("d2"); // Blue: part of the wall
        white[1] = sq("c3"); // Green: blocked, standing on Brown
        white[5] = sq("f6"); // Purple: the mover
        white[7] = sq("a8");
        white[4] = sq("d8");
        white[3] = sq("e8");
        white[0] = sq("h8");
        let mut s = KamisadoState::from_towers([black, white], 1, 5);
        let game = Kamisado;
        let mv = KamisadoMove {
            from: sq("f6"),
            to: sq("e5"),
        };
        assert!(game.legal_actions(&s).contains(&mv));
        assert_eq!(BOARD_COLOR[sq("e5") as usize], 0); // lands on Brown
        game.apply(&mut s, mv);
        assert!(game.is_terminal(&s));
        assert_eq!(game.returns(&s, 0), 1.0, "the mover caused it and loses");
    }

    #[test]
    fn state_key_tracks_side_and_obligation() {
        let game = Kamisado;
        let s0 = game.initial_state();
        let (_, s1) = play(&[("d1", "d7")]);
        let (_, s2) = play(&[("d1", "d4")]);
        assert_ne!(game.state_key(&s0), game.state_key(&s1));
        assert_ne!(game.state_key(&s1), game.state_key(&s2));
    }

    /// Every move advances a tower a rank or more, so a round can never
    /// exceed 112 actions and always ends with a winner (no draws).
    #[test]
    fn random_playouts_terminate_with_a_winner() {
        let game = Kamisado;
        let mut rng = game_core::Rng::new(7);
        for _ in 0..300 {
            let mut s = game.initial_state();
            let mut plies = 0;
            while !game.is_terminal(&s) {
                let acts = game.legal_actions(&s);
                assert!(!acts.is_empty(), "non-terminal state with no actions");
                let mv = acts[rng.below(acts.len())];
                assert_eq!(
                    s.tower_at(mv.from).map(|(p, _)| p),
                    Some(s.to_move as usize)
                );
                game.apply(&mut s, mv);
                plies += 1;
                assert!(plies <= 112, "round exceeded the progress bound");
            }
            assert!(s.winner.is_some());
            let occ: u64 = s.towers.iter().flatten().map(|&t| bit(t)).sum();
            assert_eq!(occ, s.occ, "occupancy drifted from tower positions");
        }
    }

    // ------------------------------------------------------------------
    // Cross-validation against the hamisado project's published perft
    // counts. Its convention differs from this crate's encoding in one way:
    // a blocked obligation is an explicit "pass" move (from == to) and a
    // round ends after two consecutive passes. The walker below replays that
    // convention on top of this crate's board and move generator, so the
    // color grid and movement rules are checked against an external oracle
    // (obligation colors steer the tree from depth 2 onward).
    // ------------------------------------------------------------------

    #[derive(Clone)]
    struct RefPos {
        towers: [[u8; 8]; 2],
        occ: u64,
        to_move: usize,
        last_to: Option<u8>,
        last_two_passes: (bool, bool),
    }

    impl RefPos {
        fn initial() -> Self {
            let s = Kamisado.initial_state();
            RefPos {
                towers: s.towers,
                occ: s.occ,
                to_move: 0,
                last_to: None,
                last_two_passes: (false, false),
            }
        }

        fn over(&self) -> bool {
            let reached = self
                .last_to
                .is_some_and(|t| rank(t) == goal_rank(1 - self.to_move));
            reached || (self.last_two_passes.0 && self.last_two_passes.1)
        }

        fn moves(&self) -> Vec<KamisadoMove> {
            if self.over() {
                return vec![];
            }
            let mut out = Vec::new();
            match self.last_to {
                None => {
                    for c in 0..8 {
                        push_tower_moves(
                            self.to_move,
                            self.towers[self.to_move][c],
                            self.occ,
                            &mut out,
                        );
                    }
                }
                Some(t) => {
                    let from = self.towers[self.to_move][BOARD_COLOR[t as usize] as usize];
                    push_tower_moves(self.to_move, from, self.occ, &mut out);
                    if out.is_empty() {
                        out.push(KamisadoMove { from, to: from });
                    }
                }
            }
            out
        }

        fn apply(&mut self, mv: KamisadoMove) {
            if mv.from != mv.to {
                let c = self.towers[self.to_move]
                    .iter()
                    .position(|&t| t == mv.from)
                    .unwrap();
                self.towers[self.to_move][c] = mv.to;
                self.occ ^= bit(mv.from) | bit(mv.to);
            }
            self.last_two_passes = (mv.from == mv.to, self.last_two_passes.0);
            self.last_to = Some(mv.to);
            self.to_move = 1 - self.to_move;
        }
    }

    fn ref_perft(pos: &RefPos, depth: u32) -> u64 {
        if depth == 0 {
            return 1;
        }
        let mut n = 0;
        for mv in pos.moves() {
            let mut child = pos.clone();
            child.apply(mv);
            n += ref_perft(&child, depth - 1);
        }
        n
    }

    #[test]
    fn perft_matches_hamisado() {
        let root = RefPos::initial();
        let expected: &[u64] = &[102, 1150, 11182, 105024, 901006];
        for (d, &leaves) in expected.iter().enumerate() {
            assert_eq!(ref_perft(&root, d as u32 + 1), leaves, "perft {}", d + 1);
        }
    }

    #[test]
    fn perft_6_matches_hamisado() {
        if cfg!(debug_assertions) {
            return; // ~7.4M leaves; run under --release (the repo norm)
        }
        assert_eq!(ref_perft(&RefPos::initial(), 6), 7_399_924);
    }

    #[test]
    fn eval_is_symmetric_at_the_start_and_tracks_progress() {
        let game = Kamisado;
        let s = game.initial_state();
        let e0 = KamisadoEval.eval(&game, &s, 0);
        assert!(e0.abs() < 1e-12, "symmetric start should be ~0, got {e0}");
        let (_, s) = play(&[("d1", "d5")]);
        assert!(KamisadoEval.eval(&game, &s, 0) > 0.0);
        assert!(KamisadoEval.eval(&game, &s, 1) < 0.0);
    }

    #[test]
    fn order_hint_prefers_wins_and_flags_gifts() {
        let game = Kamisado;
        // g8-b3 gifts nothing immediate; after d1-d7 White's b3-d1 wins.
        let (_, s) = play(&[("d1", "d7"), ("g8", "b3")]);
        let win = KamisadoMove {
            from: sq("b3"),
            to: sq("d1"),
        };
        assert_eq!(KamisadoSpec.order_hint(&game, &s, win), 1_000);
        for a in game.legal_actions(&s) {
            if a != win {
                assert!(KamisadoSpec.order_hint(&game, &s, a) < 1_000);
            }
        }
    }
}
