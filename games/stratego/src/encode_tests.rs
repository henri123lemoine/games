//! Encoder verification: the reference plane-test oracles ported to Rust.
//!
//! The bitset planes (threat/evade/active-adjacency/protection) are checked
//! differentially: the Python trackers in `tests/test_attack_planes.py` and
//! `tests/test_protect_planes.py` are re-implemented here verbatim and run
//! against random self-play games, comparing their expectations to the encoder
//! channels that read the kernel-maintained bitsets. The posterior, cemetery,
//! death-reason, history, and piece-id groups are checked with direct
//! constructed-position value assertions.

use crate::action::Action;
use crate::arrangement::Arrangement;
use crate::board::{Board, Color, PieceType};
use crate::encode::{
    EncoderConfig, NUM_BOARD_STATE_CHANNELS, NUM_OCCUPIABLE_CELLS, NUM_PIECE_ID, encode_infostate,
    encode_tokens, piece_id_grid,
};
use crate::{arrangement, rules};
use game_core::Rng;

// --- Reference board-string / piece-id rendering (absolute coords) ----------

const RED_CHARS: [char; 12] = ['c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'b'];
const BLUE_CHARS: [char; 12] = ['o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'n'];

/// The reference `board_str[::2]` in absolute coordinates: one char per cell,
/// `'a'` empty, `'_'` lake, red `c..b` / blue `o..n` lowercased iff visible.
fn board_str_abs(board: &Board) -> Vec<char> {
    let mut out = vec!['a'; 100];
    for (cell, slot) in out.iter_mut().enumerate() {
        let p = board.pieces[cell];
        let ch = match p.color {
            Color::Empty => 'a',
            Color::Lake => '_',
            Color::Red => RED_CHARS[p.kind as usize],
            Color::Blue => BLUE_CHARS[p.kind as usize],
        };
        *slot = if p.visible || matches!(p.color, Color::Empty | Color::Lake) {
            ch
        } else {
            ch.to_ascii_uppercase()
        };
    }
    out
}

/// The board string in the acting player's POV, matching the Python trackers'
/// `board_str[::-1]` for player 1 (the env string is absolute; player 1 reverses
/// it to its own POV).
fn board_str_pov(board: &Board, player: usize) -> Vec<char> {
    let mut bs = board_str_abs(board);
    if player == 1 {
        bs.reverse();
    }
    bs
}

/// `current_piece_ids` in the acting player's POV: own pieces `[0,39]`,
/// opponent `[60,99]`, empty `255`.
fn piece_ids_pov(board: &Board, player: usize) -> [usize; 100] {
    piece_id_grid(board, player)
}

/// Absolute -> acting-player POV move coordinates (`99 - x` for player 1).
fn pov_move(src: i32, dst: i32, player: usize) -> (i32, i32) {
    if player == 1 {
        (99 - src, 99 - dst)
    } else {
        (src, dst)
    }
}

fn is_adjacent_ref(src: i32, dst: i32) -> bool {
    let to_right = src == dst + 1 && dst % 10 < 9;
    let to_left = src == dst - 1 && dst % 10 > 0;
    let above = src == dst + 10 && dst < 90;
    let below = src == dst - 10 && dst > 9;
    to_right || to_left || above || below
}

fn is_two_squares_away(src: i32, dst: i32) -> bool {
    let r = src == dst + 2 && dst % 10 < 8;
    let l = src == dst - 2 && dst % 10 > 1;
    let a = src == dst + 20 && dst < 80;
    let b = src == dst - 20 && dst > 10;
    let ra = src == dst + 11 && dst % 10 < 9 && dst < 90;
    let la = src == dst + 9 && dst % 10 > 0 && dst < 90;
    let rb = src == dst - 9 && dst % 10 < 9 && dst > 9;
    let lb = src == dst - 11 && dst % 10 > 0 && dst > 9;
    r || l || a || b || ra || la || rb || lb
}

fn on_board(p: i32) -> bool {
    (0..100).contains(&p)
}

fn one_square_positions(pos: i32) -> Vec<i32> {
    let mut v = Vec::new();
    for m in [-10, -1, 1, 10] {
        if is_adjacent_ref(pos, pos + m) && on_board(pos + m) {
            v.push(pos + m);
        }
    }
    v
}

fn two_square_positions(pos: i32) -> Vec<i32> {
    let mut v = Vec::new();
    for m1 in [-10, -1, 1, 10] {
        for m2 in [-10, -1, 1, 10] {
            let t = pos + m1 + m2;
            if is_two_squares_away(pos, t) && on_board(t) {
                v.push(t);
            }
        }
    }
    v
}

/// Letter -> rank index `[0,11]` (red and blue share an index space).
fn letter_index(ch: char) -> usize {
    let c = ch.to_ascii_lowercase();
    if let Some(i) = RED_CHARS.iter().position(|&x| x == c) {
        return i;
    }
    BLUE_CHARS.iter().position(|&x| x == c).unwrap()
}

/// Tracker bucket index `l2i` (`test_protect_planes.py:35-39`): like
/// `letter_index` but bomb collapses onto index 10 (no separate flag bucket in
/// the 13-wide protection vector ordering used by the trackers' `-2/-1`).
fn l2i(ch: char) -> usize {
    let c = ch.to_ascii_lowercase();
    let i = letter_index(c);
    if c == 'b' || c == 'n' { i - 1 } else { i }
}

fn is_upper(ch: char) -> bool {
    ch.is_ascii_uppercase()
}

// --- Threat / Evade / Active-adjacency trackers (test_attack_planes.py) -----

const FLAG_BOMB: [char; 4] = ['m', 'b', 'y', 'n'];

struct ThreatTracker {
    info: Vec<[bool; 11]>,
    player: usize,
}

impl ThreatTracker {
    fn new(player: usize) -> Self {
        ThreatTracker {
            info: vec![[false; 11]; 40],
            player,
        }
    }

    fn opp_adj(board_str: &[char], cell: i32, player: usize) -> Vec<char> {
        let my = if player == 0 { &RED_CHARS } else { &BLUE_CHARS };
        let mut v = Vec::new();
        let mut push = |c: i32| {
            let ch = board_str[c as usize];
            let low = ch.to_ascii_lowercase();
            if low != 'a' && low != '_' && !my.contains(&low) {
                v.push(ch);
            }
        };
        if cell < 90 {
            push(cell + 10);
        }
        if cell > 10 {
            push(cell - 10);
        }
        if cell % 10 < 9 {
            push(cell + 1);
        }
        if cell % 10 > 0 {
            push(cell - 1);
        }
        v
    }

    fn update(&mut self, board: &Board, src: i32, dst: i32) {
        let bs = board_str_pov(board, self.player);
        let ids = piece_ids_pov(board, self.player);
        let (src, dst) = pov_move(src, dst, self.player);
        let piece_id = ids[src as usize];
        for p in Self::opp_adj(&bs, dst, self.player) {
            if is_upper(p) {
                self.info[piece_id][10] = true;
            } else if FLAG_BOMB.contains(&p) {
                continue;
            } else {
                self.info[piece_id][letter_index(p)] = true;
            }
        }
    }
}

struct EvadeTracker {
    info: Vec<[bool; 11]>,
    player: usize,
}

impl EvadeTracker {
    fn new(player: usize) -> Self {
        EvadeTracker {
            info: vec![[false; 11]; 40],
            player,
        }
    }

    fn update(&mut self, board: &Board, src: i32, dst: i32, last: Option<(i32, i32)>) {
        let Some((_last_src, last_dst)) = last else {
            return;
        };
        let opp = if self.player == 0 {
            &BLUE_CHARS
        } else {
            &RED_CHARS
        };
        let bs = board_str_pov(board, self.player);
        let ids = piece_ids_pov(board, self.player);
        let (src, _dst) = pov_move(src, dst, self.player);
        let last_dst = if self.player == 1 {
            99 - last_dst
        } else {
            last_dst
        };
        let piece_id = ids[src as usize];
        if is_adjacent_ref(last_dst, src) {
            let cell = bs[last_dst as usize];
            if opp.contains(&cell.to_ascii_lowercase()) {
                if is_upper(cell) {
                    self.info[piece_id][10] = true;
                } else {
                    self.info[piece_id][letter_index(cell)] = true;
                }
            }
        }
    }
}

struct ActAdjTracker {
    info: Vec<[bool; 11]>,
    player: usize,
}

impl ActAdjTracker {
    fn new(player: usize) -> Self {
        ActAdjTracker {
            info: vec![[false; 11]; 40],
            player,
        }
    }

    fn update(&mut self, board: &Board, src: i32, dst: i32) {
        let opp = if self.player == 0 {
            &BLUE_CHARS
        } else {
            &RED_CHARS
        };
        let bs = board_str_pov(board, self.player);
        let ids = piece_ids_pov(board, self.player);
        let (src, dst) = pov_move(src, dst, self.player);
        let moving_id = ids[src as usize];
        // The moving piece still sits at `src` in this pre-move board; the
        // tracker records opponents adjacent to that source cell (skipping the
        // destination), per `MyActivelyAdjacentTracker`.
        let cell = src;
        for adj in [
            (cell >= 10).then(|| cell - 10),
            (cell < 90).then(|| cell + 10),
            (cell % 10 > 0).then(|| cell - 1),
            (cell % 10 < 9).then(|| cell + 1),
        ]
        .into_iter()
        .flatten()
        {
            if adj == dst {
                continue;
            }
            let v = bs[adj as usize];
            if opp.contains(&v.to_ascii_lowercase()) {
                if is_upper(v) {
                    self.info[moving_id][10] = true;
                } else if opp[..10].contains(&v.to_ascii_lowercase()) {
                    self.info[moving_id][letter_index(v)] = true;
                }
            }
        }
        for pid in 0..100usize {
            let cell_opt = ids.iter().position(|&x| x == pid);
            let Some(cell) = cell_opt else { continue };
            if pid >= 40 || pid == moving_id {
                continue;
            }
            for p in ThreatTracker::opp_adj(&bs, cell as i32, self.player) {
                if is_upper(p) {
                    self.info[pid][10] = true;
                } else if opp[..10].contains(&p.to_ascii_lowercase()) {
                    self.info[pid][letter_index(p)] = true;
                }
            }
        }
    }
}

// --- Differential driver -----------------------------------------------------

/// Plays a random game, and at every step compares the encoder's `we_*` and
/// `they_*` planes for one bitset group against an independently-tracked
/// expectation, exactly mirroring the Python attack-plane tests.
fn random_arrangement(rng: &mut Rng) -> Arrangement {
    let letters = "CDDDDDDDDEEEEEFFFFGGGGHHHHIIIJJKLMBBBBBB";
    let mut base: Vec<char> = letters.chars().collect();
    loop {
        for i in (1..base.len()).rev() {
            let j = rng.below(i + 1);
            base.swap(i, j);
        }
        let arr = Arrangement::from_chars(&base.iter().collect::<String>()).unwrap();
        if !arr.is_terminal() {
            return arr;
        }
    }
}

fn fresh_play(rng: &mut Rng) -> Board {
    let red = random_arrangement(rng);
    let blue = random_arrangement(rng);
    crate::arrangement::board_from_arrangements(&red, &blue)
}

#[derive(Clone, Copy)]
enum Group {
    Threatened,
    Evaded,
    ActAdj,
}

fn group_offsets(g: Group) -> (usize, usize) {
    match g {
        Group::Threatened => (43, 76),
        Group::Evaded => (54, 87),
        Group::ActAdj => (65, 98),
    }
}

#[test]
fn channel_index_layout_is_exact() {
    // Smoke-check that the encoder writes the documented offsets only, by
    // confirming the infostate length and a few known constant planes.
    let cfg = EncoderConfig::default();
    assert_eq!(cfg.num_infostate_channels(), 355 + 32);
    assert_eq!(cfg.num_token_features(), 355 + 32 + 256);
    let mut rng = Rng::new(1);
    let board = fresh_play(&mut rng);
    let enc = encode_infostate(&board, 0, &cfg);
    assert_eq!(enc.len(), (355 + 32) * 100);
    // empty_bool (ch 38) is set on genuinely empty cells (the lakes are LAKE, not
    // EMPTY, so they stay 0) and the mid-board rows 4-5 start empty.
    for cell in 40..60 {
        if crate::board::LAKES.contains(&cell) {
            assert_eq!(enc[38 * 100 + cell], 0.0, "lakes are not empty_bool");
        } else {
            assert_eq!(enc[38 * 100 + cell], 1.0, "mid rows start empty");
        }
    }
}

fn run_attack_group(g: Group, seeds: u64) {
    let cfg = EncoderConfig::default();
    let (we_s, they_s) = group_offsets(g);
    for seed in 0..seeds {
        let mut rng = Rng::new(seed + 7);
        let mut board = fresh_play(&mut rng);
        let mut threat = [ThreatTracker::new(0), ThreatTracker::new(1)];
        let mut evade = [EvadeTracker::new(0), EvadeTracker::new(1)];
        let mut actadj = [ActAdjTracker::new(0), ActAdjTracker::new(1)];
        let mut last_move: Option<(i32, i32)> = None;
        let mut player = 0usize;
        for _ in 0..120 {
            if rules::is_terminal(&board, player, None) {
                break;
            }
            let enc = encode_infostate(&board, player, &cfg);

            // Verify `we_*` for the acting player's pieces.
            check_group_we(g, &board, player, we_s, &threat, &evade, &actadj, &enc);
            // Verify `they_*` for the opponent pieces.
            check_group_they(g, &board, player, they_s, &threat, &evade, &actadj, &enc);

            let mask = rules::legal_mask(&board, player);
            let legal: Vec<usize> = (0..mask.len()).filter(|&i| mask[i]).collect();
            if legal.is_empty() {
                break;
            }
            let a = legal[rng.below(legal.len())];
            let (src, dst) = Action(a as u16).to_abs(player);
            let (src, dst) = (src as i32, dst as i32);
            match g {
                Group::Threatened => threat[player].update(&board, src, dst),
                Group::Evaded => evade[player].update(&board, src, dst, last_move),
                Group::ActAdj => actadj[player].update(&board, src, dst),
            }
            last_move = Some((src, dst));
            rules::apply(&mut board, Action(a as u16), player);
            player = 1 - player;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn check_group_we(
    g: Group,
    board: &Board,
    player: usize,
    we_s: usize,
    threat: &[ThreatTracker; 2],
    evade: &[EvadeTracker; 2],
    actadj: &[ActAdjTracker; 2],
    enc: &[f32],
) {
    let bs = board_str_pov(board, player);
    let ids = piece_ids_pov(board, player);
    for piece_id in 0..40usize {
        let Some(pov_cell) = ids.iter().position(|&x| x == piece_id) else {
            continue;
        };
        if !is_upper(bs[pov_cell]) {
            for i in 0..11 {
                assert_eq!(enc[(we_s + i) * 100 + pov_cell], 0.0);
            }
            continue;
        }
        let expected = match g {
            Group::Threatened => threat[player].info[piece_id],
            Group::Evaded => evade[player].info[piece_id],
            Group::ActAdj => actadj[player].info[piece_id],
        };
        for i in 0..11 {
            assert_eq!(
                enc[(we_s + i) * 100 + pov_cell],
                f32::from(expected[i]),
                "we_* group mismatch id={piece_id} chan={i}"
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn check_group_they(
    g: Group,
    board: &Board,
    player: usize,
    they_s: usize,
    threat: &[ThreatTracker; 2],
    evade: &[EvadeTracker; 2],
    actadj: &[ActAdjTracker; 2],
    enc: &[f32],
) {
    let bs = board_str_pov(board, player);
    let ids = piece_ids_pov(board, player);
    let opp_tracker = 1 - player;
    for view_id in 60..100usize {
        let own_slot = 99 - view_id; // the opponent's own starting slot
        let Some(pov_cell) = ids.iter().position(|&x| x == view_id) else {
            continue;
        };
        if !is_upper(bs[pov_cell]) {
            for i in 0..11 {
                assert_eq!(enc[(they_s + i) * 100 + pov_cell], 0.0);
            }
            continue;
        }
        let expected = match g {
            Group::Threatened => threat[opp_tracker].info[own_slot],
            Group::Evaded => evade[opp_tracker].info[own_slot],
            Group::ActAdj => actadj[opp_tracker].info[own_slot],
        };
        for i in 0..11 {
            assert_eq!(
                enc[(they_s + i) * 100 + pov_cell],
                f32::from(expected[i]),
                "they_* group mismatch view_id={view_id} chan={i}"
            );
        }
    }
}

// --- Protection tracker (test_protect_planes.py MyProtectTracker) -----------

const EMPTY_BUCKET: usize = 11; // `-2` in the Python tracker
const UNKNOWN_BUCKET: usize = 12; // `-1` in the Python tracker

fn is_defender_wins_p(attacker: char, defender: char) -> bool {
    if defender == 'a' {
        return false;
    }
    let atk = letter_index(attacker);
    let def = letter_index(defender);
    if def == 10 {
        return false; // flag
    }
    if atk == 0 && def == 9 {
        return false; // spy beats marshal
    }
    if atk == 2 && def == 11 {
        return false; // miner defuses bomb
    }
    atk < def
}

fn is_defender_dies_p(attacker: char, defender: char) -> bool {
    let atk = letter_index(attacker);
    let def = letter_index(defender);
    if def == 10 {
        return true;
    }
    if atk == 0 && def == 9 {
        return true;
    }
    if atk == 2 && def == 11 {
        return true;
    }
    def <= atk
}

fn is_attacker_dies_p(attacker: char, defender: char) -> bool {
    if defender == 'a' {
        return false;
    }
    let atk = letter_index(attacker);
    let def = letter_index(defender);
    if def == 10 {
        return false;
    }
    if atk == 0 && def == 9 {
        return false;
    }
    if atk == 2 && def == 11 {
        return false;
    }
    atk <= def
}

struct ProtectTracker {
    protected_: Vec<[bool; 13]>,
    protected_against: Vec<[bool; 13]>,
    was_protected_by: Vec<[bool; 13]>,
    was_protected_against: Vec<[bool; 13]>,
    player: usize,
}

impl ProtectTracker {
    fn new(player: usize) -> Self {
        ProtectTracker {
            protected_: vec![[false; 13]; 40],
            protected_against: vec![[false; 13]; 40],
            was_protected_by: vec![[false; 13]; 40],
            was_protected_against: vec![[false; 13]; 40],
            player,
        }
    }

    fn our(&self) -> &'static [char; 12] {
        if self.player == 0 {
            &RED_CHARS
        } else {
            &BLUE_CHARS
        }
    }
    fn opp(&self) -> &'static [char; 12] {
        if self.player == 0 {
            &BLUE_CHARS
        } else {
            &RED_CHARS
        }
    }

    fn is_our(&self, ch: char) -> bool {
        self.our().contains(&ch.to_ascii_lowercase())
    }
    fn is_opp(&self, ch: char) -> bool {
        self.opp().contains(&ch.to_ascii_lowercase())
    }

    fn update(&mut self, board: &Board, src: i32, dst: i32, last: Option<(i32, i32)>) {
        let ids = piece_ids_pov(board, self.player);
        let (src, dst) = pov_move(src, dst, self.player);

        let Some((_last_src, last_dst)) = last else {
            // First move of the game: the moving piece protects/against the
            // unknown, per the special-cased branch.
            if is_adjacent_ref(src, dst) || is_two_squares_away(src, dst) {
                let sid = ids[src as usize];
                self.protected_[sid][EMPTY_BUCKET] = true;
                self.protected_against[sid][UNKNOWN_BUCKET] = true;
            }
            return;
        };
        let last_dst = if self.player == 1 {
            99 - last_dst
        } else {
            last_dst
        };

        let bs = board_str_pov(board, self.player);
        let piece_id = ids[src as usize];

        // Moving piece revealed an opponent it lost to.
        if self.is_opp(bs[dst as usize]) && is_defender_wins_p(bs[src as usize], bs[dst as usize]) {
            for our_pos in two_square_positions(dst) {
                if !self.is_our(bs[our_pos as usize]) {
                    continue;
                }
                for protectee_pos in one_square_positions(our_pos) {
                    let pc = bs[protectee_pos as usize];
                    if pc == '_' {
                        continue;
                    }
                    if self.is_opp(pc) {
                        continue;
                    }
                    if !(is_adjacent_ref(protectee_pos, our_pos)
                        && is_adjacent_ref(protectee_pos, dst))
                    {
                        continue;
                    }
                    let our_id = ids[our_pos as usize];
                    self.protected_against[our_id][l2i(bs[dst as usize])] = true;
                    if pc != 'a' {
                        let protectee_id = ids[protectee_pos as usize];
                        self.was_protected_against[protectee_id][l2i(bs[dst as usize])] = true;
                    }
                }
            }
        }

        // Moving piece cleared an open cell for protection.
        for opp_pos in one_square_positions(src) {
            if opp_pos == dst && !is_defender_wins_p(bs[src as usize], bs[dst as usize]) {
                continue;
            }
            let op = bs[opp_pos as usize];
            if op == '_' {
                continue;
            }
            if !self.is_opp(op) {
                continue;
            }
            for protector_pos in one_square_positions(src) {
                if !is_two_squares_away(protector_pos, opp_pos) {
                    continue;
                }
                if !self.is_our(bs[protector_pos as usize]) {
                    continue;
                }
                let prid = ids[protector_pos as usize];
                self.protected_[prid][EMPTY_BUCKET] = true;
                if is_upper(op) && opp_pos != dst {
                    self.protected_against[prid][UNKNOWN_BUCKET] = true;
                } else {
                    self.protected_against[prid][l2i(op)] = true;
                }
            }
        }

        // Moving piece is actively protecting.
        if !is_attacker_dies_p(bs[src as usize], bs[dst as usize]) {
            for opp_pos in two_square_positions(dst) {
                if !self.is_opp(bs[opp_pos as usize]) {
                    continue;
                }
                for mv in [-10, -1, 1, 10] {
                    let protectee_pos = opp_pos + mv;
                    if !(is_adjacent_ref(dst, protectee_pos)
                        && is_adjacent_ref(protectee_pos, opp_pos))
                    {
                        continue;
                    }
                    if !on_board(protectee_pos) {
                        continue;
                    }
                    let pc = bs[protectee_pos as usize];
                    if pc == '_' {
                        continue;
                    }
                    if self.is_opp(pc) {
                        continue;
                    }
                    let protectee_idx = if protectee_pos == src || pc == 'a' {
                        EMPTY_BUCKET
                    } else if is_upper(pc) {
                        UNKNOWN_BUCKET
                    } else {
                        l2i(pc)
                    };
                    self.protected_[piece_id][protectee_idx] = true;
                    if is_upper(bs[opp_pos as usize]) {
                        self.protected_against[piece_id][UNKNOWN_BUCKET] = true;
                    } else {
                        self.protected_against[piece_id][l2i(bs[opp_pos as usize])] = true;
                    }
                    if protectee_idx == EMPTY_BUCKET {
                        continue;
                    }
                    let protectee_id = ids[protectee_pos as usize];
                    if is_upper(bs[src as usize]) && is_adjacent_ref(src, dst) {
                        self.was_protected_by[protectee_id][UNKNOWN_BUCKET] = true;
                    } else {
                        self.was_protected_by[protectee_id][l2i(bs[src as usize])] = true;
                    }
                    if is_upper(bs[opp_pos as usize]) {
                        self.was_protected_against[protectee_id][UNKNOWN_BUCKET] = true;
                    } else {
                        self.was_protected_against[protectee_id][l2i(bs[opp_pos as usize])] = true;
                    }
                }
            }
        }

        // Moving piece moved into a protected position.
        for opp_pos in one_square_positions(dst) {
            if !self.is_opp(bs[opp_pos as usize]) {
                continue;
            }
            for protector_pos in one_square_positions(dst) {
                if !is_two_squares_away(protector_pos, opp_pos) {
                    continue;
                }
                if !self.is_our(bs[protector_pos as usize]) {
                    continue;
                }
                if is_attacker_dies_p(bs[src as usize], bs[dst as usize]) {
                    continue;
                }
                if src == protector_pos {
                    continue;
                }
                let protector_id = ids[protector_pos as usize];
                let protected_idx = if self.is_opp(bs[dst as usize]) || !is_adjacent_ref(src, dst) {
                    l2i(bs[src as usize])
                } else if is_upper(bs[src as usize]) {
                    UNKNOWN_BUCKET
                } else {
                    l2i(bs[src as usize])
                };
                self.protected_[protector_id][protected_idx] = true;
                if is_upper(bs[protector_pos as usize]) {
                    self.was_protected_by[piece_id][UNKNOWN_BUCKET] = true;
                } else {
                    self.was_protected_by[piece_id][l2i(bs[protector_pos as usize])] = true;
                }
                if is_upper(bs[opp_pos as usize]) {
                    self.protected_against[protector_id][UNKNOWN_BUCKET] = true;
                    self.was_protected_against[piece_id][UNKNOWN_BUCKET] = true;
                } else {
                    self.protected_against[protector_id][l2i(bs[opp_pos as usize])] = true;
                    self.was_protected_against[piece_id][l2i(bs[opp_pos as usize])] = true;
                }
            }
        }

        // Passive protection against the opponent's last move.
        if !self.is_opp(bs[last_dst as usize]) {
            return;
        }
        if dst == last_dst && is_defender_dies_p(bs[src as usize], bs[last_dst as usize]) {
            return;
        }
        for our_pos in two_square_positions(last_dst) {
            let our_id = ids[our_pos as usize];
            if !self.is_our(bs[our_pos as usize]) {
                continue;
            }
            if our_pos == src {
                continue;
            }
            for mv in [-10, -1, 1, 10] {
                let protectee_pos = our_pos + mv;
                if !on_board(protectee_pos) {
                    continue;
                }
                if protectee_pos == dst {
                    continue;
                }
                let protectee_id = ids[protectee_pos as usize];
                if !(is_adjacent_ref(protectee_pos, our_pos)
                    && is_adjacent_ref(protectee_pos, last_dst))
                {
                    continue;
                }
                let pc = bs[protectee_pos as usize];
                if pc == '_' {
                    continue;
                }
                if self.is_opp(pc) {
                    continue;
                }
                if protectee_pos == src {
                    continue;
                }
                let protectee_idx = if src == protectee_pos || pc == 'a' {
                    EMPTY_BUCKET
                } else if is_upper(pc) {
                    UNKNOWN_BUCKET
                } else {
                    l2i(pc)
                };
                self.protected_[our_id][protectee_idx] = true;
                if is_upper(bs[last_dst as usize]) && last_dst != dst {
                    self.protected_against[our_id][UNKNOWN_BUCKET] = true;
                } else {
                    self.protected_against[our_id][l2i(bs[last_dst as usize])] = true;
                }
                if protectee_pos == src || pc == 'a' {
                    continue;
                }
                if is_upper(bs[our_pos as usize]) {
                    self.was_protected_by[protectee_id][UNKNOWN_BUCKET] = true;
                } else {
                    self.was_protected_by[protectee_id][l2i(bs[our_pos as usize])] = true;
                }
                if is_upper(bs[last_dst as usize]) && last_dst != dst {
                    self.was_protected_against[protectee_id][UNKNOWN_BUCKET] = true;
                } else {
                    self.was_protected_against[protectee_id][l2i(bs[last_dst as usize])] = true;
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum ProtGroup {
    Protected,
    ProtectedAgainst,
    WasProtectedBy,
    WasProtectedAgainst,
}

fn prot_offsets(g: ProtGroup) -> (usize, usize) {
    match g {
        ProtGroup::Protected => (251, 303),
        ProtGroup::ProtectedAgainst => (264, 316),
        ProtGroup::WasProtectedBy => (277, 329),
        ProtGroup::WasProtectedAgainst => (290, 342),
    }
}

fn prot_vec(t: &ProtectTracker, g: ProtGroup, id: usize) -> [bool; 13] {
    match g {
        ProtGroup::Protected => t.protected_[id],
        ProtGroup::ProtectedAgainst => t.protected_against[id],
        ProtGroup::WasProtectedBy => t.was_protected_by[id],
        ProtGroup::WasProtectedAgainst => t.was_protected_against[id],
    }
}

/// An independently derived protection oracle that validates the production
/// encoder (`rules.rs` geometry) without copying it. It accumulates the four
/// protection bitsets, keyed by `(player, own_slot)`.
///
/// The reference semantics (`action_kernels.cu` `BoardStateKernel__Protections`
/// plus `UPDATE_PROTECT`) reduce to one geometric invariant. A protector,
/// protectee, aggressor triple is a protection exactly when the protectee is
/// orthogonally adjacent to both its protector and the aggressor, and the
/// protector and aggressor are distinct cells — the protectee sits between a
/// defender and a threat. The per-move `step` dispatch (taken from the
/// reference) recomputes only the triples touching a changed cell, across the
/// five roles a changed cell can play: previously moved enemy as aggressor,
/// surviving mover as protector, mover or tie-emptied square as protectee,
/// revealed winning defender as aggressor, and vacated source as protectee.
///
/// Crucially, the neighbour enumeration here is rebuilt from [`orth_neighbors`],
/// not from the production knight-offset tables in `rules.rs`, so a transposed
/// or missing offset in the production geometry diverges from this oracle and
/// fails the test where a byte-identical copy never could. The random-game
/// driver exercises all four groups across quiet, attack, tie, and reveal moves.
struct ProtectOracle {
    protected_: [[[bool; 13]; 40]; 2],
    protected_against: [[[bool; 13]; 40]; 2],
    was_protected_by: [[[bool; 13]; 40]; 2],
    was_protected_against: [[[bool; 13]; 40]; 2],
}

/// The orthogonal on-board neighbours of absolute cell `cell` on the 10x10 grid
/// — the independent adjacency primitive the protection geometry is rebuilt from.
fn orth_neighbors(cell: i32) -> Vec<i32> {
    let (row, col) = (cell / 10, cell % 10);
    let mut out = Vec::with_capacity(4);
    if row > 0 {
        out.push(cell - 10);
    }
    if row < 9 {
        out.push(cell + 10);
    }
    if col > 0 {
        out.push(cell - 1);
    }
    if col < 9 {
        out.push(cell + 1);
    }
    out
}

/// Type -> protection bucket index (`pieces_with_extras`): movables 0..9,
/// bomb 10, empty 11, anything else (hidden) 12.
fn protect_bucket(t: u8) -> usize {
    match t {
        0..=9 => t as usize,
        x if x == PieceType::Bomb as u8 => 10,
        x if x == PieceType::Empty as u8 => 11,
        _ => 12,
    }
}

impl ProtectOracle {
    fn new() -> Self {
        ProtectOracle {
            protected_: [[[false; 13]; 40]; 2],
            protected_against: [[[false; 13]; 40]; 2],
            was_protected_by: [[[false; 13]; 40]; 2],
            was_protected_against: [[[false; 13]; 40]; 2],
        }
    }

    /// One `UPDATE_PROTECT(a, b, c)` over the committed board for acting `player`
    /// (0-indexed).
    fn update_protect(&mut self, board: &Board, player: usize, a: i32, b: i32, c: i32) {
        if a == b || a == c || b == c {
            return;
        }
        let own = Color::of_player(player);
        let opp = Color::of_player(1 - player);
        let protector = board.pieces[a as usize];
        let protectee = board.pieces[b as usize];
        let aggressor = board.pieces[c as usize];
        if aggressor.color == opp
            && (protectee.color == own || protectee.color == Color::Empty)
            && protector.color == own
        {
            let protector_pt = protector.tracked_type();
            let protectee_pt = protectee.tracked_type();
            let aggressor_pt = aggressor.tracked_type();
            let pr_slot = protector.piece_id as usize;
            self.protected_[player][pr_slot][protect_bucket(protectee_pt)] = true;
            self.protected_against[player][pr_slot][protect_bucket(aggressor_pt)] = true;
            if protectee.color == own {
                let pe_slot = protectee.piece_id as usize;
                self.was_protected_by[player][pe_slot][protect_bucket(protector_pt)] = true;
                self.was_protected_against[player][pe_slot][protect_bucket(aggressor_pt)] = true;
            }
        }
    }

    /// Anchor = aggressor at `center`: every protectee is an orthogonal
    /// neighbour of the aggressor, and its protector is an orthogonal neighbour
    /// of the protectee other than the aggressor itself.
    fn aggressor_pattern(&mut self, board: &Board, player: usize, center: i32) {
        for protectee in orth_neighbors(center) {
            for protector in orth_neighbors(protectee) {
                if protector != center {
                    self.update_protect(board, player, protector, protectee, center);
                }
            }
        }
    }

    /// Anchor = protector at `dst`: every protectee is an orthogonal neighbour of
    /// the protector, and the aggressor is an orthogonal neighbour of the
    /// protectee other than the protector itself.
    fn protector_pattern(&mut self, board: &Board, player: usize, dst: i32) {
        for protectee in orth_neighbors(dst) {
            for aggressor in orth_neighbors(protectee) {
                if aggressor != dst {
                    self.update_protect(board, player, dst, protectee, aggressor);
                }
            }
        }
    }

    /// Anchor = protectee at `center`: the protector and the aggressor are two
    /// distinct orthogonal neighbours of the protectee.
    fn protectee_pattern(&mut self, board: &Board, player: usize, center: i32) {
        for protector in orth_neighbors(center) {
            for aggressor in orth_neighbors(center) {
                if protector != aggressor {
                    self.update_protect(board, player, protector, center, aggressor);
                }
            }
        }
    }

    /// Replay all five cases for a move, given the kernel scalars captured before
    /// the move and the now-committed `board`.
    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        board: &Board,
        player: usize,
        from_abs: i32,
        to_abs: i32,
        prev_dst_before: u8,
        last_moved_before: u8,
        to_wins: bool,
        tie: bool,
    ) {
        if last_moved_before != 0xff && prev_dst_before != 0xff {
            self.aggressor_pattern(board, player, prev_dst_before as i32);
        }
        if !(to_wins || tie) {
            self.protector_pattern(board, player, to_abs);
        }
        if !to_wins {
            self.protectee_pattern(board, player, to_abs);
        }
        if to_wins {
            self.aggressor_pattern(board, player, to_abs);
        }
        self.protectee_pattern(board, player, from_abs);
    }

    fn get(&self, g: ProtGroup, player: usize, slot: usize) -> [bool; 13] {
        match g {
            ProtGroup::Protected => self.protected_[player][slot],
            ProtGroup::ProtectedAgainst => self.protected_against[player][slot],
            ProtGroup::WasProtectedBy => self.was_protected_by[player][slot],
            ProtGroup::WasProtectedAgainst => self.was_protected_against[player][slot],
        }
    }
}

/// Drives random games checking every protection group against the independent
/// adjacency oracle over quiet, attack, tie, and reveal moves. The
/// `(attacks, ties, reveals)` tally is asserted nonzero so the test cannot pass
/// by exercising quiet slides alone.
fn run_protect_group(seeds: u64) {
    let cfg = EncoderConfig::default();
    let groups = [
        ProtGroup::Protected,
        ProtGroup::ProtectedAgainst,
        ProtGroup::WasProtectedBy,
        ProtGroup::WasProtectedAgainst,
    ];
    let (mut attacks, mut ties, mut reveals) = (0u32, 0u32, 0u32);
    for seed in 0..seeds {
        let mut rng = Rng::new(seed + 101);
        let mut board = fresh_play(&mut rng);
        let mut oracle = ProtectOracle::new();
        let mut player = 0usize;
        for _ in 0..160 {
            if rules::is_terminal(&board, player, None) {
                break;
            }
            let enc = encode_infostate(&board, player, &cfg);
            let bs = board_str_pov(&board, player);
            let ids = piece_ids_pov(&board, player);

            for g in groups {
                let (we_s, they_s) = prot_offsets(g);
                for piece_id in 0..40usize {
                    let Some(pov_cell) = ids.iter().position(|&x| x == piece_id) else {
                        continue;
                    };
                    let expect = if is_upper(bs[pov_cell]) {
                        oracle.get(g, player, piece_id)
                    } else {
                        [false; 13]
                    };
                    for i in 0..13 {
                        assert_eq!(
                            enc[(we_s + i) * 100 + pov_cell],
                            f32::from(expect[i]),
                            "our {g:?} mismatch seed={seed} id={piece_id} chan={i}"
                        );
                    }
                }
                for view_id in 60..100usize {
                    let own_slot = 99 - view_id;
                    let Some(pov_cell) = ids.iter().position(|&x| x == view_id) else {
                        continue;
                    };
                    let expect = if is_upper(bs[pov_cell]) {
                        oracle.get(g, 1 - player, own_slot)
                    } else {
                        [false; 13]
                    };
                    for i in 0..13 {
                        assert_eq!(
                            enc[(they_s + i) * 100 + pov_cell],
                            f32::from(expect[i]),
                            "their {g:?} mismatch seed={seed} view_id={view_id} chan={i}"
                        );
                    }
                }
            }

            let mask = rules::legal_mask(&board, player);
            let legal: Vec<usize> = (0..mask.len()).filter(|&i| mask[i]).collect();
            if legal.is_empty() {
                break;
            }
            let a = legal[rng.below(legal.len())];
            let (from_abs, to_abs) = Action(a as u16).to_abs(player);
            let prev_dst_before = board.prev_dst;
            let last_moved_before = board.last_moved_piece_type;
            let applied = rules::apply(&mut board, Action(a as u16), player);
            let (to_wins, tie) = battle_kind(&applied, &board, to_abs, player);
            attacks += u32::from(applied.was_attack);
            ties += u32::from(tie);
            reveals += u32::from(to_wins);
            oracle.step(
                &board,
                player,
                from_abs as i32,
                to_abs as i32,
                prev_dst_before,
                last_moved_before,
                to_wins,
                tie,
            );
            player = 1 - player;
        }
    }
    assert!(
        attacks > 0 && ties > 0 && reveals > 0,
        "protection oracle must cover attack/tie/reveal cases, got \
         attacks={attacks} ties={ties} reveals={reveals}"
    );
}

/// Recover `(to_wins, tie)` from the applied result and the post-move board: a
/// tie leaves the destination empty after an attack; a defender win leaves an
/// opponent piece there.
fn battle_kind(
    applied: &rules::Applied,
    board: &Board,
    to_abs: usize,
    player: usize,
) -> (bool, bool) {
    if !applied.was_attack {
        return (false, false);
    }
    let dst = board.pieces[to_abs];
    let own = Color::of_player(player);
    if dst.color == Color::Empty {
        (false, true) // tie
    } else if dst.color == own {
        (false, false) // attacker won
    } else {
        (true, false) // defender won
    }
}

#[test]
fn protection_groups_match_kernel_oracle() {
    run_protect_group(120);
}

/// The three protection groups for which the heuristic Python tracker is
/// faithful in non-attacking play, cross-checked over random games where every
/// move is a quiet slide. A second, fully-independent oracle for the common case.
#[test]
fn quiet_protection_matches_python_tracker() {
    let cfg = EncoderConfig::default();
    let groups = [
        ProtGroup::Protected,
        ProtGroup::WasProtectedBy,
        ProtGroup::WasProtectedAgainst,
    ];
    for seed in 0..120u64 {
        let mut rng = Rng::new(seed + 303);
        let mut board = fresh_play(&mut rng);
        let mut trackers = [ProtectTracker::new(0), ProtectTracker::new(1)];
        let mut last_move: Option<(i32, i32)> = None;
        let mut player = 0usize;
        for _ in 0..160 {
            if rules::is_terminal(&board, player, None) {
                break;
            }
            let enc = encode_infostate(&board, player, &cfg);
            let bs = board_str_pov(&board, player);
            let ids = piece_ids_pov(&board, player);
            for g in groups {
                let (we_s, they_s) = prot_offsets(g);
                for piece_id in 0..40usize {
                    let Some(pc) = ids.iter().position(|&x| x == piece_id) else {
                        continue;
                    };
                    if !is_upper(bs[pc]) {
                        continue;
                    }
                    let e = prot_vec(&trackers[player], g, piece_id);
                    for i in 0..13 {
                        assert_eq!(enc[(we_s + i) * 100 + pc], f32::from(e[i]));
                    }
                }
                for view_id in 60..100usize {
                    let Some(pc) = ids.iter().position(|&x| x == view_id) else {
                        continue;
                    };
                    if !is_upper(bs[pc]) {
                        continue;
                    }
                    let e = prot_vec(&trackers[1 - player], g, 99 - view_id);
                    for i in 0..13 {
                        assert_eq!(enc[(they_s + i) * 100 + pc], f32::from(e[i]));
                    }
                }
            }

            // Restrict to quiet (non-attacking) moves so the Python tracker stays
            // faithful; fall back to any legal move only if no quiet move exists.
            let mask = rules::legal_mask(&board, player);
            let own = Color::of_player(player);
            let quiet: Vec<usize> = (0..mask.len())
                .filter(|&i| mask[i])
                .filter(|&i| {
                    let (_s, d) = Action(i as u16).to_abs(player);
                    board.pieces[d].color == Color::Empty && board.pieces[d].color != own
                })
                .collect();
            if quiet.is_empty() {
                break;
            }
            let a = quiet[rng.below(quiet.len())];
            let (src, dst) = Action(a as u16).to_abs(player);
            trackers[player].update(&board, src as i32, dst as i32, last_move);
            rules::apply(&mut board, Action(a as u16), player);
            last_move = Some((src as i32, dst as i32));
            player = 1 - player;
        }
    }
}

#[test]
fn we_they_threatened_match_tracker() {
    run_attack_group(Group::Threatened, 60);
}

#[test]
fn we_they_evaded_match_tracker() {
    run_attack_group(Group::Evaded, 60);
}

#[test]
fn we_they_actively_adjacent_match_tracker() {
    run_attack_group(Group::ActAdj, 60);
}

// --- Own piece-type one-hot (channels 0..=11) -------------------------------

#[test]
fn own_piece_types_one_hot() {
    let cfg = EncoderConfig::default();
    let red = Arrangement::from_chars("KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD").unwrap();
    let blue = Arrangement::from_chars("KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD").unwrap();
    let board = arrangement::board_from_arrangements(&red, &blue);
    // Player 0 view: each own red piece writes a 1.0 at channel = its type, POV
    // cell = absolute cell (player 0 POV is identity).
    let enc = encode_infostate(&board, 0, &cfg);
    for cell in 0..40usize {
        let p = board.pieces[cell];
        if p.color != Color::Red {
            continue;
        }
        let t = p.kind as usize;
        assert_eq!(enc[100 * t + cell], 1.0, "own one-hot at {cell}");
        // No other own-type channel set at this cell.
        for other in 0..12usize {
            if other != t {
                assert_eq!(enc[100 * other + cell], 0.0);
            }
        }
    }
    // Player 1 view applies the 180-degree reflection.
    let enc1 = encode_infostate(&board, 1, &cfg);
    for cell in 60..100usize {
        let p = board.pieces[cell];
        if p.color != Color::Blue {
            continue;
        }
        let t = p.kind as usize;
        assert_eq!(enc1[100 * t + (99 - cell)], 1.0, "own one-hot p1 at {cell}");
    }
}

// --- "If-uniform-random" posterior (channels 12..=23) -----------------------

/// Builds a board where, from player 0's view, the opponent (blue) has a known
/// hidden composition, so the analytic posterior is hand-computable.
#[test]
fn posterior_moved_hidden_uses_movable_denominator() {
    let cfg = EncoderConfig::default();
    let mut board = Board::blank();
    // Blue hidden composition: 2 spies, 1 scout, 1 flag, 1 bomb (none moved yet
    // except the one we expose). total=5, denom = 5 - 1(flag) - 1(bomb) = 3.
    board.num_hidden[1][PieceType::Spy as usize] = 2;
    board.num_hidden[1][PieceType::Scout as usize] = 1;
    board.num_hidden[1][PieceType::Flag as usize] = 1;
    board.num_hidden[1][PieceType::Bomb as usize] = 1;
    board.num_hidden_unmoved[1] = 4; // four of the five never moved
    // A blue hidden piece that HAS moved sits at cell 50 (so it can't be flag/bomb).
    let mut moved = crate::board::Piece::new(PieceType::Spy, Color::Blue, 10);
    moved.has_moved = true;
    moved.visible = false;
    board.pieces[50] = moved;

    let enc = encode_infostate(&board, 0, &cfg);
    let base = 1200 + 50; // ch 12 starts at cell offset 1200; POV cell 50.
    // denom = 3; movables: spy 2/3, scout 1/3, rest 0.
    assert!((enc[base] - 2.0 / 3.0).abs() < 1e-6);
    assert!((enc[100 + base] - 1.0 / 3.0).abs() < 1e-6);
    for t in 2..10usize {
        assert_eq!(enc[100 * t + base], 0.0);
    }
    // Flag/bomb planes are not written for a moved piece.
    assert_eq!(enc[100 * 10 + base], 0.0);
    assert_eq!(enc[100 * 11 + base], 0.0);
}

#[test]
fn posterior_unmoved_hidden_weights_flag_and_bomb() {
    let cfg = EncoderConfig::default();
    let mut board = Board::blank();
    // Blue: 1 spy, 1 flag, 1 bomb, all hidden, all unmoved. total=3,
    // denom = 3 - 1 - 1 = 1, num_hidden_unmoved = 3.
    board.num_hidden[1][PieceType::Spy as usize] = 1;
    board.num_hidden[1][PieceType::Flag as usize] = 1;
    board.num_hidden[1][PieceType::Bomb as usize] = 1;
    board.num_hidden_unmoved[1] = 3;
    let piece = crate::board::Piece::new(PieceType::Spy, Color::Blue, 20);
    board.pieces[80] = piece; // hidden, never moved

    let enc = encode_infostate(&board, 0, &cfg);
    let base = 1200 + 80;
    // norm = (unmoved - flag - bomb) / (unmoved * denom) = (3-1-1)/(3*1) = 1/3.
    let norm = 1.0f32 / 3.0;
    assert!((enc[base] - norm).abs() < 1e-6, "spy posterior");
    for t in 1..10usize {
        assert_eq!(enc[100 * t + base], 0.0);
    }
    // flag = num_hidden[flag]/unmoved = 1/3; bomb = 1/3.
    assert!(
        (enc[100 * 10 + base] - 1.0 / 3.0).abs() < 1e-6,
        "flag posterior"
    );
    assert!(
        (enc[100 * 11 + base] - 1.0 / 3.0).abs() < 1e-6,
        "bomb posterior"
    );
}

#[test]
fn posterior_visible_opponent_is_one_hot() {
    let cfg = EncoderConfig::default();
    let mut board = Board::blank();
    let mut p = crate::board::Piece::new(PieceType::Major, Color::Blue, 5);
    p.visible = true;
    board.pieces[70] = p;
    let enc = encode_infostate(&board, 0, &cfg);
    let base = 1200 + 70;
    assert_eq!(enc[100 * (PieceType::Major as usize) + base], 1.0);
    for t in 0..12usize {
        if t != PieceType::Major as usize {
            assert_eq!(enc[100 * t + base], 0.0);
        }
    }
}

/// `test_unknown_piece_counts.py`: the hidden-type counts must equal
/// `num_hidden`. We assert that the encoder's posterior support (nonzero
/// channels) only spans types that are actually present in `num_hidden`, and
/// that visible opponents reduce the counts. Verified over random games via the
/// posterior column summing to 1 for each hidden opponent cell.
#[test]
fn posterior_columns_sum_to_one_for_hidden_opponents() {
    let cfg = EncoderConfig::default();
    for seed in 0..40u64 {
        let mut rng = Rng::new(seed + 909);
        let mut board = fresh_play(&mut rng);
        let mut player = 0usize;
        for _ in 0..80 {
            if rules::is_terminal(&board, player, None) {
                break;
            }
            let enc = encode_infostate(&board, player, &cfg);
            for cell in 0..100usize {
                let p = board.pieces[cell];
                let is_opp = p.color as u8 == 3 - (player as u8 + 1);
                if !is_opp || p.visible {
                    continue;
                }
                let pov = if player == 1 { 99 - cell } else { cell };
                let sum: f32 = (0..12).map(|t| enc[100 * t + 1200 + pov]).sum();
                // The posterior over a hidden opponent must be a distribution.
                assert!(
                    (sum - 1.0).abs() < 1e-4,
                    "seed={seed} cell={cell} posterior sum={sum}"
                );
            }
            let mask = rules::legal_mask(&board, player);
            let legal: Vec<usize> = (0..mask.len()).filter(|&i| mask[i]).collect();
            if legal.is_empty() {
                break;
            }
            let a = legal[rng.below(legal.len())];
            rules::apply(&mut board, Action(a as u16), player);
            player = 1 - player;
        }
    }
}

/// The opponent's posterior over my pieces (channels 24..=35) reflects back with
/// the extra `^ rotate`: it uses the acting player's hidden composition and lands
/// at the same POV cells as my own one-hot.
#[test]
fn posterior_over_me_uses_my_hidden_counts_rotated() {
    let cfg = EncoderConfig::default();
    let mut board = Board::blank();
    // My (red, player 0) hidden: 1 spy unmoved + flag + bomb -> mirrors the
    // unmoved posterior math, written to channels 24..=35 at my cell.
    board.num_hidden[0][PieceType::Spy as usize] = 1;
    board.num_hidden[0][PieceType::Flag as usize] = 1;
    board.num_hidden[0][PieceType::Bomb as usize] = 1;
    board.num_hidden_unmoved[0] = 3;
    board.pieces[5] = crate::board::Piece::new(PieceType::Spy, Color::Red, 5);

    let enc = encode_infostate(&board, 0, &cfg);
    let base = 2400 + 5; // ch 24, POV cell 5 (player 0 identity, rotate cancels).
    let norm = 1.0f32 / 3.0;
    assert!((enc[base] - norm).abs() < 1e-6);
    assert!((enc[100 * 10 + base] - 1.0 / 3.0).abs() < 1e-6);
    assert!((enc[100 * 11 + base] - 1.0 / 3.0).abs() < 1e-6);
}

// --- Cemetery (channels 109..=130) ------------------------------------------

#[test]
fn cemetery_marks_dead_at_starting_square_with_initial_type() {
    let cfg = EncoderConfig::default();
    // Red scout at 30 attacks blue major at 31? Build a tiny lethal setup:
    // red marshal at 30, blue scout at 31 (visible) -> red wins, blue scout dies.
    let mut board = Board::blank();
    let mut rmar = crate::board::Piece::new(PieceType::Marshal, Color::Red, 30);
    rmar.visible = true;
    board.pieces[30] = rmar;
    board.zero_types[0][30] = PieceType::Marshal as u8;
    let mut bsc = crate::board::Piece::new(PieceType::Scout, Color::Blue, 8);
    bsc.visible = true;
    board.pieces[31] = bsc;
    board.zero_types[1][8] = PieceType::Scout as u8;
    board.num_hidden_unmoved = [1, 1];

    let act = Action::from_abs(30, 31, 0).unwrap();
    rules::apply(&mut board, act, 0);
    // Blue scout (slot 8) is dead; its starting absolute cell is 99-8 = 91.
    let enc = encode_infostate(&board, 0, &cfg);
    // their_dead_scout = channel 120 + 1, at POV cell 91 (player 0 identity).
    assert_eq!(
        enc[(120 + 1) * 100 + 91],
        1.0,
        "blue scout in their cemetery"
    );
    // No other their_dead type at that cell.
    for i in 0..11usize {
        if i != 1 {
            assert_eq!(enc[(120 + i) * 100 + 91], 0.0);
        }
    }
    // The death square itself (31) holds the marshal now, not a cemetery mark.
    for i in 0..11usize {
        assert_eq!(enc[(120 + i) * 100 + 31], 0.0);
    }
}

// --- Death reasons (channels 131..=250) -------------------------------------

#[test]
fn death_reason_marked_at_death_location() {
    let cfg = EncoderConfig::default();
    // Red marshal (visible) at 30 attacks visible blue scout at 31: blue scout
    // dies as VISIBLE_DEFENDED_WEAKER at the death location 31.
    let mut board = Board::blank();
    let mut rmar = crate::board::Piece::new(PieceType::Marshal, Color::Red, 30);
    rmar.visible = true;
    board.pieces[30] = rmar;
    board.zero_types[0][30] = PieceType::Marshal as u8;
    let mut bsc = crate::board::Piece::new(PieceType::Scout, Color::Blue, 8);
    bsc.visible = true;
    board.pieces[31] = bsc;
    board.zero_types[1][8] = PieceType::Scout as u8;
    board.num_hidden_unmoved = [1, 1];

    rules::apply(&mut board, Action::from_abs(30, 31, 0).unwrap(), 0);
    let enc = encode_infostate(&board, 0, &cfg);
    // their_deathstatus: base 191, reason VISIBLE_DEFENDED_WEAKER=3, type scout=1.
    let ch = 191 + 3 * 10 + 1;
    assert_eq!(enc[ch * 100 + 31], 1.0, "scout death-reason at 31");
    // Nothing on our side.
    let our = 131 + 3 * 10 + 1;
    assert_eq!(enc[our * 100 + 31], 0.0);
}

// --- 32 src/dst history planes (channels 355..=386) -------------------------

#[test]
fn history_planes_round_trip_with_parity_flip() {
    let cfg = EncoderConfig::default();
    let red = Arrangement::from_chars("KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD").unwrap();
    let blue = Arrangement::from_chars("KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD").unwrap();
    let mut board = arrangement::board_from_arrangements(&red, &blue);
    // Make a couple of moves and confirm the most-recent plane decodes back.
    let moves = [(0usize, 30usize, 40usize), (1, 60, 50)];
    for &(p, s, d) in &moves {
        let act = Action::from_abs(s, d, p).unwrap();
        rules::apply(&mut board, act, p);
    }
    // Observer is player 0 now (to_play after two moves). Most recent move is
    // blue's 60->50, which in player-0 POV with delta=1 (odd) is flipped.
    let enc = encode_infostate(&board, 0, &cfg);
    let last_plane = NUM_BOARD_STATE_CHANNELS + cfg.history_len - 1;
    let base = last_plane * 100;
    // Reconstruct the from/to cells from the plane and re-derive the action,
    // mirroring test_far_action_history.py.
    let from_cell = (0..100).find(|&c| enc[base + c] == -1.0).unwrap();
    let to_cell = (0..100).find(|&c| enc[base + c] == 1.0).unwrap();
    // delta = 1 is odd -> the observer (player 0) differs in parity from the
    // acting player (player 1), so flip back to the acting POV.
    let (f, t) = (99 - from_cell, 99 - to_cell);
    let action = decode_history_action(f, t);
    assert_eq!(
        action,
        board.action_history[board.action_history.len() - 1] as usize
    );

    // The second-to-last plane holds red's 30->40 (delta=2, even -> no flip).
    let prev_plane = NUM_BOARD_STATE_CHANNELS + cfg.history_len - 2;
    let pbase = prev_plane * 100;
    let pf = (0..100).find(|&c| enc[pbase + c] == -1.0).unwrap();
    let pt = (0..100).find(|&c| enc[pbase + c] == 1.0).unwrap();
    let paction = decode_history_action(pf, pt);
    assert_eq!(paction, board.action_history[0] as usize);
}

/// Re-derive the action index from a (from, to) pair in the acting player's POV,
/// the inverse used by `test_far_action_history.py:88-99`.
fn decode_history_action(from_cell: usize, to_cell: usize) -> usize {
    let new_coord = if from_cell % 10 == to_cell % 10 {
        let mut nc = to_cell / 10;
        if nc > from_cell / 10 {
            nc -= 1;
        }
        nc
    } else {
        let mut nc = to_cell % 10;
        if nc > from_cell % 10 {
            nc -= 1;
        }
        nc + 9
    };
    from_cell + 100 * new_coord
}

#[test]
fn history_planes_are_empty_before_any_move() {
    let cfg = EncoderConfig::default();
    let red = Arrangement::from_chars("KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD").unwrap();
    let blue = Arrangement::from_chars("KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD").unwrap();
    let board = arrangement::board_from_arrangements(&red, &blue);
    let enc = encode_infostate(&board, 0, &cfg);
    for ch in NUM_BOARD_STATE_CHANNELS..NUM_BOARD_STATE_CHANNELS + cfg.history_len {
        for c in 0..100 {
            assert_eq!(enc[ch * 100 + c], 0.0, "history empty at game start");
        }
    }
}

// --- Piece-id one-hot (256) + token layout ----------------------------------

#[test]
fn token_layout_is_92_by_643() {
    let cfg = EncoderConfig::default();
    let mut rng = Rng::new(11);
    let board = fresh_play(&mut rng);
    let tokens = encode_tokens(&board, 0, &cfg);
    assert_eq!(
        tokens.len(),
        NUM_OCCUPIABLE_CELLS * (355 + 32 + NUM_PIECE_ID)
    );
    assert_eq!(cfg.num_token_features(), 643);
    assert_eq!(NUM_OCCUPIABLE_CELLS, 92);

    // Each token carries exactly one piece-id one-hot bit (own/opp/empty=255).
    let nfeat = cfg.num_token_features();
    let nchan = cfg.num_infostate_channels();
    for token in 0..NUM_OCCUPIABLE_CELLS {
        let base = token * nfeat + nchan;
        let on: Vec<usize> = (0..NUM_PIECE_ID)
            .filter(|&i| tokens[base + i] == 1.0)
            .collect();
        assert_eq!(on.len(), 1, "exactly one piece-id bit per token");
    }
}

#[test]
fn piece_id_ranges_match_reference() {
    let mut rng = Rng::new(13);
    let mut board = fresh_play(&mut rng);
    let mut player = 0usize;
    for _ in 0..50 {
        if rules::is_terminal(&board, player, None) {
            break;
        }
        let ids = piece_id_grid(&board, player);
        for (pov_cell, &id) in ids.iter().enumerate() {
            let abs = if player == 1 { 99 - pov_cell } else { pov_cell };
            let p = board.pieces[abs];
            match p.color {
                Color::Empty | Color::Lake => assert_eq!(id, 255),
                c if c as u8 == player as u8 + 1 => assert!(id < 40, "own id<40"),
                _ => assert!((60..100).contains(&id), "opp id in [60,100)"),
            }
        }
        let mask = rules::legal_mask(&board, player);
        let legal: Vec<usize> = (0..mask.len()).filter(|&i| mask[i]).collect();
        if legal.is_empty() {
            break;
        }
        let a = legal[rng.below(legal.len())];
        rules::apply(&mut board, Action(a as u16), player);
        player = 1 - player;
    }
}

// --- Full 643-vector hand check ---------------------------------------------

#[test]
fn full_token_vector_matches_hand_computation() {
    let cfg = EncoderConfig::default();
    let red = Arrangement::from_chars("KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD").unwrap();
    let blue = Arrangement::from_chars("KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD").unwrap();
    let board = arrangement::board_from_arrangements(&red, &blue);
    let tokens = encode_tokens(&board, 0, &cfg);
    let nfeat = cfg.num_token_features();
    let nchan = cfg.num_infostate_channels();

    // Token index for absolute cell 0 (red general 'K', piece_id 0): it is the
    // first non-lake POV cell, so token 0.
    let token0 = 0;
    let base = token0 * nfeat;
    // own one-hot: general = type 8 -> channel 8.
    assert_eq!(tokens[base + (PieceType::General as usize)], 1.0);
    // our_hidden_bool (ch 36): the piece is hidden and ours.
    assert_eq!(tokens[base + 36], 1.0);
    // empty_bool (ch 38) is 0.
    assert_eq!(tokens[base + 38], 0.0);
    // moved booleans (39,40) are 0 at game start.
    assert_eq!(tokens[base + 39], 0.0);
    assert_eq!(tokens[base + 40], 0.0);
    // max_num_moves_frac (ch 41) and attack frac (ch 42) are 0 at start.
    assert_eq!(tokens[base + 41], 0.0);
    assert_eq!(tokens[base + 42], 0.0);
    // history planes empty.
    for i in 0..cfg.history_len {
        assert_eq!(tokens[base + NUM_BOARD_STATE_CHANNELS + i], 0.0);
    }
    // piece-id one-hot: cell 0 is red general with id 0 -> bit 0.
    assert_eq!(tokens[base + nchan], 1.0);

    // A genuinely empty mid-board cell (cell 40) should be token where the only
    // signals are empty_bool=1, piece-id=255.
    let empty_token = (0..100usize)
        .filter(|c| !crate::board::LAKES.contains(c))
        .position(|c| c == 40)
        .unwrap();
    let eb = empty_token * nfeat;
    assert_eq!(tokens[eb + 38], 1.0, "empty_bool at empty cell");
    assert_eq!(tokens[eb + nchan + 255], 1.0, "empty piece-id 255");
    // No own/opp one-hot.
    for t in 0..12usize {
        assert_eq!(tokens[eb + t], 0.0);
    }
}

#[test]
fn batched_encoding_concatenates_single_encodings() {
    use crate::encode::encode_tokens_batch;
    let cfg = EncoderConfig::default();
    let mut rng = Rng::new(31);
    let b0 = fresh_play(&mut rng);
    let b1 = fresh_play(&mut rng);
    let single0 = encode_tokens(&b0, 0, &cfg);
    let single1 = encode_tokens(&b1, 1, &cfg);
    let batch = encode_tokens_batch(&[(&b0, 0), (&b1, 1)], &cfg);
    let stride = NUM_OCCUPIABLE_CELLS * cfg.num_token_features();
    assert_eq!(batch.len(), 2 * stride);
    assert_eq!(&batch[..stride], &single0[..]);
    assert_eq!(&batch[stride..], &single1[..]);
}
