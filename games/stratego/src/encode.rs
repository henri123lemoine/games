//! The byte-exact input encoder: the 355 board-state planes, the 32 src/dst
//! move-history planes, and the 256-way piece-id one-hot that together form the
//! 643-feature-per-token move-net input.
//!
//! Every channel offset, POV convention, and arithmetic detail is ported from
//! the reference CUDA kernels (`infostate_kernels.cu`, orchestrated in
//! `stratego.cu:1184-1336`) and verified against the repo plane-test oracles in
//! `tests.rs`. See `ENCODING_SPEC.md` for the authoritative map.
//!
//! Memory conventions (matching `FeatureOrchestrator`,
//! `feature_orchestration.py`):
//! * The board/history tensor is channel-major over POV cells:
//!   `plane[channel * 100 + pov_cell]`, the whole tensor zero-filled first.
//! * `pov_cell = 99 - cell` for player 2 (the 180-degree point reflection); the
//!   posterior-over-me planes apply an extra `^ rotate`.
//! * The per-token layout drops the eight lake cells (symmetric under the
//!   reflection) and concatenates, per surviving cell, the 355 board planes, the
//!   `history_len` history planes, then the 256 piece-id one-hot — giving the
//!   `(92, 643)` matrix the transformer consumes.

use crate::board::{Board, HIDDEN_PIECE, LAKES, NUM_CELLS, PieceType, bitset_get};

/// Number of board-state planes (channels `0..=354`).
pub const NUM_BOARD_STATE_CHANNELS: usize = 355;
/// Move-net history length (`plane_history_len`, `train.log`).
pub const MOVE_NET_HISTORY_LEN: usize = 32;
/// Piece-id one-hot width (`N_PIECE_ID`, `constants.py:40`).
pub const NUM_PIECE_ID: usize = 256;
/// Empty-cell piece-id (`EMPTY_PIECE_ID`, `constants.py`).
pub const EMPTY_PIECE_ID: usize = 255;
/// Occupiable (non-lake) cells; the token count for the move net.
pub const NUM_OCCUPIABLE_CELLS: usize = NUM_CELLS - LAKES.len();

/// The per-square configuration the encoder needs beyond the board itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncoderConfig {
    /// How many src/dst history planes to emit (`plane_history_len`). The move
    /// net uses 32; the belief variant uses 86. The board planes are unchanged.
    pub history_len: usize,
    /// Denominator of the `max_num_moves_frac` constant plane (channel 41).
    pub max_num_moves: u32,
    /// Denominator of the `max_num_moves_between_attacks_frac` plane (channel 42).
    pub max_num_moves_between_attacks: u32,
}

impl Default for EncoderConfig {
    /// The final-run move-net configuration: 32 history planes, `max_num_moves =
    /// 4000`, `max_num_moves_between_attacks = 100`. The constant-plane
    /// denominators must match the values used during training: `rl_main.py:51-52`
    /// (the RL entry that produced `pretrained/final_run`) overrides the
    /// `StrategoConf` default of 200 with 100, matching `MAX_NUM_MOVES_BETWEEN_ATTACKS`
    /// in the M1 termination logic.
    fn default() -> Self {
        EncoderConfig {
            history_len: MOVE_NET_HISTORY_LEN,
            max_num_moves: crate::rules::MAX_NUM_MOVES,
            max_num_moves_between_attacks: crate::rules::MAX_NUM_MOVES_BETWEEN_ATTACKS,
        }
    }
}

impl EncoderConfig {
    /// Total board + history channels (`355 + history_len`); the raw infostate
    /// channel count.
    pub fn num_infostate_channels(&self) -> usize {
        NUM_BOARD_STATE_CHANNELS + self.history_len
    }

    /// Per-token feature width (`num_infostate_channels + 256`).
    pub fn num_token_features(&self) -> usize {
        self.num_infostate_channels() + NUM_PIECE_ID
    }
}

/// The point-reflection POV cell for `for_player` (1 or 2, reference numbering).
#[inline]
fn pov(cell: usize, for_player: u8) -> usize {
    if for_player == 2 { 99 - cell } else { cell }
}

/// Reference color value for player index `p` (0 -> red(1), 1 -> blue(2)).
#[inline]
fn color_val(p: usize) -> u8 {
    (p + 1) as u8
}

/// Writes the 355 board-state planes plus `cfg.history_len` history planes into
/// a freshly-zeroed channel-major tensor (`out[channel * 100 + pov_cell]`), seen
/// from `to_play`'s perspective. Returns a `Vec<f32>` of length
/// `num_infostate_channels * 100`.
pub fn encode_infostate(board: &Board, to_play: usize, cfg: &EncoderConfig) -> Vec<f32> {
    let n = cfg.num_infostate_channels();
    let mut out = vec![0.0f32; n * 100];
    write_board_state(board, to_play, cfg, &mut out);
    write_history(board, to_play, cfg, &mut out);
    out
}

/// Builds the per-token `(NUM_OCCUPIABLE_CELLS, num_token_features)` matrix the
/// move net consumes, row-major (`token * num_token_features + feature`).
/// Token order is the POV cell index with the eight lakes removed.
pub fn encode_tokens(board: &Board, to_play: usize, cfg: &EncoderConfig) -> Vec<f32> {
    let infostate = encode_infostate(board, to_play, cfg);
    let piece_ids = piece_id_grid(board, to_play);

    let n_chan = cfg.num_infostate_channels();
    let n_feat = cfg.num_token_features();
    let mut out = vec![0.0f32; NUM_OCCUPIABLE_CELLS * n_feat];

    let mut token = 0usize;
    for pov_cell in 0..NUM_CELLS {
        if LAKES.contains(&pov_cell) {
            continue;
        }
        let base = token * n_feat;
        for ch in 0..n_chan {
            out[base + ch] = infostate[ch * 100 + pov_cell];
        }
        let pid = piece_ids[pov_cell];
        out[base + n_chan + pid] = 1.0;
        token += 1;
    }
    debug_assert_eq!(token, NUM_OCCUPIABLE_CELLS);
    out
}

/// Batched token encoding: stacks `encode_tokens` for each `(board, to_play)`
/// into a flat `(batch, NUM_OCCUPIABLE_CELLS, num_token_features)` buffer,
/// row-major (`((b * 92) + token) * num_token_features + feature`).
pub fn encode_tokens_batch(states: &[(&Board, usize)], cfg: &EncoderConfig) -> Vec<f32> {
    let stride = NUM_OCCUPIABLE_CELLS * cfg.num_token_features();
    let mut out = vec![0.0f32; states.len() * stride];
    for (b, &(board, to_play)) in states.iter().enumerate() {
        let one = encode_tokens(board, to_play, cfg);
        out[b * stride..(b + 1) * stride].copy_from_slice(&one);
    }
    out
}

/// The per-(POV)-cell piece-id field used by the one-hot. Each piece is numbered
/// by its absolute starting cell as seen by the observer: own pieces keep their
/// starting slot `[0, 39]`, opponent pieces map to `99 - slot` (`[60, 99]`),
/// empty/lake cells are `255` (`test_piece_id.py:106-157`).
pub fn piece_id_grid(board: &Board, to_play: usize) -> [usize; NUM_CELLS] {
    let for_player = color_val(to_play);
    let mut ids = [EMPTY_PIECE_ID; NUM_CELLS];
    for cell in 0..NUM_CELLS {
        let p = board.pieces[cell];
        let pc = p.color as u8;
        if pc != 1 && pc != 2 {
            continue; // empty or lake
        }
        let pov_cell = pov(cell, for_player);
        ids[pov_cell] = if pc == for_player {
            p.piece_id as usize
        } else {
            99 - p.piece_id as usize
        };
    }
    ids
}

pub const DEPLOY_TYPE_WIDTH: usize = 14;

/// The setup-net feature vector: a `HOME_CELLS * DEPLOY_TYPE_WIDTH` one-hot of
/// the placed-so-far piece types, unfilled slots all-zero.
pub fn deploy_obs(current: &crate::arrangement::DeploymentState) -> Vec<f32> {
    let mut obs = vec![0.0f32; crate::board::HOME_CELLS * DEPLOY_TYPE_WIDTH];
    for (slot, &kind) in current.placed.iter().enumerate() {
        obs[slot * DEPLOY_TYPE_WIDTH + kind as usize] = 1.0;
    }
    obs
}

fn write_board_state(board: &Board, to_play: usize, cfg: &EncoderConfig, out: &mut [f32]) {
    let for_player = color_val(to_play);
    own_piece_types(board, for_player, out);
    // 12..23: opponent posterior in my POV.
    prob_types(board, for_player, false, 1200, out);
    // 24..35: opponent's posterior over my pieces, rotated back.
    prob_types(board, 3 - for_player, true, 2400, out);
    basics(board, for_player, cfg, out);
    threat_evade_actadj(board, for_player, out);
    cemetery(board, for_player, out);
    death_reasons(board, for_player, out);
    protections(board, for_player, out);
}

/// Channels 0..=11: one-hot own piece types (`BoardStateKernel__OwnPieceTypes`).
fn own_piece_types(board: &Board, for_player: u8, out: &mut [f32]) {
    for cell in 0..NUM_CELLS {
        let p = board.pieces[cell];
        if (p.kind as u8) >= PieceType::Lake as u8 || p.color as u8 != for_player {
            continue;
        }
        let pc = pov(cell, for_player);
        out[100 * (p.kind as usize) + pc] = 1.0;
    }
}

/// Channels 12..=23 / 24..=35: the analytic "if-uniform-random" opponent
/// posterior (`BoardStateKernel__ProbTypes`). `for_player` is whose pieces are
/// considered opponent-relative; `rotate` flips the POV again for the
/// posterior-over-me planes; `shift` is the channel-cell offset (1200 or 2400).
fn prob_types(board: &Board, for_player: u8, rotate: bool, shift: usize, out: &mut [f32]) {
    let counts_owner = (2 - for_player) as usize; // num_hidden index = 2 - for_player
    let num_hidden = &board.num_hidden[counts_owner];
    let num_hidden_unmoved = board.num_hidden_unmoved[counts_owner] as f32;
    let total: u32 = num_hidden.iter().map(|&x| x as u32).sum();

    let flag = PieceType::Flag as usize;
    let bomb = PieceType::Bomb as usize;
    let nf = num_hidden[flag] as f32;
    let nb = num_hidden[bomb] as f32;
    let denom = total as f32 - nf - nb;

    let reflect = (for_player == 2) ^ rotate;
    for cell in 0..NUM_CELLS {
        let p = board.pieces[cell];
        if (p.kind as u8) >= PieceType::Lake as u8 || p.color as u8 == for_player {
            continue;
        }
        let pc = if reflect { 99 - cell } else { cell };
        let base = shift + pc;
        if p.visible {
            out[100 * (p.kind as usize) + base] = 1.0;
            continue;
        }
        if p.has_moved {
            // Hidden and has moved: cannot be flag/bomb, denom > 0.
            for t in 0..10usize {
                out[100 * t + base] = num_hidden[t] as f32 / denom;
            }
        } else {
            // Hidden, never moved: bombs/flags weight toward immobility.
            if total != num_hidden[flag] as u32 + num_hidden[bomb] as u32 {
                let norm = (num_hidden_unmoved - nf - nb) / (num_hidden_unmoved * denom);
                for t in 0..10usize {
                    out[100 * t + base] = num_hidden[t] as f32 * norm;
                }
            }
            out[100 * flag + base] = nf / num_hidden_unmoved;
            out[100 * bomb + base] = nb / num_hidden_unmoved;
        }
    }
}

/// Channels 36..=42: hidden/empty/moved booleans and the two constant fraction
/// planes (`BoardStateKernel__InvisiblesEmptyAndMoved`).
fn basics(board: &Board, for_player: u8, cfg: &EncoderConfig, out: &mut [f32]) {
    let moves_frac = board.num_moves as f32 / cfg.max_num_moves as f32;
    let attack_frac =
        board.num_moves_since_last_attack as f32 / cfg.max_num_moves_between_attacks as f32;
    for cell in 0..NUM_CELLS {
        let p = board.pieces[cell];
        let pc = pov(cell, for_player);
        let is_player = p.color as u8 == for_player;
        out[3600 + pc] = f32::from(!p.visible && is_player);
        out[3700 + pc] = f32::from(!p.visible && !is_player);
        out[3800 + pc] = f32::from(p.kind == PieceType::Empty);
        out[3900 + pc] = f32::from(p.has_moved && is_player);
        out[4000 + pc] = f32::from(p.has_moved && !is_player);
        out[4100 + pc] = moves_frac;
        out[4200 + pc] = attack_frac;
    }
}

const TEA_TYPES: [u8; 11] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, HIDDEN_PIECE];

/// Channels 43..=108: threat / evade / active-adjacency bitsets, reported only
/// for hidden pieces (`BoardStateKernel__ThreatEvadeActiveAdj`).
fn threat_evade_actadj(board: &Board, for_player: u8, out: &mut [f32]) {
    for cell in 0..NUM_CELLS {
        let p = board.pieces[cell];
        let pc = pov(cell, for_player);
        let we_hidden = p.color as u8 == for_player && !p.visible;
        let they_hidden = p.color as u8 == 3 - for_player && !p.visible;
        for (i, &t) in TEA_TYPES.iter().enumerate() {
            if we_hidden {
                out[(43 + i) * 100 + pc] = f32::from(bitset_get(p.threatened, t));
                out[(54 + i) * 100 + pc] = f32::from(bitset_get(p.evaded, t));
                out[(65 + i) * 100 + pc] = f32::from(bitset_get(p.actively_adjacent, t));
            }
            if they_hidden {
                out[(76 + i) * 100 + pc] = f32::from(bitset_get(p.threatened, t));
                out[(87 + i) * 100 + pc] = f32::from(bitset_get(p.evaded, t));
                out[(98 + i) * 100 + pc] = f32::from(bitset_get(p.actively_adjacent, t));
            }
        }
    }
}

const CEMETERY_TYPES: [u8; 11] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, PieceType::Bomb as u8];

/// Channels 109..=130: cemetery one-hot, at each dead piece's starting square,
/// typed from the initial arrangement (`BoardStateKernel__Deaths`). Home rows
/// only (`[40, 60)` skipped).
fn cemetery(board: &Board, for_player: u8, out: &mut [f32]) {
    for cell in 0..NUM_CELLS {
        if (40..60).contains(&cell) {
            continue;
        }
        let rel = if cell < 40 { cell } else { 99 - cell };
        let side = usize::from(cell >= 60); // 0 = red home, 1 = blue home
        let is_dead = board.is_dead(side, rel as u8);
        if !is_dead {
            continue;
        }
        let zero_type = board.zero_types[side][rel];
        let pc = pov(cell, for_player);
        let dead_color = color_val(side);
        for (i, &t) in CEMETERY_TYPES.iter().enumerate() {
            if zero_type != t {
                continue;
            }
            if dead_color == for_player {
                out[(109 + i) * 100 + pc] = 1.0;
            } else {
                out[(120 + i) * 100 + pc] = 1.0;
            }
        }
    }
}

/// Channels 131..=250: the six death-reason planes per side, at the death
/// location (`BoardStateKernel__DeathReasons`).
fn death_reasons(board: &Board, for_player: u8, out: &mut [f32]) {
    // our side = death_status[for_player - 1]; their side = death_status[2 - for_player].
    let our_side = (for_player - 1) as usize;
    let their_side = (2 - for_player) as usize;
    for slot in 0..40usize {
        let ds = board.death_status[our_side][slot];
        if ds.is_dead && ds.piece_type <= PieceType::Marshal as u8 {
            let pc = pov(ds.death_location as usize, for_player);
            let ch = 131 + ds.reason as usize * 10 + ds.piece_type as usize;
            out[ch * 100 + pc] = 1.0;
        }
        let ds = board.death_status[their_side][slot];
        if ds.is_dead && ds.piece_type <= PieceType::Marshal as u8 {
            let pc = pov(ds.death_location as usize, for_player);
            let ch = 191 + ds.reason as usize * 10 + ds.piece_type as usize;
            out[ch * 100 + pc] = 1.0;
        }
    }
}

const PROTECT_TYPES: [u8; 13] = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    PieceType::Bomb as u8,
    PieceType::Empty as u8,
    HIDDEN_PIECE,
];

/// Channels 251..=354: the eight protection bitset groups, reported only for
/// hidden pieces (`BoardStateKernel__Protections`).
fn protections(board: &Board, for_player: u8, out: &mut [f32]) {
    for cell in 0..NUM_CELLS {
        let p = board.pieces[cell];
        let pc = pov(cell, for_player);
        let we_hidden = p.color as u8 == for_player && !p.visible;
        let they_hidden = p.color as u8 == 3 - for_player && !p.visible;
        for (i, &t) in PROTECT_TYPES.iter().enumerate() {
            if we_hidden {
                out[(251 + i) * 100 + pc] = f32::from(bitset_get(p.protected_, t));
                out[(264 + i) * 100 + pc] = f32::from(bitset_get(p.protected_against, t));
                out[(277 + i) * 100 + pc] = f32::from(bitset_get(p.was_protected_by, t));
                out[(290 + i) * 100 + pc] = f32::from(bitset_get(p.was_protected_against, t));
            }
            if they_hidden {
                out[(303 + i) * 100 + pc] = f32::from(bitset_get(p.protected_, t));
                out[(316 + i) * 100 + pc] = f32::from(bitset_get(p.protected_against, t));
                out[(329 + i) * 100 + pc] = f32::from(bitset_get(p.was_protected_by, t));
                out[(342 + i) * 100 + pc] = f32::from(bitset_get(p.was_protected_against, t));
            }
        }
    }
}

/// Channels `355..355+history_len`: the src/dst move-history planes
/// (`InjectInfostateSrcDstKernel`). Plane index `i` corresponds to
/// `delta = history_len - i` moves ago; the most recent move lands in the last
/// plane. Each rendered move marks `src = -1`, `dst = +1` in the observer's POV,
/// applying the per-move parity flip.
fn write_history(board: &Board, _to_play: usize, cfg: &EncoderConfig, out: &mut [f32]) {
    let history_len = cfg.history_len;
    let num_moves = board.action_history.len();
    for i in 0..history_len {
        let delta = history_len - i; // in [1, history_len]
        if delta > num_moves {
            continue; // not enough history (also covers the episode-boundary guard)
        }
        let action = board.action_history[num_moves - delta] as usize;
        let from_cell = action % 100;
        let from_row = (from_cell / 10) as i32;
        let from_col = (from_cell % 10) as i32;
        let direction = action >= 900; // false = vertical, true = horizontal
        let new_coord = ((action / 100) % 9) as i32;

        let (to_row, to_col) = if !direction {
            (new_coord + i32::from(new_coord >= from_row), from_col)
        } else {
            (from_row, new_coord + i32::from(new_coord >= from_col))
        };

        let requires_flip = delta % 2 == 1;
        let (fr, fc, tr, tc) = if requires_flip {
            (9 - from_row, 9 - from_col, 9 - to_row, 9 - to_col)
        } else {
            (from_row, from_col, to_row, to_col)
        };

        let base = (NUM_BOARD_STATE_CHANNELS + i) * 100;
        out[base + (10 * fr + fc) as usize] = -1.0;
        out[base + (10 * tr + tc) as usize] = 1.0;
    }
}
