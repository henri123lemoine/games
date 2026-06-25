//! Move/attack legality, battle resolution, and the termination/reward logic —
//! the move-phase rules engine.
//!
//! Ports `LegalActionsMaskKernel`, `ApplyActionsKernel`'s battle table and
//! capture bookkeeping, `SaturatedNumMovementDirectionsKernel`, and the
//! `IncrementTerminationCounterKernel` / `ComputeRewardPl0Kernel` machines.

use crate::action::{Action, NUM_ACTIONS};
use crate::board::{
    Board, Color, DeathReason, DeathStatus, HIDDEN_PIECE, MoveSummary, NO_ATTACK_DST_CODE, Piece,
    PieceType,
};

/// Final-run termination limits (`rl_main.py:50-52`).
pub const MAX_NUM_MOVES: u32 = 4000;
pub const MAX_NUM_MOVES_BETWEEN_ATTACKS: u32 = 100;

/// Resolves a battle: returns `true` when the *defender* (the piece being
/// attacked) wins and the attacker dies. Exactly the reference battle table
/// (`action_kernels.cu:333-337`): spy beats marshal only when attacking, miner
/// defuses a bomb, every other tie/over-rank follows numeric order, and a bomb
/// kills any non-miner.
pub fn defender_wins(from: PieceType, to: PieceType) -> bool {
    let (f, t) = (from as u8, to as u8);
    (t < PieceType::Flag as u8 && t > f && !(to == PieceType::Marshal && from == PieceType::Spy))
        || (to == PieceType::Bomb && from != PieceType::Miner)
}

/// Whether attacker and defender mutually destroy (equal rank).
pub fn is_tie(from: PieceType, to: PieceType) -> bool {
    from == to
}

/// Outcome of a resolved attack on an occupied enemy cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Battle {
    /// Attacker survives, defender removed (or flag captured).
    AttackerWins,
    /// Both removed.
    Tie,
    /// Defender survives, attacker removed.
    DefenderWins,
}

pub fn resolve(from: PieceType, to: PieceType) -> Battle {
    if defender_wins(from, to) {
        Battle::DefenderWins
    } else if is_tie(from, to) {
        Battle::Tie
    } else {
        Battle::AttackerWins
    }
}

/// Builds the raw legal-action mask for `player` from physical movement only
/// (scout slides, blocked by own pieces and lakes), before the chase/two-square
/// restrictions are subtracted. Mirrors `LegalActionsMaskKernel`.
pub fn raw_legal_mask(board: &Board, player: usize) -> Box<[bool; NUM_ACTIONS]> {
    let mut out = Box::new([false; NUM_ACTIONS]);
    let own = Color::of_player(player);
    for cell in 0..100usize {
        let piece = board.pieces[cell];
        if !piece.kind.is_movable() || piece.color != own {
            continue;
        }
        write_cell_moves(board, cell, player, own, &mut out);
    }
    out
}

fn occupied_by_own(board: &Board, cell: usize, own: Color) -> bool {
    board.pieces[cell].color == own
}

fn blocked(board: &Board, cell: usize, own: Color) -> bool {
    let c = board.pieces[cell].color;
    c == own || c == Color::Lake
}

fn write_cell_moves(
    board: &Board,
    cell: usize,
    player: usize,
    own: Color,
    out: &mut [bool; NUM_ACTIONS],
) {
    let row = cell / 10;
    let col = cell % 10;
    let is_scout = board.pieces[cell].kind.is_scout();

    // Absolute reach bounds along each direction: the farthest cell the piece
    // may stop on (scouts slide over empties; everyone is blocked by own/lake).
    let mut left = col as i32 - 1;
    let mut right = col as i32 + 1;
    let mut up = row as i32 + 1;
    let mut down = row as i32 - 1;

    while is_scout && left > 0 && board.pieces[row * 10 + left as usize].color == Color::Empty {
        left -= 1;
    }
    if left < 0 || blocked(board, row * 10 + left.max(0) as usize, own) {
        left += 1;
    }
    while is_scout && right < 9 && board.pieces[row * 10 + right as usize].color == Color::Empty {
        right += 1;
    }
    if right > 9 || blocked(board, row * 10 + right.min(9) as usize, own) {
        right -= 1;
    }
    while is_scout && up < 9 && board.pieces[up as usize * 10 + col].color == Color::Empty {
        up += 1;
    }
    if up > 9 || blocked(board, up.min(9) as usize * 10 + col, own) {
        up -= 1;
    }
    while is_scout && down > 0 && board.pieces[down as usize * 10 + col].color == Color::Empty {
        down -= 1;
    }
    if down < 0 || blocked(board, down.max(0) as usize * 10 + col, own) {
        down += 1;
    }

    let _ = occupied_by_own; // referenced for documentation symmetry

    // Convert absolute reach to POV reach, then to the 9-slot displacement
    // ranges, exactly as the kernel does for each (pov_row/col <= k) offset.
    let (mut left, mut right, mut up, mut down) = (left, right, up, down);
    if player == 1 {
        let (l, r, u, d) = (left, right, up, down);
        left = 9 - r;
        right = 9 - l;
        down = 9 - u;
        up = 9 - d;
    }
    let pov_cell = if player == 1 { 99 - cell } else { cell };
    let pov_row = (pov_cell / 10) as i32;
    let pov_col = (pov_cell % 10) as i32;

    for k in 0..9i32 {
        let off_v = i32::from(pov_row <= k);
        if down <= k + off_v && up >= k + off_v {
            out[(k as usize) * 100 + pov_cell] = true;
        }
        let off_h = i32::from(pov_col <= k);
        if left <= k + off_h && right >= k + off_h {
            out[(900 + k as usize * 100) + pov_cell] = true;
        }
    }
}

/// The legal-action mask after applying the active player's two-square and
/// continuous-chase restrictions.
pub fn legal_mask(board: &Board, player: usize) -> Box<[bool; NUM_ACTIONS]> {
    let mut mask = raw_legal_mask(board, player);
    board.twosquare[player].remove_actions(&mut mask);
    remove_chase_moves(board, player, &mut mask);
    mask
}

/// Subtracts chase-rule violations from `mask` for the active player. For each
/// currently-legal non-attack move, asks the oracle whether it reproduces an
/// earlier threatening position.
fn remove_chase_moves(board: &Board, player: usize, mask: &mut [bool; NUM_ACTIONS]) {
    let oracle = &board.chase_oracle[player];
    for (idx, legal) in mask.iter_mut().enumerate() {
        if !*legal {
            continue;
        }
        let (src, dst) = Action(idx as u16).to_abs(player);
        if board.pieces[dst].color != Color::Empty {
            continue; // attacks reset the chase; never illegal on this account
        }
        if oracle.would_violate(src, dst) {
            *legal = false;
        }
    }
}

/// The collected effects of applying one move, surfaced for the [`Game`] layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Applied {
    pub summary: MoveSummary,
    pub was_attack: bool,
    /// Whether this move captured the opponent flag.
    pub flag_captured: bool,
    /// Absolute death cell for each color, or `0xff`.
    pub red_death: u8,
    pub blue_death: u8,
}

/// Applies `action` for `player` to `board`, resolving any battle and updating
/// all counters, visibility, death bookkeeping, and the restriction machines.
/// Mirrors `ApplyActionsKernel` (rules-relevant parts; the threat/protection
/// bitset geometry is deferred — see `TODO(m2)`).
pub fn apply(board: &mut Board, action: Action, player: usize) -> Applied {
    board.num_moves += 1;
    board.num_moves_since_last_attack += 1;

    let (from_abs, to_abs) = action.to_abs(player);
    let from_pov = if player == 1 { 99 - from_abs } else { from_abs };
    let to_pov = if player == 1 { 99 - to_abs } else { to_abs };

    let mut from_piece = board.pieces[from_abs];
    let mut to_piece = board.pieces[to_abs];
    let to_was_visible = to_piece.visible;

    let summary = MoveSummary {
        src_rel: from_pov as u8,
        dst_rel: to_pov as u8,
        src_code: encode_piece(
            if to_piece.kind == PieceType::Empty && !from_piece.visible {
                HIDDEN_PIECE
            } else {
                from_piece.kind as u8
            },
            from_piece.visible,
            from_piece.has_moved,
        ),
        dst_code: encode_piece(to_piece.kind as u8, to_piece.visible, to_piece.has_moved),
        src_id: from_piece.piece_id,
        dst_id: to_piece.piece_id,
    };
    let was_attack = summary.dst_code != NO_ATTACK_DST_CODE;

    if !from_piece.has_moved && !from_piece.visible {
        board.num_hidden_unmoved[player] -= 1;
    }
    from_piece.has_moved = true;

    let is_attack_target = to_piece.color != Color::Empty && to_piece.color != Color::Lake;
    if is_attack_target {
        board.num_moves_since_last_attack = 0;
        if !from_piece.visible {
            board.num_hidden[player][from_piece.kind as usize] -= 1;
            from_piece.visible = true;
        }
        if !to_piece.visible {
            let tp = to_piece.color.player().unwrap();
            board.num_hidden[tp][to_piece.kind as usize] -= 1;
            to_piece.visible = true;
            if !to_piece.has_moved {
                board.num_hidden_unmoved[tp] -= 1;
            }
        }
    }

    let step_len = (to_abs as i32 / 10 - from_abs as i32 / 10).abs()
        + (to_abs as i32 % 10 - from_abs as i32 % 10).abs();
    if step_len >= 2 && !from_piece.visible {
        board.num_hidden[player][from_piece.kind as usize] -= 1;
        from_piece.visible = true;
    }

    board.pieces[from_abs] = Piece::EMPTY;

    let outcome = if is_attack_target {
        resolve(from_piece.kind, to_piece.kind)
    } else {
        Battle::AttackerWins
    };

    let to_row = (to_abs / 10) as u8;
    let to_col = (to_abs % 10) as u8;
    let mut flag_captured = false;

    let dest_after: Piece = match outcome {
        Battle::DefenderWins => {
            board.mark_dead(player, from_piece.piece_id);
            record_death(
                board,
                player,
                from_piece.piece_id,
                if to_was_visible {
                    DeathReason::AttackedVisibleStronger
                } else {
                    DeathReason::AttackedHidden
                },
                from_piece.kind,
                to_abs,
            );
            to_piece
        }
        Battle::Tie => {
            let tp = to_piece.color.player().unwrap();
            board.mark_dead(player, from_piece.piece_id);
            board.mark_dead(tp, to_piece.piece_id);
            record_death(
                board,
                player,
                from_piece.piece_id,
                if to_was_visible {
                    DeathReason::AttackedVisibleTie
                } else {
                    DeathReason::AttackedHidden
                },
                from_piece.kind,
                to_abs,
            );
            record_death(
                board,
                tp,
                to_piece.piece_id,
                if to_was_visible {
                    DeathReason::VisibleDefendedTie
                } else {
                    DeathReason::HiddenDefended
                },
                to_piece.kind,
                to_abs,
            );
            Piece::EMPTY
        }
        Battle::AttackerWins => {
            if is_attack_target {
                if to_piece.kind == PieceType::Flag {
                    flag_captured = true;
                } else {
                    let tp = to_piece.color.player().unwrap();
                    board.mark_dead(tp, to_piece.piece_id);
                    record_death(
                        board,
                        tp,
                        to_piece.piece_id,
                        if to_was_visible {
                            DeathReason::VisibleDefendedWeaker
                        } else {
                            DeathReason::HiddenDefended
                        },
                        to_piece.kind,
                        to_abs,
                    );
                }
            }
            from_piece
        }
    };

    board.pieces[to_abs] = dest_after;

    let red_death = death_cell(Color::Red, player, &to_piece, &dest_after, to_abs);
    let blue_death = death_cell(Color::Blue, player, &to_piece, &dest_after, to_abs);

    // Restriction machines: two-square (per active player), chase (both
    // oracles), and the rolling prev-dst trail.
    board.twosquare[player].update_move(from_pov as u8, to_pov as u8);
    if red_death != 0xff {
        let pov = red_death as usize; // red POV == absolute
        board.twosquare[0].update_death(pov as u8);
    }
    if blue_death != 0xff {
        board.twosquare[1].update_death(blue_death);
    }

    for p in 0..2 {
        board.chase_oracle[p].update(from_abs, to_abs, was_attack, p != player);
    }

    board.last_moved_piece_type = match outcome {
        Battle::DefenderWins | Battle::Tie => 0xff,
        Battle::AttackerWins if from_piece.visible => from_piece.kind as u8,
        Battle::AttackerWins => HIDDEN_PIECE,
    };
    board.prev_prev_dst = board.prev_dst;
    board.prev_dst = to_abs as u8;
    let _ = (to_row, to_col);

    Applied {
        summary,
        was_attack,
        flag_captured,
        red_death,
        blue_death,
    }
}

#[inline]
fn encode_piece(type_index: u8, visible: bool, has_moved: bool) -> u8 {
    type_index + ((visible as u8) << 4) + ((has_moved as u8) << 5)
}

fn record_death(
    board: &mut Board,
    player: usize,
    piece_id: u8,
    reason: DeathReason,
    kind: PieceType,
    location: usize,
) {
    board.death_status[player][piece_id as usize] = DeathStatus {
        is_dead: true,
        reason: reason as u8,
        piece_type: kind as u8,
        death_location: location as u8,
    };
}

/// Computes the per-color absolute death cell reported to the two-square death
/// reset (`ApplyActionsKernel:704-705`). Red is reported in absolute coords,
/// blue point-reflected.
fn death_cell(
    color: Color,
    player: usize,
    to_piece: &Piece,
    dest_after: &Piece,
    to_abs: usize,
) -> u8 {
    let c = color as u8;
    let to_color = to_piece.color as u8;
    let acting_color = (player + 1) as u8;
    let dest_color = dest_after.color as u8;
    if (to_color == c || acting_color == c) && dest_color != c {
        if color == Color::Red {
            to_abs as u8
        } else {
            (99 - to_abs) as u8
        }
    } else {
        0xff
    }
}

/// Saturated count (0/1/2) of free orthogonal directions for `player`'s pieces,
/// summed over the board: the physical-mobility proxy used in the off-turn
/// stuck check (`SaturatedNumMovementDirectionsKernel`).
pub fn saturated_movement(board: &Board, player: usize) -> u32 {
    let own = Color::of_player(player);
    let mut total = 0u32;
    for cell in 0..100usize {
        let piece = board.pieces[cell];
        if !piece.kind.is_movable() || piece.color != own {
            continue;
        }
        let row = cell / 10;
        let col = cell % 10;
        let free = |r: usize, c: usize| board.pieces[r * 10 + c].color != own;
        let mut n = 0u32;
        if row > 0 && free(row - 1, col) {
            n += 1;
        }
        if row < 9 && free(row + 1, col) {
            n += 1;
        }
        if col > 0 && free(row, col - 1) {
            n += 1;
        }
        if col < 9 && free(row, col + 1) {
            n += 1;
        }
        total += n.min(2);
    }
    total
}

/// Whether `player` (the one about to act) has any legal move, applying the full
/// chase + two-square restrictions for that player.
pub fn has_any_legal(board: &Board, player: usize) -> bool {
    legal_mask(board, player).iter().any(|&b| b)
}

/// The `has_legal_movement` code (`UpdateHasLegalMovement_`): bit 1 set when red
/// can move, bit 2 when blue can. The *to-play* player gets the full restricted
/// check; the off-turn player a relaxed physical-mobility check (chase is not
/// applied off-turn), minus any two-square-precluded direction.
pub fn has_legal_movement(board: &Board, to_play: usize) -> u8 {
    let red = player_can_move(board, 0, to_play);
    let blue = player_can_move(board, 1, to_play);
    (red as u8) | ((blue as u8) << 1)
}

fn player_can_move(board: &Board, player: usize, to_play: usize) -> bool {
    if player == to_play {
        return has_any_legal(board, player);
    }
    let mut mobility = saturated_movement(board, player);
    if board.twosquare[player].is_precluding_direction() {
        mobility = mobility.saturating_sub(1);
    }
    mobility >= 1
}

/// Whether the position is terminal for a state about to be acted on by
/// `to_play`, given the supplied `flag_captured` player index (or `None`).
/// Mirrors `IncrementTerminationCounterKernel`'s predicate.
pub fn is_terminal(board: &Board, to_play: usize, flag_captured: Option<usize>) -> bool {
    has_legal_movement(board, to_play) < 3
        || flag_captured.is_some()
        || board.num_moves > MAX_NUM_MOVES
        || board.num_moves_since_last_attack > MAX_NUM_MOVES_BETWEEN_ATTACKS
}

/// Reward to player 0 at a terminal state (`ComputeRewardPl0Kernel`): +1 / -1
/// flag capture, +1 / -1 / 0 for stuck players, 0 on either timeout.
pub fn reward_pl0(board: &Board, to_play: usize, flag_captured: Option<usize>) -> f64 {
    let not_timeout = board.num_moves <= MAX_NUM_MOVES
        && board.num_moves_since_last_attack <= MAX_NUM_MOVES_BETWEEN_ATTACKS;

    if not_timeout && let Some(p) = flag_captured {
        // p captured the flag: red (0) capturing -> +1, blue (1) -> -1.
        return if p == 0 { 1.0 } else { -1.0 };
    }
    if not_timeout {
        let x = has_legal_movement(board, to_play);
        // x==0 both stuck -> 0; x==1 only red -> +1; x==2 only blue -> -1.
        return match x {
            0 => 0.0,
            1 => 1.0,
            _ => -1.0,
        };
    }
    0.0
}
