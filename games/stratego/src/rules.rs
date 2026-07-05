//! Move/attack legality, battle resolution, and the termination/reward logic —
//! the move-phase rules engine.
//!
//! Ports `LegalActionsMaskKernel`, `ApplyActionsKernel`'s battle table and
//! capture bookkeeping, `SaturatedNumMovementDirectionsKernel`, and the
//! `IncrementTerminationCounterKernel` / `ComputeRewardPl0Kernel` machines.

use crate::action::{Action, NUM_ACTIONS};
use crate::board::{
    Board, Color, DeathReason, DeathStatus, HIDDEN_PIECE, MoveSummary, NO_ATTACK_DST_CODE, Piece,
    PieceType, bitset_set, is_adjacent,
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
/// all counters, visibility, death bookkeeping, the restriction machines, and
/// the threat/evade/active-adjacency/protection bitsets that feed the encoder.
/// A faithful port of `ApplyActionsKernel` (`action_kernels.cu`).
pub fn apply(board: &mut Board, action: Action, player: usize) -> Applied {
    board.num_moves += 1;
    board.num_moves_since_last_attack += 1;
    board.action_history.push(action.0);

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

    // The trackers below read the *previous* move's trail, captured before this
    // move overwrites them.
    let last_moved = board.last_moved_piece_type;
    let prev_dst = board.prev_dst;
    let prev_prev_dst = board.prev_prev_dst;

    update_actively_adjacent(board, player, last_moved, prev_dst, prev_prev_dst);

    // The act-adj pass may have flipped bits on the from/to cells; re-read.
    from_piece = board.pieces[from_abs];
    to_piece = board.pieces[to_abs];

    if last_moved != 0xff
        && to_abs as u8 != prev_dst
        && prev_dst != 0xff
        && is_adjacent(from_abs, prev_dst as usize)
    {
        bitset_set(&mut from_piece.evaded, last_moved);
    }

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
    let to_wins = outcome == Battle::DefenderWins;
    let tie = outcome == Battle::Tie;

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
            } else {
                // Non-attack slide onto an empty square: the moving piece now
                // threatens every adjacent opponent. The reference writes this
                // into `from_piece` (the about-to-land piece) before it is
                // committed to the destination cell.
                update_threatened(board, player, &mut from_piece, to_abs);
            }
            from_piece
        }
    };

    board.pieces[to_abs] = dest_after;

    update_protections(board, player, from_abs, to_abs, to_wins, tie);

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

    Applied {
        summary,
        was_attack,
        flag_captured,
        red_death,
        blue_death,
    }
}

/// Orthogonal neighbours of an absolute cell, as `(cell, exists)` would be — we
/// just yield the in-bounds ones.
#[inline]
fn neighbours(cell: usize) -> impl Iterator<Item = usize> {
    let (r, c) = (cell / 10, cell % 10);
    let mut out = [usize::MAX; 4];
    let mut n = 0;
    if r > 0 {
        out[n] = cell - 10;
        n += 1;
    }
    if r < 9 {
        out[n] = cell + 10;
        n += 1;
    }
    if c > 0 {
        out[n] = cell - 1;
        n += 1;
    }
    if c < 9 {
        out[n] = cell + 1;
        n += 1;
    }
    out.into_iter().take(n)
}

/// `actively_adjacent` update (`action_kernels.cu:169-254`), run before the move
/// is applied. Marks pieces adjacent to the previous and previous-previous
/// destinations as having been close to the relevant piece type during the
/// turn.
fn update_actively_adjacent(
    board: &mut Board,
    player: usize,
    last_moved: u8,
    prev_dst: u8,
    prev_prev_dst: u8,
) {
    let own = Color::of_player(player);
    let opp = Color::of_player(1 - player);

    if last_moved != 0xff && prev_dst != 0xff {
        for nb in neighbours(prev_dst as usize) {
            if board.pieces[nb].color == own {
                bitset_set(&mut board.pieces[nb].actively_adjacent, last_moved);
            }
        }
    }

    if prev_prev_dst != 0xff {
        let here = prev_prev_dst as usize;
        let center = board.pieces[here];
        if center.color == own {
            // Our piece survived at the previous-previous destination; record the
            // opponent types now adjacent to it.
            for nb in neighbours(here) {
                let p = board.pieces[nb];
                if p.color == opp {
                    let pt = p.tracked_type();
                    bitset_set(&mut board.pieces[here].actively_adjacent, pt);
                }
            }
        } else if center.color == opp && center.visible {
            // A now-revealed opponent sits there; our neighbours were actively
            // adjacent to it.
            let pt = center.kind as u8;
            for nb in neighbours(here) {
                if board.pieces[nb].color == own {
                    bitset_set(&mut board.pieces[nb].actively_adjacent, pt);
                }
            }
        }
    }
}

/// `threatened` update for a non-attacking slide (`action_kernels.cu:420-442`):
/// the moving piece threatens every opponent adjacent to its destination.
fn update_threatened(board: &Board, player: usize, from_piece: &mut Piece, to_abs: usize) {
    let opp = Color::of_player(1 - player);
    for nb in neighbours(to_abs) {
        let p = board.pieces[nb];
        if p.color == opp {
            bitset_set(&mut from_piece.threatened, p.tracked_type());
        }
    }
}

/// One application of the `UPDATE_PROTECT` macro (`action_kernels.cu:448-468`):
/// if `aggressor` is an enemy, `protectee` is ours-or-empty, and `protector` is
/// ours, record the four-way protection relationship by piece type.
fn update_protect(
    board: &mut Board,
    player: usize,
    protector: usize,
    protectee: usize,
    aggressor: usize,
) {
    let own = Color::of_player(player);
    let opp = Color::of_player(1 - player);
    let ag = board.pieces[aggressor];
    let pe = board.pieces[protectee];
    let pr = board.pieces[protector];
    if ag.color == opp && (pe.color == own || pe.color == Color::Empty) && pr.color == own {
        let protector_pt = pr.tracked_type();
        let protectee_pt = pe.tracked_type();
        let aggressor_pt = ag.tracked_type();
        bitset_set(&mut board.pieces[protector].protected_, protectee_pt);
        bitset_set(&mut board.pieces[protector].protected_against, aggressor_pt);
        bitset_set(&mut board.pieces[protectee].was_protected_by, protector_pt);
        bitset_set(
            &mut board.pieces[protectee].was_protected_against,
            aggressor_pt,
        );
    }
}

/// The five-case protection geometry (`action_kernels.cu:469-696`), run after
/// the destination cell has been committed.
fn update_protections(
    board: &mut Board,
    player: usize,
    from_abs: usize,
    to_abs: usize,
    to_wins: bool,
    tie: bool,
) {
    let last_moved = board.last_moved_piece_type;
    let prev_dst = board.prev_dst;

    // Case 1: the previously-moved piece is the aggressor.
    if last_moved != 0xff && prev_dst != 0xff {
        protect_aggressor_pattern(board, player, prev_dst as usize);
    }

    // Case 2: the moving piece is the protector (survived its move).
    if !(to_wins || tie) {
        protect_protector_pattern(board, player, to_abs);
    }

    // Case 3: the moving piece (or the empty square left by a tie) is protectee.
    if !to_wins {
        protect_protectee_pattern(board, player, to_abs);
    }

    // Case 4: protection against a newly-revealed defender that just won.
    if to_wins {
        protect_aggressor_pattern(board, player, to_abs);
    }

    // Case 5: the abandoned source square becomes a protectee.
    protect_protectee_pattern(board, player, from_abs);
}

/// `UPDATE_PROTECT(center-2step, center-1step, center)` over the eight
/// two-step / knight-shaped geometries used by cases 1 and 4 (`:474-517`).
fn protect_aggressor_pattern(board: &mut Board, player: usize, center: usize) {
    let row = center / 10;
    let col = center % 10;
    let c = center as i32;
    let go =
        |b: &mut Board, a: i32, m: i32| update_protect(b, player, a as usize, m as usize, center);

    if row >= 2 {
        go(board, c - 20, c - 10);
    }
    if row < 8 {
        go(board, c + 20, c + 10);
    }
    if col >= 2 {
        go(board, c - 2, c - 1);
    }
    if col < 8 {
        go(board, c + 2, c + 1);
    }
    if row < 9 && col >= 1 {
        go(board, c + 9, c - 1);
        go(board, c + 9, c + 10);
    }
    if row < 9 && col < 9 {
        go(board, c + 11, c + 1);
        go(board, c + 11, c + 10);
    }
    if row > 0 && col >= 1 {
        go(board, c - 11, c - 1);
        go(board, c - 11, c - 10);
    }
    if row > 0 && col < 9 {
        go(board, c - 9, c + 1);
        go(board, c - 9, c - 10);
    }
}

/// Case 2 geometry (`:520-567`): the moving piece at `dst` is the protector,
/// shielding the cell one step away from an aggressor two steps away.
fn protect_protector_pattern(board: &mut Board, player: usize, dst: usize) {
    let row = dst / 10;
    let col = dst % 10;
    let d = dst as i32;
    let go = |b: &mut Board, m: i32, a: i32| update_protect(b, player, dst, m as usize, a as usize);

    if row >= 2 {
        go(board, d - 10, d - 20);
    }
    if row < 8 {
        go(board, d + 10, d + 20);
    }
    if col >= 2 {
        go(board, d - 1, d - 2);
    }
    if col < 8 {
        go(board, d + 1, d + 2);
    }
    if row < 9 && col >= 1 {
        go(board, d - 1, d + 9);
        go(board, d + 10, d + 9);
    }
    if row < 9 && col < 9 {
        go(board, d + 1, d + 11);
        go(board, d + 10, d + 11);
    }
    if row > 0 && col >= 1 {
        go(board, d - 1, d - 11);
        go(board, d - 10, d - 11);
    }
    if row > 0 && col < 9 {
        go(board, d + 1, d - 9);
        go(board, d - 10, d - 9);
    }
}

/// Cases 3 and 5 geometry (`:568-608`, `:658-695`): the cell `center` is the
/// protectee, flanked by a protector and an aggressor on opposite adjacent
/// sides.
fn protect_protectee_pattern(board: &mut Board, player: usize, center: usize) {
    let row = center / 10;
    let col = center % 10;
    let c = center as i32;
    let go = |b: &mut Board, pr: i32, ag: i32| {
        update_protect(b, player, pr as usize, center, ag as usize)
    };

    if row < 9 && row > 0 {
        go(board, c + 10, c - 10);
        go(board, c - 10, c + 10);
    }
    if row < 9 && col > 0 {
        go(board, c + 10, c - 1);
        go(board, c - 1, c + 10);
    }
    if row < 9 && col < 9 {
        go(board, c + 10, c + 1);
        go(board, c + 1, c + 10);
    }
    if row > 0 && col > 0 {
        go(board, c - 10, c - 1);
        go(board, c - 1, c - 10);
    }
    if row > 0 && col < 9 {
        go(board, c - 10, c + 1);
        go(board, c + 1, c - 10);
    }
    if col < 9 && col > 0 {
        go(board, c - 1, c + 1);
        go(board, c + 1, c - 1);
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
/// `to_play`, given the supplied `flag_captured` player index (or `None`) and
/// the reference-parity attack clock. Mirrors `IncrementTerminationCounterKernel`'s
/// predicate.
pub fn is_terminal(board: &Board, to_play: usize, flag_captured: Option<usize>) -> bool {
    is_terminal_with_clock(board, to_play, flag_captured, MAX_NUM_MOVES_BETWEEN_ATTACKS)
}

/// [`is_terminal`] with an explicit no-attack draw clock instead of the
/// reference-parity constant — the knob a training-only self-play loop anneals
/// (e.g. the [`Simulator`](crate::sim::Simulator)'s configurable clock); every
/// other caller (the `Game` trait, `lab`, tests) goes through [`is_terminal`]
/// and stays reference-faithful.
pub fn is_terminal_with_clock(
    board: &Board,
    to_play: usize,
    flag_captured: Option<usize>,
    attack_clock: u32,
) -> bool {
    has_legal_movement(board, to_play) < 3
        || flag_captured.is_some()
        || board.num_moves > MAX_NUM_MOVES
        || board.num_moves_since_last_attack > attack_clock
}

/// Reward to player 0 at a terminal state (`ComputeRewardPl0Kernel`): +1 / -1
/// flag capture, +1 / -1 / 0 for stuck players, 0 on either timeout.
pub fn reward_pl0(board: &Board, to_play: usize, flag_captured: Option<usize>) -> f64 {
    reward_pl0_with_clock(board, to_play, flag_captured, MAX_NUM_MOVES_BETWEEN_ATTACKS)
}

/// [`reward_pl0`] with an explicit no-attack draw clock; see
/// [`is_terminal_with_clock`].
pub fn reward_pl0_with_clock(
    board: &Board,
    to_play: usize,
    flag_captured: Option<usize>,
    attack_clock: u32,
) -> f64 {
    let not_timeout =
        board.num_moves <= MAX_NUM_MOVES && board.num_moves_since_last_attack <= attack_clock;

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
