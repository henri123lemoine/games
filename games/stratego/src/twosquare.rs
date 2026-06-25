//! Two-square rule: a piece may not cross the same cell border more than three
//! times in a row (`twosquare_state.cu`).
//!
//! State is the moving piece's last four cells `{A newest, B, C, D oldest}` in
//! the moving player's *relative* (POV) coordinates, `0xff` for missing. The
//! chain extends only while the same piece oscillates along the same axis; any
//! other move, or a death on the tracked cell, resets it. The trigger is a
//! strict single-axis zig-zag across those four cells. Scouts get the relaxed
//! "precluding direction" variant: only the destinations *past* the prior
//! turning point are forbidden, not the whole axis.

use crate::action::{Action, NUM_ACTIONS};

const MISSING: u8 = 0xff;

#[inline]
fn col(x: u8) -> u8 {
    x % 10
}
#[inline]
fn row(x: u8) -> u8 {
    x / 10
}

/// The four tracked POV cells of the last-moved piece.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TwosquareState {
    pub a: u8,
    pub b: u8,
    pub c: u8,
    pub d: u8,
}

impl Default for TwosquareState {
    fn default() -> Self {
        TwosquareState {
            a: MISSING,
            b: MISSING,
            c: MISSING,
            d: MISSING,
        }
    }
}

impl TwosquareState {
    fn clear(&mut self) {
        *self = TwosquareState::default();
    }

    /// Folds one move (in this player's POV cells) into the machine
    /// (`UpdateTwosquareActionKernel`). `src_pov`/`dst_pov` are relative cells;
    /// the chain continues only if it resumes from the prior destination along
    /// the same axis.
    pub fn update_move(&mut self, src_pov: u8, dst_pov: u8) {
        let is_vertical = col(src_pov) == col(dst_pov);
        let last_vertical = self.a != MISSING && self.b != MISSING && col(self.a) == col(self.b);

        if self.a == src_pov && is_vertical == last_vertical {
            self.d = self.c;
            self.c = self.b;
        } else {
            self.d = MISSING;
            self.c = MISSING;
        }
        self.a = dst_pov;
        self.b = src_pov;
    }

    /// Resets the machine if the tracked piece died on its current cell
    /// (`UpdateTwosquareDeathKernel`). `dead_pov` is the death cell in this
    /// player's POV, or `0xff` for no death.
    pub fn update_death(&mut self, dead_pov: u8) {
        if dead_pov != MISSING && self.a == dead_pov {
            self.clear();
        }
    }

    /// Whether the four cells form a strict single-axis zig-zag — the rule is
    /// active for the tracked piece (`IsTwosquareRuleTriggeredKernel`).
    pub fn is_triggered(&self) -> bool {
        if self.d == MISSING {
            return false;
        }
        let (a, b, c, d) = (self.a, self.b, self.c, self.d);
        // Horizontal zig-zags
        (col(d) < col(c) && col(c) > col(b) && col(b) < col(a) && col(d) < col(a))
            || (col(d) > col(c) && col(c) < col(b) && col(b) > col(a) && col(d) > col(a))
            // Vertical zig-zags
            || (row(d) < row(c) && row(c) > row(b) && row(b) < row(a) && row(d) < row(a))
            || (row(d) > row(c) && row(c) < row(b) && row(b) > row(a) && row(d) > row(a))
    }

    /// Whether a non-scout is wholly blocked along the oscillation axis — for
    /// scouts, whether even a one-cell move toward the turning point is barred
    /// (`IsTwosquareRulePrecludingDirectionKernel`). Used by the off-turn
    /// stuck-check.
    pub fn is_precluding_direction(&self) -> bool {
        if self.d == MISSING {
            return false;
        }
        let (a, b, c, d) = (self.a, self.b, self.c, self.d);
        (col(d) < col(c)
            && col(c) > col(b)
            && col(b) < col(a)
            && col(d) < col(a)
            && col(a) <= col(c))
            || (col(d) > col(c)
                && col(c) < col(b)
                && col(b) > col(a)
                && col(d) > col(a)
                && col(a) >= col(c))
            || (row(d) < row(c)
                && row(c) > row(b)
                && row(b) < row(a)
                && row(d) < row(a)
                && row(a) <= row(c))
            || (row(d) > row(c)
                && row(c) < row(b)
                && row(b) > row(a)
                && row(d) > row(a)
                && row(a) >= row(c))
    }

    /// Clears the action-mask bits for destinations forbidden by the rule
    /// (`RemoveTwosquareActionsKernel`). For non-scouts this removes the single
    /// reverse destination; for scouts it removes every destination at or past
    /// the prior turning point along the axis.
    pub fn remove_actions(&self, legal: &mut [bool; NUM_ACTIONS]) {
        if self.d == MISSING {
            return;
        }
        let (a, b, c, d) = (self.a, self.b, self.c, self.d);

        // Horizontal LTR: cannot move left of min(col(A), col(C)).
        if col(d) < col(c) && col(c) > col(b) && col(b) < col(a) && col(d) < col(a) {
            let mut idx = 900 + a as usize;
            for _ in 0..col(c).min(col(a)) {
                legal[idx] = false;
                idx += 100;
            }
        }
        // Horizontal RTL: cannot move right of max(col(A), col(C)).
        if col(d) > col(c) && col(c) < col(b) && col(b) > col(a) && col(d) > col(a) {
            let start = col(c).max(col(a));
            let mut idx = 900 + a as usize + 100 * start as usize;
            for _ in start..9 {
                legal[idx] = false;
                idx += 100;
            }
        }
        // Vertical TTB: cannot move below min(row(A), row(C)).
        if row(d) < row(c) && row(c) > row(b) && row(b) < row(a) && row(d) < row(a) {
            let mut idx = a as usize;
            for _ in 0..row(c).min(row(a)) {
                legal[idx] = false;
                idx += 100;
            }
        }
        // Vertical BTT: cannot move above max(row(A), row(C)).
        if row(d) > row(c) && row(c) < row(b) && row(b) > row(a) && row(d) > row(a) {
            let start = row(c).max(row(a));
            let mut idx = a as usize + 100 * start as usize;
            for _ in start..9 {
                legal[idx] = false;
                idx += 100;
            }
        }
    }
}

/// Rebuilds both players' two-square machines from a move log, used when a
/// state is reconstructed without carrying the live machines
/// (`TwosquareStateFromEnvState`). `moves` are `(action, player)` pairs in
/// chronological order; only the trailing six are consulted.
pub fn from_move_log(moves: &[(Action, usize)]) -> [TwosquareState; 2] {
    let mut states = [TwosquareState::default(); 2];
    let start = moves.len().saturating_sub(6);
    for &(action, player) in &moves[start..] {
        let (src_abs, dst_abs) = action.to_abs(player);
        let src_pov = if player == 1 { 99 - src_abs } else { src_abs } as u8;
        let dst_pov = if player == 1 { 99 - dst_abs } else { dst_abs } as u8;
        states[player].update_move(src_pov, dst_pov);
    }
    states
}
