//! Action encoding: the 1800-slot source-displacement space and the
//! conversions between it and absolute board coordinates.
//!
//! `action = 100*c + src_pov`, where `src_pov` is the source cell in the acting
//! player's 180deg POV and `c in [0, 18)` selects the displacement:
//! `c in [0, 9)` = vertical (`c%9` indexes the destination row, skipping the
//! source row); `c in [9, 18)` = horizontal (`c%9` indexes the destination
//! column, skipping the source column). All in POV; blue mirrors via `9 - idx`.
//! (`action_kernels.cu:129-147`.)

pub const NUM_ACTIONS: usize = 1800;

/// A move-phase action, stored as its index in the 1800-slot space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Action(pub u16);

#[inline]
fn pov(cell: usize, player: usize) -> usize {
    if player == 1 { 99 - cell } else { cell }
}

impl Action {
    /// Decodes to absolute `(src_cell, dst_cell)` for the acting `player`,
    /// reproducing `ApplyActionsKernel:130-147`.
    pub fn to_abs(self, player: usize) -> (usize, usize) {
        let action = self.0 as usize;
        let from_cell = action % 100;
        let from_row = from_cell / 10;
        let from_col = from_cell % 10;

        let raw = action / 100;
        let horizontal = raw >= 9;
        let mut coord = raw % 9;
        coord +=
            usize::from((horizontal && coord >= from_col) || (!horizontal && coord >= from_row));

        let (pov_to_row, pov_to_col) = if horizontal {
            (from_row, coord)
        } else {
            (coord, from_col)
        };

        let from_abs = pov(from_cell, player);
        let to_pov = 10 * pov_to_row + pov_to_col;
        let to_abs = pov(to_pov, player);
        (from_abs, to_abs)
    }

    /// Encodes an absolute single-axis move for `player` into the action space.
    /// Returns `None` if the move is not a straight orthogonal slide.
    /// Mirrors `AbsCoordinatesToActions` / `ComputeIllegalChaseMovesKernel:266`.
    pub fn from_abs(src_abs: usize, dst_abs: usize, player: usize) -> Option<Action> {
        if src_abs == dst_abs {
            return None;
        }
        let src_pov = pov(src_abs, player);
        let dst_pov = pov(dst_abs, player);
        let (sr, sc) = (src_pov / 10, src_pov % 10);
        let (dr, dc) = (dst_pov / 10, dst_pov % 10);

        let (c, src_cell) = if sc == dc {
            let coord = dr - usize::from(dr > sr);
            (coord, src_pov)
        } else if sr == dr {
            let coord = dc - usize::from(dc > sc);
            (9 + coord, src_pov)
        } else {
            return None;
        };
        Some(Action((100 * c + src_cell) as u16))
    }
}
