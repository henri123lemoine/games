//! Continuous-chasing rule: you may not, during a chase, reproduce an earlier
//! threatening position — except you may undo your immediately previous move.
//!
//! The reference ships two equivalent implementations: the GPU kernels diff the
//! board against its state `delta` moves ago over a circular history ring
//! (`chase_state.cu`), and the Python `StateMachine`
//! (`tests/continuous_chase.py`) is the abstract per-move oracle the diff
//! reproduces. We port the oracle: it is exact, needs no board-history ring,
//! and keeps [`Board`](crate::board::Board) cheap to clone.
//!
//! Each player runs a [`ChaseOracle`] tracking the *opponent's* chasing piece
//! (`chaser`) and *our* fleeing piece (`evader`), plus a history of threatening
//! (chaser-cell, evader-cell) adjacency pairs. A pair is keyed by direction so
//! the same border crossed from either side is distinct.

use crate::board::is_adjacent;

/// Compact bitset state machine state from `chase_state.h` — kept as struct
/// fields the rules reference, though the live rule logic uses [`ChaseOracle`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChaseState {
    pub last_dst: u8,
    pub last_src: u8,
    pub chase_length: i32,
}

impl Default for ChaseState {
    fn default() -> Self {
        ChaseState {
            last_dst: 0xee,
            last_src: 0xee,
            chase_length: 0,
        }
    }
}

/// Directional adjacency key for the threatening-pair history: encodes the move
/// from `x` to its neighbour `y` (`pair_to_int`, `continuous_chase.py:26-36`).
fn pair_key(x: usize, y: usize) -> usize {
    if y + 1 == x {
        x
    } else if y == x + 1 {
        100 + x
    } else if y + 10 == x {
        200 + x
    } else if y == x + 10 {
        300 + x
    } else {
        unreachable!("pair_key on non-adjacent cells {x} {y}")
    }
}

/// The abstract chase oracle for one player, ported from the reference
/// `StateMachine` (`continuous_chase.py:50-97`). `is_opp` in [`update`] is true
/// when the *other* player (the chaser, from this machine's perspective) moved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChaseOracle {
    /// Threatening (chaser -> evader) directional pairs already seen.
    pair_history: Vec<bool>,
    chaser_pos: Option<usize>,
    evader_pos: Option<usize>,
    last_chaser_pos: Option<usize>,
}

impl Default for ChaseOracle {
    fn default() -> Self {
        ChaseOracle {
            pair_history: vec![false; 400],
            chaser_pos: None,
            evader_pos: None,
            last_chaser_pos: None,
        }
    }
}

impl ChaseOracle {
    fn clear(&mut self) {
        self.pair_history.iter_mut().for_each(|b| *b = false);
        self.chaser_pos = None;
        self.evader_pos = None;
        self.last_chaser_pos = None;
    }

    /// Folds one applied move into the machine and returns whether that move was
    /// a chase-rule *violation* (a repeated threatening reproduction by us).
    ///
    /// `src`/`dst` are absolute cells of the move; `was_battle` true if it was
    /// an attack; `is_opp` true when the acting player is this machine's chaser
    /// (i.e. the opponent of the player this oracle defends).
    pub fn update(&mut self, src: usize, dst: usize, was_battle: bool, is_opp: bool) -> bool {
        let needs_reset = if is_opp {
            self.evader_pos.is_some_and(|e| e != src)
                || self.chaser_pos.is_some_and(|c| !is_adjacent(c, src))
        } else {
            self.chaser_pos != Some(src) || self.evader_pos.is_some_and(|e| !is_adjacent(dst, e))
        } || was_battle;

        if needs_reset {
            self.clear();
        }

        let mut violation = false;
        if is_opp {
            if let Some(c) = self.chaser_pos
                && self.evader_pos.is_none()
            {
                self.pair_history[pair_key(c, src)] = true;
            }
            self.evader_pos = Some(dst);
        } else {
            if let (Some(_chaser), Some(evader)) = (self.chaser_pos, self.evader_pos)
                && self.last_chaser_pos != Some(dst)
            {
                violation = self.pair_history[pair_key(dst, evader)];
            }
            if self.chaser_pos.is_some()
                && let Some(evader) = self.evader_pos
            {
                debug_assert!(is_adjacent(evader, dst));
                self.pair_history[pair_key(dst, evader)] = true;
            }
            self.last_chaser_pos = self.chaser_pos;
            self.chaser_pos = Some(dst);
        }
        violation
    }

    /// Would applying the (non-attack) move `src -> dst` by the player this
    /// oracle defends be a chase violation, without mutating the machine?
    pub fn would_violate(&self, src: usize, dst: usize) -> bool {
        let needs_reset =
            self.chaser_pos != Some(src) || self.evader_pos.is_some_and(|e| !is_adjacent(dst, e));
        if needs_reset {
            return false;
        }
        if let (Some(_chaser), Some(evader)) = (self.chaser_pos, self.evader_pos)
            && self.last_chaser_pos != Some(dst)
        {
            return self.pair_history[pair_key(dst, evader)];
        }
        false
    }
}
