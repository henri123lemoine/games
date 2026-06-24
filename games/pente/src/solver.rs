//! A capture-aware VCF-style forcing-sequence solver for Pente — the tactical
//! terminal solver the net-guided search leans on so it never misses, or
//! hallucinates, a forced win in the sharp capture-and-five endgame.
//!
//! [`winning_move`] runs a depth- and node-bounded search over *forcing* moves
//! only: at each attacker ply the attacker plays a move that threatens to win
//! outright next ply (complete a five, or capture the fifth pair), and the
//! defender must answer every such threat. The Pente twist over a gomoku VCF is
//! that the defender's answers include *capturing a stone out of the threat* —
//! a four can be undone by taking one of its stones — and the attacker's
//! threats include captures, not just lines. Both fall out for free: the search
//! uses the real [`Game::apply`] for ground truth, so every offensive and
//! defensive capture is resolved exactly.
//!
//! It is sound — it only ever returns a move it has *proven* wins by force, and
//! a node-budget cutoff yields "not proven", never a false positive. It is not
//! complete: it searches fours and capture-wins (VCF), not the broader open-three
//! threats (VCT), so it can miss some forced wins. Missing a win only costs
//! tactical sharpness; it never plays an unsound move.

use game_core::Game;

use crate::{PAIRS_TO_WIN, Pente, PenteAction, PenteState, completes_line};

#[derive(Clone, Copy)]
pub struct VcfConfig {
    /// Maximum attacker plies in a forcing line.
    pub max_depth: u32,
    /// Node budget; on exhaustion the search reports "not proven" (sound).
    pub max_nodes: u64,
}

impl Default for VcfConfig {
    fn default() -> VcfConfig {
        VcfConfig {
            max_depth: 12,
            max_nodes: 200_000,
        }
    }
}

struct Budget {
    nodes: u64,
    max_nodes: u64,
}

impl Budget {
    fn spend(&mut self) -> bool {
        self.nodes += 1;
        self.nodes <= self.max_nodes
    }
}

/// The forced win for the side to move, if the bounded forcing search proves
/// one. Returns the winning first move.
pub fn winning_move(game: &Pente, state: &PenteState, cfg: VcfConfig) -> Option<PenteAction> {
    if game.is_terminal(state) {
        return None;
    }
    let attacker = state.to_move as u8;
    let mut budget = Budget {
        nodes: 0,
        max_nodes: cfg.max_nodes,
    };
    attack(game, state, attacker, cfg.max_depth, &mut budget)
}

/// Whether placing `color` at empty `p` wins outright — completes a five, or
/// captures the fifth pair. `color` need not be the side to move.
fn wins_at(game: &Pente, state: &PenteState, p: usize, color: u8) -> bool {
    let size = game.size();
    let (row, col) = (p / size, p % size);
    state.pairs[color as usize] + game.capture_pairs_at(state, p, color) >= PAIRS_TO_WIN
        || completes_line(&state.cells, size, row, col, color)
}

/// The first move in `moves` by which `color` wins outright, if any.
fn first_win(
    game: &Pente,
    state: &PenteState,
    moves: &[PenteAction],
    color: u8,
) -> Option<PenteAction> {
    moves
        .iter()
        .copied()
        .find(|a| wins_at(game, state, a.0 as usize, color))
}

/// How many moves in `moves` win outright for `color` — the threat count that
/// orders forcing moves (strongest, e.g. double-threats, first).
fn count_wins(game: &Pente, state: &PenteState, moves: &[PenteAction], color: u8) -> usize {
    moves
        .iter()
        .filter(|a| wins_at(game, state, a.0 as usize, color))
        .count()
}

/// Attacker (the side to move in `state`) to move: a forcing move proving a win
/// within `depth` attacker plies, or `None`.
fn attack(
    game: &Pente,
    state: &PenteState,
    attacker: u8,
    depth: u32,
    budget: &mut Budget,
) -> Option<PenteAction> {
    if depth == 0 || !budget.spend() {
        return None;
    }
    let moves = game.legal_actions(state);
    if let Some(m) = first_win(game, state, &moves, attacker) {
        return Some(m);
    }
    // Forcing moves: those that leave the attacker threatening an immediate win
    // the defender must parry. Order stronger threats (double-threats) first.
    let mut forcing: Vec<(PenteAction, usize)> = Vec::new();
    for &a in &moves {
        let mut next = state.clone();
        game.apply(&mut next, a);
        if next.over {
            continue; // a winning placement is already an immediate win above
        }
        let threats = count_wins(game, &next, &game.legal_actions(&next), attacker);
        if threats >= 1 {
            forcing.push((a, threats));
        }
    }
    forcing.sort_by_key(|&(_, threats)| std::cmp::Reverse(threats));
    for (m, _) in forcing {
        let mut next = state.clone();
        game.apply(&mut next, m);
        if defends_fail(game, &next, attacker, depth - 1, budget) {
            return Some(m);
        }
    }
    None
}

/// Defender (the side to move in `state`) faces a standing attacker threat.
/// Returns true iff every defense still loses — i.e. the attacker forces a win.
fn defends_fail(
    game: &Pente,
    state: &PenteState,
    attacker: u8,
    depth: u32,
    budget: &mut Budget,
) -> bool {
    if !budget.spend() {
        return false;
    }
    let defender = attacker ^ 1;
    let moves = game.legal_actions(state);
    // The defender escapes by winning first (its own five or fifth pair).
    if first_win(game, state, &moves, defender).is_some() {
        return false;
    }
    for &d in &moves {
        let mut next = state.clone();
        game.apply(&mut next, d);
        if next.over {
            // The defender cannot win on this move (checked above), so an ended
            // game here is a draw: the attacker did not force a win.
            return false;
        }
        if first_win(game, &next, &game.legal_actions(&next), attacker).is_some() {
            continue; // d failed to remove the threat — not a real defense
        }
        // A real defense (block or a capture that took a threat stone away).
        if attack(game, &next, attacker, depth, budget).is_none() {
            return false; // this defense holds
        }
    }
    // No legal move removed the threat (an unstoppable double-threat), or every
    // defense that did still lost: the attacker forces the win.
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vcf(g: &Pente, s: &PenteState) -> Option<PenteAction> {
        winning_move(g, s, VcfConfig::default())
    }

    #[test]
    fn creating_an_open_four_is_a_forced_win() {
        // Black to move with the open three b5 c5 d5; playing e5 makes the open
        // four b5..e5 (ends a5 and f5 both open) — a forced win.
        let g = Pente::new(9);
        let s = g.parse_state(
            &[
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". X X X . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
            ],
            0,
            [0, 0],
        );
        assert_eq!(vcf(&g, &s), Some(PenteAction(g.point("e5").unwrap())));
    }

    #[test]
    fn a_double_four_in_one_move_is_a_forced_win() {
        // Black has a horizontal three b5 c5 d5 and a vertical three e2 e3 e4; e5
        // completes both into fours at once. The defender cannot parry both.
        let g = Pente::new(9);
        let mut s = g.parse_state(&[". . . . . . . . ."; 9], 0, [0, 0]);
        for c in ["b5", "c5", "d5", "e2", "e3", "e4"] {
            s.cells[g.point(c).unwrap() as usize] = crate::BLACK;
        }
        assert_eq!(vcf(&g, &s), Some(PenteAction(g.point("e5").unwrap())));
    }

    #[test]
    fn capturing_the_fifth_pair_is_a_forced_win() {
        // Black sits on four pairs; the X O O . arm lets e2 take the fifth.
        let g = Pente::new(9);
        let s = g.parse_state(
            &[
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". X O O . . . . .",
                ". . . . . . . . .",
            ],
            0,
            [4, 0],
        );
        assert_eq!(vcf(&g, &s), Some(PenteAction(g.point("e2").unwrap())));
    }

    #[test]
    fn a_blockable_simple_four_is_not_a_forced_win() {
        // White flank at a5 caps the left end, so b5 c5 d5 + e5 is only a simple
        // four; the defender blocks f5 and survives. No forcing follow-up.
        let g = Pente::new(9);
        let s = g.parse_state(
            &[
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                "O X X X . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
            ],
            0,
            [0, 0],
        );
        assert_eq!(vcf(&g, &s), None);
    }

    #[test]
    fn a_quiet_position_has_no_forced_win() {
        let g = Pente::new(9);
        let s = g.parse_state(
            &[
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . X . O . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
            ],
            0,
            [0, 0],
        );
        assert_eq!(vcf(&g, &s), None);
    }
}
