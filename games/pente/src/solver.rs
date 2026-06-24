//! A capture-aware forcing-sequence solver for Pente — the tactical terminal
//! solver the net-guided search leans on so it never misses, or hallucinates, a
//! forced win in the sharp capture-and-five endgame.
//!
//! [`winning_move`] runs a depth- and node-bounded AND/OR search over *forcing*
//! moves only. At each attacker ply the attacker plays a move that *threatens*
//! to win, and the defender must answer it; the attacker wins only if it wins
//! against **every** defender reply. Two threat levels are supported:
//!
//! * **VCF** ([`Level::Vcf`]) — Victory by Continuous Fours. A forcing move is
//!   one after which the attacker has an *immediate* win available next ply
//!   (complete a five, or capture the fifth pair): a four, or a capture-to-fifth.
//! * **VCT** ([`Level::Vct`]) — Victory by Continuous Threats. A forcing move is
//!   one after which the attacker can win within a small horizon of *free*
//!   attacker plies (a null-move analysis, the defender passing). This subsumes
//!   VCF (horizon 1) and adds open-three → open-four → five sequences and
//!   capture threats that ripen into an immediate win within the horizon.
//!
//! The Pente twist over a gomoku VCF/VCT is that the defender's answers include
//! *capturing a stone out of the threat* — a four or three can be undone by
//! taking one of its stones — and the attacker's threats include captures, not
//! just lines. Both fall out for free: the search uses the real [`Game::apply`]
//! for ground truth, so every offensive and defensive capture is resolved
//! exactly, and the threat horizon is recomputed on the true post-capture board.
//!
//! **Soundness is the whole point.** The solver only ever returns a move it has
//! *proven* wins by force; a node-budget or depth cutoff yields "not proven",
//! never a false positive. Threat detection only ever *restricts which attacker
//! moves are tried* (an OR-node pruning) — the defender side always enumerates
//! **all** legal replies and the attacker must beat every one, so narrowing the
//! attacker's move set can only miss wins, never invent them. The defender's own
//! immediate wins and counter-threats are checked first: if the defender can win
//! or simply escape (reach a non-loss) the attacker has not forced a win.

use game_core::Game;

use crate::{EMPTY, PAIRS_TO_WIN, Pente, PenteAction, PenteState, completes_line};

/// Which class of threats counts as "forcing". [`Level::Vcf`] is the legacy
/// fours-and-capture-wins solver; [`Level::Vct`] widens forcing moves to any
/// threat that wins within [`VcfConfig::threat_horizon`] free attacker plies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    /// Victory by Continuous Fours: forcing moves threaten an immediate win.
    Vcf,
    /// Victory by Continuous Threats: forcing moves threaten a win within
    /// [`VcfConfig::threat_horizon`] free attacker plies.
    Vct,
}

#[derive(Clone, Copy)]
pub struct VcfConfig {
    /// Maximum attacker plies in a forcing line.
    pub max_depth: u32,
    /// Node budget; on exhaustion the search reports "not proven" (sound).
    pub max_nodes: u64,
    /// Whether forcing moves are fours-only (VCF) or any bounded threat (VCT).
    pub level: Level,
    /// VCT only: how many *free* consecutive attacker plies (the defender
    /// passing) a move may take to reach a win and still count as a threat. 1 is
    /// VCF-equivalent (an immediate win on the next ply); 2 admits open-threes
    /// and capture threats that become an immediate win after one more attacker
    /// move; higher admits longer ripening threats. Ignored at [`Level::Vcf`].
    pub threat_horizon: u32,
}

impl Default for VcfConfig {
    fn default() -> VcfConfig {
        VcfConfig {
            max_depth: 12,
            max_nodes: 200_000,
            level: Level::Vcf,
            threat_horizon: 1,
        }
    }
}

impl VcfConfig {
    /// A VCT configuration: forcing moves are any threat that wins within
    /// `threat_horizon` free attacker plies (2 covers open-threes; clamped to at
    /// least 1, which is VCF-equivalent).
    pub fn vct(max_depth: u32, max_nodes: u64, threat_horizon: u32) -> VcfConfig {
        VcfConfig {
            max_depth,
            max_nodes,
            level: Level::Vct,
            threat_horizon: threat_horizon.max(1),
        }
    }

    /// The effective threat horizon: VCF is always horizon 1 (immediate-win
    /// threats); VCT uses `threat_horizon`, never below 1.
    fn horizon(&self) -> u32 {
        match self.level {
            Level::Vcf => 1,
            Level::Vct => self.threat_horizon.max(1),
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

/// The hybrid move choice a net-MCTS Pente bot makes at the root: try the
/// sound forcing solver first and play a proven forced win immediately,
/// otherwise defer to `chooser` (the net-guided search). Keeping the
/// composition here — beside the solver it leans on — lets a caller drive any
/// search (`solvers::azero::Search`, MCTS, …) without forking the solver, and
/// makes the "forced-win-or-fallback" decision testable with a stub chooser.
pub fn hybrid_move<F: FnOnce() -> PenteAction>(
    game: &Pente,
    state: &PenteState,
    cfg: VcfConfig,
    chooser: F,
) -> PenteAction {
    winning_move(game, state, cfg).unwrap_or_else(chooser)
}

/// The forced win for the side to move, if the bounded forcing search proves
/// one. Returns the winning first move. At [`Level::Vcf`] this is the classic
/// VCF solver; at [`Level::Vct`] forcing moves widen to bounded continuous
/// threats. Sound at either level: never returns a move that is not a proven
/// forced win within budget.
pub fn winning_move(game: &Pente, state: &PenteState, cfg: VcfConfig) -> Option<PenteAction> {
    if game.is_terminal(state) {
        return None;
    }
    let attacker = state.to_move as u8;
    let mut budget = Budget {
        nodes: 0,
        max_nodes: cfg.max_nodes,
    };
    // Iterative deepening over attacker plies. A forced win is a *shallow* fact —
    // a double-three mates in three attacker moves regardless of how deep the
    // budget would allow — so searching depth 1, 2, 3, … finds the shortest win
    // with a small tree before a deep, hopeless forcing line can drain the
    // budget. Each iteration is independent and sound; the first that proves a
    // win returns it. The node budget is shared across iterations, so the total
    // work is still capped (re-search overhead is the classic ID constant, tiny
    // next to the branching it tames).
    for depth in 1..=cfg.max_depth {
        if let Some(m) = attack(game, state, attacker, cfg, depth, &mut budget) {
            return Some(m);
        }
        if budget.nodes >= budget.max_nodes {
            break;
        }
    }
    None
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

/// How many moves in `moves` win outright for `color` — the immediate-threat
/// count that orders forcing moves (strongest, e.g. double-threats, first).
fn count_wins(game: &Pente, state: &PenteState, moves: &[PenteAction], color: u8) -> usize {
    moves
        .iter()
        .filter(|a| wins_at(game, state, a.0 as usize, color))
        .count()
}

/// Empty intersections adjacent (Chebyshev distance 1) to a `side` stone — the
/// only points at which a free `side` placement can extend one of its own lines
/// or land a capturing flank. The null-move threat probe walks these so its cost
/// is the size of the attacker's frontier, not the whole board. (Soundness is
/// unaffected: the probe only *classifies* a move as forcing; a candidate set
/// that misses a free win merely under-counts threats, never invents one.)
fn frontier(game: &Pente, state: &PenteState, side: u8) -> Vec<usize> {
    let size = game.size() as i32;
    let mut seen = vec![false; state.cells.len()];
    let mut out = Vec::new();
    for p in 0..state.cells.len() {
        if state.cells[p] != side {
            continue;
        }
        let (row, col) = ((p / game.size()) as i32, (p % game.size()) as i32);
        for dr in -1..=1 {
            for dc in -1..=1 {
                let (r, c) = (row + dr, col + dc);
                if r < 0 || c < 0 || r >= size || c >= size {
                    continue;
                }
                let q = (r * size + c) as usize;
                if state.cells[q] == EMPTY && !seen[q] {
                    seen[q] = true;
                    out.push(q);
                }
            }
        }
    }
    out
}

/// Null-move threat analysis: the fewest *free* consecutive `attacker`
/// placements (the defender always passing) that reach an outright win from
/// `state`, capped at `max_h`. `Some(1)` means the attacker already has an
/// immediate win; `Some(k)` means k free attacker moves win; `None` means no win
/// within `max_h` free plies. `steps` bounds the probe so classification stays
/// cheap; it is a private counter, *not* the search's node budget.
///
/// This drives only the *classification* of a move as forcing (an OR-node
/// pruning heuristic). It is never used to declare a win, so the optimistic
/// "defender passes" assumption cannot make the solver unsound — a move that
/// looks forcing here still has to beat every real defender reply in `attack`.
fn null_move_horizon(
    game: &Pente,
    state: &PenteState,
    attacker: u8,
    max_h: u32,
    steps: &mut u32,
) -> Option<u32> {
    if max_h == 0 || *steps == 0 {
        return None;
    }
    *steps -= 1;
    // An immediate win uses the full action set (a capturing flank can sit off
    // the attacker's own frontier, e.g. the fifth-pair capture of a defender
    // pair); the cheap frontier is only for the *extension* recursion below.
    let moves = game.legal_actions(state);
    if first_win(game, state, &moves, attacker).is_some() {
        return Some(1);
    }
    if max_h == 1 {
        return None;
    }
    // Give the attacker one more free move (the defender passes), then recurse,
    // walking only the attacker's frontier so the probe stays near-linear.
    let mut best: Option<u32> = None;
    for p in frontier(game, state, attacker) {
        let mut next = state.clone();
        game.apply(&mut next, PenteAction(p as u16));
        if next.over {
            // `apply` ends the game only on a win; a winning placement is already
            // caught by `first_win` above, so this is defensive.
            return Some(1);
        }
        // The defender passes: hand the move straight back to the attacker.
        next.to_move = attacker as usize;
        if let Some(k) = null_move_horizon(game, &next, attacker, max_h - 1, steps) {
            let h = k + 1;
            best = Some(best.map_or(h, |b| b.min(h)));
            if best == Some(2) {
                break; // can't beat a two-ply threat at this node
            }
        }
    }
    best
}

/// Step ceiling for one threat-classification probe — generous enough to walk
/// the attacker's frontier at the configured horizon, bounded so classifying a
/// single move can never blow up. Independent of the search's node budget.
const PROBE_STEPS: u32 = 4_096;

/// Whether placing `a` is a *forcing* move for `attacker` from `state`: after
/// the move (resolved on the true board) the attacker threatens a win within the
/// configured horizon if the defender does nothing. Returns the resulting state
/// and the threat's null-move horizon (smaller = stronger) for move ordering.
fn forcing_after(
    game: &Pente,
    state: &PenteState,
    a: PenteAction,
    attacker: u8,
    cfg: VcfConfig,
) -> Option<(PenteState, u32)> {
    let mut next = state.clone();
    game.apply(&mut next, a);
    if next.over {
        return None; // an outright win is handled by `first_win`, not as a threat
    }
    // After the attacker's move it is the defender's turn; measure the threat
    // from the attacker's seat (the defender passing) over the remaining horizon.
    let mut probe = next.clone();
    probe.to_move = attacker as usize;
    let mut steps = PROBE_STEPS;
    null_move_horizon(game, &probe, attacker, cfg.horizon(), &mut steps).map(|h| (next, h))
}

/// Attacker (the side to move in `state`) to move: a forcing move proving a win
/// within `depth` attacker plies, or `None`. An OR node — one forcing move that
/// wins against every defense suffices.
fn attack(
    game: &Pente,
    state: &PenteState,
    attacker: u8,
    cfg: VcfConfig,
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
    // Forcing moves: those that leave the attacker threatening a win within the
    // horizon the defender must parry. Order stronger threats first (shorter
    // horizon, then more immediate-win follow-ups).
    let mut forcing: Vec<(PenteAction, PenteState, u32, usize)> = Vec::new();
    for &a in &moves {
        if let Some((next, horizon)) = forcing_after(game, state, a, attacker, cfg) {
            let immediate = count_wins(game, &next, &game.legal_actions(&next), attacker);
            forcing.push((a, next, horizon, immediate));
        }
    }
    forcing.sort_by(|x, y| x.2.cmp(&y.2).then(y.3.cmp(&x.3)));
    for (m, next, _, _) in forcing {
        // `next` has the defender to move facing the threat `m` created. The
        // attacker wins with `m` iff every defense fails.
        if defends_fail(game, &next, attacker, cfg, depth - 1, budget) {
            return Some(m);
        }
    }
    None
}

/// Defender (the side to move in `state`) faces a standing attacker threat.
/// Returns true iff every defense still loses — i.e. the attacker forces a win.
/// An AND node: the attacker wins only if **all** defender replies lose.
///
/// The defender is searched **exhaustively** — every legal reply, blocks and
/// captures and counter-attacks alike. This is the soundness keystone: the
/// threat classification gates only the *attacker's* moves (an OR-node pruning,
/// which can miss wins but never invent them), while the defender is never
/// pruned by any heuristic about which replies "look relevant". A
/// horizon-based defender filter is unsound in Pente — a reply that does not
/// change the attacker's immediate threat can still set up a delayed capture
/// that refutes it once the attacker commits — so the only pruning here is the
/// exact one: a reply after which the attacker has an *immediate* win has
/// already lost, and is dispatched without a recursive search.
fn defends_fail(
    game: &Pente,
    state: &PenteState,
    attacker: u8,
    cfg: VcfConfig,
    depth: u32,
    budget: &mut Budget,
) -> bool {
    if !budget.spend() {
        return false;
    }
    let defender = attacker ^ 1;
    let moves = game.legal_actions(state);
    // The defender escapes outright by winning first (its own five or fifth pair).
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
        // Exact, sound prune: if the attacker has an immediate win after this
        // reply, the defender lost on the spot — the attacker just plays it.
        // This branch fails for the defender with no recursive search, and it
        // is the workhorse cut: a reply that ignores a four-threat (most of
        // them) is dispatched in O(moves) instead of a whole subtree.
        if first_win(game, &next, &game.legal_actions(&next), attacker).is_some() {
            continue;
        }
        // A genuine defense (a block, or a capture that took a threat stone
        // away): does the attacker still force a win? An empty result means this
        // defense holds — the win is not proven, so (soundly) report it.
        if attack(game, &next, attacker, cfg, depth, budget).is_none() {
            return false;
        }
    }
    // No defender reply removed the threat (an unstoppable multi-threat), or
    // every defense that did still lost: the attacker forces the win.
    !moves.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vcf(g: &Pente, s: &PenteState) -> Option<PenteAction> {
        winning_move(g, s, VcfConfig::default())
    }

    fn vct(g: &Pente, s: &PenteState) -> Option<PenteAction> {
        winning_move(g, s, VcfConfig::vct(12, 400_000, 2))
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

    #[test]
    fn hybrid_plays_the_forced_win_over_the_chooser() {
        // The open-four position has a proven VCF win at e5; the hybrid must take
        // it and never consult the (here-poisoned) fallback chooser.
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
        let chosen = hybrid_move(&g, &s, VcfConfig::default(), || {
            panic!("the chooser must not run when the VCF proves a win")
        });
        assert_eq!(chosen, PenteAction(g.point("e5").unwrap()));
    }

    #[test]
    fn hybrid_defers_to_the_chooser_in_a_quiet_position() {
        // No forcing win here, so the hybrid falls through to whatever the
        // net-guided search would play — modeled by a sentinel-returning stub.
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
        let sentinel = PenteAction(g.point("e6").unwrap());
        let chosen = hybrid_move(&g, &s, VcfConfig::default(), || sentinel);
        assert_eq!(chosen, sentinel);
    }

    // ---- VCT-specific tests ------------------------------------------------
    //
    // A note on the VCF/VCT boundary in this solver. An *open three* whose only
    // extension makes an open four is already a VCF win here, because the
    // open-four it creates carries two immediate-win completions the defender
    // cannot both parry — so a lone `. X X X .` is *not* a VCF/VCT separator.
    // The true separator is a position with **no four-making move at all**, where
    // a *three*-making move forces the win: a double-three. That is what these
    // tests turn on.

    /// A double open three: the empty pivot g6 turns the horizontal pair e6 f6
    /// and the vertical pair g5 g7 into two open threes at once, neither a four.
    /// No move on the board makes a four, so VCF is blind; VCT proves the win
    /// (block one three, the other becomes an open four → five).
    fn double_three(g: &Pente) -> PenteState {
        g.parse_state(
            &[
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . X . .", // g7
                ". . . . X X . . .", // e6 f6
                ". . . . . . X . .", // g5
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
            ],
            0,
            [0, 0],
        )
    }

    #[test]
    fn vct_wins_a_double_three_that_vcf_misses() {
        // The headline case: a forced win VCT proves and VCF cannot. The pivot
        // g6 makes a double open three; no four exists yet, so VCF finds nothing,
        // but VCT drives the open-three → open-four → five sequence.
        let g = Pente::new(9);
        let s = double_three(&g);
        assert_eq!(vcf(&g, &s), None, "no four on the board → VCF is blind");
        assert_eq!(
            vct(&g, &s),
            Some(PenteAction(g.point("g6").unwrap())),
            "VCT proves the double-three forces the win at g6"
        );
    }

    #[test]
    fn vct_double_three_with_a_white_cap_is_not_a_win() {
        // The same double three, but one white stone at d6 caps the *left* end of
        // the horizontal three: after Black plays g6 the horizontal e6 f6 g6 can
        // only extend right (d6 is blocked), so it is no longer an open three —
        // the defender blocks the still-open vertical three and survives. A lone
        // refuting stone collapses the win: VCT must return None. Contrast the
        // capped-three soundness guard — this proves the search re-evaluates the
        // *whole* refuted structure, not just the pivot.
        let g = Pente::new(9);
        let s = g.parse_state(
            &[
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . X . .", // g7
                ". . . O X X . . .", // d6=O caps the horizontal three, e6 f6
                ". . . . . . X . .", // g5
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
            ],
            0,
            [0, 0],
        );
        // The base double three (no d6) is a VCT win at g6; the single cap
        // downgrades it to a blockable single threat.
        assert_eq!(
            vct(&g, &double_three(&g)),
            Some(PenteAction(g.point("g6").unwrap()))
        );
        assert_eq!(
            vct(&g, &s),
            None,
            "one capping stone breaks the double three"
        );
    }

    /// Exhaustive ground truth for the capture-defense test: does `attacker`
    /// (to move) have a forced win within `depth` plies under *full* minimax over
    /// every move — no threat heuristics? An independent oracle to check the
    /// solver against on capture-laden positions.
    fn attacker_forces_win(g: &Pente, s: &PenteState, attacker: u8, depth: u32) -> bool {
        if depth == 0 {
            return false;
        }
        g.legal_actions(s).into_iter().any(|a| {
            let mut next = s.clone();
            g.apply(&mut next, a);
            if g.is_terminal(&next) {
                next.winner() == Some(attacker as usize)
            } else {
                every_defense_loses(g, &next, attacker, depth - 1)
            }
        })
    }

    fn every_defense_loses(g: &Pente, s: &PenteState, attacker: u8, depth: u32) -> bool {
        if depth == 0 {
            return false;
        }
        let moves = g.legal_actions(s);
        !moves.is_empty()
            && moves.into_iter().all(|d| {
                let mut next = s.clone();
                g.apply(&mut next, d);
                if g.is_terminal(&next) {
                    next.winner() == Some(attacker as usize)
                } else {
                    attacker_forces_win(g, &next, attacker, depth - 1)
                }
            })
    }

    #[test]
    fn vct_capture_aware_defense_matches_ground_truth() {
        // Capture-aware *defense*, the keystone soundness test. The solver must
        // treat a defender capture exactly like any other refutation — neither
        // miss it (claiming a false win) nor over-weight it (missing a real one).
        // We check that against an independent exhaustive minimax on several
        // small, capture-dense boards: any divergence is a capture-handling bug.
        // (Small board + shallow depth keep the oracle cheap.)
        let boards: [(&[&str], usize, [u8; 2]); 3] = [
            (
                // Black three c3 d3 e3 (open) but c-file holds a capturable black
                // pair c3,c4 flanked by c2=O and c5: White can capture instead of
                // (or as well as) block. Solver must match the truth either way.
                &[
                    ". . O . .",
                    ". . X . .", // c4 = X
                    ". . X X X", // c3 d3 e3 — an open-ended three
                    ". . O . .", // c2 = O (with c5=O above, c3 c4 is a capturable pair)
                    ". . . . .",
                ],
                0,
                [0, 0],
            ),
            (
                // A four with a capturable pair: b3 c3 d3 e3 four, c3,c4 a
                // capturable pair (c2=O, c5=O). White's only saves involve the
                // capture and/or the block — the solver must agree with truth.
                &[
                    ". . O . .",
                    ". . X . .",
                    ". X X X X", // b3 c3 d3 e3
                    ". . O . .",
                    ". . . . .",
                ],
                0,
                [0, 0],
            ),
            (
                // Quiet-ish capture-laden board: scattered pairs, no forced win.
                &[
                    "X O O X .",
                    ". . . . .",
                    ". O X O .",
                    ". . . . .",
                    ". X O O X",
                ],
                0,
                [1, 1],
            ),
        ];
        for (rows, to_move, pairs) in boards {
            let g = Pente::new(5);
            let s = g.parse_state(rows, to_move, pairs);
            let truth = attacker_forces_win(&g, &s, s.to_move() as u8, 6);
            let solved = winning_move(&g, &s, VcfConfig::vct(6, 200_000, 2)).is_some();
            assert_eq!(
                solved, truth,
                "solver disagrees with exhaustive ground truth on a capture-laden board"
            );
        }
    }

    #[test]
    fn vct_capture_move_wins_when_vcf_cannot() {
        // A capture-driven VCT win: the *forcing move itself is a capture* that
        // VCF cannot prove. Black e5 captures the white pair e3 e4 (flanked by
        // e2=X) and the same stone completes two open threes — horizontal
        // c5 d5 e5 and vertical e5 e6 e7 — a double three made *by a capture*.
        // After the capture there is no four (only two threes), so VCF is blind;
        // VCT proves the win and the winning move removes a pair.
        let g = Pente::new(9);
        let s = g.parse_state(
            &[
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . X . . . .", // e7
                ". . . . X . . . .", // e6
                ". . X X . . . . .", // c5 d5  (e5 empty pivot)
                ". . . . O . . . .", // e4 = O
                ". . . . O . . . .", // e3 = O
                ". . . . X . . . .", // e2 = X (capturing flank)
                ". . . . . . . . .",
            ],
            0,
            [0, 0],
        );
        let e5 = g.point("e5").unwrap();
        assert_eq!(
            vcf(&g, &s),
            None,
            "capture makes threes, not a four → VCF blind"
        );
        assert_eq!(
            vct(&g, &s),
            Some(PenteAction(e5)),
            "VCT proves the capture-and-double-three win at e5"
        );
        // Confirm the winning move genuinely captures a pair (the e3 e4 stones).
        let mut after = s.clone();
        g.apply(&mut after, PenteAction(e5));
        assert_eq!(after.pairs(), [1, 0], "e5 captured the e3 e4 pair");
    }

    #[test]
    fn vct_capture_to_fifth_pair_is_a_forced_win() {
        // A capture double-threat to the fifth pair. Black sits on four pairs
        // with two `. O O .` arms sharing the empty pivot e5: playing e5 stands
        // as the flank of both white pairs (f5 g5 and e6 e7), so each arm becomes
        // a `X O O .` capture-to-fifth the defender must answer — and one move
        // cannot save both. Captures drive the whole win.
        let g = Pente::new(9);
        let s = g.parse_state(
            &[
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . O . . . .", // e7 = O
                ". . . . O . . . .", // e6 = O
                ". . . . . O O . .", // f5 g5 = O   (e5 empty pivot)
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
            ],
            0,
            [4, 0],
        );
        assert_eq!(
            vct(&g, &s),
            Some(PenteAction(g.point("e5").unwrap())),
            "the double capture-to-fifth threat forces the win"
        );
    }

    #[test]
    fn vct_is_sound_on_a_quiet_position() {
        // A lone stone each: no threat at any horizon. VCT must return None.
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
        assert_eq!(vct(&g, &s), None, "quiet position: no VCT win");
    }

    #[test]
    fn vct_is_sound_on_a_doubly_capped_three() {
        // A three boxed by white on both relevant ends (O X X X . O): no
        // four-making move, no continuing threat, no capture — not a forced win
        // at VCT. Guards against the solver hallucinating a win from a mere
        // three.
        let g = Pente::new(9);
        let s = g.parse_state(
            &[
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                "O X X X . O . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
            ],
            0,
            [0, 0],
        );
        assert_eq!(vcf(&g, &s), None);
        assert_eq!(vct(&g, &s), None, "a capped three forces nothing");
    }

    #[test]
    fn vcf_wins_remain_vct_wins() {
        // Every position VCF proves a win in, VCT proves too (horizon 1 ⊆ the
        // wider horizon). Checked on an open four — an immediate VCF win.
        let g = Pente::new(9);
        let s = g.parse_state(
            &[
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". X X X X . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
            ],
            0,
            [0, 0],
        );
        assert!(
            vcf(&g, &s).is_some(),
            "an open four is an immediate VCF win"
        );
        assert!(vct(&g, &s).is_some(), "VCT must also win wherever VCF wins");
    }
}
