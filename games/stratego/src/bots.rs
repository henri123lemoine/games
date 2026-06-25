//! A competent heuristic baseline `Agent<Stratego>` — no neural net.
//!
//! It plays both phases: during deployment it fills its home rows with a random
//! *legal* arrangement (square-by-square, honouring supply and flag handedness
//! through the same [`DeploymentState`](crate::arrangement::DeploymentState) the
//! game uses), and during the move phase it picks the highest-scoring move from a
//! one-ply greedy search over a material+position evaluation with a
//! suspected-rank belief over hidden enemy pieces.
//!
//! The belief never reads a hidden opponent rank off the board (that would be
//! cheating): an unknown enemy is scored by the distribution of ranks it could
//! still be, derived from the public hidden-piece counts and narrowed by what its
//! movements have revealed (a piece that has moved cannot be a Flag or Bomb).
//! Attacks on *revealed* enemies use the exact battle table; attacks on unknowns
//! use the belief's expected outcome, so the bot throws the minimal sufficient
//! rank at a likely-weaker target and keeps its strong pieces away from
//! aggressive unknowns that might outrank them.

use game_core::{Agent, Rng};

use crate::action::Action;
use crate::board::{Board, Color, PieceType};
use crate::game::{Move, State, Stratego};
use crate::rules::{self, Battle};

/// Material value of a piece to its owner. The Spy and Scout are priced above
/// their bare rank for their special roles (the Spy is the only answer to an
/// enemy Marshal; Scouts reveal and strike at range), the Flag is effectively
/// the game, and Bombs are worth a solid defensive piece.
fn piece_value(kind: PieceType) -> f64 {
    match kind {
        PieceType::Flag => 1000.0,
        PieceType::Marshal => 100.0,
        PieceType::General => 70.0,
        PieceType::Colonel => 45.0,
        PieceType::Major => 30.0,
        PieceType::Captain => 22.0,
        PieceType::Lieutenant => 16.0,
        PieceType::Sergeant => 12.0,
        PieceType::Miner => 18.0,
        PieceType::Scout => 10.0,
        PieceType::Spy => 28.0,
        PieceType::Bomb => 20.0,
        PieceType::Lake | PieceType::Empty => 0.0,
    }
}

/// The heuristic baseline agent.
pub struct HeuristicBot;

impl Agent<Stratego> for HeuristicBot {
    fn act(&self, game: &Stratego, state: &State, player: usize, rng: &mut Rng) -> usize {
        use game_core::Game;
        let actions = game.legal_actions(state);
        match state {
            // Deployment is a random *legal* fill. A fixed bunker template was
            // tried and measured worse for this one-ply bot: pinning the flag to
            // a known corner makes it easier for an opponent to stumble onto,
            // and front-loading the line parks the heavy pieces out of the fight.
            // A varied random line places strength where it can actually battle,
            // and the move-phase flag defence guards the flag during play.
            State::Deploy { .. } => rng.below(actions.len()),
            State::Play { board, to_play, .. } => {
                debug_assert_eq!(*to_play, player);
                best_move(board, player, &actions, rng)
            }
        }
    }
}

/// Per-opponent-piece belief: the set of ranks a hidden enemy could still be and
/// the expected material value an attacker faces. Built once per decision from
/// the public hidden-piece counts so every candidate move shares it.
struct Belief {
    /// Remaining hidden enemy count per [`PieceType`] index `[0, 12)`.
    hidden_counts: [u32; 12],
    /// Total hidden enemy pieces.
    total_hidden: u32,
    /// Hidden movable pieces (everything that is not a Flag or Bomb).
    movable_hidden: u32,
}

impl Belief {
    fn new(board: &Board, opponent: usize) -> Belief {
        let counts = board.num_hidden[opponent];
        let mut hidden_counts = [0u32; 12];
        let mut total = 0;
        let mut movable = 0;
        for (t, &c) in counts.iter().enumerate() {
            hidden_counts[t] = c as u32;
            total += c as u32;
            if PieceType::from_u8(t as u8).is_movable() {
                movable += c as u32;
            }
        }
        Belief {
            hidden_counts,
            total_hidden: total,
            movable_hidden: movable,
        }
    }

    /// The rank distribution of a hidden enemy on `cell`, given whether it has
    /// moved (a moved piece is never a Flag or Bomb). Returns `(weight, rank)`
    /// pairs over [`PieceType`]s with nonzero belief; empty when the public
    /// counts say nothing is hidden (the caller then declines to guess).
    fn distribution(&self, has_moved: bool) -> Vec<(f64, PieceType)> {
        let mut out = Vec::new();
        let denom = if has_moved {
            self.movable_hidden
        } else {
            self.total_hidden
        };
        if denom == 0 {
            return out;
        }
        for t in 0..12u8 {
            let kind = PieceType::from_u8(t);
            if self.hidden_counts[t as usize] == 0 {
                continue;
            }
            if has_moved && !kind.is_movable() {
                continue;
            }
            out.push((self.hidden_counts[t as usize] as f64 / denom as f64, kind));
        }
        out
    }

    /// Expected material swing of attacking a hidden enemy with `attacker`,
    /// scored from the attacker owner's perspective: `+enemy value` when the
    /// attacker is expected to win, `-attacker value` when it loses, and the
    /// trade value on a tie, weighted by the belief.
    fn expected_attack_value(&self, attacker: PieceType, has_moved: bool) -> f64 {
        let dist = self.distribution(has_moved);
        if dist.is_empty() {
            return 0.0;
        }
        let mut expected = 0.0;
        for (w, defender) in dist {
            expected += w * match rules::resolve(attacker, defender) {
                Battle::AttackerWins => piece_value(defender),
                Battle::DefenderWins => -piece_value(attacker),
                Battle::Tie => piece_value(defender) - piece_value(attacker),
            };
        }
        expected
    }
}

/// Static material balance (own minus enemy, enemy hidden pieces valued by the
/// belief's average) — the resting value of a position before any move bonus.
fn material(board: &Board, player: usize, belief: &Belief) -> f64 {
    let own = Color::of_player(player);
    let mut score = 0.0;
    for cell in 0..100usize {
        let p = &board.pieces[cell];
        match p.color {
            c if c == Color::Empty || c == Color::Lake => {}
            c if c == own => score += piece_value(p.kind),
            _ => {
                if p.visible {
                    score -= piece_value(p.kind);
                } else if belief.total_hidden > 0 {
                    score -= belief.average_hidden_value();
                }
            }
        }
    }
    score
}

impl Belief {
    /// Mean material value of an as-yet-unidentified enemy piece.
    fn average_hidden_value(&self) -> f64 {
        if self.total_hidden == 0 {
            return 0.0;
        }
        let mut sum = 0.0;
        for t in 0..12u8 {
            sum += self.hidden_counts[t as usize] as f64 * piece_value(PieceType::from_u8(t));
        }
        sum / self.total_hidden as f64
    }
}

/// Scores one candidate move from `player`'s perspective: the immediate
/// material expectation of any battle it starts, plus small positional nudges
/// (advance Scouts to reveal, keep movable pieces off the back row guarding the
/// flag region, avoid walking a strong piece next to an aggressive unknown).
fn score_move(board: &Board, player: usize, act: Action, belief: &Belief) -> f64 {
    let opponent = 1 - player;
    let opp = Color::of_player(opponent);
    let (from_abs, to_abs) = act.to_abs(player);
    let mover = board.pieces[from_abs];
    let target = board.pieces[to_abs];

    let mut score = 0.0;

    let is_attack = target.color == opp;
    if is_attack {
        if target.visible {
            score += match rules::resolve(mover.kind, target.kind) {
                Battle::AttackerWins => {
                    if target.kind == PieceType::Flag {
                        1e6
                    } else {
                        piece_value(target.kind)
                    }
                }
                Battle::DefenderWins => -piece_value(mover.kind),
                Battle::Tie => piece_value(target.kind) - piece_value(mover.kind),
            };
        } else {
            score += belief.expected_attack_value(mover.kind, target.has_moved);
        }
    } else {
        // A quiet slide. Reward Scouts probing forward (they reveal and pressure
        // cheaply), and discourage marching a valuable piece next to a hidden
        // enemy that has shown it will move — it might outrank us.
        if mover.kind.is_scout() {
            score += advance_bonus(from_abs, to_abs, player) * 2.0;
        } else {
            score += advance_bonus(from_abs, to_abs, player);
        }
        if piece_value(mover.kind) >= piece_value(PieceType::Colonel) {
            score -= aggressive_unknown_adjacency(board, to_abs, opp, belief);
        }
    }

    score += flag_defense_delta(board, player, from_abs, to_abs, is_attack);

    score
}

/// Value the danger to a flag in *threats* — one per enemy piece orthogonally
/// adjacent to it (each is a single move from capturing it). Each such threat is
/// near-catastrophic, so the move scorer treats reducing it as worth far more
/// than a piece.
const FLAG_THREAT_VALUE: f64 = 200.0;

/// How a move changes the danger to our own flag, scored so that capturing the
/// enemy piece next to the flag, or interposing one of our pieces on an open
/// flag-adjacent square, is strongly preferred; vacating a square that was the
/// flag's only shield is strongly penalised. This is the bot's defensive
/// backbone — without it a loose enemy walks straight onto the flag.
fn flag_defense_delta(
    board: &Board,
    player: usize,
    from_abs: usize,
    to_abs: usize,
    is_attack: bool,
) -> f64 {
    let own = Color::of_player(player);
    let opp = Color::of_player(1 - player);
    let Some(flag) = find_flag(board, own) else {
        return 0.0;
    };
    let flag_adjacent: Vec<usize> = orthogonal(flag).collect();
    let mut delta = 0.0;

    // Capturing an enemy that sits next to our flag removes a live threat.
    if is_attack && flag_adjacent.contains(&to_abs) && board.pieces[to_abs].color == opp {
        delta += FLAG_THREAT_VALUE;
    }
    // Sliding our own piece onto an empty flag-adjacent square blocks that
    // approach; leaving such a square (when it was our shield) opens one.
    let to_shields = flag_adjacent.contains(&to_abs) && board.pieces[to_abs].color == Color::Empty;
    let from_shielded = flag_adjacent.contains(&from_abs);
    if to_shields {
        delta += FLAG_THREAT_VALUE * 0.15;
    }
    if from_shielded && !flag_adjacent.contains(&to_abs) {
        delta -= FLAG_THREAT_VALUE * 0.25;
    }
    delta
}

/// The cell of `color`'s flag, if it is still on the board.
fn find_flag(board: &Board, color: Color) -> Option<usize> {
    (0..100).find(|&c| board.pieces[c].color == color && board.pieces[c].kind == PieceType::Flag)
}

/// Forward progress toward the enemy home, in the mover's own orientation:
/// player 0 advances up the board (increasing row), player 1 down.
fn advance_bonus(from_abs: usize, to_abs: usize, player: usize) -> f64 {
    let from_row = (from_abs / 10) as i32;
    let to_row = (to_abs / 10) as i32;
    let delta = if player == 0 {
        to_row - from_row
    } else {
        from_row - to_row
    };
    delta as f64 * 0.4
}

/// Penalty for parking a strong piece next to a hidden enemy that has already
/// moved (so could be anything movable, including a rank that beats us). Scales
/// with the expected loss if that unknown turns out to outrank our piece.
fn aggressive_unknown_adjacency(board: &Board, cell: usize, opp: Color, belief: &Belief) -> f64 {
    let mover_here = board.pieces[cell];
    let mut penalty = 0.0;
    for nb in orthogonal(cell) {
        let p = &board.pieces[nb];
        if p.color == opp && !p.visible && p.has_moved {
            for (w, rank) in belief.distribution(true) {
                if rules::resolve(rank, mover_here.kind) == Battle::AttackerWins {
                    penalty += w * piece_value(mover_here.kind) * 0.25;
                }
            }
        }
    }
    penalty
}

/// Orthogonal in-bounds neighbours of an absolute cell.
fn orthogonal(cell: usize) -> impl Iterator<Item = usize> {
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

/// One-ply greedy choice over the legal moves: the index (into `actions`) of the
/// highest-scoring move. Ties are broken uniformly at random so play is not
/// deterministically exploitable and self-play games diverge.
fn best_move(board: &Board, player: usize, actions: &[Move], rng: &mut Rng) -> usize {
    let belief = Belief::new(board, 1 - player);
    let base = material(board, player, &belief);
    let mut best_score = f64::NEG_INFINITY;
    let mut best: Vec<usize> = Vec::new();
    for (i, action) in actions.iter().enumerate() {
        let Move::Step(act) = action else {
            continue;
        };
        let score = base + score_move(board, player, *act, &belief);
        if score > best_score + 1e-9 {
            best_score = score;
            best.clear();
            best.push(i);
        } else if (score - best_score).abs() <= 1e-9 {
            best.push(i);
        }
    }
    if best.is_empty() {
        return rng.below(actions.len());
    }
    best[rng.below(best.len())]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Stratego;
    use game_core::{Game, RandomAgent, Rng, play_n, winrate_vs_field};

    #[test]
    fn heuristic_game_terminates() {
        let game = Stratego;
        let agents: [&dyn Agent<Stratego>; 2] = [&HeuristicBot, &HeuristicBot];
        let mut rng = Rng::new(2024);
        let terminal = play_n(&game, &agents, &mut rng);
        assert!(
            game.is_terminal(&terminal),
            "heuristic self-play terminates"
        );
        let r0 = game.returns(&terminal, 0);
        assert!(
            [-1.0, 0.0, 1.0].contains(&r0),
            "well-formed terminal reward"
        );
    }

    #[test]
    fn heuristic_beats_random_by_a_clear_margin() {
        // The true win share against a field of random agents is ~0.74 (fair is
        // 0.5); 400 games tightens the estimate enough that 0.68 clears it with
        // margin at every seed measured. A one-ply greedy bot in a deep
        // hidden-info game won't post a crushing 0.9, but this is unambiguous.
        let game = Stratego;
        let share = winrate_vs_field(&game, &HeuristicBot, &RandomAgent, 400, 7);
        assert!(
            share > 0.68,
            "heuristic should clearly beat random; win share was {share}"
        );
    }

    #[test]
    fn heuristic_takes_a_free_capture_of_a_revealed_weaker_piece() {
        // Red Marshal next to a revealed Blue Captain it can take for free.
        let mut board = Board::blank();
        board.pieces[55] = crate::board::Piece::new(PieceType::Marshal, Color::Red, 0);
        board.pieces[56] = crate::board::Piece::new(PieceType::Captain, Color::Blue, 0);
        board.pieces[56].visible = true;
        board.pieces[0] = crate::board::Piece::new(PieceType::Flag, Color::Red, 1);
        board.pieces[99] = crate::board::Piece::new(PieceType::Flag, Color::Blue, 1);
        board.num_hidden[0][PieceType::Marshal as usize] = 1;
        board.num_hidden_unmoved[0] = 1;

        let state = State::Play {
            board: Box::new(board),
            to_play: 0,
            flag_captured: None,
        };
        let game = Stratego;
        let actions = game.legal_actions(&state);
        let mut rng = Rng::new(1);
        let idx = HeuristicBot.act(&game, &state, 0, &mut rng);
        let (src, dst) = match actions[idx] {
            Move::Step(a) => a.to_abs(0),
            _ => panic!("move phase"),
        };
        assert_eq!(
            (src, dst),
            (55, 56),
            "the Marshal must take the free Captain"
        );
    }
}
