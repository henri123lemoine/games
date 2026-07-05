//! Behavior-cloning dataset generation for the move-RL warm start: plays
//! [`HeuristicBot`] against itself through the real engine (not
//! [`Simulator`](crate::sim::Simulator) — a completed game's outcome is exact,
//! so there is no λ-return/bootstrap machinery to route through) and records
//! every move-phase decision's encoded obs, legal set, and chosen action,
//! labeled with the acting player's true final outcome.
//!
//! The heuristic's deployment is already a uniform-random legal fill
//! ([`bots::HeuristicBot`]'s `State::Deploy` arm), so it carries no learnable
//! setup policy beyond "any legal arrangement" — BC only clones the move phase.

use game_core::{Agent, Game, Rng, Turn};

use crate::bots::HeuristicBot;
use crate::encode::{EncoderConfig, encode_tokens};
use crate::evaluator::two_hot;
use crate::game::{Move, State, Stratego};
use crate::rules;

/// One recorded move-phase decision: the encoded `(92, feat)` obs (flattened
/// row-major), the legal action indices, which one was chosen, and the acting
/// player's outcome one-hot over [`crate::evaluator::VALUE_CATEGORIES`]
/// (`[-1, 0, 1]`), resolved once the game that produced it has finished.
pub struct BcRow {
    pub obs: Vec<f32>,
    pub legal: Vec<u16>,
    pub action: u16,
    pub outcome: [f32; 3],
}

/// Plays `num_games` HeuristicBot-vs-HeuristicBot games to completion (real
/// engine termination, reference-parity clock) in parallel, returning every
/// recorded move-phase decision across all of them. `epsilon` is the
/// probability either seat ignores the heuristic's pick and plays a uniformly
/// random legal action instead — jitter so a deterministic bot doesn't produce
/// one repeated line per opening (deployment is already randomized).
pub fn generate_games(
    num_games: usize,
    seed: u64,
    epsilon: f32,
    cfg: &EncoderConfig,
) -> Vec<BcRow> {
    use rayon::prelude::*;
    (0..num_games)
        .into_par_iter()
        .flat_map(|i| {
            play_one_game(
                seed ^ (i as u64).wrapping_mul(0x9e3779b97f4a7c15),
                epsilon,
                cfg,
            )
        })
        .collect()
}

fn play_one_game(seed: u64, epsilon: f32, cfg: &EncoderConfig) -> Vec<BcRow> {
    let game = Stratego;
    let bot = HeuristicBot;
    let mut rng = Rng::new(seed);
    let mut state = game.initial_state();
    // (player, obs, legal, action) — outcome filled in once the game ends.
    let mut pending: Vec<(usize, Vec<f32>, Vec<u16>, u16)> = Vec::new();

    while !game.is_terminal(&state) {
        let Turn::Player(player) = game.turn(&state) else {
            unreachable!("Stratego has no chance nodes")
        };
        let actions = game.legal_actions(&state);
        let is_move = matches!(state, State::Play { .. });
        let jitter = is_move && rng.below(1_000_000) < (epsilon * 1_000_000.0) as usize;
        let idx = if jitter {
            rng.below(actions.len())
        } else {
            bot.act(&game, &state, player, &mut rng)
        };

        if is_move {
            let State::Play { board, to_play, .. } = &state else {
                unreachable!("is_move implies State::Play")
            };
            let mask = rules::legal_mask(board, *to_play);
            let legal: Vec<u16> = (0..mask.len())
                .filter(|&i| mask[i])
                .map(|i| i as u16)
                .collect();
            let obs = encode_tokens(board, *to_play, cfg);
            let Move::Step(action) = actions[idx] else {
                unreachable!("move phase yields Move::Step")
            };
            pending.push((player, obs, legal, action.0));
        }

        game.apply(&mut state, actions[idx]);
    }

    let State::Play {
        board,
        to_play,
        flag_captured,
    } = &state
    else {
        unreachable!("is_terminal is only ever true in State::Play")
    };
    let reward_pl0 = rules::reward_pl0(board, *to_play, *flag_captured);

    pending
        .into_iter()
        .map(|(player, obs, legal, action)| {
            let value = if player == 0 { reward_pl0 } else { -reward_pl0 };
            BcRow {
                obs,
                legal,
                action,
                outcome: two_hot(value as f32),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_one_row_per_recorded_move_decision() {
        let cfg = EncoderConfig::default();
        let rows = generate_games(4, 1, 0.1, &cfg);
        assert!(!rows.is_empty());
        for row in &rows {
            assert_eq!(
                row.obs.len(),
                cfg.num_token_features() * crate::encode::NUM_OCCUPIABLE_CELLS
            );
            assert!(!row.legal.is_empty());
            assert!(row.legal.contains(&row.action));
            let sum: f32 = row.outcome.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "outcome must be a distribution: {:?}",
                row.outcome
            );
        }
    }

    #[test]
    fn outcomes_are_zero_sum_between_the_two_players_last_decisions() {
        // Regenerate deterministically and check at least one game produced a
        // decisive (non-tie) outcome — otherwise epsilon/seed choices above
        // would be silently testing nothing.
        let cfg = EncoderConfig::default();
        let rows = generate_games(20, 7, 0.05, &cfg);
        let decisive = rows.iter().any(|r| r.outcome[1] < 0.99);
        assert!(
            decisive,
            "expected at least one decisive game in 20 heuristic games"
        );
    }

    #[test]
    fn epsilon_zero_is_deterministic_given_a_seed() {
        let cfg = EncoderConfig::default();
        let a = generate_games(3, 42, 0.0, &cfg);
        let b = generate_games(3, 42, 0.0, &cfg);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.action, y.action);
            assert_eq!(x.legal, y.legal);
        }
    }
}
