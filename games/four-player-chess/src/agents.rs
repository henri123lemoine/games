use game_core::{Agent, Game, Rng};

use crate::{FourPlayerChess, State, legal_moves_for};

/// One-ply FFA baseline: maximize immediate points, then retain activity.
/// It is intentionally ordinary game knowledge, not a stand-in for the
/// shipped neural agent.
#[derive(Debug, Default, Clone, Copy)]
pub struct GreedyAgent;

impl Agent<FourPlayerChess> for GreedyAgent {
    fn act(&self, game: &FourPlayerChess, state: &State, player: usize, rng: &mut Rng) -> usize {
        let actions = game.legal_actions(state);
        let mut best = i64::MIN;
        let mut choices = Vec::new();
        for (index, &action) in actions.iter().enumerate() {
            let mut next = state.clone();
            let before = next.scores[player];
            game.apply(&mut next, action);
            let gained = i64::from(next.scores[player] - before);
            let activity = if next.is_active(crate::Color::from_index(player)) {
                legal_moves_for(&next, crate::Color::from_index(player)).len() as i64
            } else {
                -1_000
            };
            let value = gained * 10_000 + activity;
            if value > best {
                best = value;
                choices.clear();
                choices.push(index);
            } else if value == best {
                choices.push(index);
            }
        }
        choices[rng.below(choices.len())]
    }
}

/// Activity-only deterministic-ish baseline used as a second field style.
#[derive(Debug, Default, Clone, Copy)]
pub struct MobilityAgent;

impl Agent<FourPlayerChess> for MobilityAgent {
    fn act(&self, game: &FourPlayerChess, state: &State, player: usize, rng: &mut Rng) -> usize {
        let color = crate::Color::from_index(player);
        let actions = game.legal_actions(state);
        let mut best = 0usize;
        let mut choices = Vec::new();
        for (index, &action) in actions.iter().enumerate() {
            let mut next = state.clone();
            game.apply(&mut next, action);
            let mobility = legal_moves_for(&next, color).len();
            if mobility > best || choices.is_empty() {
                best = mobility;
                choices.clear();
                choices.push(index);
            } else if mobility == best {
                choices.push(index);
            }
        }
        choices[rng.below(choices.len())]
    }
}
