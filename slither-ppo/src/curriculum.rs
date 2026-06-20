//! The self-play curriculum: a small ladder of stages that decides, per arena
//! reset, how the world is built (learner size vs prey size) and which pool
//! opponents are eligible. The blueprint's recipe — start the learner OVERSIZED
//! against small scripted prey (encircling only pays when you're bigger), then ramp
//! to mixed-size even self-play against the heuristic and past snapshots.
//!
//! Stages don't decay the encircle shaping; that anneal is the trainer's job,
//! keyed off real kills appearing. Here we only move the *matchup* distribution.

use slither_rl::Rng;
use slither_rl::world::{START_LENGTH, World, WorldConfig};

use crate::opponent::{Pool, PoolKind};
use crate::rollout::prey_cluster_world;

/// Fraction of even-self-play seats forced to the heuristic regardless of its
/// PFSP weight, so the gating opponent never falls out of the pool.
const HEURISTIC_FLOOR: f32 = 0.35;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Learner oversized, small fleeing/heuristic prey clustered around it. The
    /// learner is big enough that walling a foe off is geometrically possible —
    /// where encircling first emerges.
    OversizedVsPrey,
    /// Learner still somewhat bigger, prey a bit larger and the heuristic in the
    /// mix at full strength — the trap must now beat a competent evader.
    Mixed,
    /// Even self-play: learner the same size as the field, opponents drawn from the
    /// heuristic + past snapshots. The end state — genuine league play.
    EvenSelfPlay,
}

impl Stage {
    /// Build the arena world for this stage. Seat 0 is the learner.
    pub fn build_world(self, cfg: WorldConfig, seed: u64) -> World {
        match self {
            Stage::OversizedVsPrey => prey_cluster_world(cfg, seed, 220.0, START_LENGTH),
            Stage::Mixed => prey_cluster_world(cfg, seed, 140.0, START_LENGTH + 40.0),
            Stage::EvenSelfPlay => {
                // Symmetric spawn: every worm the same modest size, scattered.
                World::new(
                    seed,
                    WorldConfig {
                        seat0_length: START_LENGTH + 30.0,
                        prey_jitter: 40.0,
                        ..cfg
                    },
                )
            }
        }
    }

    /// Which pool entries this stage will draw opponents from, as a PFSP sample.
    /// Early stages bias toward prey/heuristic; the even stage opens to snapshots.
    pub fn sample_opponent(self, pool: &mut Pool, rng: &mut Rng) -> usize {
        match self {
            Stage::OversizedVsPrey => {
                // Mostly scripted prey, sometimes the heuristic, never a snapshot —
                // keep the foes genuinely smaller/weaker so encircling can land.
                pick_kind(
                    pool,
                    rng,
                    &[PoolKind::Prey, PoolKind::Heuristic],
                    &[0.7, 0.3],
                )
            }
            Stage::Mixed => pick_kind(
                pool,
                rng,
                &[PoolKind::Prey, PoolKind::Heuristic],
                &[0.4, 0.6],
            ),
            Stage::EvenSelfPlay => {
                // Reserve a fixed fraction of seats for the heuristic every
                // iteration. Pure PFSP drops it once the learner beats it (its
                // `p*(1-p)` weight collapses as p→1), and the learner then trains
                // only against its own snapshots and *forgets how to beat the
                // heuristic* — the regression we saw (0.88→0.66). A guaranteed
                // floor keeps the gating opponent always present so winrate-vs-
                // heuristic climbs-or-plateaus instead of regressing.
                if rng.unit() < HEURISTIC_FLOOR {
                    pool.entries
                        .iter()
                        .position(|e| e.kind == PoolKind::Heuristic)
                        .unwrap_or(0)
                } else {
                    pool.sample()
                }
            }
        }
    }
}

/// Sample one of the listed pool kinds by the given (unnormalized) probabilities,
/// returning its pool index. Falls back to the heuristic entry if a kind is absent.
fn pick_kind(pool: &Pool, rng: &mut Rng, kinds: &[PoolKind], probs: &[f32]) -> usize {
    let total: f32 = probs.iter().sum();
    let mut r = rng.unit() * total;
    let mut chosen = kinds[0];
    for (k, p) in kinds.iter().zip(probs) {
        r -= p;
        if r <= 0.0 {
            chosen = *k;
            break;
        }
    }
    pool.entries
        .iter()
        .position(|e| e.kind == chosen)
        .or_else(|| {
            pool.entries
                .iter()
                .position(|e| e.kind == PoolKind::Heuristic)
        })
        .unwrap_or(0)
}
