//! The self-play curriculum: a small ladder of stages that decides, per arena
//! reset, how the world is built (learner size vs prey size) and which pool
//! opponents are eligible. The blueprint's recipe — start the learner OVERSIZED
//! against small scripted prey (encircling only pays when you're bigger), then ramp
//! to mixed-size even self-play against the heuristic and past snapshots.
//!
//! Stages don't decay the encircle shaping; that anneal is the trainer's job,
//! keyed off real kills appearing. Here we only move the *matchup* distribution.

use slither_rl::Rng;
use slither_rl::geometry::Vec2;
use slither_rl::world::{START_LENGTH, WORLD, World, WorldConfig, Worm};

use crate::opponent::{Pool, PoolKind};
use crate::rollout::prey_cluster_world;

/// Fraction of even-self-play seats forced to the heuristic regardless of its
/// PFSP weight, so the gating opponent never falls out of the pool.
const HEURISTIC_FLOOR: f32 = 0.35;

/// Fraction of even-self-play *arenas* that spawn as a tight equal-size cluster
/// against a wall (the encircle/cut-off practice geometry) rather than scattered.
const CLOSE_ENCOUNTER_FRAC: f32 = 0.4;

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
                let even_cfg = WorldConfig {
                    seat0_length: START_LENGTH + 30.0,
                    prey_jitter: 40.0,
                    ..cfg
                };
                // A fraction of arenas spawn as a tight, equal-size cluster pinned
                // against a wall instead of scattered open-field. In the open field
                // the learner can always just run and out-grow, so the cut-off
                // never has to fire; a close equal-size encounter with a wall to
                // pin against is where encircling is geometrically on the table and
                // the policy can *discover* the finisher. The rest stay scattered so
                // it doesn't overfit to the spawn geometry — real league play too.
                if seed_unit(seed) < CLOSE_ENCOUNTER_FRAC {
                    close_encounter_world(even_cfg, seed)
                } else {
                    World::new(seed, even_cfg)
                }
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

/// A deterministic `[0,1)` value from a seed, for per-arena spawn-variant choice
/// without threading an extra RNG through `build_world`.
fn seed_unit(seed: u64) -> f32 {
    let mut z = seed.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^= z >> 31;
    (z >> 40) as f32 / (1u64 << 24) as f32
}

/// A tight equal-size cluster pinned against a randomly chosen wall: every worm
/// the same length (`seat0_length`), packed within a small radius near the wall
/// with headings roughly along it. Close enough that a cut-off is immediately on
/// the table and the wall is a surface to pin a foe against — the encircle
/// practice geometry the open field never forces.
fn close_encounter_world(cfg: WorldConfig, seed: u64) -> World {
    let mut rng = Rng::new(seed ^ 0xE7C1_2C1E);
    let len = cfg.seat0_length;
    let margin = 360.0;

    // Pick a wall and a center anchored near it; the wall-parallel axis spreads
    // the cluster, the inward axis sets distance from the wall.
    let along = rng.range(margin, WORLD - margin);
    let inward = rng.range(margin, margin + 420.0);
    let (center, base_heading) = match rng.below(4) {
        0 => (Vec2::new(inward, along), 0.0), // left wall, face right
        1 => (Vec2::new(WORLD - inward, along), std::f32::consts::PI), // right wall, face left
        2 => (Vec2::new(along, inward), std::f32::consts::FRAC_PI_2), // top wall, face down
        _ => (
            Vec2::new(along, WORLD - inward),
            -std::f32::consts::FRAC_PI_2,
        ), // bottom, face up
    };

    let cluster_r = 260.0;
    let mut worms = Vec::with_capacity(cfg.worms);
    for _ in 0..cfg.worms {
        let ang = rng.range(0.0, std::f32::consts::TAU);
        let r = rng.range(60.0, cluster_r);
        let pos = Vec2::new(
            (center.x + ang.cos() * r).clamp(20.0, WORLD - 20.0),
            (center.y + ang.sin() * r).clamp(20.0, WORLD - 20.0),
        );
        // Headings roughly along the wall (±) so worms cross paths quickly rather
        // than all driving straight into the wall.
        let heading = base_heading + rng.range(-1.0, 1.0);
        worms.push(Worm::spawn(pos, heading, len));
    }
    World::from_worms(seed, worms, cfg.pellet_target)
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
