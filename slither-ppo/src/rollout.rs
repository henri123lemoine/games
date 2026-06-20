//! Vectorized self-play rollout over many parallel [`Env`] arenas, cleanrl-style:
//! every arena steps in lockstep, a dead learner auto-resets its arena mid-rollout
//! so the trajectory stays a fixed `T × N` block, and only the learner seat's
//! transitions are recorded for PPO. Opponent seats are filled from the
//! [`Pool`]; scripted opponents act CPU-side, neural snapshots are batched.
//!
//! The learner sits in seat 0 of every arena. The collector does NOT own the
//! learner net — the trainer drives the forward passes (learner and each neural
//! opponent) so all tch work stays on the trainer's thread and one GPU stream. The
//! collector hands out the observation batches that need a forward and takes back
//! the sampled actions.

use rayon::prelude::*;

use slither_rl::env::{Action, Env, StepOut};
use slither_rl::obs::Obs;
use slither_rl::world::{World, WorldConfig, Worm};
use slither_rl::{Rng, geometry::Vec2};

use crate::curriculum::Stage;
use crate::opponent::{Pool, SeatPolicy};

/// One arena's result from a parallel rollout step: the learner transition plus
/// the side data the serial fold needs.
struct ArenaStep {
    transition: Transition,
    kills: u64,
    /// `Some((learner_won, distinct_pool_indices_faced))` if the episode ended.
    outcome: Option<(bool, Vec<usize>)>,
}

/// One recorded learner transition (seat 0 of one arena at one step). Stored flat;
/// the trainer reshapes into `[T, N]` for GAE.
#[derive(Clone)]
pub struct Transition {
    pub obs: Obs,
    pub turn: i64,
    pub boost: i64,
    pub log_prob: f32,
    pub value: f32,
    pub reward: f32,
    /// True if seat 0 of this arena terminated *this* step (death) — a GAE episode
    /// boundary. The arena is reset for the next step.
    pub done: bool,
}

/// Per-arena live state during a rollout.
struct Arena {
    env: Env,
    seats: Vec<SeatPolicy>,
    /// Pool index each non-learner seat was drawn from (for win-rate updates).
    seat_pool_idx: Vec<usize>,
    /// Last observation for every worm (seat-major), refreshed each step. Seat 0 is
    /// the learner's next-input obs.
    last_obs: Vec<Obs>,
    /// Learner kills so far this episode (for the match outcome vs the opponent).
    learner_kills: u32,
}

/// The rollout collector: owns the arenas and their opponent assignments, but not
/// the nets. Driven step-by-step by the trainer.
pub struct Collector {
    arenas: Vec<Arena>,
    cfg: WorldConfig,
    stage: Stage,
    rng: Rng,
    seed_ctr: u64,
    /// Outcome ledger this rollout: (pool_idx, learner_won) appended whenever a
    /// learner-vs-opponent episode ends. Drained by the trainer to update PFSP
    /// win-rates.
    pub outcomes: Vec<(usize, bool)>,
    /// Learner kills accumulated across all arenas this rollout — the honest kill
    /// count, summed straight from `StepOut::kills[0]`, that drives the shaping
    /// anneal and the headline metric.
    pub learner_kills: u64,
}

/// What the trainer must forward this step: the learner's obs batch (seat 0 of
/// every arena, in arena order) and, for each neural opponent pool-neural-index,
/// the obs of every seat using it (with back-references so actions route home).
pub struct StepInputs {
    pub learner_obs: Vec<Obs>,
    /// One bucket per neural snapshot index actually in play this step.
    pub neural: Vec<NeuralBatch>,
}

pub struct NeuralBatch {
    /// Index into [`Pool::neural`].
    pub neural_idx: usize,
    pub obs: Vec<Obs>,
    /// `(arena, seat)` each row routes back to.
    pub routes: Vec<(usize, usize)>,
}

/// Actions the trainer produces for the learner seats (one per arena, arena order)
/// plus the sampled extras PPO will store.
pub struct LearnerActions {
    pub turn: Vec<i64>,
    pub boost: Vec<i64>,
    pub log_prob: Vec<f32>,
    pub value: Vec<f32>,
}

impl Collector {
    pub fn new(
        num_arenas: usize,
        cfg: WorldConfig,
        stage: Stage,
        pool: &mut Pool,
        seed: u64,
    ) -> Self {
        let mut rng = Rng::new(seed);
        let mut seed_ctr = seed.wrapping_mul(0x9e3779b9).wrapping_add(1);
        let arenas = (0..num_arenas)
            .map(|_| {
                let s = seed_ctr;
                seed_ctr = seed_ctr.wrapping_add(0x1000193);
                Arena::new(cfg, stage, pool, &mut rng, s)
            })
            .collect();
        Self {
            arenas,
            cfg,
            stage,
            rng,
            seed_ctr,
            outcomes: Vec::new(),
            learner_kills: 0,
        }
    }

    /// Switch the curriculum stage for arenas reset from now on (live arenas keep
    /// their stage until they reset).
    pub fn set_stage(&mut self, stage: Stage) {
        self.stage = stage;
    }

    /// Gather the observation batches that need a net forward this step.
    pub fn step_inputs(&self, pool: &Pool) -> StepInputs {
        let learner_obs: Vec<Obs> = self.arenas.iter().map(|a| a.last_obs[0].clone()).collect();

        let mut buckets: Vec<NeuralBatch> = Vec::new();
        for (ai, a) in self.arenas.iter().enumerate() {
            for (seat, sp) in a.seats.iter().enumerate() {
                if let SeatPolicy::Neural(ni) = sp {
                    // Skip dead seats: a dead worm doesn't act and its obs are zero.
                    if a.env.world().worms[seat].dead {
                        continue;
                    }
                    let bucket = match buckets.iter_mut().find(|b| b.neural_idx == *ni) {
                        Some(b) => b,
                        None => {
                            buckets.push(NeuralBatch {
                                neural_idx: *ni,
                                obs: Vec::new(),
                                routes: Vec::new(),
                            });
                            buckets.last_mut().unwrap()
                        }
                    };
                    bucket.obs.push(a.last_obs[seat].clone());
                    bucket.routes.push((ai, seat));
                }
            }
        }
        let _ = pool;
        StepInputs {
            learner_obs,
            neural: buckets,
        }
    }

    /// Apply the learner actions and the neural-opponent actions, run the scripted
    /// opponents, step every arena, record learner transitions, and auto-reset any
    /// arena whose learner died. `neural_actions[k]` parallels `inputs.neural[k]`'s
    /// rows: `(turn, boost)` per route.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        pool: &mut Pool,
        inputs: &StepInputs,
        learner: &LearnerActions,
        neural_actions: &[(Vec<i64>, Vec<i64>)],
        out: &mut Vec<Transition>,
        shaping: f32,
    ) {
        let n = self.arenas.len();

        // Scatter neural-opponent actions into a per-arena seat->Action map.
        let mut neural_seat_action: Vec<std::collections::HashMap<usize, Action>> =
            vec![std::collections::HashMap::new(); n];
        for (bucket, (turns, boosts)) in inputs.neural.iter().zip(neural_actions) {
            for (row, &(ai, seat)) in bucket.routes.iter().enumerate() {
                neural_seat_action[ai].insert(
                    seat,
                    Action {
                        turn: turns[row] as u8,
                        boost: boosts[row] != 0,
                    },
                );
            }
        }

        // Step every arena in parallel — the env step plus the CPU-side scripted
        // opponents are independent across arenas (the throughput-critical part of
        // the rollout). Each arena returns its learner transition and the side
        // data the serial fold needs (kills, episode outcome).
        let learner = &learner;
        let neural_seat_action = &neural_seat_action;
        let results: Vec<ArenaStep> = self
            .arenas
            .par_iter_mut()
            .enumerate()
            .map(|(ai, a)| {
                a.env.set_shaping(shaping);

                let mut actions: Vec<Action> = Vec::with_capacity(a.seats.len());
                actions.push(Action {
                    turn: learner.turn[ai] as u8,
                    boost: learner.boost[ai] != 0,
                });
                for (seat, sp) in a.seats.iter_mut().enumerate().skip(1) {
                    let act = match sp {
                        SeatPolicy::Scripted(s) => s.act(a.env.world(), seat),
                        SeatPolicy::Neural(_) => neural_seat_action[ai]
                            .get(&seat)
                            .copied()
                            .unwrap_or_default(),
                    };
                    actions.push(act);
                }

                let StepOut {
                    obs,
                    reward,
                    done,
                    kills,
                    ..
                } = a.env.step(&actions);

                a.learner_kills += kills[0];

                let transition = Transition {
                    obs: a.last_obs[0].clone(),
                    turn: learner.turn[ai],
                    boost: learner.boost[ai],
                    log_prob: learner.log_prob[ai],
                    value: learner.value[ai],
                    reward: reward[0],
                    done: done[0],
                };
                a.last_obs = obs;

                // On the learner's death, settle the match against each distinct
                // opponent it faced (seat 0's `usize::MAX` placeholder excluded).
                let outcome = if done[0] {
                    let learner_won = learner_outcome(a);
                    let mut faced: Vec<usize> = a
                        .seat_pool_idx
                        .iter()
                        .copied()
                        .filter(|&pi| pi != usize::MAX)
                        .collect();
                    faced.sort_unstable();
                    faced.dedup();
                    Some((learner_won, faced))
                } else {
                    None
                };

                ArenaStep {
                    transition,
                    kills: kills[0] as u64,
                    outcome,
                }
            })
            .collect();

        let mut resets: Vec<usize> = Vec::new();
        for (ai, r) in results.into_iter().enumerate() {
            self.learner_kills += r.kills;
            if let Some((won, faced)) = r.outcome {
                for pi in faced {
                    self.outcomes.push((pi, won));
                }
                resets.push(ai);
            }
            out.push(r.transition);
        }

        for ai in resets {
            let s = self.seed_ctr;
            self.seed_ctr = self.seed_ctr.wrapping_add(0x1000193);
            self.arenas[ai] = Arena::new(self.cfg, self.stage, pool, &mut self.rng, s);
        }
    }

    /// Bootstrap value input: seat-0 obs of every arena after the last step, for
    /// the GAE tail. (Dead-and-reset arenas carry a fresh obs, but their last
    /// transition was `done`, so the bootstrap is masked out anyway.)
    pub fn bootstrap_obs(&self) -> Vec<Obs> {
        self.arenas.iter().map(|a| a.last_obs[0].clone()).collect()
    }
}

/// Did the learner "win" the episode it just ended? In slither there is no formal
/// win; we score the learner ahead if it got at least one kill, or outlived /
/// out-grew where no kills happened. A kill is the decisive predatory outcome the
/// whole project is about, so it dominates.
fn learner_outcome(a: &Arena) -> bool {
    if a.learner_kills > 0 {
        return true;
    }
    // No kill: count it a loss unless the learner is still the biggest thing alive
    // (it survived and dominated on length), which is a weak but non-zero signal.
    let me_len = a.env.world().worms[0].length;
    let biggest_foe = a
        .env
        .world()
        .worms
        .iter()
        .enumerate()
        .filter(|(j, w)| *j != 0 && !w.dead)
        .map(|(_, w)| w.length)
        .fold(0.0f32, f32::max);
    me_len > biggest_foe * 1.2
}

impl Arena {
    fn new(cfg: WorldConfig, stage: Stage, pool: &mut Pool, rng: &mut Rng, seed: u64) -> Self {
        let world = stage.build_world(cfg, seed);
        let mut env = Env::new(cfg);
        let n = world.worms.len();

        // Sample one pool entry per non-learner seat. Drawing per seat (not per
        // arena) lets one arena mix a heuristic and a snapshot, widening the
        // experience the AlphaStar way.
        let mut seats: Vec<SeatPolicy> = Vec::with_capacity(n);
        let mut seat_pool_idx: Vec<usize> = Vec::with_capacity(n);
        // Seat 0 placeholder (the learner; never read from `seats`).
        seats.push(SeatPolicy::Scripted(crate::opponent::Scripted::Prey));
        seat_pool_idx.push(usize::MAX);
        for seat in 1..n {
            let pi = stage.sample_opponent(pool, rng);
            let seat_seed = seed
                .wrapping_mul(0x100000001b3)
                .wrapping_add(seat as u64 * 0x9e3779b9);
            seats.push(pool.instantiate(pi, seat_seed));
            seat_pool_idx.push(pi);
        }

        let last_obs = env.reset_world(world);
        Self {
            env,
            seats,
            seat_pool_idx,
            last_obs,
            learner_kills: 0,
        }
    }
}

/// Build a world where seat 0 (learner) is freshly placed near a cluster of small
/// prey — used by the prey curriculum stage. Kept here because it needs the
/// concrete [`Worm`]/[`World`] construction the env exposes.
pub fn prey_cluster_world(cfg: WorldConfig, seed: u64, predator_len: f32, prey_len: f32) -> World {
    let mut rng = Rng::new(seed ^ 0xCAFE);
    let margin = 500.0;
    let center = Vec2::new(
        rng.range(margin, slither_rl::world::WORLD - margin),
        rng.range(margin, slither_rl::world::WORLD - margin),
    );
    let mut worms = Vec::with_capacity(cfg.worms);
    worms.push(Worm::spawn(
        center,
        rng.range(0.0, std::f32::consts::TAU),
        predator_len,
    ));
    for _ in 1..cfg.worms {
        let ang = rng.range(0.0, std::f32::consts::TAU);
        let r = rng.range(200.0, 600.0);
        let pos = Vec2::new(
            (center.x + ang.cos() * r).clamp(20.0, slither_rl::world::WORLD - 20.0),
            (center.y + ang.sin() * r).clamp(20.0, slither_rl::world::WORLD - 20.0),
        );
        worms.push(Worm::spawn(
            pos,
            rng.range(0.0, std::f32::consts::TAU),
            prey_len,
        ));
    }
    World::from_worms(seed, worms, cfg.pellet_target)
}
