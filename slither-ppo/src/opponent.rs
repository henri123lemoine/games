//! The self-play opponent pool (PFSP-lite). Every non-learner seat in every arena
//! is filled by an [`Opponent`] drawn from the pool. Two kinds:
//!
//!   * **Scripted** — the hand-coded encircle [`Heuristic`] (the competent
//!     teacher the pool is seeded with), a fleeing-prey scripted policy for the
//!     opening curriculum, and plain random. These read the [`World`] directly and
//!     emit an [`Action`] on the CPU, per arena, so they cost nothing on the GPU.
//!   * **Neural** — a frozen past checkpoint of the learner. Its observations are
//!     batched across all arenas using it and run through its net in one forward
//!     pass, exactly like the learner, so a snapshot opponent is as cheap as the
//!     learner itself regardless of how many arenas it sits in.
//!
//! PFSP-lite: each pool entry tracks a win-rate against the current learner and is
//! sampled with weight emphasizing near-even matchups (the opponents that teach
//! most), so the learner spends its games against beatable-but-not-trivial foes —
//! the AlphaStar prioritized-fictitious-self-play idea, trimmed to a single main
//! agent.

use slither_rl::env::{Action, TURN_BUCKETS};
use slither_rl::geometry::angle_diff;
use slither_rl::world::World;
use slither_rl::{Heuristic, Rng, random_action};

use tch::nn;

use crate::net::Policy;

/// A scripted opponent's behavior; cheap, CPU-side, reads the world directly.
pub enum Scripted {
    /// The hand-coded encircle predator — the pool's teacher.
    Heuristic(Box<Heuristic>),
    /// Flee directly away from the nearest other worm. The opening curriculum's
    /// small prey: something for an oversized learner to learn to run down and
    /// wall off.
    Prey,
    /// Uniform random turns, occasional boost. The floor the learner must crush.
    Random(Rng),
}

impl Scripted {
    pub fn act(&mut self, world: &World, idx: usize) -> Action {
        match self {
            Scripted::Heuristic(h) => h.act(world, idx),
            Scripted::Prey => flee_action(world, idx),
            Scripted::Random(rng) => random_action(rng),
        }
    }
}

/// A frozen neural opponent: a past checkpoint of the learner, run in inference.
pub struct Neural {
    /// Owns the snapshot's weights; `policy` borrows from this store, so it must
    /// outlive every forward. Held, not read.
    #[allow(dead_code)]
    pub vs: nn::VarStore,
    pub policy: Policy,
    /// Which training iteration produced it (for logs / pool identity).
    pub iter: u64,
}

/// One pool member as it is assigned to a seat in an arena. Scripted members are
/// instantiated per seat (they carry per-arena RNG / heuristic state); neural
/// members are referenced by their pool index so their obs can be batched.
pub enum SeatPolicy {
    Scripted(Scripted),
    /// Index into [`Pool::neural`]; the rollout batches all seats sharing an index.
    Neural(usize),
}

/// Identity of a pool entry, used for win-rate bookkeeping and sampling weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolKind {
    Heuristic,
    Prey,
    Random,
    /// A past checkpoint, identified by its training iteration.
    Snapshot(u64),
}

/// A pool entry plus the running win-rate of the *learner* against it. PFSP weights
/// are derived from this win-rate so the sampler favors instructive matchups.
pub struct PoolEntry {
    pub kind: PoolKind,
    /// Learner wins / games played against this entry (smoothed; see `update`).
    pub learner_winrate: f32,
    pub games: u64,
}

impl PoolEntry {
    fn new(kind: PoolKind) -> Self {
        Self {
            kind,
            learner_winrate: 0.5,
            games: 0,
        }
    }

    /// Fold one matchup outcome into the running win-rate with an EMA so the weight
    /// tracks the *current* learner, not its whole history.
    pub fn update(&mut self, learner_won: bool) {
        let x = if learner_won { 1.0 } else { 0.0 };
        let alpha = 0.05;
        self.learner_winrate = (1.0 - alpha) * self.learner_winrate + alpha * x;
        self.games += 1;
    }

    /// PFSP weight: a matchup the learner wins ~half the time teaches most; near-0
    /// (hopeless) and near-1 (trivially won) teach least. `p*(1-p)` peaks at 0.5.
    /// A floor keeps every opponent in rotation so win-rates stay fresh.
    fn weight(&self) -> f32 {
        let p = self.learner_winrate;
        0.05 + p * (1.0 - p)
    }
}

/// The opponent pool: scripted teachers (always present) plus a growing set of
/// frozen learner snapshots. Holds the neural snapshots' VarStores so their
/// forward passes run during rollout.
pub struct Pool {
    pub entries: Vec<PoolEntry>,
    /// Neural snapshots, parallel to the `Snapshot` entries; a [`SeatPolicy::Neural`]
    /// indexes here.
    pub neural: Vec<Neural>,
    rng: Rng,
}

impl Pool {
    /// Seed with the scripted teachers. `with_prey` adds the fleeing-prey policy
    /// (used in the opening oversized curriculum); `with_random` adds the random
    /// floor.
    pub fn seeded(seed: u64, with_prey: bool, with_random: bool) -> Self {
        let mut entries = vec![PoolEntry::new(PoolKind::Heuristic)];
        if with_prey {
            entries.push(PoolEntry::new(PoolKind::Prey));
        }
        if with_random {
            entries.push(PoolEntry::new(PoolKind::Random));
        }
        Self {
            entries,
            neural: Vec::new(),
            rng: Rng::new(seed),
        }
    }

    /// Add a frozen snapshot of the current learner to the pool.
    pub fn add_snapshot(&mut self, neural: Neural) {
        self.entries
            .push(PoolEntry::new(PoolKind::Snapshot(neural.iter)));
        self.neural.push(neural);
    }

    /// PFSP-sample a pool entry index, weighted toward near-even matchups.
    pub fn sample(&mut self) -> usize {
        let weights: Vec<f32> = self.entries.iter().map(PoolEntry::weight).collect();
        let total: f32 = weights.iter().sum();
        let mut r = self.rng.unit() * total;
        for (i, w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return i;
            }
        }
        weights.len() - 1
    }

    /// Map a sampled pool index plus a per-seat RNG seed to a concrete seat policy.
    /// Scripted entries are freshly instantiated (per-seat state); a snapshot entry
    /// resolves to the matching neural index.
    pub fn instantiate(&self, pool_idx: usize, seat_seed: u64) -> SeatPolicy {
        match self.entries[pool_idx].kind {
            PoolKind::Heuristic => {
                SeatPolicy::Scripted(Scripted::Heuristic(Box::new(Heuristic::new(seat_seed))))
            }
            PoolKind::Prey => SeatPolicy::Scripted(Scripted::Prey),
            PoolKind::Random => SeatPolicy::Scripted(Scripted::Random(Rng::new(seat_seed))),
            PoolKind::Snapshot(iter) => {
                let ni = self
                    .neural
                    .iter()
                    .position(|n| n.iter == iter)
                    .expect("snapshot entry without matching neural");
                SeatPolicy::Neural(ni)
            }
        }
    }
}

/// Aim worm `idx` directly away from the nearest other living worm (the prey
/// policy). Falls back to straight-ahead when alone.
fn flee_action(world: &World, idx: usize) -> Action {
    let me = &world.worms[idx];
    let head = me.head();
    let mut nearest: Option<usize> = None;
    let mut best = f32::MAX;
    for (j, w) in world.worms.iter().enumerate() {
        if j == idx || w.dead {
            continue;
        }
        let d = head.dist2(w.head());
        if d < best {
            best = d;
            nearest = Some(j);
        }
    }
    let Some(j) = nearest else {
        return Action::default();
    };
    let t = world.worms[j].head();
    let away = (head.y - t.y).atan2(head.x - t.x);
    aim_to_action(me.angle, away)
}

/// Quantize a desired absolute heading into the nearest turn bucket — the same
/// mapping the heuristic uses, so scripted policies share the learner's action
/// space.
fn aim_to_action(current: f32, aim: f32) -> Action {
    let mid = (TURN_BUCKETS / 2) as i32;
    let max_turn = 1.2f32;
    let d = angle_diff(current, aim).clamp(-max_turn, max_turn);
    let bucket = (mid as f32 + (d / max_turn) * mid as f32).round() as i32;
    Action {
        turn: bucket.clamp(0, (TURN_BUCKETS - 1) as i32) as u8,
        boost: false,
    }
}
