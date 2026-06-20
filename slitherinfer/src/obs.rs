//! Drive the net straight from a [`slither_rl`] world, so the browser bot's
//! features are bit-identical to training. [`act`] builds worm `idx`'s
//! egocentric observation with [`slither_rl::obs::observe`] (the same call the
//! trainer uses), forwards it, and decodes greedy actions — argmax turn bucket,
//! boost when its logit clears zero, matching `Policy::act_greedy`.

use slither_rl::env::{Action, TURN_BUCKETS};
use slither_rl::obs::{SEMANTIC_CHANNELS, observe};
use slither_rl::world::World;

use crate::Model;

/// Per-worm scratch the bot must persist across frames: the previous step's
/// 5-channel semantic grid, so the observation's delta channels are filled the
/// same way the env fills them. One per bot worm; zero-initialized at spawn.
pub struct ObsMemory {
    prev_semantic: Vec<f32>,
}

impl ObsMemory {
    pub fn new() -> Self {
        Self {
            prev_semantic: vec![0.0; SEMANTIC_CHANNELS * crate::GRID * crate::GRID],
        }
    }
}

impl Default for ObsMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// Greedy action for worm `idx` from the trained net: build its observation,
/// forward, take the argmax turn bucket and boost-if-`logit ≥ 0`. Mutates
/// `mem.prev_semantic` for next frame's delta channels.
pub fn act(model: &Model, world: &World, idx: usize, mem: &mut ObsMemory) -> Action {
    let obs = observe(world, idx, &mut mem.prev_semantic);
    let out = model.forward(&obs.grid, &obs.scalars);
    let turn = argmax(&out.turn_logits) as u8;
    debug_assert!((turn as usize) < TURN_BUCKETS);
    Action {
        turn,
        boost: out.boost_logit >= 0.0,
    }
}

fn argmax(xs: &[f32]) -> usize {
    let mut best = 0;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use slither_rl::world::{World, WorldConfig};

    /// `act` returns a legal action and advances the obs memory — a smoke check
    /// that the obs↔net wiring runs end to end against a real world.
    #[test]
    fn act_produces_a_legal_action() {
        // A zero net: every turn logit is 0, so argmax is bucket 0; boost logit
        // 0 ≥ 0 so boost is true. The point is that it runs and stays in range.
        let model = {
            // Reuse the test buffer builder by parsing a zero export.
            let bytes = zero_export();
            Model::parse(&bytes).expect("parse")
        };
        let world = World::new(7, WorldConfig::default());
        let mut mem = ObsMemory::new();
        let a = act(&model, &world, 1, &mut mem);
        assert_eq!(a.turn, 0, "all-equal turn logits → argmax bucket 0");
        assert!(a.boost, "boost logit 0 ≥ 0 → boost on");
    }

    /// With the *real* exported net, a large net-driven predator dropped next
    /// to small prey closes the distance — the hunting behavior the policy was
    /// trained for (it beats the encircle heuristic 0.88). Guards that the
    /// browser bot plays the trained policy, not noise. Skipped if the weights
    /// have not been exported yet (fresh clone before `slither-ppo export`).
    #[test]
    fn trained_predator_closes_on_prey() {
        use slither_rl::geometry::Vec2;
        use slither_rl::world::{WORLD, World, Worm, WormControl};

        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../web/app/public/slither/slither.weights"
        );
        let Ok(bytes) = std::fs::read(path) else {
            eprintln!("skipping: {path} not exported yet");
            return;
        };
        let model = Model::parse(&bytes).expect("parse real export");

        // Predator at seat 0 (big), prey just ahead of it (small), both clear of
        // walls. The predator is net-driven; the prey flees straight.
        let mid = WORLD * 0.5;
        let predator = Worm::spawn(Vec2::new(mid, mid), 0.0, 220.0);
        let prey = Worm::spawn(Vec2::new(mid + 360.0, mid + 40.0), 0.0, 24.0);
        let mut world = World::from_worms(1, vec![predator, prey], 200);
        let mut mem = ObsMemory::new();

        let dist = |w: &World| w.worms[0].head().dist(w.worms[1].head());
        let start = dist(&world);
        let mut closest = start;
        for _ in 0..200 {
            if world.worms[0].dead || world.worms[1].dead {
                break;
            }
            let pred = act(&model, &world, 0, &mut mem).control(world.worms[0].angle);
            // Prey flees straight ahead at base speed (no boost), a fixed target.
            let flee = WormControl {
                aim: world.worms[1].angle,
                boost: false,
            };
            world.step(&[pred, flee]);
            closest = closest.min(dist(&world));
        }
        // The predator should get meaningfully closer than it started (or eat the
        // prey outright). Equal top speed makes a clean kill hard without a wall
        // or boost edge (the policy/heuristic both struggle to convert the final
        // gap), so the bar is "it hunts" — it cuts the gap by a clear margin — not
        // "it always kills". A wandering net would not close at all.
        assert!(
            world.worms[1].dead || closest < start * 0.85,
            "predator did not close: start {start:.0}, closest {closest:.0}"
        );
    }

    fn zero_export() -> Vec<u8> {
        // Mirror lib's test `buf(0.0)` without exposing it: recompute lengths.
        use crate::{CHANNELS, GRID, SCALARS, TURN_BUCKETS};
        const KERNEL: usize = 3;
        const HIDDEN: usize = 256;
        let specs = [(CHANNELS, 32usize, 1usize), (32, 64, 2), (64, 64, 2)];
        let mut hw = GRID;
        let mut floats = 0usize;
        for (cin, cout, stride) in specs {
            floats += cout * cin * KERNEL * KERNEL + cout;
            hw = (hw + 2 * ((KERNEL - 1) / 2) - KERNEL) / stride + 1;
        }
        let conv_flat = 64 * hw * hw;
        floats += HIDDEN * (conv_flat + SCALARS) + HIDDEN;
        floats += TURN_BUCKETS * HIDDEN + TURN_BUCKETS;
        floats += HIDDEN + 1;
        floats += HIDDEN + 1;
        let mut b = Vec::new();
        b.extend_from_slice(b"SLNET1");
        b.extend_from_slice(&(CHANNELS as u32).to_le_bytes());
        b.extend_from_slice(&(GRID as u32).to_le_bytes());
        b.extend_from_slice(&(SCALARS as u32).to_le_bytes());
        for _ in 0..floats {
            b.extend_from_slice(&0.0f32.to_le_bytes());
        }
        b
    }
}
