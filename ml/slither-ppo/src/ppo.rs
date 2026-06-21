//! Slither's thin adapter over `ppo-core`: the PPO math (GAE(λ), clipped surrogate,
//! value clipping, entropy, per-minibatch advantage normalization, gradient
//! clipping, the minibatch/epoch loop) lives in [`ppo_core`]; this module only
//! supplies the env-specific pieces — turning the slither [`Transition`] buffer into
//! the core's per-step `(reward, value, done, log_prob)` view, and a [`Policy`]
//! adapter that packs the obs/action minibatch and runs the tch forward.
//!
//! The transition buffer is a flat `T*N` block in step-major order (step 0 for all
//! arenas, then step 1, …). Done flags mark episode boundaries; the env auto-resets
//! a dead learner, so a `done` at step t means the value bootstrap from t+1 is cut.

use tch::{Device, Tensor, nn};

use ppo_core::{AdvNorm, Minibatch, Step};

use crate::net::Policy;
use crate::obs_batch;
use crate::rollout::Transition;

pub use ppo_core::UpdateStats;

/// Slither's PPO knobs. The cleanrl-faithful configuration (clipped value loss,
/// per-minibatch advantage normalization, shuffled minibatches, no BC anchor) is
/// fixed in [`PpoConfig::to_core`]; only the tunable hyperparameters are fields.
pub struct PpoConfig {
    pub gamma: f32,
    pub lambda: f32,
    pub clip: f64,
    pub value_coef: f64,
    pub entropy_coef: f64,
    pub max_grad_norm: f64,
    pub epochs: usize,
    pub minibatches: usize,
    /// `T`: rollout length (steps per arena). `N` = buffer_len / T is the arena
    /// count; the trainer guarantees the buffer is exactly `T*N` step-major.
    pub steps: usize,
}

impl PpoConfig {
    fn to_core(&self) -> ppo_core::PpoConfig {
        ppo_core::PpoConfig {
            gamma: self.gamma,
            lambda: self.lambda,
            clip: self.clip,
            value_coef: self.value_coef,
            entropy_coef: self.entropy_coef,
            max_grad_norm: self.max_grad_norm,
            epochs: self.epochs,
            steps: self.steps,
            value_clip: true,
            adv_norm: AdvNorm::PerMinibatch,
            minibatch: Minibatch::Shuffled {
                count: self.minibatches,
            },
            bc_anchor: None,
        }
    }
}

/// Per-transition `(reward, value, done)` for the core's GAE, in the buffer's
/// step-major order.
fn steps(buf: &[Transition]) -> Vec<Step> {
    buf.iter()
        .map(|t| Step {
            reward: t.reward,
            value: t.value,
            done: t.done,
        })
        .collect()
}

/// Wraps the learner net plus the rollout's obs/action minibatch, pre-packed once
/// onto the device. [`ppo_core::update`] hands it transition indices; it
/// `index_select`s the stored rows and runs the tch forward to produce the
/// log-prob/entropy/value the surrogate needs.
struct LearnerAdapter<'a> {
    policy: &'a Policy,
    grid_all: Tensor,
    scalars_all: Tensor,
    turn_all: Tensor,
    boost_all: Tensor,
}

impl ppo_core::Policy for LearnerAdapter<'_> {
    fn evaluate(&self, idx: &Tensor) -> ppo_core::Eval {
        let grid = self.grid_all.index_select(0, idx);
        let scalars = self.scalars_all.index_select(0, idx);
        let turn = self.turn_all.index_select(0, idx);
        let boost = self.boost_all.index_select(0, idx);
        let ev = self.policy.evaluate(&grid, &scalars, &turn, &boost);
        ppo_core::Eval {
            log_prob: ev.log_prob,
            entropy: ev.entropy,
            value: ev.value,
        }
    }
}

/// Run the PPO update over one rollout. Mutates the policy via `opt`. Returns
/// averaged diagnostics. `bootstrap_values` comes from a value-only forward on the
/// post-rollout observations (the trainer computes it).
pub fn update(
    policy: &Policy,
    opt: &mut nn::Optimizer,
    device: Device,
    buf: &[Transition],
    bootstrap_values: &[f32],
    cfg: &PpoConfig,
) -> UpdateStats {
    // Pack the obs/action minibatch once for the whole update; the adapter indexes
    // it per minibatch (the same one-pack-then-index_select pattern as before).
    let obs: Vec<_> = buf.iter().map(|t| t.obs.clone()).collect();
    let (grid_all, scalars_all) = obs_batch::pack(&obs, device);
    let turn_all =
        Tensor::from_slice(&buf.iter().map(|t| t.turn).collect::<Vec<_>>()).to_device(device);
    let boost_all =
        Tensor::from_slice(&buf.iter().map(|t| t.boost).collect::<Vec<_>>()).to_device(device);
    let adapter = LearnerAdapter {
        policy,
        grid_all,
        scalars_all,
        turn_all,
        boost_all,
    };

    let old_log_prob: Vec<f32> = buf.iter().map(|t| t.log_prob).collect();

    ppo_core::update(
        &adapter,
        opt,
        device,
        &steps(buf),
        &old_log_prob,
        bootstrap_values,
        &cfg.to_core(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ppo_core::gae;
    use slither_rl::obs::Obs;

    fn tr(reward: f32, value: f32, done: bool) -> Transition {
        Transition {
            obs: Obs::zeros(),
            turn: 0,
            boost: 0,
            log_prob: 0.0,
            value,
            reward,
            done,
        }
    }

    /// GAE on a single arena with no episode boundary reduces to the textbook
    /// discounted-TD-residual sum; check it against a hand-rolled reference. Runs
    /// through the slither `Transition` -> `ppo_core::Step` adapter and the core
    /// GAE, guarding the wiring the trainer relies on.
    #[test]
    fn gae_matches_reference_single_arena() {
        let gamma = 0.99;
        let lambda = 0.95;
        let buf = vec![
            tr(1.0, 0.5, false),
            tr(0.0, 0.4, false),
            tr(2.0, 0.6, false),
        ];
        let boot = vec![0.7f32];
        let (adv, ret) = gae(&steps(&buf), &boot, buf.len(), gamma, lambda);

        // Reference: deltas then the GAE recursion, T=3, N=1.
        let v: Vec<f32> = buf.iter().map(|t| t.value).collect();
        let r: Vec<f32> = buf.iter().map(|t| t.reward).collect();
        let next = [v[1], v[2], boot[0]];
        let delta: Vec<f32> = (0..3).map(|t| r[t] + gamma * next[t] - v[t]).collect();
        let mut ref_adv = [0.0f32; 3];
        let mut acc = 0.0;
        for t in (0..3).rev() {
            acc = delta[t] + gamma * lambda * acc;
            ref_adv[t] = acc;
        }
        for t in 0..3 {
            assert!(
                (adv[t] - ref_adv[t]).abs() < 1e-5,
                "adv[{t}] {} vs {}",
                adv[t],
                ref_adv[t]
            );
            assert!((ret[t] - (ref_adv[t] + v[t])).abs() < 1e-5);
        }
    }

    /// A `done` flag must cut the bootstrap and restart the GAE recursion: the
    /// advantage at a terminal step is exactly its own TD residual with no future.
    #[test]
    fn gae_done_cuts_bootstrap() {
        let gamma = 0.99;
        let lambda = 0.95;
        // Two steps, single arena; step 0 is terminal (learner died, arena reset).
        let buf = vec![tr(3.0, 1.0, true), tr(0.5, 0.2, false)];
        let boot = vec![0.9f32];
        let (adv, _ret) = gae(&steps(&buf), &boot, buf.len(), gamma, lambda);
        // Step 0 terminal: delta = r - v (mask zeroes the next-value), and the
        // recursion from step 1 cannot leak back because mask=0 at step 0.
        let expected0 = 3.0 - 1.0;
        assert!((adv[0] - expected0).abs() < 1e-5, "got {}", adv[0]);
    }

    /// Step-major indexing: two arenas interleaved must not bleed advantage across
    /// arena boundaries.
    #[test]
    fn gae_two_arenas_independent() {
        let gamma = 0.99;
        let lambda = 0.95;
        // T=2, N=2, step-major: [s0a0, s0a1, s1a0, s1a1].
        let buf = vec![
            tr(1.0, 0.0, false), // arena 0, step 0
            tr(5.0, 0.0, false), // arena 1, step 0
            tr(1.0, 0.0, false), // arena 0, step 1
            tr(5.0, 0.0, false), // arena 1, step 1
        ];
        let boot = vec![0.0f32, 0.0f32];
        let (adv, _) = gae(&steps(&buf), &boot, 2, gamma, lambda);
        // Arena 0 sees only 1.0 rewards, arena 1 only 5.0 — a 5x gap at every step.
        assert!((adv[1] / adv[0] - 5.0).abs() < 1e-4);
        assert!((adv[3] / adv[2] - 5.0).abs() < 1e-4);
    }
}
