//! Generalized Advantage Estimation, GAE(λ), over a step-major `[T, N]` rollout.
//!
//! The transition buffer is a flat `T*N` block in step-major order (step 0 for all
//! actors, then step 1, …) so the `[T, N]` reshape GAE needs is just a view. Done
//! flags mark episode boundaries; a `done` at step t means the value bootstrap from
//! t+1 must be cut and the recursion restarts (cleanrl's mask) — the collector is
//! expected to have auto-reset the dead actor, so the buffer stays a fixed block.
//!
//! The core works on plain `f32` slices, not a concrete transition type: each
//! adapter extracts `(reward, value, done)` per stored step and hands them here,
//! keeping env `Obs`/`Action` types out of the PPO math.

/// One stored step's GAE inputs, in the same step-major order as the rollout buffer.
#[derive(Clone, Copy, Debug)]
pub struct Step {
    pub reward: f32,
    /// V(s_t) recorded when the action was taken (the rollout's value estimate).
    pub value: f32,
    /// True if the actor terminated *this* step — a GAE episode boundary.
    pub done: bool,
}

/// Compute GAE(λ) advantages and returns over the step-major `[T, N]` buffer.
/// `bootstrap_values[n]` is V(s_T) for actor n (the value after the last stored
/// step). Returns `(advantages, returns)` flat in the same step-major order.
///
/// `steps` is `T`; `N = buf.len() / steps` is the actor count. The caller
/// guarantees the buffer is exactly `T*N` step-major.
pub fn gae(
    buf: &[Step],
    bootstrap_values: &[f32],
    steps: usize,
    gamma: f32,
    lambda: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = buf.len() / steps;
    let mut adv = vec![0.0f32; buf.len()];
    let idx = |t: usize, j: usize| t * n + j;

    for j in 0..n {
        let mut last_gae = 0.0f32;
        for t in (0..steps).rev() {
            let tr = &buf[idx(t, j)];
            let next_value = if t == steps - 1 {
                bootstrap_values[j]
            } else {
                buf[idx(t + 1, j)].value
            };
            // A `done` at t ends the episode: no bootstrap past it, and the GAE
            // recursion restarts (the env already reset the actor).
            let mask = if tr.done { 0.0 } else { 1.0 };
            let delta = tr.reward + gamma * next_value * mask - tr.value;
            last_gae = delta + gamma * lambda * mask * last_gae;
            adv[idx(t, j)] = last_gae;
        }
    }

    let returns: Vec<f32> = adv
        .iter()
        .zip(buf.iter())
        .map(|(a, tr)| a + tr.value)
        .collect();
    (adv, returns)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(reward: f32, value: f32, done: bool) -> Step {
        Step {
            reward,
            value,
            done,
        }
    }

    /// GAE on a single actor with no episode boundary reduces to the textbook
    /// discounted-TD-residual sum; check it against a hand-rolled reference.
    #[test]
    fn gae_matches_reference_single_arena() {
        let gamma = 0.99;
        let lambda = 0.95;
        let buf = vec![
            step(1.0, 0.5, false),
            step(0.0, 0.4, false),
            step(2.0, 0.6, false),
        ];
        let boot = vec![0.7f32];
        let (adv, ret) = gae(&buf, &boot, buf.len(), gamma, lambda);

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
        // Two steps, single actor; step 0 is terminal (learner died, arena reset).
        let buf = vec![step(3.0, 1.0, true), step(0.5, 0.2, false)];
        let boot = vec![0.9f32];
        let (adv, _ret) = gae(&buf, &boot, buf.len(), gamma, lambda);
        // Step 0 terminal: delta = r - v (mask zeroes the next-value), and the
        // recursion from step 1 cannot leak back because mask=0 at step 0.
        let expected0 = 3.0 - 1.0;
        assert!((adv[0] - expected0).abs() < 1e-5, "got {}", adv[0]);
    }

    /// Step-major indexing: two actors interleaved must not bleed advantage across
    /// actor boundaries.
    #[test]
    fn gae_two_arenas_independent() {
        let gamma = 0.99;
        let lambda = 0.95;
        // T=2, N=2, step-major: [s0a0, s0a1, s1a0, s1a1].
        let buf = vec![
            step(1.0, 0.0, false), // actor 0, step 0
            step(5.0, 0.0, false), // actor 1, step 0
            step(1.0, 0.0, false), // actor 0, step 1
            step(5.0, 0.0, false), // actor 1, step 1
        ];
        let boot = vec![0.0f32, 0.0f32];
        let (adv, _) = gae(&buf, &boot, 2, gamma, lambda);
        // Actor 0 sees only 1.0 rewards, actor 1 only 5.0 — a 5x gap at every step.
        assert!((adv[1] / adv[0] - 5.0).abs() < 1e-4);
        assert!((adv[3] / adv[2] - 5.0).abs() < 1e-4);
    }
}
