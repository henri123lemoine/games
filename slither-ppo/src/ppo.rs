//! PPO on the collected learner transitions, following cleanrl's single-file PPO
//! (the known-correct reference) rather than an invented variant: GAE(λ) advantage
//! estimation, the clipped surrogate objective, value-function clipping, an entropy
//! bonus, advantage normalization, and several minibatch epochs over the rollout.
//!
//! The transition buffer is a flat `T*N` block in step-major order (step 0 for all
//! arenas, then step 1, …) so the `[T, N]` reshape GAE needs is just a view. Done
//! flags mark episode boundaries; the env auto-resets a dead learner, so a `done`
//! at step t means the value bootstrap from t+1 must be cut (cleanrl's mask).

use tch::{Device, Kind, Tensor, nn};

use crate::net::Policy;
use crate::obs_batch;
use crate::rollout::Transition;

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

#[derive(Default, Clone)]
pub struct UpdateStats {
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
    pub approx_kl: f32,
    pub clip_frac: f32,
    pub explained_variance: f32,
}

/// Compute GAE(λ) advantages and returns over the step-major `[T, N]` buffer.
/// `bootstrap_values[n]` is V(s_T) for arena n (the value after the last stored
/// step). Returns `(advantages, returns)` flat in the same step-major order.
fn gae(
    buf: &[Transition],
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
            // recursion restarts (the env already reset the arena).
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
    let (adv, returns) = gae(buf, bootstrap_values, cfg.steps, cfg.gamma, cfg.lambda);
    let batch = buf.len();

    // Static tensors for the whole update (obs, actions, old log-probs, old values,
    // advantages, returns). Advantages are normalized once over the full batch.
    let obs: Vec<_> = buf.iter().map(|t| t.obs.clone()).collect();
    let (grid_all, scalars_all) = obs_batch::pack(&obs, device);
    let turn_all =
        Tensor::from_slice(&buf.iter().map(|t| t.turn).collect::<Vec<_>>()).to_device(device);
    let boost_all =
        Tensor::from_slice(&buf.iter().map(|t| t.boost).collect::<Vec<_>>()).to_device(device);
    let old_logp_all =
        Tensor::from_slice(&buf.iter().map(|t| t.log_prob).collect::<Vec<_>>()).to_device(device);
    let old_value_all =
        Tensor::from_slice(&buf.iter().map(|t| t.value).collect::<Vec<_>>()).to_device(device);
    let returns_all = Tensor::from_slice(&returns).to_device(device);

    let adv_t = Tensor::from_slice(&adv).to_device(device);

    let mut stats = UpdateStats::default();
    let mut updates = 0u32;

    let mb_size = batch / cfg.minibatches;
    for _ in 0..cfg.epochs {
        let perm = Tensor::randperm(batch as i64, (Kind::Int64, device));
        for mb in 0..cfg.minibatches {
            let lo = (mb * mb_size) as i64;
            let hi = if mb == cfg.minibatches - 1 {
                batch as i64
            } else {
                ((mb + 1) * mb_size) as i64
            };
            let idx = perm.slice(0, lo, hi, 1);

            let grid = grid_all.index_select(0, &idx);
            let scalars = scalars_all.index_select(0, &idx);
            let turn = turn_all.index_select(0, &idx);
            let boost = boost_all.index_select(0, &idx);
            let old_logp = old_logp_all.index_select(0, &idx);
            let old_value = old_value_all.index_select(0, &idx);
            let ret = returns_all.index_select(0, &idx);

            // Per-minibatch advantage normalization (cleanrl default).
            let mb_adv = adv_t.index_select(0, &idx);
            let mean = mb_adv.mean(Kind::Float);
            let std = mb_adv.std(true) + 1e-8;
            let norm_adv = (&mb_adv - &mean) / &std;

            let ev = policy.evaluate(&grid, &scalars, &turn, &boost);

            let log_ratio = &ev.log_prob - &old_logp;
            let ratio = log_ratio.exp();

            // Clipped surrogate: -min(ratio*A, clip(ratio)*A), maximized via -loss.
            let surr1 = &ratio * &norm_adv;
            let surr2 = ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip) * &norm_adv;
            let policy_loss = -surr1.minimum(&surr2).mean(Kind::Float);

            // Value clipping (cleanrl): max of unclipped and clipped squared error.
            let v_unclipped = (&ev.value - &ret).pow_tensor_scalar(2);
            let v_clipped = &old_value + (&ev.value - &old_value).clamp(-cfg.clip, cfg.clip);
            let v_clipped = (&v_clipped - &ret).pow_tensor_scalar(2);
            let value_loss = 0.5 * v_unclipped.maximum(&v_clipped).mean(Kind::Float);

            let entropy = ev.entropy.mean(Kind::Float);

            let value_term: Tensor = cfg.value_coef * &value_loss;
            let entropy_term: Tensor = cfg.entropy_coef * &entropy;
            let loss = &policy_loss + value_term - entropy_term;

            opt.zero_grad();
            loss.backward();
            opt.clip_grad_norm(cfg.max_grad_norm);
            opt.step();

            // Diagnostics (detached).
            tch::no_grad(|| {
                stats.policy_loss += f32::try_from(&policy_loss).unwrap();
                stats.value_loss += f32::try_from(&value_loss).unwrap();
                stats.entropy += f32::try_from(&entropy).unwrap();
                let kl = f32::try_from((&old_logp - &ev.log_prob).mean(Kind::Float)).unwrap();
                stats.approx_kl += kl;
                let clipped = ratio
                    .gt(1.0 + cfg.clip)
                    .logical_or(&ratio.lt(1.0 - cfg.clip))
                    .to_kind(Kind::Float)
                    .mean(Kind::Float);
                stats.clip_frac += f32::try_from(&clipped).unwrap();
            });
            updates += 1;
        }
    }

    let u = updates.max(1) as f32;
    stats.policy_loss /= u;
    stats.value_loss /= u;
    stats.entropy /= u;
    stats.approx_kl /= u;
    stats.clip_frac /= u;
    stats.explained_variance = explained_variance(&old_value_all, &returns_all);
    stats
}

/// Fraction of the return variance the value head explains — 1.0 perfect, ≤0
/// no better than predicting the mean. A standard PPO health metric.
fn explained_variance(values: &Tensor, returns: &Tensor) -> f32 {
    tch::no_grad(|| {
        let var_ret = f32::try_from(returns.var(true)).unwrap();
        if var_ret < 1e-8 {
            return 0.0;
        }
        let var_resid = f32::try_from((returns - values).var(true)).unwrap();
        1.0 - var_resid / var_ret
    })
}

#[cfg(test)]
mod tests {
    use super::*;
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
    /// discounted-TD-residual sum; check it against a hand-rolled reference.
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
        // Two steps, single arena; step 0 is terminal (learner died, arena reset).
        let buf = vec![tr(3.0, 1.0, true), tr(0.5, 0.2, false)];
        let boot = vec![0.9f32];
        let (adv, _ret) = gae(&buf, &boot, buf.len(), gamma, lambda);
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
        let (adv, _) = gae(&buf, &boot, 2, gamma, lambda);
        // Arena 0 sees only 1.0 rewards, arena 1 only 5.0 — a 5x gap at every step.
        assert!((adv[1] / adv[0] - 5.0).abs() < 1e-4);
        assert!((adv[3] / adv[2] - 5.0).abs() < 1e-4);
    }
}
