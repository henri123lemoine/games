//! The PPO update: GAE(λ) advantages, then several minibatch epochs of the clipped
//! surrogate objective over the rollout, following cleanrl's single-file PPO (the
//! known-correct reference): clipped surrogate, (optional) value-function clipping,
//! an entropy bonus, advantage normalization, and gradient clipping.
//!
//! The core is env-agnostic: it takes per-transition `(reward, value, done)` and the
//! old log-probs as plain `f32`, computes GAE, and drives the minibatch loop by
//! handing transition indices to a [`Policy`]. The policy owns its obs/action
//! tensors and re-evaluates the indexed rows under the current params.

use tch::{Device, Kind, Tensor, nn};

use crate::config::{AdvNorm, Minibatch, PpoConfig};
use crate::gae::{Step, gae};
use crate::policy::Policy;

/// Averaged diagnostics from one PPO update — the standard PPO health metrics.
#[derive(Default, Clone)]
pub struct UpdateStats {
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
    pub approx_kl: f32,
    pub clip_frac: f32,
    pub explained_variance: f32,
}

/// Run the PPO update over one rollout. Mutates the policy via `opt`. Returns
/// averaged diagnostics.
///
/// - `buf`: per-transition `(reward, value, done)` in step-major `[T, N]` order.
/// - `old_log_prob`: the action log-prob recorded at rollout time, same order.
/// - `bootstrap_values`: V(s_T) per actor, for the GAE tail (a value-only forward
///   on the post-rollout observations the adapter computes).
pub fn update<P: Policy>(
    policy: &P,
    opt: &mut nn::Optimizer,
    device: Device,
    buf: &[Step],
    old_log_prob: &[f32],
    bootstrap_values: &[f32],
    cfg: &PpoConfig,
) -> UpdateStats {
    let (adv, returns) = gae(buf, bootstrap_values, cfg.steps, cfg.gamma, cfg.lambda);
    let batch = buf.len();

    // Static tensors for the whole update (old log-probs, old values, advantages,
    // returns). The policy holds the obs/actions itself and indexes them.
    let old_logp_all = Tensor::from_slice(old_log_prob).to_device(device);
    let old_value_all =
        Tensor::from_slice(&buf.iter().map(|t| t.value).collect::<Vec<_>>()).to_device(device);
    let returns_all = Tensor::from_slice(&returns).to_device(device);

    let mut adv_t = Tensor::from_slice(&adv).to_device(device);
    // Per-rollout normalization (when selected) happens once, before the epochs.
    if cfg.adv_norm == AdvNorm::PerRollout {
        let mean = adv_t.mean(Kind::Float);
        let std = adv_t.std(true) + 1e-8;
        adv_t = (&adv_t - &mean) / &std;
    }

    let mut stats = UpdateStats::default();
    let mut updates = 0u32;

    for _ in 0..cfg.epochs {
        for idx in minibatch_indices(cfg, batch, device) {
            let old_logp = old_logp_all.index_select(0, &idx);
            let old_value = old_value_all.index_select(0, &idx);
            let ret = returns_all.index_select(0, &idx);

            let mb_adv = adv_t.index_select(0, &idx);
            let norm_adv = match cfg.adv_norm {
                AdvNorm::PerMinibatch => {
                    // Per-minibatch advantage normalization (cleanrl default).
                    let mean = mb_adv.mean(Kind::Float);
                    let std = mb_adv.std(true) + 1e-8;
                    (&mb_adv - &mean) / &std
                }
                AdvNorm::PerRollout | AdvNorm::Off => mb_adv.shallow_clone(),
            };

            let ev = policy.evaluate(&idx);

            let log_ratio = &ev.log_prob - &old_logp;
            let ratio = log_ratio.exp();

            // Clipped surrogate: -min(ratio*A, clip(ratio)*A), maximized via -loss.
            let surr1 = &ratio * &norm_adv;
            let surr2 = ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip) * &norm_adv;
            let policy_loss = -surr1.minimum(&surr2).mean(Kind::Float);

            let value_loss = if cfg.value_clip {
                // Value clipping (cleanrl): max of unclipped and clipped sq error.
                let v_unclipped = (&ev.value - &ret).pow_tensor_scalar(2);
                let v_clipped = &old_value + (&ev.value - &old_value).clamp(-cfg.clip, cfg.clip);
                let v_clipped = (&v_clipped - &ret).pow_tensor_scalar(2);
                0.5 * v_unclipped.maximum(&v_clipped).mean(Kind::Float)
            } else {
                0.5 * (&ev.value - &ret).pow_tensor_scalar(2).mean(Kind::Float)
            };

            let entropy = ev.entropy.mean(Kind::Float);

            let value_term: Tensor = cfg.value_coef * &value_loss;
            let entropy_term: Tensor = cfg.entropy_coef * &entropy;
            let mut loss = &policy_loss + value_term - entropy_term;
            if let Some(anchor) = cfg.bc_anchor
                && let Some(bc) = policy.bc_term(&idx)
            {
                loss += anchor.coef * bc;
            }

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

/// The minibatch index tensors for one epoch, per the configured strategy.
fn minibatch_indices(cfg: &PpoConfig, batch: usize, device: Device) -> Vec<Tensor> {
    match cfg.minibatch {
        Minibatch::Shuffled { count } => {
            let perm = Tensor::randperm(batch as i64, (Kind::Int64, device));
            let mb_size = batch / count;
            (0..count)
                .map(|mb| {
                    let lo = (mb * mb_size) as i64;
                    let hi = if mb == count - 1 {
                        batch as i64
                    } else {
                        ((mb + 1) * mb_size) as i64
                    };
                    perm.slice(0, lo, hi, 1)
                })
                .collect()
        }
        Minibatch::BpttWindows { window } => bptt_windows(batch, cfg.steps, window, device),
    }
}

/// BPTT-style minibatches: each actor's `T` steps in time order, split into
/// contiguous `window`-length segments (the tail window absorbs the remainder).
/// Built against the step-major `[T, N]` layout, so step t of actor j is index
/// `t*N + j`. Designed-for sequence/recurrent policies (doom's adapter may use it).
fn bptt_windows(batch: usize, steps: usize, window: usize, device: Device) -> Vec<Tensor> {
    let n = batch / steps;
    let window = window.max(1).min(steps);
    let mut out = Vec::new();
    for j in 0..n {
        let mut t0 = 0;
        while t0 < steps {
            let t1 = (t0 + window).min(steps);
            let idx: Vec<i64> = (t0..t1).map(|t| (t * n + j) as i64).collect();
            out.push(Tensor::from_slice(&idx).to_device(device));
            t0 = t1;
        }
    }
    out
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
