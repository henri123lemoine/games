"""The co-trained setup loss (ATARAXOS_SPEC §4.2, reference `rl.py:616-702` +
`arrangement/buffer.py:process_data`).

Pure Monte-Carlo (arr_td_lambda = arr_gae_lambda = 1.0, no advantage filtering).
Under λ=1.0 the reference's per-slot value/advantage traces telescope, so the
exact targets reduce to closed forms (`buffer.py:311-352`):

  value target / grounded advantage:
    val_est[k]     = MC return  (so the categorical value-CE target is two-hot(return) at every slot)
    grounded adv[k] = return - value_baseline[k]

  entropy regularization (`ents = reg_norm * ent_pred_base`; nll = -logp of placed type):
    reg_val_est[k] (ent-pred MSE target, normalized) = (sum_{j>=k} nll[j]) / reg_norm
    reg advantage[k] (added to policy adv, scaled by reg_temp)
                     = sum_{j>=k} nll[j] - ents_base[k]

Three head losses (coefs policy 1.0 / ent_pred 1.0 / vf 0.5 / kl 0.1):
  PPO-clip placement + categorical value-CE + conditional-entropy MSE + 0.1*rev-KL.
The behavior baseline (value/ent/log-prob predictions) is frozen at the start of
the iteration (the reference's generation-time actor), held across the 5 epochs.
"""

import mlx.core as mx

import stratego_nets as S
from .move_loss import two_hot

PIECE_COUNTS = mx.array(list(S.spec.CLASSIC_PIECE_COUNTS), dtype=mx.float32)
CATS = mx.array(S.spec.CATEGORICAL_AGGREGATION, dtype=mx.float32)  # [-1, 0, 1]
# The net fills illegal placement logits with finfo.min (~-3.4e38). Differentiating
# through that magnitude produces inf/NaN gradients, so clamp to a finite floor —
# exp(-1e9) is still 0 (the type stays impossible) but the backward stays finite.
NEG = -1e9


def _log_softmax_legal(logits):
    """Numerically-safe log-softmax over the legal placement types.

    Returns `(log_probs, legal)` where `legal` is the boolean mask of placeable
    types (the net set illegal logits to finfo.min). Illegal log-probs are a
    finite large-negative so gradients never explode; `legal` lets callers zero
    illegal contributions in cross-entropy / KL sums.
    """
    legal = logits > (mx.finfo(logits.dtype).max * -1.0)  # True where not finfo.min
    safe = mx.where(legal, logits, NEG)
    return safe - mx.logsumexp(safe, axis=-1, keepdims=True), legal


def setup_baseline(net, seq, reg_norm):
    """Freeze the behavior policy from the current net (pre-update actor).

    Returns stop-gradient MLX arrays:
      log_probs (B,40,14)  log-softmax placement log-probs (data policy)
      value_scalar (B,40)  softmax(value)@[-1,0,1] per-slot value baseline
      reg_returns (B,40)   normalized future-nll = ent-pred MSE target
      reg_adv (B,40)       entropy-regularization advantage (telescoped λ=1.0)
    """
    out = net(seq, PIECE_COUNTS)
    logits = out["logits"].astype(mx.float32)
    log_probs, _ = _log_softmax_legal(logits)
    value_logp = out["value"].astype(mx.float32)
    value_logp = value_logp - mx.logsumexp(value_logp, axis=-1, keepdims=True)
    value_scalar = (mx.exp(value_logp) * CATS).sum(-1)
    ent_pred = out["ent_pred"].squeeze(-1)  # (B,40) normalized prediction

    placed = mx.argmax(seq, axis=-1)
    nll = -mx.take_along_axis(log_probs, placed[..., None], axis=-1).squeeze(-1)  # (B,40)

    # future-nll suffix sum: future_nll[k] = sum_{j>=k} nll[j].
    future_nll = mx.cumsum(nll[:, ::-1], axis=1)[:, ::-1]
    reg_returns = future_nll / reg_norm  # ent-pred MSE target (normalized)
    ents_base = reg_norm * ent_pred  # denormalized baseline entropy prediction
    reg_adv = future_nll - ents_base  # reg advantage (λ=1.0 telescoped)

    return {
        "log_probs": mx.stop_gradient(log_probs),
        "value_scalar": mx.stop_gradient(value_scalar),
        "reg_returns": mx.stop_gradient(reg_returns),
        "reg_adv": mx.stop_gradient(reg_adv),
    }


def setup_loss_and_stats(net, batch, cfg):
    """Scalar setup loss + stats over a minibatch of completed arrangements.

    `batch` MLX arrays:
      seq (B,40,14), outcome (B,), reg_temp (scalar),
      old_log_probs (B,40,14), old_value_scalar (B,40), reg_returns (B,40), reg_adv (B,40).
    """
    seq = batch["seq"]
    outcome = batch["outcome"]
    reg_temp = batch["reg_temp"]
    old_log_probs = batch["old_log_probs"]
    old_value_scalar = batch["old_value_scalar"]
    reg_returns = batch["reg_returns"]
    reg_adv = batch["reg_adv"]

    out = net(seq, PIECE_COUNTS)
    logits = out["logits"].astype(mx.float32)
    log_probs, legal = _log_softmax_legal(logits)
    value_logp = out["value"].astype(mx.float32)
    value_logp = value_logp - mx.logsumexp(value_logp, axis=-1, keepdims=True)
    ent_pred = out["ent_pred"].squeeze(-1)  # (B,40)

    placed = mx.argmax(seq, axis=-1)
    log_prob = mx.take_along_axis(log_probs, placed[..., None], axis=-1).squeeze(-1)
    old_lp = mx.take_along_axis(old_log_probs, placed[..., None], axis=-1).squeeze(-1)
    # Clamp the log-ratio before exp: across the 5 epochs on a frozen baseline the
    # policy can drift far enough that exp(log_prob - old_lp) overflows to inf
    # (then inf*adv -> NaN). PPO already clips the ratio to [1-eps, 1+eps] for the
    # advantage, so bounding the raw ratio to a wide finite range changes nothing
    # in the trusted region while keeping the backward finite.
    log_ratio = mx.clip(log_prob - old_lp, -10.0, 10.0)
    ratio = mx.exp(log_ratio)  # (B,40)

    grounded_adv = outcome[:, None] - old_value_scalar  # (B,40)
    advantages = grounded_adv + reg_temp * reg_adv

    clipped = mx.clip(ratio, 1 - cfg.arr_clip_range, 1 + cfg.arr_clip_range)
    policy_loss = -mx.minimum(advantages * ratio, advantages * clipped).mean()

    # categorical value-CE: two-hot(outcome) shared across all 40 slots.
    target = mx.broadcast_to(two_hot(outcome)[:, None, :], value_logp.shape)
    value_loss = -(target * value_logp).sum(-1).mean()

    entropy_loss = ((ent_pred - reg_returns) ** 2).mean()
    # rev-KL to data policy, summed over legal types only (illegal log-probs are a
    # finite floor whose ~0 prob would otherwise leak a huge-magnitude gradient).
    kl_terms = mx.exp(log_probs) * (log_probs - old_log_probs)
    kl_loss = mx.where(legal, kl_terms, 0.0).sum(-1).mean()

    loss = (
        cfg.arr_policy_coef * policy_loss
        + cfg.arr_ent_pred_coef * entropy_loss
        + cfg.arr_vf_coef * value_loss
        + cfg.arr_kl_coef * kl_loss
    )
    stats = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
        "kl_loss": kl_loss,
        "loss": loss,
    }
    return loss, stats
