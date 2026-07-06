"""The move-RL loss (ATARAXOS_SPEC §4.1, reference `rl.py:547-579`).

PPO-clip policy + temperature(t)*magnet-KL + categorical value-CE + 0.1*rev-KL-to-data.
Advantage filtering keeps `|adv| >= max(quantile(|adv|, 0.75), 0.01)`. The value-CE
target `ret` is the genuine categorical λ-return the Rust buffer computes
(`buffer.rs::process_data`'s `use_cat_vf` path: a per-category vector cumsum of
`target_value_probs - value_probs`, bootstrapped from the next position's actual
softmax(W/L/D) distribution) — NOT a two-hot projection of a scalar return, which
the reference's categorical bootstrap is not equivalent to whenever a bootstrapped
value isn't a bare category anchor.
"""

import mlx.core as mx
import numpy as np

import stratego_nets as S

CATS = mx.array(S.spec.CATEGORICAL_AGGREGATION, dtype=mx.float32)  # [-1, 0, 1]
NEG_INF = -1e30


def _sanitize(x):
    """NaN -> 0 (a genuine bf16-overflow NaN carries no salvageable value; the
    subsequent clip bounds the result), +-inf preserved for clip to bound next."""
    return mx.where(mx.isnan(x), mx.zeros_like(x), x)


def advantage_filter_mask(advantage_np, rate=0.75, thresh=0.01, min_keep=0):
    """Spec filter `|adv| >= max(quantile(|adv|, rate), thresh)` (`buffer.py:233-241`).

    Returns `(mask, n_threshold)` where `n_threshold` is how many rows actually passed
    the spec threshold BEFORE the anti-starve floor — the real starvation signal the
    watchdog watches (a floored `mask.sum()` would hide it).

    `min_keep` is an anti-starve floor: as the policy sharpens, |adv| shrinks below the
    abs floor for most rows so the threshold keep collapses toward 0 (the full1 freeze:
    12762 kept at iter46 -> 945 at iter50). When fewer than `min_keep` rows pass the
    threshold, retain the `min_keep` largest-|adv| rows instead so the pass keeps a
    stable batch size; the quantile still does the real filtering when advantages are large.
    """
    abs_adv = np.abs(advantage_np)
    n = abs_adv.size
    if n == 0:
        return np.zeros(0, dtype=bool), 0
    threshold = max(float(np.quantile(abs_adv, rate)), thresh)
    mask = abs_adv >= threshold
    n_threshold = int(mask.sum())  # rows passing the spec filter, before any floor
    k = min(min_keep, n)
    if k > 0 and n_threshold < k:
        keep_idx = np.argpartition(abs_adv, n - k)[n - k:]
        mask = np.zeros(n, dtype=bool)
        mask[keep_idx] = True
    return mask, n_threshold


def two_hot(scalar, cats=CATS):
    """Two-hot encode a scalar value over ordered category points (here [-1, 0, 1]).

    For a value v in [cats[i], cats[i+1]] places (1-w) on i and w on i+1, so
    `two_hot(v) @ cats == v`. For the integer outcomes {-1, 0, 1} this is the
    one-hot the reference uses for terminal returns.
    """
    v = mx.clip(scalar, cats[0], cats[-1])
    n = cats.shape[0]
    # locate the upper bin edge
    ge = (v[:, None] >= cats[None, :]).astype(mx.int32)  # (B, n)
    lower = mx.clip(mx.sum(ge, axis=-1) - 1, 0, n - 2)  # (B,) index of lower edge
    lo = cats[lower]
    hi = cats[lower + 1]
    w = (v - lo) / (hi - lo)
    onehot_lo = (mx.arange(n)[None, :] == lower[:, None]).astype(mx.float32)
    onehot_hi = (mx.arange(n)[None, :] == (lower + 1)[:, None]).astype(mx.float32)
    return onehot_lo * (1.0 - w)[:, None] + onehot_hi * w[:, None]


def move_loss_and_stats(net, batch, magnet_coef, cfg, anchor_log_probs=None, anchor_coef=0.0):
    """Compute the scalar move loss (and a stats dict) over a filtered minibatch.

    `batch` holds MLX arrays already restricted to the kept (filtered) rows:
      obs (B,92,F), legal (B,1800) bool, action (B,) int, old_log_prob (B,),
      data_log_prob (B,1800), advantage (B,), ret (B,3) categorical value target.

    `anchor_log_probs` (B,1800), if given, is a FROZEN reference policy's
    log-probs on this same minibatch's obs (the BC warm-start checkpoint, held
    fixed all run) — a trust-region term, same reverse-KL shape as the existing
    data-KL, distinct from it (data-KL anchors to the ROLLING behavior policy
    that generated the batch; this anchors to the FIXED BC init, resisting
    long-horizon drift the rolling term can't see). `anchor_coef` decays over
    training (`TrainConfig.anchor_coef_at`) so it constrains early exploration
    without capping how far RL can ultimately surpass the teacher.
    """
    obs = batch["obs"]
    legal = batch["legal"]
    action = batch["action"]
    advantages = batch["advantage"]  # raw, per reference rl.py:541 (no standardization)
    old_log_prob = batch["old_log_prob"]
    data_log_prob = batch["data_log_prob"]
    ret = batch["ret"]

    out = net(obs, legal_mask=legal)
    # bf16 forward hardening: an entropy-collapsed policy occasionally drives the
    # bf16 net's raw logits to +/-inf (marathon1 iter159: H 2.2->0.01 in one step,
    # then 5 straight NaN'd bf16 forwards). NaN->0 + clip bounds it to a range
    # log_softmax/exp can't overflow, without touching a healthy forward's math
    # (legal logits stay far inside +/-60; only already-broken values move).
    move_logits = mx.clip(_sanitize(out["move_logits"].astype(mx.float32)), -60.0, 60.0)
    log_probs = move_logits - mx.logsumexp(move_logits, axis=-1, keepdims=True)  # (B,1800)
    value_logp = mx.clip(_sanitize(out["value_logp"].astype(mx.float32)), -60.0, 0.0)  # (B,3) log-softmax

    # PPO-clip policy loss on the chosen action. Clamp the log-ratio before exp so
    # a policy that drifts far from the data policy can't overflow exp() to inf
    # (PPO clips the ratio to [1-eps, 1+eps] for the advantage anyway, so the wide
    # clamp is inert in the trusted region and only keeps the backward finite).
    chosen_logp = mx.take_along_axis(log_probs, action[:, None], axis=-1).squeeze(-1)
    ratio = mx.exp(mx.clip(chosen_logp - old_log_prob, -10.0, 10.0))
    clipped = mx.clip(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range)
    policy_loss = -mx.minimum(advantages * ratio, advantages * clipped).mean()

    # Probabilities over the legal set (illegal log-probs are -inf -> prob 0).
    legal_f = legal.astype(mx.float32)
    probs = mx.exp(log_probs) * legal_f

    # Reverse-KL to the data policy: (probs * (log_probs - data_log_prob)).sum(-1).
    data_lp = mx.where(legal, data_log_prob.astype(mx.float32), 0.0)
    kl_loss = (probs * mx.where(legal, log_probs - data_lp, 0.0)).sum(-1).mean()

    # Reverse-KL to the frozen BC-anchor policy, same shape as the data-KL above.
    # Zero (not skipped) when there's no anchor, so `stats["anchor_kl"]` is always
    # a real, loggable number.
    if anchor_log_probs is not None:
        anchor_lp = mx.where(legal, anchor_log_probs.astype(mx.float32), 0.0)
        anchor_kl = (probs * mx.where(legal, log_probs - anchor_lp, 0.0)).sum(-1).mean()
    else:
        anchor_kl = mx.array(0.0)

    # Categorical value-CE: the buffer's categorical λ-return vs log_softmax(value).
    value_loss = -(ret * value_logp).sum(-1).mean()

    # Magnet reverse-KL to the flat-uniform legal magnet.
    entropy = -(probs * mx.where(legal, log_probs, 0.0)).sum(-1)
    magnet = legal_f / legal_f.sum(-1, keepdims=True)
    xe = -(probs * mx.log(mx.clip(magnet, 1e-10, None))).sum(-1)
    magnet_kl = (xe - entropy).mean()

    loss = (
        cfg.policy_coef * policy_loss
        + magnet_coef * magnet_kl
        + cfg.vf_coef * value_loss
        + cfg.kl_coef * kl_loss
        + anchor_coef * anchor_kl
    )
    stats = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "kl_loss": kl_loss,
        "magnet_kl": magnet_kl,
        "anchor_kl": anchor_kl,
        "entropy": entropy.mean(),
        "loss": loss,
    }
    return loss, stats
