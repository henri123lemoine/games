"""Test-time search (ATARAXOS_SPEC §5): belief-sampled depth-D rollouts under the
move net + a single magnetic-mirror-descent (MMD) update at decision time.

This is **not** MCTS — no tree, no iteration loop. One `search(...)` call:

  1. reads the root (legal actions + the hidden-opponent belief inputs) from the
     `stratego_sim.Searcher` bridge session;
  2. samples `n_sample = min(num_envs // L, max_samples)` belief assignments
     (`L` = #legal root actions), tiles them to fill the rollout batch, and
     assigns each legal root action ≈`num_envs / L` worlds;
  3. drives depth-D rollouts under the move net (both players sample its policy)
     via the bridge, getting each world's λ=1 leaf value and its root action;
  4. scatters the leaf values per root action → categorical q̂ → `scalar_q =
     q @ [-1, 0, 1]`;
  5. forms the MMD search policy `π_search ∝ exp((log π_bp + α·q̂ + α·τ·log
     π_magnet) / (1 + α·τ))` over the legal actions and samples one.

Belief flavors (analytic, the learned belief net is milestone 7 and plugs into
the same `Belief` interface):
  * `UniformBelief`  — the combinatorial posterior: uniform over feasible type
    assignments, autoregressively masked by remaining count + movability.
  * `MarginalizedBelief` — autoregressive from the encoder's `their_*_prob`
    marginals (the marginalized-uniform posterior), same masking.
  * `belief=None` — ground-truth determinization ("perfect search"), an ablation
    that skips sampling and uses the true hidden ranks the bridge exposes.

The reference `pyengine/core/search.py::SearchBot` + `compute_search_policy`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

import stratego_sim as sim

N_PIECE_TYPE = sim.N_PIECE_TYPE  # 14
N_BOARD_CELL = 100
N_ACTION = sim.N_ACTION  # 1800
CATS = np.array([-1.0, 0.0, 1.0], dtype=np.float32)  # CATEGORICAL_AGGREGATION

# Movability: ranks 0..9 are movable; flag (10) and bomb (11) are immovable.
_MOVABLE = np.arange(N_PIECE_TYPE) < 10
_IMMOVABLE_TYPES = (10, 11)


# --------------------------------------------------------------------------- #
# Belief sampling
# --------------------------------------------------------------------------- #
class Belief(Protocol):
    """A belief samples opponent hidden-piece type assignments consistent with the
    public state. `sample(root, n_sample, rng)` returns `(n_sample, n_hidden)`
    uint8 assignments — one `PieceType` per hidden piece, in the root's row-major
    POV rank order. The learned belief net (M7) implements the same contract."""

    def sample(self, root: dict, n_sample: int, rng: np.random.Generator) -> np.ndarray: ...


def _feasible_mask(remaining: np.ndarray, has_moved_i: bool, remaining_unmoved: int,
                   remaining_immovable: int) -> np.ndarray:
    """Per-draw feasibility mask over the 14 types for one hidden piece, matching
    the reference `belief/masking.py::create_mask` constraints:

      1. count: a type is allowed only while its remaining supply is positive;
      2. movability: a *moved* piece must be a movable rank; an *unmoved* piece
         may take a movable rank only if doing so would not later force an
         immovable rank onto a moved piece (i.e. unless every remaining unmoved
         slot is needed for the remaining immovable pieces).
    """
    mask = remaining > 0
    if has_moved_i:
        # Moved => cannot be flag/bomb (immovable).
        mask = mask & _MOVABLE
    else:
        # Unmoved: if every remaining unmoved slot is needed by the remaining
        # immovable pieces, this slot must take an immovable type.
        if remaining_immovable == remaining_unmoved and remaining_immovable > 0:
            mask = mask & ~_MOVABLE
    return mask


def _sample_assignment(marginal_logits: np.ndarray, counts: np.ndarray,
                       has_moved: np.ndarray, n_sample: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Autoregressive masked sampling shared by the uniform & marginalized beliefs.

    `marginal_logits` `(n_hidden, 14)` are the per-piece type scores (uniform =
    all-equal feasible; marginalized = log of the analytic marginals). Pieces are
    filled in row-major POV rank order; the count + movability mask
    (`_feasible_mask`) is recomputed per draw from the running assignment. Returns
    `(n_sample, n_hidden)` uint8.
    """
    n_hidden = len(has_moved)
    out = np.zeros((n_sample, n_hidden), dtype=np.uint8)
    # Pad the 12-wide per-rank hidden counts to the 14-wide type space.
    counts14 = np.zeros(N_PIECE_TYPE, dtype=np.int64)
    counts14[: len(counts)] = counts.astype(np.int64)
    counts = counts14
    total_immovable = int(counts[list(_IMMOVABLE_TYPES)].sum())

    for s in range(n_sample):
        remaining = counts.astype(np.int64).copy()
        # remaining unmoved slots after this one (for the movability look-ahead).
        unmoved_after = np.zeros(n_hidden, dtype=np.int64)
        run = 0
        for i in range(n_hidden - 1, -1, -1):
            unmoved_after[i] = run
            if not has_moved[i]:
                run += 1
        remaining_immovable = total_immovable
        for i in range(n_hidden):
            # remaining_unmoved INCLUDING the current piece if it is unmoved.
            remaining_unmoved = int(unmoved_after[i]) + (0 if has_moved[i] else 1)
            mask = _feasible_mask(remaining, bool(has_moved[i]),
                                  remaining_unmoved, remaining_immovable)
            logits = np.where(mask, marginal_logits[i], -np.inf)
            m = logits.max()
            p = np.exp(logits - m)
            p_sum = p.sum()
            if not np.isfinite(p_sum) or p_sum <= 0:
                # Degenerate (no feasible type with positive marginal mass): fall
                # back to a uniform draw over the feasible set.
                p = mask.astype(np.float64)
                p_sum = p.sum()
            p = p / p_sum
            t = int(rng.choice(N_PIECE_TYPE, p=p))
            out[s, i] = t
            remaining[t] -= 1
            if t in _IMMOVABLE_TYPES:
                remaining_immovable -= 1
    return out


@dataclass(frozen=True)
class UniformBelief:
    """The combinatorial posterior: uniform over feasible assignments (count +
    movability masked), autoregressive."""

    def sample(self, root: dict, n_sample: int, rng: np.random.Generator) -> np.ndarray:
        counts = np.asarray(root["hidden_counts"])
        has_moved = np.asarray(root["hidden_has_moved"])
        n_hidden = int(root["n_hidden"])
        if n_hidden == 0:
            return np.zeros((n_sample, 0), dtype=np.uint8)
        flat = np.zeros((n_hidden, N_PIECE_TYPE), dtype=np.float32)  # uniform logits
        return _sample_assignment(flat, counts, has_moved, n_sample, rng)


@dataclass(frozen=True)
class MarginalizedBelief:
    """The marginalized-uniform posterior: autoregressive sampling from the
    encoder's `their_*_prob` marginals, count + movability masked."""

    def sample(self, root: dict, n_sample: int, rng: np.random.Generator) -> np.ndarray:
        counts = np.asarray(root["hidden_counts"])
        has_moved = np.asarray(root["hidden_has_moved"])
        n_hidden = int(root["n_hidden"])
        if n_hidden == 0:
            return np.zeros((n_sample, 0), dtype=np.uint8)
        marg = np.asarray(root["marginal"], dtype=np.float64)  # (n_hidden, 14)
        tiny = np.finfo(np.float64).tiny
        logits = np.log(marg + tiny).astype(np.float32)
        return _sample_assignment(logits, counts, has_moved, n_sample, rng)


# --------------------------------------------------------------------------- #
# The MMD search policy
# --------------------------------------------------------------------------- #
def get_weighted_uniform_policy(legal_mask: np.ndarray) -> np.ndarray:
    """The per-origin uniform magnet (`utils.helper.get_weighted_uniform_policy`):
    a legal action's mass is `1 / (#legal actions sharing its origin square)`,
    then renormalized over all legal actions. `legal_mask` is `(1800,)` bool."""
    origin = np.arange(N_ACTION) % N_BOARD_CELL
    counts = np.zeros(N_BOARD_CELL, dtype=np.int64)
    np.add.at(counts, origin, legal_mask.astype(np.int64))
    per = np.clip(counts[origin], 1, None)
    unnorm = legal_mask.astype(np.float64) / per
    total = unnorm.sum()
    if total <= 0:
        return unnorm.astype(np.float32)
    return (unnorm / total).astype(np.float32)


def compute_search_policy(q: np.ndarray, bp_logits: np.ndarray, legal_mask: np.ndarray,
                          temperature: float, stepsize: float,
                          uniform_magnet: bool = False) -> np.ndarray:
    """The MMD closed form (`compute_search_policy`, `core/search.py:431-454`),
    over the L legal actions:

      uniform_magnet:  π ∝ exp((log π_bp + α·q̂) / (1 + α·τ))
      weighted magnet: π ∝ exp((log π_bp + α·q̂ + α·τ·log π_magnet) / (1 + α·τ))

    where α = stepsize, τ = temperature, π_bp = move-net behavior policy (its
    log-probs over the legal set), π_magnet = `get_weighted_uniform_policy`.
    Returns the full `(1800,)` policy (0 off-legal)."""
    legal_idx = np.nonzero(legal_mask)[0]
    logits_l = bp_logits[legal_idx]
    q_l = q[legal_idx]
    temp_ss = temperature * stepsize
    if uniform_magnet:
        search_logits = (logits_l + stepsize * q_l) / (1.0 + temp_ss)
    else:
        magnet = get_weighted_uniform_policy(legal_mask)[legal_idx]
        log_magnet = np.log(magnet + np.finfo(np.float64).tiny)
        search_logits = (logits_l + stepsize * q_l + temp_ss * log_magnet) / (1.0 + temp_ss)
    m = search_logits.max()
    p = np.exp(search_logits - m)
    p = p / p.sum()
    out = np.zeros(N_ACTION, dtype=np.float32)
    out[legal_idx] = p.astype(np.float32)
    return out


def _sample_deterministic(policy: np.ndarray, total: int) -> np.ndarray:
    """`sample_deterministic` (`core/search.py:21-40`): split `total` samples
    across categories proportional to `policy`, rounding then fixing the residual
    so the counts sum to exactly `total`. Returns a `(total,)` int array of the
    category each sample belongs to (here: the legal-action row index)."""
    expected = np.round(policy * total).astype(np.int64)
    # Fix the rounding residual to hit `total` exactly.
    while expected.sum() < total:
        disc = policy * total - expected
        expected[int(np.argmax(disc))] += 1
    while expected.sum() > total:
        disc = expected - policy * total
        expected[int(np.argmax(disc))] -= 1
    return np.repeat(np.arange(len(policy)), expected)


def _move_forward(move_net, obs_np, legal_np):
    """Move-net forward on a rollout batch -> (logits_np (B,1800), values_np (B,))
    with the value head reduced to a scalar (search-player POV `softmax(W/L/D) @
    [-1,0,1]`). Lazily imports MLX so the belief/policy math stays torch/MLX-free."""
    import mlx.core as mx

    if obs_np.shape[0] == 0:
        return np.zeros((0, N_ACTION), np.float32), np.zeros(0, np.float32)
    out = move_net(mx.array(obs_np), legal_mask=mx.array(legal_np))
    logits = np.array(out["move_logits"].astype(mx.float32))
    vlogp = np.array(out["value_logp"])
    vals = (np.exp(vlogp) * CATS).sum(-1).astype(np.float32)
    return logits, vals


@dataclass
class SearchResult:
    """The search outputs for inspection / eval: the sampled action plus the
    diagnostics (`q`, `bp`, `search` policies, per-action world counts)."""
    action: int
    q: np.ndarray            # (1800,) scalar q̂ per action (0 off-legal)
    bp_policy: np.ndarray    # (1800,) move-net behavior policy
    search_policy: np.ndarray  # (1800,) the MMD search policy
    counts: np.ndarray       # (1800,) #rollout worlds per root action
    n_sample: int


def search(move_net, searcher, depth: int = 10, stepsize: float = 10.0,
           temperature: float = 1e-3, max_samples: int = 200,
           num_envs: int = 1024, belief: Optional[Belief] = MarginalizedBelief(),
           uniform_magnet: bool = False, seed: int = 0) -> SearchResult:
    """Run §5 test-time search from a `stratego_sim.Searcher` root and return the
    sampled action + diagnostics.

    Args:
      move_net: the MLX move net (forward -> {"move_logits", "value_logp"}).
      searcher: a `stratego_sim.Searcher` opened at the search root.
      depth: rollout depth (even, ≥ 2). Eval default 10.
      stepsize (α): MMD step size. Eval default 10.
      temperature (τ): MMD temperature. Eval default ~1e-3.
      max_samples: cap on belief samples (`n_sample = min(num_envs//L, this)`).
      num_envs: rollout-world budget (the total rollouts ≈ this).
      belief: a `Belief` to sample determinizations, or `None` for ground-truth
        ("perfect search") — the ablation.
      uniform_magnet: drop the per-origin magnet term (flat-uniform magnet).
      seed: RNG seed for belief sampling + the deterministic world split.
    """
    if depth < 2 or depth % 2 != 0:
        raise ValueError("depth must be even and >= 2")
    rng = np.random.default_rng(seed)
    root = searcher.root()
    legal_mask = np.asarray(root["legal"])
    legal_idx = np.nonzero(legal_mask)[0]
    n_legal = len(legal_idx)
    if n_legal == 0:
        raise ValueError("no legal actions at the search root")

    # 1. n_sample beliefs; tile to fill the rollout batch; assign each legal root
    #    action ≈ num_envs / L worlds (the per-action world split).
    n_sample = max(1, min(num_envs // n_legal, max_samples))
    n_hidden = int(root["n_hidden"])

    if belief is None:
        # Perfect search: the single ground-truth determinization, tiled.
        assignment = np.asarray(searcher.true_hidden(), dtype=np.uint8)  # (n_hidden,)
        beliefs = np.broadcast_to(assignment, (n_sample, n_hidden)).copy()
    else:
        beliefs = belief.sample(root, n_sample, rng)

    # The per-world root action: a uniform deterministic split over legal actions
    # (mirrors the reference `sample_deterministic(uniform, num_envs)`).
    uniform_over_legal = np.ones(n_legal, dtype=np.float64) / n_legal
    world_action_row = _sample_deterministic(uniform_over_legal, num_envs)  # (num_envs,) -> legal row
    root_actions = legal_idx[world_action_row].astype(np.int64)  # (num_envs,) -> 1800 idx

    # Tile beliefs across the worlds (one belief per world, cycling).
    if n_hidden == 0:
        assignments = np.zeros((num_envs, 0), dtype=np.uint8)
    else:
        reps = num_envs // n_sample + 1
        assignments = np.tile(beliefs, (reps, 1))[:num_envs].astype(np.uint8)

    # 2. Drive depth-D rollouts under the move net.
    searcher.begin(assignments, root_actions, depth, seed + 1)
    while not searcher.is_done():
        b = searcher.collect()
        logits, vals = _move_forward(move_net, b["obs"], b["legal"])
        searcher.commit(logits, vals)
    res = searcher.finish()
    world_root_action = np.asarray(res["root_action"])  # (num_envs,)
    leaf = np.asarray(res["leaf"], dtype=np.float32)     # (num_envs,) search-POV λ=1 value

    # 3. q̂ per root action via scatter. The bridge already returns each rollout's
    #    *scalar* λ-return (terminal reward in {-1,0,1}, or the value head's
    #    softmax(W/L/D) @ [-1,0,1] bootstrap), so scatter-add the scalar leaves
    #    per action and divide by the per-action world counts. This equals the
    #    reference's `cat_q @ [-1,0,1]` exactly: averaging then projecting through
    #    the linear aggregation is the same as projecting then averaging.
    cum = np.zeros(N_ACTION, dtype=np.float64)
    counts = np.zeros(N_ACTION, dtype=np.float64)
    np.add.at(cum, world_root_action, leaf.astype(np.float64))
    np.add.at(counts, world_root_action, 1.0)
    q = np.zeros(N_ACTION, dtype=np.float32)
    nz = counts > 0
    q[nz] = (cum[nz] / counts[nz]).astype(np.float32)

    # 4. The move-net behavior policy at the root (its log-probs over the legal
    #    set), then the MMD closed form, then sample.
    bp_log = _root_bp_logprobs(move_net, searcher, legal_mask)
    search_policy = compute_search_policy(q, bp_log, legal_mask, temperature, stepsize,
                                          uniform_magnet)
    bp_policy = np.zeros(N_ACTION, dtype=np.float32)
    bp_policy[legal_idx] = np.exp(bp_log[legal_idx]).astype(np.float32)

    p = search_policy.astype(np.float64)
    p = p / p.sum()
    action = int(rng.choice(N_ACTION, p=p))
    return SearchResult(action=action, q=q, bp_policy=bp_policy,
                        search_policy=search_policy, counts=counts.astype(np.int64),
                        n_sample=n_sample)


def _root_bp_logprobs(move_net, searcher, legal_mask: np.ndarray) -> np.ndarray:
    """The move net's behavior log-probs over the legal actions at the root, the
    `log π_bp` of the MMD closed form. Encodes the *true* root (no
    determinization — the actual infostate the acting player sees)."""
    import mlx.core as mx

    obs = np.asarray(searcher.root_obs())[None, ...]  # (1, 92, F)
    lg = move_net(mx.array(obs), legal_mask=mx.array(legal_mask[None, :]))["move_logits"]
    logits = np.array(lg.astype(mx.float32))[0]
    out = np.full(N_ACTION, -np.inf, dtype=np.float64)
    idx = np.nonzero(legal_mask)[0]
    lg_l = logits[idx].astype(np.float64)
    m = lg_l.max()
    out[idx] = lg_l - (m + np.log(np.exp(lg_l - m).sum()))
    return out.astype(np.float32)
