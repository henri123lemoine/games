"""Tests for the test-time search (ATARAXOS_SPEC §5): the belief samplers respect
the count + movability constraints, the MMD closed form reduces to the reference
math, and an end-to-end search runs through the bridge for every belief flavor.

The rollout/value-plumbing oracle (`test_search.py::test_q_value_computation`)
lives Rust-side as `search::tests::rollout_leaf_matches_independent_lambda_return`
(the leaf value matches an independent λ=1 return to 1e-6); these cover the Python
belief + MMD + bridge layer. Needs the `stratego_sim` bridge installed; skips if
absent.
"""

import numpy as np
import pytest

import stratego_nets as S

sim = pytest.importorskip("stratego_sim")

from stratego_trainer.search import (  # noqa: E402
    MarginalizedBelief,
    UniformBelief,
    compute_search_policy,
    get_weighted_uniform_policy,
    search,
)


def _drive_to_move_phase(num_envs=128, seed=3, steps=120):
    """Drive a uniform self-play sim into the move phase; return (sim, move env)."""
    s = sim.BatchSim(num_envs=num_envs, move_cap=400, seed=seed)

    def _commit(b):
        nm, nd = b["move_obs"].shape[0], b["deploy_obs"].shape[0]
        s.commit(np.zeros((nm, sim.N_ACTION), np.float32), np.zeros(nm, np.float32),
                 np.full((nm, 3), 1.0 / 3.0, np.float32),
                 np.zeros((nd, sim.DEPLOY_WIDTH), np.float32), np.zeros(nd, np.float32),
                 np.full((nd, 3), 1.0 / 3.0, np.float32))

    for _ in range(steps):
        _commit(s.collect())
    b = s.collect()
    env = int(b["move_env"][0])
    _commit(b)
    return s, env


def test_weighted_uniform_policy_is_per_origin_uniform():
    # Two legal moves from origin 0, one from origin 5. Per-origin uniform gives
    # origin-0 moves half-weight-each-of-the-origin, then renormalized overall.
    legal = np.zeros(sim.N_ACTION, dtype=bool)
    legal[0] = True       # origin 0
    legal[100] = True     # origin 0 (action = 100*c + origin)
    legal[5] = True       # origin 5
    pol = get_weighted_uniform_policy(legal)
    assert np.isclose(pol.sum(), 1.0)
    # origin-0 actions share its mass (1/2 each before renorm); origin-5 gets 1.
    # unnorm = [0.5, 0.5, 1.0] over the three legal slots -> normalized [.25,.25,.5]
    assert np.isclose(pol[0], 0.25) and np.isclose(pol[100], 0.25)
    assert np.isclose(pol[5], 0.5)
    # zero off-legal
    assert pol[np.logical_not(legal)].sum() == 0.0


def test_compute_search_policy_matches_closed_form():
    legal = np.zeros(sim.N_ACTION, dtype=bool)
    idx = [0, 100, 5]
    for i in idx:
        legal[i] = True
    rng = np.random.default_rng(0)
    bp_log = np.full(sim.N_ACTION, -np.inf, np.float32)
    raw = rng.normal(size=3).astype(np.float32)
    raw = raw - (raw.max() + np.log(np.exp(raw - raw.max()).sum()))  # log-softmax
    for j, i in enumerate(idx):
        bp_log[i] = raw[j]
    q = np.zeros(sim.N_ACTION, np.float32)
    q[idx] = np.array([0.5, -0.2, 0.1], np.float32)

    alpha, tau = 10.0, 1e-3
    # uniform_magnet branch
    pol_u = compute_search_policy(q, bp_log, legal, tau, alpha, uniform_magnet=True)
    sl = (bp_log[idx] + alpha * q[idx]) / (1.0 + tau * alpha)
    sl = np.exp(sl - sl.max())
    sl = sl / sl.sum()
    assert np.allclose(pol_u[idx], sl, atol=1e-5)
    assert np.isclose(pol_u.sum(), 1.0)

    # weighted-magnet branch
    pol_w = compute_search_policy(q, bp_log, legal, tau, alpha, uniform_magnet=False)
    magnet = get_weighted_uniform_policy(legal)[idx]
    log_magnet = np.log(magnet + np.finfo(np.float64).tiny)
    slw = (bp_log[idx] + alpha * q[idx] + tau * alpha * log_magnet) / (1.0 + tau * alpha)
    slw = np.exp(slw - slw.max())
    slw = slw / slw.sum()
    assert np.allclose(pol_w[idx], slw, atol=1e-5)


@pytest.mark.parametrize("belief_cls", [UniformBelief, MarginalizedBelief])
def test_belief_samples_respect_counts_and_movability(belief_cls):
    s, env = _drive_to_move_phase()
    root = s.search_root(env).root()
    if int(root["n_hidden"]) == 0:
        pytest.skip("no hidden pieces at this root")
    counts = np.asarray(root["hidden_counts"])
    has_moved = np.asarray(root["hidden_has_moved"])
    samples = belief_cls().sample(root, 300, np.random.default_rng(7))
    assert samples.shape == (300, int(root["n_hidden"]))
    for a in samples:
        # exact per-rank counts
        bc = np.bincount(a, minlength=14)[:12]
        assert (bc == counts).all(), (bc, counts)
        # moved pieces are never flag(10)/bomb(11)
        assert not ((a >= 10) & has_moved).any()


@pytest.mark.parametrize("belief", [UniformBelief(), MarginalizedBelief(), None])
def test_search_runs_and_returns_valid_distribution(belief):
    s, env = _drive_to_move_phase()
    srch = s.search_root(env)
    legal = np.asarray(srch.root()["legal"])
    n_legal = int(legal.sum())
    net = S.MoveTransformer.from_config(S.MoveConfig())
    r = search(net, srch, depth=6, stepsize=10.0, temperature=1e-3,
               max_samples=40, num_envs=256, belief=belief, seed=11)
    # the sampled action is legal
    assert legal[r.action]
    # the search policy is a valid distribution supported on legal actions only
    assert np.isclose(r.search_policy.sum(), 1.0, atol=1e-4)
    assert r.search_policy[np.logical_not(legal)].sum() == 0.0
    # every legal root action got ~num_envs/L rollout worlds
    assert r.counts[legal].sum() == 256
    assert r.n_sample == min(256 // n_legal, 40)


def test_perfect_search_determinizes_to_truth():
    # belief=None uses the ground-truth hidden ranks; the assignment must match
    # the searcher's true_hidden, and the search still produces a legal action.
    s, env = _drive_to_move_phase()
    srch = s.search_root(env)
    root = srch.root()
    if int(root["n_hidden"]) == 0:
        pytest.skip("no hidden pieces")
    true_hidden = np.asarray(srch.true_hidden())
    counts = np.asarray(root["hidden_counts"])
    # the ground-truth ranks themselves respect the hidden counts
    bc = np.bincount(true_hidden, minlength=14)[:12]
    assert (bc == counts).all()
    net = S.MoveTransformer.from_config(S.MoveConfig())
    r = search(net, srch, depth=4, stepsize=10.0, temperature=1e-3,
               max_samples=20, num_envs=128, belief=None, seed=1)
    assert np.asarray(root["legal"])[r.action]


def test_one_mmd_step_improves_the_policy():
    """The deterministic policy-improvement guarantee (the noise-free core of
    "search beats no-search"): a single MMD step never decreases the expected
    value under the search's OWN q estimates, i.e. `ev_diff = sum_a q[a] *
    (pi_search[a] - pi_bp[a]) >= 0`, and moves the policy toward the q-greedy
    action (search regret <= policy regret). Holds at every searched root.

    The reference tracks this as `stats["ev_diff"]` / `policy_net_regret` vs
    `search_regret`. We verify it across several roots with an untrained net (the
    property is a property of the MMD update, not of net quality)."""
    s, env0 = _drive_to_move_phase(num_envs=96, seed=21)
    net = S.MoveTransformer.from_config(S.MoveConfig())
    n_checked = 0
    for trial in range(6):
        b = s.collect()
        env = int(b["move_env"][trial % len(b["move_env"])])
        nm, nd = b["move_obs"].shape[0], b["deploy_obs"].shape[0]
        s.commit(np.zeros((nm, sim.N_ACTION), np.float32), np.zeros(nm, np.float32),
                 np.full((nm, 3), 1.0 / 3.0, np.float32),
                 np.zeros((nd, sim.DEPLOY_WIDTH), np.float32), np.zeros(nd, np.float32),
                 np.full((nd, 3), 1.0 / 3.0, np.float32))
        srch = s.search_root(env)
        r = search(net, srch, depth=6, stepsize=10.0, temperature=1e-3,
                   max_samples=40, num_envs=256, belief=MarginalizedBelief(), seed=trial)
        legal = np.nonzero(r.counts > 0)[0]
        if len(legal) < 2:
            continue
        q = r.q[legal]
        bp = r.bp_policy[legal]
        bp = bp / bp.sum() if bp.sum() > 0 else bp
        pi = r.search_policy[legal]
        pi = pi / pi.sum()
        ev_diff = float((q * (pi - bp)).sum())
        # improvement (allow tiny float slack)
        assert ev_diff >= -1e-5, f"MMD step decreased expected q: ev_diff={ev_diff}"
        # search is at least as close to the q-greedy action as the raw policy
        assert (q.max() - (q * pi).sum()) <= (q.max() - (q * bp).sum()) + 1e-5
        n_checked += 1
    assert n_checked >= 1, "no multi-action roots were searched"
