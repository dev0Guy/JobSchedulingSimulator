"""paper/train.py: the max_ticks cut-off penalty (starvation must stay expensive)."""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "paper"))
from ptrmcts import make_env, truncation_penalty  # noqa: E402
from train import run_episode  # noqa: E402


def _always_wait(env):
    wait = env.dims.n_actions - 1

    def choose(obs, mask, prefix, expect, step):
        a = wait if mask[wait] else int(np.flatnonzero(mask)[0])
        pi = np.zeros(len(mask), np.float32)
        pi[a] = 1
        return a, pi
    return choose


def test_penalty_formula():
    env = make_env(max_ticks=50)
    env.reset(seed=1)
    wait = env.dims.n_actions - 1
    done = False
    while not done:
        done = any(env.step(wait)[2:4])  # never place: cut off at tick 50
    raw = env.raw()
    dur = raw["duration"][raw["valid"]].astype(float)
    assert len(dur) > 0 and raw["now"] == 50
    want = -sum((50 + d) / d + 10 for d in dur)
    assert truncation_penalty(env) == pytest.approx(want)
    assert truncation_penalty(env, 0.0) == 0.0
    assert truncation_penalty(env, 2.0) == pytest.approx(2 * want)
    env.close()


def test_cut_off_episode_gets_penalty_but_logged_return_does_not():
    env = make_env(max_ticks=50)
    gamma, scale = 0.999, 10.0
    s0, stats, ret0 = run_episode(env, 1, _always_wait(env), gamma, scale, trunc_penalty=0.0)
    s1, _, ret1 = run_episode(env, 1, _always_wait(env), gamma, scale, trunc_penalty=1.0)
    assert stats["placed"] < 80  # really starved, really cut off
    assert ret0 == pytest.approx(ret1)                      # logged return: true reward only
    assert s1[0][3] < s0[0][3] - 1.0                        # training target: much worse
    env.close()


def test_work_conserving_mask_blocks_wait_only_when_a_job_fits():
    from sched_env import SchedEnv
    kw = dict(machines="deeprm:n=1", jobs="poisson:n=40,rate=0.8", n_res=2, n_ticks=20, max_jobs=40,
              reward="slowdown", max_ticks=1000)
    env, ref = SchedEnv(**kw, work_conserving=True), SchedEnv(**kw)
    env.reset(seed=3), ref.reset(seed=3)
    rng = np.random.default_rng(0)
    seen_blocked = seen_free = False
    done = False
    while not done:
        m, r = env.action_masks(), ref.action_masks()
        assert (m[:-1] == r[:-1]).all()            # placements unchanged
        assert m[-1] == (r[-1] and not r[:-1].any())  # wait only when nothing fits
        seen_blocked |= r[:-1].any()
        seen_free |= not r[:-1].any()
        a = int(rng.choice(np.flatnonzero(m)))
        done = any(env.step(a)[2:4])
        ref.step(a)
    assert seen_blocked and seen_free
    env.close(), ref.close()



def test_age_keeps_growing_after_wait_clips():
    env = make_env(max_ticks=200)
    obs, _ = env.reset(seed=1)
    wait_a = env.dims.n_actions - 1
    first = None
    for t in range(60):
        if obs["valid"].any() and first is None:
            first = int(np.flatnonzero(obs["valid"])[0])
        obs, *_ = env.step(wait_a)  # never place anything
        if first is not None and t == 30:
            w30, a30 = obs["wait"][first], obs["age"][first]
    assert w30 == obs["wait"][first] == 1.0           # wait saturates at n_ticks
    assert obs["age"][first] > a30 > 1.0              # age does not
    from ptrmcts import featurize
    assert featurize(obs)["jobs"].shape[1] == env.dims.n_res * env.dims.n_ticks + 3
    env.close()


def test_pointer_net_shared_grid_encoder():
    import torch
    from ptrmcts import JOB_EXTRA, PointerNet, collate, featurize
    env = make_env()
    obs, _ = env.reset(seed=2)
    net = PointerNet.for_env(env).eval()
    d = env.dims
    assert net.grid.conv.kernel_size == (1, d.n_ticks)       # one time row per resource
    # one encoder for both: the same grid gives the same embedding
    g = torch.rand(1, 1, d.n_res * d.n_ticks)
    assert torch.equal(net.grid(g), net.grid(g.clone()))
    assert not any("job_in" in k or "mach_in" in k for k, _ in net.named_parameters())
    # logits cover every (job, machine) pair + wait; jobs are a set
    f = featurize(obs)
    assert f["jobs"].shape[1] == d.n_res * d.n_ticks + JOB_EXTRA
    mask = env.action_masks()
    with torch.no_grad():
        logits, v = net(*collate([f], [mask], "cpu"))
        perm = np.random.default_rng(0).permutation(d.max_jobs)
        fp = {"jobs": f["jobs"][perm], "mach": f["mach"], "valid": f["valid"][perm]}
        mp = np.concatenate([mask[:-1].reshape(d.max_jobs, -1)[perm].ravel(), mask[-1:]])
        lp, vp = net(*collate([fp], [mp], "cpu"))
    assert logits.shape == (1, d.n_actions)
    lj = logits[0, :-1].reshape(d.max_jobs, -1)[perm]
    assert torch.allclose(lj, lp[0, :-1].reshape(d.max_jobs, -1), atol=1e-4, equal_nan=True)
    assert torch.allclose(v, vp, atol=1e-4)
    env.close()


def test_value_transform_round_trips():
    from ptrmcts import value_h, value_h_inv
    for x in [-2000.0, -15.0, -0.3, 0.0, 0.7, 40.0]:
        assert value_h_inv(value_h(x)) == pytest.approx(x, rel=1e-4, abs=1e-4)
    assert abs(value_h(-2000.0)) < 50  # squashed


def test_bootstrapped_targets():
    from train import bootstrapped, discounted
    r, g, sc = [-1.0, -2.0, -3.0, -4.0], 0.5, 1.0
    v = [None, 10.0, 20.0, None]
    assert bootstrapped(r, v, g, sc, 0) == pytest.approx(discounted(r, g, sc))
    t = bootstrapped(r, v, g, sc, 2)
    assert t[0] == pytest.approx(-1 + 0.5 * -2 + 0.25 * 20)       # bootstraps from v[2]
    assert t[1] == pytest.approx(discounted(r, g, sc)[1])          # v[3] unknown -> full return
    assert t[2:] == pytest.approx(discounted(r, g, sc)[2:])        # near the end -> full return


def test_batched_search_matches_visit_budget():
    import torch
    from ptrmcts import ENV_KW, MCTS, PointerNet, fingerprint
    env = make_env()
    env.reset(seed=1)
    prefix = []
    for _ in range(30):
        m = env.action_masks()
        a = int(np.flatnonzero(m)[0])
        env.step(a)
        prefix.append(a)
    net = PointerNet.for_env(env).eval()
    for b in (1, 5, 8):
        mc = MCTS(net, ENV_KW, sims=21, batch=b)
        with torch.no_grad():
            c = mc.search(1, prefix, fingerprint(env), noise=True)
        assert c.sum() == 21                                 # every simulation counted once
        assert (c[~env.action_masks()] == 0).all()           # only legal actions
        assert np.isfinite(mc.root_value)
    env.close()


def test_fit_mixes_fixed_teacher_share_into_every_batch(monkeypatch):
    import types
    import train
    seen = []
    monkeypatch.setattr(train, "train_step", lambda net, opt, batch, dev, vc: (seen.append(batch), (0.0, 0.0))[1])
    args = types.SimpleNamespace(batch=8, updates=3, bc_mix=0.25, value_coef=0.5)
    buffer, anchor = [("self",)] * 20, [("teacher",)] * 5
    train.fit(None, None, buffer, args, "cpu", np.random.default_rng(0), anchor)
    assert len(seen) == 3
    for b in seen:
        assert len(b) == 8 and sum(x == ("teacher",) for x in b) == 2


def test_ppo_train_env_uses_training_seeds_and_scales_reward():
    from ppo import TrainEnv
    from ptrmcts import ENV_KW, EVAL_SEED_START, train_seed_range
    seeds = train_seed_range(1, 100)
    env = TrainEnv(dict(ENV_KW), seeds, offset=3, stride=2, scale=10.0, trunc_penalty=0.0)
    ref = make_env()
    obs, _ = env.reset()
    ref_obs, _ = ref.reset(seed=seeds[3])
    assert np.array_equal(obs["jobs"], ref_obs["jobs"]) and seeds[3] < EVAL_SEED_START
    a = int(np.flatnonzero(env.action_masks())[0])
    _, r, *_ = env.step(a)
    _, r_ref, *_ = ref.step(a)
    assert r == pytest.approx(r_ref / 10.0)


def test_ppo_ptr_policy_only_picks_legal_actions():
    import torch
    from ppo import PtrPolicy
    from stable_baselines3.common.utils import obs_as_tensor
    env = make_env()
    pol = PtrPolicy(env.observation_space, env.action_space, lambda _: 1e-4, ptr_kw={"d": 32})
    obs, _ = env.reset(seed=3)
    for _ in range(30):
        mask = env.action_masks()
        o = {k: v[None] for k, v in obs_as_tensor(obs, "cpu").items()}
        a, v, lp = pol(o, action_masks=mask[None])
        assert mask[int(a)] and torch.isfinite(lp).all() and v.shape == (1, 1)
        obs, _, term, trunc, _ = env.step(int(a))
        if term or trunc:
            break


def test_tournament_groups_pad_and_maps_back():
    from tournament import tournament_action, window_mask, window_obs
    W, M, J = 10, 1, 25
    rng = np.random.default_rng(0)
    obs = {"jobs": rng.random((J, 2, 4)).astype(np.float32), "valid": np.ones(J, np.float32),
           "wait": np.zeros(J, np.float32), "age": np.arange(J, dtype=np.float32), "machines": np.zeros((1, 2, 4))}
    mask = np.ones(J * M + 1, bool)
    calls = []

    def probs(o, m):  # prefers the job with the largest age; padded slots must be masked
        calls.append(int(m[:-1].sum()))
        assert o["jobs"].shape[0] == W and not o["valid"][int(m[:-1].sum()):].any()
        p = np.where(m[:-1], o["age"] + 1, 0)
        return np.append(p, 0.0) / p.sum()

    assert tournament_action(probs, obs, mask, W, M) == 24
    assert calls == [10, 10, 5, 3]  # groups 10,10,5 then a final round of 3 winners
    wm = window_mask(mask, np.array([3, 7]), W, M)
    assert wm[:2].all() and not wm[2:-1].any() and wm[-1]
    assert window_obs(obs, np.array([3, 7]), W)["age"][:2].tolist() == [3, 7]
