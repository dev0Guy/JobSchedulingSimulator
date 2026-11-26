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
    assert truncation_penalty(-0.5, 0.999) == pytest.approx(-0.5 * 999)
    assert truncation_penalty(-0.5, 0.999, 0.0) == 0.0
    with pytest.raises(ValueError):
        truncation_penalty(-1.0, 1.0)


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
