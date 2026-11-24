"""Smoke tests for the Python binding:  PYTHONPATH=python pytest python/tests"""

import numpy as np
from gymnasium.utils.env_checker import check_env

from sched_env import SchedEnv

KW = dict(machines="deeprm:n=2", jobs="poisson:n=30,rate=0.5", n_res=2, n_ticks=20, max_jobs=8)


def test_check_env():
    env = SchedEnv(**KW)
    check_env(env, skip_render_check=True)
    env.close()


def test_masked_episode_places_every_job():
    env = SchedEnv(**KW)
    obs, _ = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    done = False
    while not done:
        mask = env.action_masks()
        assert mask[-1]  # wait is always legal
        a = int(np.flatnonzero(mask)[0])
        obs, r, term, trunc, info = env.step(a)
        assert not info["illegal"]
        done = term or trunc
    assert info["stats"]["placed"] == info["stats"]["arrived"]
    env.close()


def test_builtin_policy_and_determinism():
    env = SchedEnv(**KW)
    a = env.run_policy("sjf", 7)
    b = env.run_policy("sjf", 7)
    assert a == b and a["placed"] == a["arrived"]
    env.close()


def test_policy_action_matches_run_policy():
    env = SchedEnv(**KW)
    want = env.run_policy("sjf", 3)
    env.reset(seed=3)
    done = False
    while not done:
        _, _, term, trunc, info = env.step(env.policy_action("sjf"))
        done = term or trunc
    assert info["stats"]["avg_wait"] == want["avg_wait"]
    assert info["stats"]["makespan"] == want["makespan"]
    env.close()


def test_viz_record_and_render(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from sched_env.viz import animate, record

    env = SchedEnv(**KW)
    frames = record(env, "sjf", seed=0)
    assert len(frames) > 1
    assert sum(f.placed_now for f in frames) == frames[-1].stats["placed"]
    assert all(0 <= f.used.min() and f.used.max() <= 1 for f in frames)
    animate(env, frames[:5], str(tmp_path / "a.gif"), fps=10)
    assert (tmp_path / "a.gif").stat().st_size > 0
    env.close()


def test_machine_view_follows_sliding_window():
    # long episode → the window compacts several times
    env = SchedEnv(**{**KW, "jobs": "poisson:n=150,rate=0.4"})
    cap = env.capacity()
    obs, _ = env.reset(seed=2)
    done, ticks = False, 0
    while not done:
        free = env.raw()["free"]
        assert (free <= cap[:, :, None]).all()
        a = env.policy_action("fifo")
        ticks += a == env.action_space.n - 1
        obs, _, term, trunc, _ = env.step(a)
        done = term or trunc
    assert ticks > 5 * KW["n_ticks"]
    assert np.allclose(obs["machines"], cap[:, :, None] / env.dims.max_cap)  # all jobs finished → empty
    env.close()


def test_reward_specs_and_new_adapters():
    # slowdown reward sums to minus the total slowdown of finished jobs
    env = SchedEnv(**{**KW, "reward": "slowdown"})
    env.reset(seed=4)
    total, done = 0.0, False
    while not done:
        _, r, term, trunc, info = env.step(env.policy_action("tetris"))
        total += r
        done = term or trunc
    assert total < 0 and info["stats"]["placed"] == info["stats"]["arrived"]
    env.close()
    for jobs in ("mmpp:n=50,rate=0.5", "pareto:n=50,rate=0.5,alpha=1.3", "selfsimilar:n=50,rate=0.5"):
        e = SchedEnv(**{**KW, "jobs": jobs})
        st = e.run_policy("tetris", 1)
        assert st["placed"] == st["arrived"] > 0
        e.close()
    import pytest
    with pytest.raises(ValueError):
        SchedEnv(**{**KW, "reward": "nope"})
