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
