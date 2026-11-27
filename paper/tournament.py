"""Tournament action selection (evaluation only, no training).

A policy trained on an env with W job slots (e.g. ppo.py --jobs 10) is run
unchanged on an env with more jobs:

  1. The jobs that can be placed now are split, in order, into groups of W
     (25 jobs, W=10 -> 10, 10, 5). A short group is padded with empty slots
     (valid = 0, all features 0, masked out), so the policy always sees W.
  2. In each group the policy's most probable placement wins; its job moves on.
  3. Repeat on the winners until at most W remain, then a final round over
     the winners decides -- here "wait" is allowed again, and the machine of
     the winning job is chosen too.

Padding is masking, not +inf/max-float values: a padded slot can never be
chosen and is excluded from attention, while huge feature values would only
push activations out of range.
"""

from __future__ import annotations

import numpy as np

JOB_KEYS = ("jobs", "valid", "wait", "age")


def window_obs(obs: dict, idx, W: int) -> dict:
    """Observation with only the job slots `idx` (len <= W), padded to W."""
    out = {"machines": obs["machines"]}
    for k in JOB_KEYS:
        a = np.zeros((W, *obs[k].shape[1:]), dtype=obs[k].dtype)
        a[:len(idx)] = obs[k][idx]
        out[k] = a
    return out


def window_mask(mask: np.ndarray, idx, W: int, M: int, allow_wait: bool = True) -> np.ndarray:
    """Action mask of the window: slot k <- job idx[k]; padded slots illegal."""
    m = np.zeros(W * M + 1, dtype=bool)
    full = mask[:-1].reshape(-1, M)
    m[:len(idx) * M] = full[idx].reshape(-1)
    m[-1] = allow_wait and bool(mask[-1])
    return m


def placeable(mask: np.ndarray, M: int) -> np.ndarray:
    """Job slots with at least one legal placement, in slot order."""
    return np.flatnonzero(mask[:-1].reshape(-1, M).any(1))


def tournament_action(probs_fn, obs: dict, mask: np.ndarray, W: int, M: int) -> int:
    """`probs_fn(window_obs, window_mask) -> probabilities over W·M+1` ."""
    cand = placeable(mask, M)
    if len(cand) == 0:
        return len(mask) - 1  # nothing fits: wait
    while len(cand) > W:
        winners = []
        for g in range(0, len(cand), W):
            idx = cand[g:g + W]
            wm = window_mask(mask, idx, W, M, allow_wait=False)
            p = np.where(wm, probs_fn(window_obs(obs, idx, W), wm), -1.0)
            winners.append(idx[int(np.argmax(p)) // M])
        cand = np.asarray(winners)
    wm = window_mask(mask, cand, W, M)
    a = int(np.argmax(np.where(wm, probs_fn(window_obs(obs, cand, W), wm), -1.0)))
    if a == W * M:
        return len(mask) - 1
    return int(cand[a // M]) * M + a % M
