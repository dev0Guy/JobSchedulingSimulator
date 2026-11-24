"""Gymnasium environment backed by the Zig engine."""

from __future__ import annotations

import ctypes as C
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import _lib


class SchedEnv(gym.Env):
    """Job-to-machine scheduling.

    Observation (Dict, float32, everything in [0, 1]):
      machines  (n_machines, n_res, n_ticks)  free capacity / max_cap
      jobs      (max_jobs,   n_res, n_ticks)  demand / max_cap (0 = empty slot)
      valid     (max_jobs,)                   1 if the slot holds a waiting job
      wait      (max_jobs,)                   ticks waited / n_ticks, clipped to 1

    Action: Discrete(max_jobs * n_machines + 1)
      a <  max_jobs*n_machines -> place job slot a // n_machines on machine a % n_machines
      a == max_jobs*n_machines -> wait (advance time one tick)

    `action_masks()` returns the legal actions (used by sb3_contrib MaskablePPO).
    Illegal actions are penalized (`illegal_penalty`) and treated as wait.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        machines: str = "deeprm:n=2",
        jobs: str = "poisson:n=100,rate=0.5",
        n_res: int = 2,
        n_ticks: int = 20,
        max_jobs: int = 10,
        reward: str = "neg_in_system",
        max_ticks: int | None = None,
        illegal_penalty: float = -1.0,
        skip_empty: bool = True,
        drop_impossible: bool = True,
        render_mode: str | None = None,
    ):
        self._lib = _lib.load()
        self._spec = (machines.encode(), jobs.encode())  # keep bytes alive
        cfg = _lib.SimConfig(
            self._spec[0], self._spec[1], n_res, n_ticks, max_jobs, _lib.REWARDS[reward],
            -1 if max_ticks is None else max_ticks, illegal_penalty, int(skip_empty), int(drop_impossible))
        self._h = self._lib.sim_create(C.byref(cfg))
        if not self._h:
            raise ValueError("sim_create failed (see stderr for the reason)")

        d = _lib.SimDims()
        self._lib.sim_dims(self._h, C.byref(d))
        self.dims = d
        self.render_mode = render_mode

        # zero-copy views into engine memory (always current)
        self._mptr = self._lib.sim_machines_ptr(self._h)
        self._jobs = _lib.grid_view(self._lib.sim_jobs_ptr(self._h), d.max_jobs, d)
        self._arrival = np.ctypeslib.as_array(self._lib.sim_arrival_ptr(self._h), shape=(d.max_jobs,))
        self._valid = np.zeros(d.max_jobs, dtype=np.uint8)
        self._mask = np.zeros(d.n_actions, dtype=np.uint8)
        self._res = _lib.SimStep()
        self._scale = 1.0 / max(1, d.max_cap)

        self.action_space = spaces.Discrete(d.n_actions)
        self.observation_space = spaces.Dict({
            "machines": spaces.Box(0, 1, (d.n_machines, d.n_res, d.n_ticks), np.float32),
            "jobs": spaces.Box(0, 1, (d.max_jobs, d.n_res, d.n_ticks), np.float32),
            "valid": spaces.Box(0, 1, (d.max_jobs,), np.float32),
            "wait": spaces.Box(0, 1, (d.max_jobs,), np.float32),
        })

    # ------------------------------------------------------------------ gym API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        s = int(self.np_random.integers(0, 2**63)) if seed is None else seed
        if self._lib.sim_reset(self._h, s) != 0:
            raise RuntimeError("sim_reset failed")
        return self._obs(), {}

    def step(self, action):
        rc = self._lib.sim_step(self._h, int(action), C.byref(self._res))
        if rc != 0:
            raise RuntimeError("step() after the episode ended; call reset()")
        r = self._res
        info = {"illegal": bool(r.illegal), "placed": bool(r.placed)}
        if r.terminated or r.truncated:
            info["stats"] = self.stats()
        return self._obs(), float(r.reward), bool(r.terminated), bool(r.truncated), info

    def action_masks(self) -> np.ndarray:
        self._lib.sim_action_mask(self._h, self._mask.ctypes.data)
        return self._mask.astype(bool)

    def render(self):
        s = self.stats()
        return f"t={s['now']} waiting={s['waiting']} placed={s['placed']}/{s['arrived']}"

    def close(self):
        if getattr(self, "_h", None):
            self._lib.sim_destroy(self._h)
            self._h = None

    def __del__(self):
        self.close()

    # ------------------------------------------------------------------ extras

    def stats(self) -> dict[str, Any]:
        st = _lib.SimStats()
        self._lib.sim_stats(self._h, C.byref(st))
        return st.to_dict()

    def run_policy(self, name: str, seed: int) -> dict[str, Any]:
        """Run a whole episode with a built-in Zig policy (fifo, sjf, best_fit, ...)."""
        st = _lib.SimStats()
        if self._lib.sim_run_policy(self._h, name.encode(), seed, C.byref(st)) != 0:
            raise ValueError(f"unknown policy {name!r}")
        return st.to_dict()

    @property
    def _machines(self) -> np.ndarray:
        """Zero-copy view of the machine horizon (rebuilt: the window head moves)."""
        d = self.dims
        return _lib.grid_view(self._mptr, d.n_machines, d, block=d.m_block, stride=d.m_stride,
                              offset=self._lib.sim_machine_head(self._h))

    def policy_action(self, name: str) -> int:
        """Action a built-in Zig policy would take now (step it yourself)."""
        a = self._lib.sim_policy_action(self._h, name.encode())
        if a < 0:
            raise ValueError(f"unknown policy {name!r}")
        return int(a)

    def capacity(self) -> np.ndarray:
        """(n_machines, n_res) capacities."""
        d = self.dims
        return np.ctypeslib.as_array(self._lib.sim_capacity_ptr(self._h),
                                     shape=(d.n_machines * d.n_res,)).reshape(d.n_machines, d.n_res).copy()

    def raw(self) -> dict[str, np.ndarray]:
        """Unnormalized current state (copies): free, jobs, duration, arrival, valid, now."""
        self._lib.sim_slot_valid(self._h, self._valid.ctypes.data)
        dur = np.ctypeslib.as_array(self._lib.sim_duration_ptr(self._h), shape=(self.dims.max_jobs,))
        return {"free": self._machines.copy(), "jobs": self._jobs.copy(), "duration": dur.copy(),
                "arrival": self._arrival.copy(), "valid": self._valid.astype(bool),
                "now": self._lib.sim_now(self._h)}

    def _obs(self) -> dict[str, np.ndarray]:
        self._lib.sim_slot_valid(self._h, self._valid.ctypes.data)
        now = self._lib.sim_now(self._h)
        valid = self._valid.astype(np.float32)
        wait = np.clip((now - self._arrival) / self.dims.n_ticks, 0, 1).astype(np.float32) * valid
        return {
            "machines": self._machines.astype(np.float32) * self._scale,
            "jobs": self._jobs.astype(np.float32) * self._scale,
            "valid": valid,
            "wait": wait,
        }
