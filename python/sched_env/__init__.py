"""Gymnasium wrapper around the Zig scheduling simulator (libsched)."""

import gymnasium as gym

from .env import SchedEnv

gym.register(id="Sched-v0", entry_point="sched_env.env:SchedEnv")

__all__ = ["SchedEnv"]
