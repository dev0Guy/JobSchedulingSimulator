"""ctypes bindings for libsched (src/app/capi.zig). Build it with `zig build lib`."""

from __future__ import annotations

import ctypes as C
import os
import sys
from pathlib import Path

import numpy as np


class SimConfig(C.Structure):
    _fields_ = [
        ("machines", C.c_char_p),
        ("jobs", C.c_char_p),
        ("n_res", C.c_uint32),
        ("n_ticks", C.c_uint32),
        ("max_jobs", C.c_uint32),
        ("reward", C.c_uint32),
        ("max_ticks", C.c_int64),
        ("illegal_penalty", C.c_float),
        ("skip_empty", C.c_uint8),
        ("drop_impossible", C.c_uint8),
    ]


class SimDims(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in
                ("n_machines", "n_res", "n_ticks", "max_jobs", "n_actions", "stride", "block", "max_cap",
                 "m_stride", "m_block")]


class SimStep(C.Structure):
    _fields_ = [
        ("reward", C.c_float),
        ("terminated", C.c_uint8),
        ("truncated", C.c_uint8),
        ("placed", C.c_uint8),
        ("illegal", C.c_uint8),
    ]


class SimStats(C.Structure):
    _fields_ = [
        ("now", C.c_int64), ("arrived", C.c_uint64), ("placed", C.c_uint64),
        ("rejected", C.c_uint64), ("illegal", C.c_uint64), ("steps", C.c_uint64),
        ("waiting", C.c_uint64), ("makespan", C.c_int64), ("avg_wait", C.c_double),
        ("max_wait", C.c_int64), ("avg_util", C.c_double), ("total_reward", C.c_double),
    ]

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name, _ in self._fields_}


REWARDS = {"neg_in_system": 0, "neg_waiting": 1, "zero": 2}


def _find_lib() -> Path:
    if env := os.environ.get("SCHED_LIB"):
        return Path(env)
    name = {"darwin": "libsched.dylib", "win32": "sched.dll"}.get(sys.platform, "libsched.so")
    root = Path(__file__).resolve().parents[2]  # repo root
    for p in (root / "zig-out" / "lib" / name, root / "zig-out" / "bin" / name):
        if p.exists():
            return p
    raise FileNotFoundError(f"{name} not found; run `zig build lib` in {root} or set SCHED_LIB")


def load() -> C.CDLL:
    lib = C.CDLL(str(_find_lib()))
    P = C.c_void_p
    u16p = C.POINTER(C.c_uint16)
    sig = {
        "sim_create": ([C.POINTER(SimConfig)], P),
        "sim_destroy": ([P], None),
        "sim_reset": ([P, C.c_uint64], C.c_int),
        "sim_dims": ([P, C.POINTER(SimDims)], None),
        "sim_step": ([P, C.c_uint32, C.POINTER(SimStep)], C.c_int),
        "sim_action_mask": ([P, C.c_void_p], None),
        "sim_slot_valid": ([P, C.c_void_p], None),
        "sim_machines_ptr": ([P], u16p),
        "sim_machine_head": ([P], C.c_uint32),
        "sim_capacity_ptr": ([P], u16p),
        "sim_jobs_ptr": ([P], u16p),
        "sim_duration_ptr": ([P], u16p),
        "sim_arrival_ptr": ([P], C.POINTER(C.c_int64)),
        "sim_now": ([P], C.c_int64),
        "sim_stats": ([P, C.POINTER(SimStats)], None),
        "sim_policy_action": ([P, C.c_char_p], C.c_int64),
        "sim_run_policy": ([P, C.c_char_p, C.c_uint64, C.POINTER(SimStats)], C.c_int),
    }
    for name, (args, res) in sig.items():
        f = getattr(lib, name)
        f.argtypes, f.restype = args, res
    return lib


def grid_view(ptr, n_grids: int, d: SimDims, *, block=None, stride=None, offset=0) -> np.ndarray:
    """Zero-copy (n_grids, n_res, n_ticks) uint16 view; strides skip the padding.

    Job pool: defaults. Machine pool: block=d.m_block, stride=d.m_stride,
    offset=sim_machine_head (the view is only valid until time advances)."""
    block = d.block if block is None else block
    stride = d.stride if stride is None else stride
    flat = np.ctypeslib.as_array(ptr, shape=(n_grids * block,))
    e = flat.itemsize
    return np.lib.stride_tricks.as_strided(
        flat[offset:], shape=(n_grids, d.n_res, d.n_ticks),
        strides=(block * e, stride * e, e), writeable=False)
