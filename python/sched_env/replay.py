"""Replay a trace file written by the `trace` observer — no simulator needed.

Record:
    zig build run -- --observe trace:path=run.jsonl ...          (command line)
    env.start_trace("run.jsonl"); env.reset(seed=0); ...; env.stop_trace()   (Python)

Replay:
    from sched_env.replay import load_trace
    from sched_env.viz import animate
    episodes = load_trace("run.jsonl")
    ep = episodes[0]
    print(ep.stats)                    # placed, arrived, avg_wait, makespan, ...
    animate(ep, ep.frames, "run.gif")   # same visualization as a live run

Each episode is rebuilt event by event into one Frame per tick (state at the
start of the tick, after arrivals), identical to what sched_env.viz.record
captures live with skip_empty=False.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .viz import Frame


@dataclass
class _Job:
    id: int
    slot: int
    arrival: int
    demand: np.ndarray  # (n_res, dur)


@dataclass
class Episode:
    seed: int
    caps: np.ndarray                          # (n_machines, n_res)
    dims: SimpleNamespace                     # n_machines, n_res, n_ticks, max_jobs, max_cap
    frames: list[Frame] = field(default_factory=list)
    placements: list[dict] = field(default_factory=list)  # every "place" event
    rejected: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    truncated: bool = False


class _Rebuild:
    """Mutable state while reading one episode."""

    def __init__(self, ep: Episode):
        self.ep = ep
        self.now = 0
        self.waiting: dict[int, _Job] = {}      # slot -> job
        self.running: list[tuple[int, int, np.ndarray]] = []  # (machine, start, demand)
        self.arrived = self.placed = self.illegal = 0
        self.total_wait = 0
        self.max_wait = 0
        self.makespan = 0
        self.util_sum = 0.0
        self.placed_now = 0
        self.started = False  # first frame (tick 0) taken?

    def stats(self) -> dict:
        n = max(1, len(self.ep.frames))
        return {
            "now": self.now, "arrived": self.arrived, "placed": self.placed,
            "rejected": len(self.ep.rejected), "illegal": self.illegal,
            "waiting": len(self.waiting), "makespan": self.makespan,
            "avg_wait": self.total_wait / self.placed if self.placed else 0.0,
            "max_wait": self.max_wait, "avg_util": self.util_sum / n,
        }

    def snapshot(self) -> None:
        d = self.ep.dims
        nm, nr, nt = d.n_machines, d.n_res, d.n_ticks
        self.running = [r for r in self.running if r[1] + r[2].shape[1] > self.now]
        used = np.zeros((nm, nr, nt), dtype=np.float32)
        for m, start, dem in self.running:
            off = self.now - start                       # ticks already run
            k = min(nt, dem.shape[1] - off)
            used[m, :, :k] += dem[:, off:off + k]
        used /= np.maximum(self.ep.caps, 1)[:, :, None]
        pool = np.full((d.max_jobs, nr, nt), np.nan, dtype=np.float32)
        wait = np.full(d.max_jobs, -1)
        for s, j in self.waiting.items():
            pool[s, :, :j.demand.shape[1]] = j.demand / max(1, d.max_cap)
            wait[s] = self.now - j.arrival
        self.util_sum += float(used[:, :, 0].mean())
        self.ep.frames.append(Frame(self.now, used, pool.reshape(-1, nt), wait, 0, {}))
        self.ep.frames[-1].stats = self.stats()

    def flush_placed(self) -> None:
        if self.ep.frames:
            self.ep.frames[-1].placed_now += self.placed_now
            self.ep.frames[-1].stats = {**self.stats(), "now": self.ep.frames[-1].now,
                                        "waiting": self.ep.frames[-1].stats["waiting"]}
        self.placed_now = 0

    def event(self, e: dict) -> None:
        ty = e["type"]
        if ty in ("arrive", "reject"):
            if ty == "arrive":
                dem = np.asarray(e["demand"], dtype=np.float32).reshape(self.ep.dims.n_res, e["dur"])
                self.waiting[e["slot"]] = _Job(e["id"], e["slot"], e["t"], dem)
                self.arrived += 1
            else:
                self.ep.rejected.append(e)
            return
        if not self.started:  # arrivals at tick 0 are in: that is frame 0
            self.started = True
            self.snapshot()
        if ty == "place":
            j = self.waiting.pop(e["slot"])
            self.running.append((e["machine"], e["t"], j.demand))
            w = e["t"] - e["arrival"]
            self.placed += 1
            self.placed_now += 1
            self.total_wait += w
            self.max_wait = max(self.max_wait, w)
            self.makespan = max(self.makespan, e["t"] + e["dur"])
            self.ep.placements.append(e)
        elif ty == "illegal":
            self.illegal += 1
        elif ty == "tick":
            self.flush_placed()
            self.now = e["t"]
            self.snapshot()
        elif ty == "done":
            self.flush_placed()
            self.now = e["t"]
            self.ep.truncated = bool(e["truncated"])
            self.ep.stats = self.stats()


def load_trace(path: str | Path) -> list[Episode]:
    """Read every episode of a trace file."""
    episodes: list[Episode] = []
    rb: _Rebuild | None = None
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            if e["type"] == "episode":
                if e.get("v", 1) != 1:
                    raise ValueError(f"unsupported trace version {e.get('v')}")
                caps = np.asarray(e["caps"], dtype=np.int64)
                dims = SimpleNamespace(n_machines=caps.shape[0], n_res=e["n_res"], n_ticks=e["n_ticks"],
                                       max_jobs=e["max_jobs"], max_cap=int(caps.max()))
                episodes.append(Episode(e["seed"], caps, dims))
                rb = _Rebuild(episodes[-1])
            elif rb is None:
                raise ValueError("trace does not start with an episode line")
            else:
                rb.event(e)
    for ep in episodes:  # episode cut short (file not closed / truncated write)
        if not ep.stats and ep.frames:
            ep.stats = ep.frames[-1].stats
    return episodes
