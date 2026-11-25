"""Matplotlib visualization of a scheduling episode.

    from sched_env import SchedEnv
    from sched_env.viz import record, animate
    env = SchedEnv(machines="deeprm:n=2", jobs="poisson:n=60,rate=1.0")
    frames = record(env, policy="sjf", seed=0)        # or policy=callable(env, obs) -> action
    animate(env, frames, "sjf.gif")                   # or path=None for a live window

One frame per tick, taken at the start of the tick (after arrivals, before
the policy's placements), annotated with how many jobs it then placed:
  * one heatmap per machine: used capacity [resource × future tick], 0 = now
  * the waiting pool: each slot's demand per resource over its duration
  * time series: waiting jobs, jobs placed this tick, utilization now
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .env import SchedEnv


@dataclass
class Frame:
    now: int
    used: np.ndarray          # (n_machines, n_res, n_ticks) used / capacity in [0, 1]
    pool: np.ndarray          # (max_jobs * n_res, n_ticks) demand / max_cap, row = slot*n_res + r; NaN = empty
    wait: np.ndarray          # (max_jobs,) ticks waited, -1 for empty
    placed_now: int
    stats: dict = field(default_factory=dict)


def _frame(env: SchedEnv, cap: np.ndarray, placed_now: int) -> Frame:
    r = env.raw()
    c = np.maximum(cap, 1)[:, :, None].astype(np.float32)
    used = (cap[:, :, None] - r["free"]) / c
    pool = r["jobs"].astype(np.float32) / max(1, env.dims.max_cap)  # (slots, res, ticks)
    pool[~r["valid"]] = np.nan
    for s in np.flatnonzero(r["valid"]):
        pool[s, :, r["duration"][s]:] = np.nan
    pool = pool.reshape(-1, pool.shape[-1])
    wait = np.where(r["valid"], r["now"] - r["arrival"], -1)
    return Frame(r["now"], used, pool, wait, placed_now, env.stats())


def record(env: SchedEnv, policy: str | Callable = "sjf", seed: int = 0,
           max_frames: int = 5000) -> list[Frame]:
    """Run one episode and capture a frame per tick.

    policy: a built-in Zig policy name (fifo, fifo_strict, sjf, best_fit,
    random) or a callable (env, obs) -> action, e.g. a trained SB3 model:
        lambda e, o: model.predict(o, action_masks=e.action_masks(), deterministic=True)[0]
    """
    cap = env.capacity()
    obs, _ = env.reset(seed=seed)
    wait_a = env.action_space.n - 1
    frames, done = [_frame(env, cap, 0)], False
    while not done and len(frames) < max_frames:
        a = env.policy_action(policy) if isinstance(policy, str) else int(policy(env, obs))
        obs, _, term, trunc, info = env.step(a)
        done = term or trunc
        if a == wait_a or done:  # a new tick starts
            frames.append(_frame(env, cap, 0))
        else:
            frames[-1].placed_now += int(info["placed"])
    return frames


def animate(env: SchedEnv, frames: list[Frame], path: str | None = None,
            fps: int = 6, title: str = ""):
    """Animate frames. path ending in .gif / .mp4 saves; None shows a window."""
    import matplotlib.pyplot as plt
    from matplotlib import animation

    d = env.dims
    nm = d.n_machines
    nr = d.n_res
    pool_h = 1.3 * max(1.0, d.max_jobs * nr / 20)
    fig = plt.figure(figsize=(3.2 * max(nm, 2) + 4, 6.2 + pool_h), constrained_layout=True)
    gs = fig.add_gridspec(3, max(nm, 2), height_ratios=[1.1, pool_h, 1])

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")
    mach_im = []
    for m in range(nm):
        ax = fig.add_subplot(gs[0, m])
        im = ax.imshow(frames[0].used[m], vmin=0, vmax=1, cmap="magma_r", aspect="auto",
                       interpolation="nearest")
        ax.set_title(f"machine {m}: used / cap")
        ax.set_xlabel("ticks ahead")
        ax.set_ylabel("resource")
        ax.set_yticks(range(d.n_res))
        mach_im.append(im)
    fig.colorbar(mach_im[-1], ax=fig.axes[:nm], shrink=0.8)

    axp = fig.add_subplot(gs[1, :])
    pool_im = axp.imshow(frames[0].pool, vmin=0, vmax=1, cmap=cmap, aspect="auto",
                         interpolation="nearest")
    axp.set_title("waiting pool: demand / max cap per resource over the job's duration, grey = empty")
    axp.set_xlabel("job tick")
    axp.set_ylabel("slot / resource")
    axp.set_yticks([s * nr + (nr - 1) / 2 for s in range(d.max_jobs)],
                   [f"slot {s}" for s in range(d.max_jobs)])
    axp.set_yticks(range(d.max_jobs * nr), [f"r{r}" for _ in range(d.max_jobs) for r in range(nr)],
                   minor=True)
    axp.tick_params(axis="y", which="major", pad=18, length=0)
    axp.tick_params(axis="y", which="minor", labelsize=6, length=2)
    for s in range(1, d.max_jobs):
        axp.axhline(s * nr - 0.5, color="white", lw=2)
    fig.colorbar(pool_im, ax=axp, shrink=0.8)
    wait_txt = [axp.text(d.n_ticks - 0.5, s * nr + (nr - 1) / 2, "", ha="right", va="center",
                         fontsize=7, color="k") for s in range(d.max_jobs)]

    axt = fig.add_subplot(gs[2, :])
    t = np.array([f.now for f in frames])
    waiting = np.array([f.stats["waiting"] for f in frames])
    placed = np.array([f.placed_now for f in frames])
    util = np.array([f.used[:, :, 0].mean() for f in frames])
    axt.plot(t, waiting, color="tab:red", lw=1, alpha=.35)
    axt.bar(t, placed, color="tab:green", alpha=.35, width=1.0)
    (lw,) = axt.plot([], [], color="tab:red", lw=2, label="waiting")
    (lp,) = axt.plot([], [], color="tab:green", lw=0, marker="|", ms=10, label="placed this tick")
    axt.set_xlabel("tick")
    axt.xaxis.get_major_locator().set_params(integer=True)
    axt.yaxis.get_major_locator().set_params(integer=True)
    axt.set_ylabel("jobs")
    axu = axt.twinx()
    axu.plot(t, util, color="tab:blue", lw=1, alpha=.35)
    (lu,) = axu.plot([], [], color="tab:blue", lw=2, label="utilization")
    axu.set_ylim(0, 1.05)
    axu.set_ylabel("utilization now")
    cursor = axt.axvline(t[0], color="k", lw=1)
    axt.legend(handles=[lw, lp, lu], loc="upper right", fontsize=8)
    head = fig.suptitle("")

    def update(i):
        f = frames[i]
        for m, im in enumerate(mach_im):
            im.set_data(f.used[m])
        pool_im.set_data(f.pool)
        for s, txt in enumerate(wait_txt):
            txt.set_text(f"wait {f.wait[s]}" if f.wait[s] >= 0 else "")
        lw.set_data(t[: i + 1], waiting[: i + 1])
        lp.set_data(t[: i + 1], placed[: i + 1])
        lu.set_data(t[: i + 1], util[: i + 1])
        cursor.set_xdata([f.now, f.now])
        s = f.stats
        head.set_text(f"{title}  tick {f.now}   placed {s['placed']}/{s['arrived']}   "
                      f"waiting {s['waiting']}   avg wait {s['avg_wait']:.2f}")
        return [*mach_im, pool_im, lw, lp, lu, cursor, head, *wait_txt]

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    if path is None:
        plt.show()
    elif path.endswith(".gif"):
        anim.save(path, writer=animation.PillowWriter(fps=fps))
    else:
        anim.save(path, fps=fps)
    return anim


def snapshot(env: SchedEnv, frame: Frame, path: str, title: str = ""):
    """Save a single frame as a PNG."""
    import matplotlib.pyplot as plt
    d = env.dims
    fig, axes = plt.subplots(1, d.n_machines + 1, figsize=(3.2 * (d.n_machines + 1), 3),
                             constrained_layout=True)
    for m in range(d.n_machines):
        axes[m].imshow(frame.used[m], vmin=0, vmax=1, cmap="magma_r", aspect="auto")
        axes[m].set_title(f"machine {m}")
    axes[-1].imshow(frame.pool, vmin=0, vmax=1, aspect="auto")
    axes[-1].set_title("waiting pool")
    fig.suptitle(f"{title} tick {frame.now}")
    fig.savefig(path)
    plt.close(fig)
