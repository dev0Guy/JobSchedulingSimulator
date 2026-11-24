"""Replay a recorded trace as an animation — no simulator needed.

Record (command line):
    zig build run -- --machines deeprm:n=1 --jobs poisson:n=60,rate=0.8 --res 2 --ticks 20 \\
        --pool 10 --policy sjf --observe trace:path=run.jsonl
Record (Python):
    env.start_trace("run.jsonl"); env.reset(seed=0); ...steps...; env.stop_trace()

Replay:
    PYTHONPATH=python python python/examples/replay.py run.jsonl --out run.gif
    PYTHONPATH=python python python/examples/replay.py run.jsonl --episode 1      # live window
    PYTHONPATH=python python python/examples/replay.py run.jsonl --list
"""

import argparse

from sched_env.replay import load_trace
from sched_env.viz import animate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out", help=".gif or .mp4 (default: live window)")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--from-tick", type=int, default=0)
    ap.add_argument("--to-tick", type=int)
    ap.add_argument("--list", action="store_true", help="summarize the episodes and exit")
    a = ap.parse_args()

    episodes = load_trace(a.trace)
    if a.list or not episodes:
        for i, ep in enumerate(episodes):
            s = ep.stats
            print(f"episode {i}: seed {ep.seed}, {len(ep.frames)} ticks, placed {s['placed']}/{s['arrived']}, "
                  f"rejected {s['rejected']}, avg wait {s['avg_wait']:.2f}, makespan {s['makespan']}"
                  + ("  (truncated)" if ep.truncated else ""))
        return
    ep = episodes[a.episode]
    frames = [f for f in ep.frames if f.now >= a.from_tick and (a.to_tick is None or f.now <= a.to_tick)]
    s = ep.stats
    print(f"episode {a.episode} (seed {ep.seed}): {len(frames)} frames, placed {s['placed']}/{s['arrived']}, "
          f"avg wait {s['avg_wait']:.2f}, makespan {s['makespan']}")
    animate(ep, frames, a.out, fps=a.fps, title=f"{a.trace} #{a.episode}")
    if a.out:
        print("saved", a.out)


if __name__ == "__main__":
    main()
