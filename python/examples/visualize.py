"""Visualize one episode of a built-in policy (or a trained SB3 model).

    zig build lib
    PYTHONPATH=python python python/examples/visualize.py --policy sjf --out sjf.gif
    PYTHONPATH=python python python/examples/visualize.py --policy fifo            # live window
    PYTHONPATH=python python python/examples/visualize.py --model ppo_sched.zip --out ppo.gif
"""

import argparse

from sched_env import SchedEnv
from sched_env.viz import animate, record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="sjf", help="fifo | fifo_strict | sjf | best_fit | random")
    ap.add_argument("--model", help="MaskablePPO .zip to visualize instead of --policy")
    ap.add_argument("--machines", default="deeprm:n=1")
    ap.add_argument("--jobs", default="poisson:n=60,rate=0.8")
    ap.add_argument("--res", type=int, default=2)
    ap.add_argument("--ticks", type=int, default=20)
    ap.add_argument("--pool", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--out", help=".gif or .mp4 (default: live window)")
    a = ap.parse_args()

    env = SchedEnv(machines=a.machines, jobs=a.jobs, n_res=a.res, n_ticks=a.ticks,
                   max_jobs=a.pool, max_ticks=5000)
    if a.model:
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(a.model)
        policy = lambda e, o: model.predict(o, action_masks=e.action_masks(), deterministic=True)[0]
        name = "ppo"
    else:
        policy, name = a.policy, a.policy

    frames = record(env, policy, seed=a.seed)
    s = frames[-1].stats
    print(f"{name}: {len(frames)} ticks, placed {s['placed']}/{s['arrived']}, "
          f"avg wait {s['avg_wait']:.2f}, makespan {s['makespan']}, util {s['avg_util']:.1%}")
    animate(env, frames, a.out, fps=a.fps, title=name)
    if a.out:
        print("saved", a.out)


if __name__ == "__main__":
    main()
