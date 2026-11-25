"""Train MaskablePPO (sb3-contrib) on the Zig simulator and compare it with
the built-in heuristics on the same seeds.

    zig build lib
    pip install -r python/requirements.txt
    PYTHONPATH=python python python/examples/train_sb3.py --steps 100000
"""

import argparse

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env

from sched_env import SchedEnv

ENV_KW = dict(
    machines="deeprm:n=1",
    jobs="poisson:n=60,rate=0.7",  # loaded enough that policies differ
    n_res=2,
    n_ticks=20,
    max_jobs=10,
    reward="neg_in_system",
    max_ticks=2000,
)


def make():
    # ActionMasker lets MaskablePPO find the mask through any wrappers.
    return ActionMasker(SchedEnv(**ENV_KW), lambda e: e.action_masks())


def evaluate(model, seeds):
    env = SchedEnv(**ENV_KW)
    out = []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
        out.append(info["stats"])
    env.close()
    return out


def summary(name, stats):
    w = np.mean([s["avg_wait"] for s in stats])
    r = np.mean([s["total_reward"] for s in stats])
    m = np.mean([s["makespan"] for s in stats])
    print(f"{name:12s} avg_wait {w:7.2f}  makespan {m:7.1f}  reward {r:9.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--save", default="ppo_sched")
    ap.add_argument("--device", default="auto",
                    help="torch device for the network: auto | cpu | cuda | mps")
    args = ap.parse_args()

    venv = make_vec_env(make, n_envs=args.envs, seed=0)
    model = MaskablePPO("MultiInputPolicy", venv, n_steps=256, batch_size=256,
                        gamma=0.99, verbose=1, seed=0, device=args.device)
    model.learn(total_timesteps=args.steps)
    model.save(args.save)

    seeds = list(range(1000, 1020))
    summary("ppo", evaluate(model, seeds))
    base = SchedEnv(**ENV_KW)
    for p in ("fifo", "sjf", "best_fit", "random"):
        summary(p, [base.run_policy(p, s) for s in seeds])


if __name__ == "__main__":
    main()
