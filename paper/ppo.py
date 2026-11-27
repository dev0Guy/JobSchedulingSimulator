"""Train a MaskablePPO baseline (sb3-contrib) on the same environment as train.py.

    zig build lib
    PYTHONPATH=python:paper python paper/ppo.py --steps 2000000 --out paper/out/ppo.zip
    PYTHONPATH=python:paper python paper/evaluate.py paper/out/ptr.pt --ppo paper/out/ppo.zip

Same env (ENV_KW), same reward (slowdown, scaled by 1/--reward-scale), and the
same max_ticks cut-off penalty as train.py. Illegal actions are masked through
env.action_masks(). The policy is sb3's MultiInputPolicy: every observation
array is flattened and fed to an MLP (fixed job/machine counts, unlike the
pointer net).

Training episodes use seeds from the --seed block (seed·10⁶ + 0, 1, ...),
which never reach the evaluation seeds; a sidecar <out>.json records it so
evaluate.py can check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym

from ptrmcts import TRAIN_SEED_BLOCK, env_config, train_seed_range, truncation_penalty
from sched_env import SchedEnv


class TrainEnv(gym.Wrapper):
    """Seeds each episode from the training block, scales the reward and adds
    the cut-off penalty on truncation (training signal only)."""

    def __init__(self, env_kw: dict, seeds: range, offset: int, stride: int, scale: float, trunc_penalty: float):
        super().__init__(SchedEnv(**env_kw))
        self.seeds, self.k, self.stride = seeds, offset, stride
        self.scale, self.trunc_penalty = scale, trunc_penalty

    def reset(self, *, seed=None, options=None):
        s = self.seeds[self.k % len(self.seeds)]
        self.k += self.stride
        return self.env.reset(seed=s, options=options)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(int(action))
        if trunc and not term and self.trunc_penalty:
            r += truncation_penalty(self.env, self.trunc_penalty)
        return obs, r / self.scale, term, trunc, info

    def action_masks(self):
        return self.env.action_masks()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=2_000_000, help="total environment steps")
    ap.add_argument("--envs", type=int, default=8, help="parallel environments")
    ap.add_argument("--n-steps", type=int, default=1024, help="rollout length per env per update")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--net", default="256,256", help="MLP hidden sizes (policy and value)")
    ap.add_argument("--reward-scale", type=float, default=10.0)
    ap.add_argument("--trunc-penalty", type=float, default=1.0)
    ap.add_argument("--psplib", default=None)
    ap.add_argument("--seed", type=int, default=1, help="training seed block (seed·10⁶ ...)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="paper/out/ppo.zip")
    args = ap.parse_args()

    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env_kw = env_config(args.psplib)
    seeds = train_seed_range(args.seed, TRAIN_SEED_BLOCK)
    venv = DummyVecEnv([
        (lambda i=i: TrainEnv(env_kw, seeds, i, args.envs, args.reward_scale, args.trunc_penalty))
        for i in range(args.envs)])
    hidden = [int(x) for x in args.net.split(",") if x]
    model = MaskablePPO(
        "MultiInputPolicy", venv, learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch,
        n_epochs=args.epochs, gamma=args.gamma, gae_lambda=args.gae_lambda, ent_coef=args.ent_coef,
        policy_kwargs={"net_arch": {"pi": hidden, "vf": hidden}}, seed=args.seed, device=args.device, verbose=1)
    model.learn(total_timesteps=args.steps)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    meta = {"env": env_kw, "args": vars(args), "seeds": [seeds.start, seeds.stop]}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
