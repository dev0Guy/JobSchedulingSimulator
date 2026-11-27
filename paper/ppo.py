"""Train a MaskablePPO baseline (sb3-contrib) on the same environment as train.py.

    zig build lib
    PYTHONPATH=python:paper python paper/ppo.py --steps 2000000 --out paper/out/ppo.zip
    PYTHONPATH=python:paper python paper/evaluate.py paper/out/ptr.pt --ppo paper/out/ppo.zip

Same env (ENV_KW), same reward (slowdown, scaled by 1/--reward-scale), and the
same max_ticks cut-off penalty as train.py. Illegal actions are masked through
env.action_masks(). The policy is the pointer net itself (PtrPolicy: same
PointerNet as train.py, its logits as the action distribution and its value
head as the critic), so ptr vs PPO compares training algorithms, not
architectures. `--policy mlp` uses sb3's flat MultiInputPolicy instead.
Rewards are normalized by a running return std (VecNormalize).

Training episodes use seeds from the --seed block (seed·10⁶ + 0, 1, ...),
which never reach the evaluation seeds; a sidecar <out>.json records it so
evaluate.py can check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym

import torch
from sb3_contrib.common.maskable.distributions import MaskableCategoricalDistribution
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from ptrmcts import TRAIN_SEED_BLOCK, PointerNet, env_config, with_jobs, train_seed_range, truncation_penalty
from sched_env import SchedEnv


def obs_to_net(obs: dict) -> tuple:
    """Batched SchedEnv observation tensors -> PointerNet inputs (as featurize)."""
    B, J = obs["jobs"].shape[:2]
    jobs = torch.cat([obs["jobs"].reshape(B, J, -1), obs["wait"][..., None], obs["age"][..., None],
                      obs["valid"][..., None]], -1)
    mach = obs["machines"].reshape(B, obs["machines"].shape[1], -1)
    return jobs, mach, obs["valid"] > 0.5


class PtrPolicy(MaskableActorCriticPolicy):
    """MaskablePPO policy whose actor and critic are one PointerNet."""

    def __init__(self, *args, ptr_kw: dict | None = None, **kw):
        self.ptr_kw = ptr_kw or {}
        super().__init__(*args, **kw)

    def _build(self, lr_schedule) -> None:
        n_res, n_ticks = self.observation_space["jobs"].shape[1:]
        self.ptr = PointerNet(n_res, n_ticks, **self.ptr_kw)
        self.action_dist = MaskableCategoricalDistribution(int(self.action_space.n))
        self.optimizer = self.optimizer_class(self.ptr.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        return {**super()._get_constructor_parameters(), "ptr_kw": self.ptr_kw}

    def _run(self, obs):
        jobs, mach, valid = obs_to_net(obs)
        mask = torch.ones(jobs.shape[0], int(self.action_space.n), dtype=torch.bool, device=jobs.device)
        return self.ptr(jobs, mach, valid, mask)  # masking is applied by the distribution

    def _dist(self, logits, action_masks):
        d = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            d.apply_masking(action_masks)
        return d

    def forward(self, obs, deterministic=False, action_masks=None):
        logits, v = self._run(obs)
        d = self._dist(logits, action_masks)
        a = d.get_actions(deterministic=deterministic)
        return a, v[:, None], d.log_prob(a)

    def evaluate_actions(self, obs, actions, action_masks=None):
        logits, v = self._run(obs)
        d = self._dist(logits, action_masks)
        return v[:, None], d.log_prob(actions), d.entropy()

    def get_distribution(self, obs, action_masks=None):
        return self._dist(self._run(obs)[0], action_masks)

    def predict_values(self, obs):
        return self._run(obs)[1][:, None]


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
    ap.add_argument("--steps", type=int, default=5_000_000, help="total environment steps")
    ap.add_argument("--envs", type=int, default=8, help="parallel environments")
    ap.add_argument("--n-steps", type=int, default=2048, help="rollout length per env per update")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.98)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.001, help="small: up to J·M+1 actions")
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--no-norm-reward", action="store_true", help="disable VecNormalize reward scaling")
    ap.add_argument("--policy", choices=["ptr", "mlp"], default="ptr")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--net", default="256,256", help="--policy mlp: hidden sizes")
    ap.add_argument("--reward-scale", type=float, default=10.0)
    ap.add_argument("--trunc-penalty", type=float, default=1.0)
    ap.add_argument("--psplib", default=None)
    ap.add_argument("--jobs", type=int, default=None,
                    help="jobs per episode (= job slots); evaluate on more with evaluate.py --jobs (tournament)")
    ap.add_argument("--seed", type=int, default=1, help="training seed block (seed·10⁶ ...)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="paper/out/ppo.zip")
    args = ap.parse_args()

    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env_kw = with_jobs(env_config(args.psplib), args.jobs)
    seeds = train_seed_range(args.seed, TRAIN_SEED_BLOCK)
    venv = DummyVecEnv([
        (lambda i=i: TrainEnv(env_kw, seeds, i, args.envs, args.reward_scale, args.trunc_penalty))
        for i in range(args.envs)])
    if not args.no_norm_reward:  # scale rewards by the running std of the discounted return
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=args.gamma)
    if args.policy == "ptr":
        import ppo  # pickle the class as ppo.PtrPolicy, not __main__.PtrPolicy, so evaluate.py can load it
        policy, pkw = ppo.PtrPolicy, {"ptr_kw": {"d": args.d_model, "layers": args.layers}}
    else:
        hidden = [int(x) for x in args.net.split(",") if x]
        policy, pkw = "MultiInputPolicy", {"net_arch": {"pi": hidden, "vf": hidden}}
    model = MaskablePPO(
        policy, venv, learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch,
        n_epochs=args.epochs, gamma=args.gamma, gae_lambda=args.gae_lambda, clip_range=args.clip,
        ent_coef=args.ent_coef, vf_coef=args.vf_coef, max_grad_norm=args.max_grad_norm,
        policy_kwargs=pkw, seed=args.seed, device=args.device, verbose=1)
    model.learn(total_timesteps=args.steps)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    meta = {"env": env_kw, "args": vars(args), "seeds": [seeds.start, seeds.stop]}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
