"""Pointer network + Monte Carlo tree search for the scheduling simulator.

Shared by paper/train.py and paper/evaluate.py.

Model (`PointerNet`)
    Waiting jobs and machines are encoded as one set with a transformer
    encoder (jobs carry their demand grid, wait time and a valid flag;
    machines their free-capacity grid). A glimpse over the set gives a
    context vector q. Like a pointer network (Vinyals et al., 2015), the
    output points at input elements instead of a fixed vocabulary:

        job pointer   u_j  = vᵀ tanh(W1 h_j + W2 q)
        pair logit    l_jm = u_j + ⟨W3 h_j, W4 h_m⟩ / √d
        wait logit    l_w  = wᵀ q

    so the action space (max_jobs·n_machines + 1, same order as SchedEnv)
    is scored by attention and illegal actions are masked out. A value head
    on q estimates the (scaled, discounted) return for MCTS.

Search (`MCTS`)
    AlphaZero-style PUCT with the network as prior and leaf evaluator.
    The engine has no clone(), so every simulation restores the root by
    replaying it: reset(seed) + the episode's actions so far. This only
    works because the Zig engine is deterministic for a given seed; every
    restore checks that the replayed state matches the recorded one.
"""

from __future__ import annotations

import ctypes as C
import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sched_env import SchedEnv
from sched_env import _lib

# Environment used in the paper experiments. Small enough for MCTS on a CPU.
ENV_KW = dict(
    machines="deeprm:n=1",
    jobs="poisson:n=80,rate=0.8",
    n_res=2,
    n_ticks=20,
    max_jobs=10,
    reward="slowdown",
)

# Seed ranges: training and evaluation never share a workload.
EVAL_SEED_START = 10**12  # training uses seed·10⁶ + episode


def make_env(**overrides) -> SchedEnv:
    return SchedEnv(**{**ENV_KW, **overrides})


# ----------------------------------------------------------------------------- features


def featurize(obs: dict) -> dict[str, np.ndarray]:
    """SchedEnv observation -> per-element feature arrays (unbatched)."""
    jobs = obs["jobs"].reshape(obs["jobs"].shape[0], -1)
    jobs = np.concatenate([jobs, obs["wait"][:, None], obs["valid"][:, None]], axis=1)
    mach = obs["machines"].reshape(obs["machines"].shape[0], -1)
    return {"jobs": jobs.astype(np.float32), "mach": mach.astype(np.float32),
            "valid": obs["valid"] > 0}


def collate(feats: list[dict], masks: list[np.ndarray], device) -> tuple:
    j = torch.as_tensor(np.stack([f["jobs"] for f in feats]), device=device)
    m = torch.as_tensor(np.stack([f["mach"] for f in feats]), device=device)
    v = torch.as_tensor(np.stack([f["valid"] for f in feats]), device=device)
    a = torch.as_tensor(np.stack(masks), device=device)
    return j, m, v, a


# ----------------------------------------------------------------------------- model


class PointerNet(nn.Module):
    def __init__(self, job_dim: int, mach_dim: int, d: int = 128, heads: int = 4, layers: int = 2):
        super().__init__()
        self.d = d
        self.job_in = nn.Sequential(nn.Linear(job_dim, d), nn.ReLU(), nn.Linear(d, d))
        self.mach_in = nn.Sequential(nn.Linear(mach_dim, d), nn.ReLU(), nn.Linear(d, d))
        self.type_emb = nn.Parameter(torch.zeros(2, d))  # job / machine
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.glimpse = nn.MultiheadAttention(d, heads, batch_first=True)
        self.q0 = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        # pointer
        self.W1 = nn.Linear(d, d, bias=False)
        self.W2 = nn.Linear(d, d)
        self.v = nn.Linear(d, 1, bias=False)
        self.W3 = nn.Linear(d, d, bias=False)
        self.W4 = nn.Linear(d, d, bias=False)
        self.wait = nn.Linear(d, 1)
        self.value = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, jobs, mach, valid, mask):
        """jobs (B,J,Fj), mach (B,M,Fm), valid (B,J) bool, mask (B,J·M+1) bool
        -> logits (B,J·M+1) with illegal = -inf, value (B,)"""
        B, J, _ = jobs.shape
        M = mach.shape[1]
        x = torch.cat([self.job_in(jobs) + self.type_emb[0], self.mach_in(mach) + self.type_emb[1]], 1)
        pad = torch.cat([~valid, torch.zeros(B, M, dtype=torch.bool, device=jobs.device)], 1)
        h = self.enc(x, src_key_padding_mask=pad)
        q, _ = self.glimpse(self.q0.expand(B, 1, self.d), h, h, key_padding_mask=pad)
        q = q[:, 0]
        hj, hm = h[:, :J], h[:, J:]
        u = self.v(torch.tanh(self.W1(hj) + self.W2(q)[:, None])).squeeze(-1)          # (B,J)
        pair = torch.einsum("bjd,bmd->bjm", self.W3(hj), self.W4(hm)) / math.sqrt(self.d)
        logits = torch.cat([(u[:, :, None] + pair).reshape(B, J * M), self.wait(q)], 1)
        logits = logits.masked_fill(~mask, float("-inf"))
        return logits, self.value(q).squeeze(-1)

    @classmethod
    def for_env(cls, env: SchedEnv, **kw) -> "PointerNet":
        d = env.dims
        return cls(d.n_res * d.n_ticks + 2, d.n_res * d.n_ticks, **kw)


@torch.no_grad()
def evaluate_net(net: PointerNet, obs: dict, mask: np.ndarray, device) -> tuple[np.ndarray, float]:
    """Prior probabilities over actions and value for one state."""
    logits, v = net(*collate([featurize(obs)], [mask], device))
    return torch.softmax(logits[0], 0).cpu().numpy(), float(v[0])


# ----------------------------------------------------------------------------- replay


class Replayer:
    """Restores an episode state by reset(seed) + replaying actions.

    Uses its own engine handle (so the caller's env is untouched) and raw
    ctypes steps (no observation copies) for speed.
    """

    def __init__(self, **env_kw):
        self.env = make_env(**env_kw)
        self._res = _lib.SimStep()
        self._st = _lib.SimStats()

    def fingerprint(self) -> tuple:
        self.env._lib.sim_stats(self.env._h, C.byref(self._st))
        s = self._st
        return (s.now, s.arrived, s.placed, s.rejected, s.waiting, round(s.total_reward, 4))

    def restore(self, seed: int, actions: list[int], expect: tuple | None = None) -> None:
        lib, h = self.env._lib, self.env._h
        if lib.sim_reset(h, seed) != 0:
            raise RuntimeError("sim_reset failed")
        for a in actions:
            lib.sim_step(h, a, C.byref(self._res))
        if expect is not None and self.fingerprint() != expect:
            raise RuntimeError(f"replay diverged (seed {seed}): {self.fingerprint()} != {expect}. "
                               "The engine is not deterministic for this seed.")

    def step(self, a: int) -> tuple[float, bool]:
        self.env._lib.sim_step(self.env._h, a, C.byref(self._res))
        r = self._res
        return float(r.reward), bool(r.terminated or r.truncated)


# ----------------------------------------------------------------------------- MCTS


@dataclass
class Node:
    prior: float
    reward: float = 0.0          # reward of the transition into this node
    terminal: bool = False
    n: int = 0
    w: float = 0.0
    children: dict[int, "Node"] = field(default_factory=dict)

    @property
    def expanded(self) -> bool:
        return bool(self.children) or self.terminal

    def q(self) -> float:
        return self.w / self.n if self.n else 0.0


class MinMax:
    def __init__(self):
        self.lo, self.hi = math.inf, -math.inf

    def update(self, x: float):
        self.lo, self.hi = min(self.lo, x), max(self.hi, x)

    def norm(self, x: float) -> float:
        return (x - self.lo) / (self.hi - self.lo) if self.hi > self.lo else 0.5


class MCTS:
    """PUCT search. Rewards are divided by `reward_scale`; Q values are
    min-max normalized inside the tree (MuZero), so the scale only matters
    for the value head's targets."""

    def __init__(self, net: PointerNet, device="cpu", sims: int = 32, c_puct: float = 1.5,
                 gamma: float = 0.98, reward_scale: float = 10.0, dirichlet: float = 0.3,
                 noise_frac: float = 0.25, rng: np.random.Generator | None = None, **env_kw):
        self.net, self.device = net, device
        self.sims, self.c, self.gamma, self.scale = sims, c_puct, gamma, reward_scale
        self.dirichlet, self.noise_frac = dirichlet, noise_frac
        self.rng = rng or np.random.default_rng(0)
        self.rp = Replayer(**env_kw)

    def _expand(self, node: Node) -> float:
        env = self.rp.env
        obs, mask = env._obs(), env.action_masks()
        p, v = evaluate_net(self.net, obs, mask, self.device)
        for a in np.flatnonzero(mask):
            node.children[int(a)] = Node(prior=float(p[a]))
        return v

    def _select(self, node: Node, mm: MinMax) -> tuple[int, Node]:
        sq = math.sqrt(max(1, node.n))
        best, best_s = None, -math.inf
        for a, ch in node.children.items():
            q = mm.norm(ch.reward / self.scale + self.gamma * ch.q()) if ch.n else 0.0
            s = q + self.c * ch.prior * sq / (1 + ch.n)
            if s > best_s:
                best, best_s = (a, ch), s
        return best

    def search(self, seed: int, prefix: list[int], expect: tuple, noise: bool) -> np.ndarray:
        """Visit counts over actions at the state reached by `prefix`."""
        root = Node(prior=1.0)
        mm = MinMax()
        self.rp.restore(seed, prefix, expect)
        self._expand(root)
        if noise and len(root.children) > 1:
            eta = self.rng.dirichlet([self.dirichlet] * len(root.children))
            for e, ch in zip(eta, root.children.values()):
                ch.prior = (1 - self.noise_frac) * ch.prior + self.noise_frac * e
        for _ in range(self.sims):
            self.rp.restore(seed, prefix)
            node, path = root, [root]
            while node.expanded and not node.terminal:
                a, node = self._select(node, mm)
                r, done = self.rp.step(a)
                node.reward, node.terminal = r, done
                path.append(node)
            g = 0.0 if node.terminal else self._expand(node)
            for nd in reversed(path):
                nd.n += 1
                nd.w += g
                if nd is not root:
                    mm.update(nd.reward / self.scale + self.gamma * nd.q())
                g = nd.reward / self.scale + self.gamma * g
        counts = np.zeros(self.rp.env.dims.n_actions, dtype=np.float32)
        for a, ch in root.children.items():
            counts[a] = ch.n
        return counts


# ----------------------------------------------------------------------------- episodes


def fingerprint(env: SchedEnv) -> tuple:
    s = env.stats()
    return (s["now"], s["arrived"], s["placed"], s["rejected"], s["waiting"], round(s["total_reward"], 4))


def act_greedy(net, obs, mask, device) -> int:
    p, _ = evaluate_net(net, obs, mask, device)
    return int(np.argmax(p))
