"""Train the pointer network with MCTS self-play (AlphaZero style).

    zig build lib
    PYTHONPATH=python:paper python paper/train.py --iters 20 --out paper/out/ptr.pt

1. (optional) Behaviour cloning warm start: imitate a built-in heuristic
   (default tetris_srpt) so search starts from a sensible prior.
2. Self-play: each step runs MCTS from the current state; the visit
   distribution π is the policy target, the discounted (scaled) return the
   value target. Actions are sampled from π for the first `--explore-steps`
   steps of an episode, greedy afterwards.
3. Train on a replay buffer with  CE(π, net) + value_coef · Huber(v, G).

Every episode uses its own training seed (seed·10⁶ + 0, 1, 2, ...);
evaluation seeds (paper/evaluate.py) are disjoint, and both scripts check it. All randomness (torch, numpy, engine) is
seeded from --seed, so a run is reproducible on the same machine.
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import multiprocessing as mp
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ptrmcts import (MCTS, PointerNet, atomic_write, collate, env_config, evaluate_net, featurize, fingerprint,
                     train_seed_range, truncation_penalty, value_h)
from sched_env import SchedEnv


def discounted(rewards: list[float], gamma: float, scale: float) -> list[float]:
    g, out = 0.0, []
    for r in reversed(rewards):
        g = r / scale + gamma * g
        out.append(g)
    return out[::-1]


def bootstrapped(rewards: list[float], values: list[float | None], gamma: float, scale: float,
                 n: int) -> list[float]:
    """n-step targets: G_t = sum_{k<n} gamma^k r_{t+k}/scale + gamma^n v_{t+n},
    v = the search value of the state n steps later (falls back to the full
    discounted return near the end of the episode, or when no value is
    known). Much less noisy than the full return when gamma is near 1."""
    mc = discounted(rewards, gamma, scale)
    if n <= 0:
        return mc
    T = len(rewards)
    out = []
    for t in range(T):
        if t + n >= T or values[t + n] is None:
            out.append(mc[t])
            continue
        g = sum(gamma ** k * rewards[t + k] / scale for k in range(n))
        out.append(g + gamma ** n * values[t + n])
    return out


def run_episode(env, seed, choose, gamma, scale, trunc_penalty=1.0, n_step=0):
    """Play one episode; `choose(obs, mask, prefix, expect, step)` -> (action, π)
    or (action, π, search value)."""
    obs, _ = env.reset(seed=seed)
    samples, rewards, prefix, values = [], [], [], []
    raw = 0.0  # true episode return (logged), without the cut-off penalty
    done = False
    while not done:
        mask = env.action_masks()
        out = choose(obs, mask, prefix, fingerprint(env), len(prefix))
        a, pi = out[0], out[1]
        values.append(out[2] if len(out) > 2 else None)
        samples.append((featurize(obs), mask, pi))
        obs, r, term, trunc, info = env.step(a)
        raw += r
        if trunc and not term and trunc_penalty:
            r += truncation_penalty(env, trunc_penalty)  # training target only
        rewards.append(r)
        prefix.append(a)
        done = term or trunc
    rets = bootstrapped(rewards, values, gamma, scale, n_step)
    return [(f, m, p, g) for (f, m, p), g in zip(samples, rets)], info["stats"], raw


def train_step(net, opt, batch, device, value_coef):
    feats, masks, pis, rets = zip(*batch)
    j, m, v, a = collate(list(feats), list(masks), device)
    pi = torch.as_tensor(np.stack(pis), device=device)
    g = value_h(torch.as_tensor(np.asarray(rets, dtype=np.float32), device=device))  # net predicts h(G)
    logits, val = net(j, m, v, a)
    logp = torch.log_softmax(logits, 1).masked_fill(~a, 0.0)
    pl = -(pi * logp).sum(1).mean()
    vl = F.smooth_l1_loss(val, g)
    loss = pl + value_coef * vl
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    return pl.item(), vl.item()


# ----------------------------------------------------------------------------- parallel self-play
# Each self-play episode is independent given the network weights, so they run
# in worker processes (CPU, one thread each: MCTS evaluates one state at a time,
# which a GPU does not speed up). Training stays on --device. Every episode's
# search randomness comes from its own seed, so results do not depend on the
# number of workers.

_W: dict = {}


def _worker_init(env_kw: dict, net_kw: dict, threads: int):
    torch.set_num_threads(threads)
    env = SchedEnv(**env_kw)
    _W["env"], _W["env_kw"] = env, env_kw
    _W["net"] = PointerNet.for_env(env, **net_kw).eval()


def self_play_episode(job):
    """(state_dict or None, seed, args dict) -> (samples, stats, return).
    Runs in a worker (or in-process when state_dict is None and _W is set)."""
    state, seed, a = job
    if state is not None:
        _W["net"].load_state_dict(state)
    env, net = _W["env"], _W["net"]
    rng = np.random.default_rng([a["seed"], seed])
    mcts = MCTS(net, _W["env_kw"], "cpu", sims=a["sims"], gamma=a["gamma"], reward_scale=a["reward_scale"], rng=rng,
                trunc_penalty=a["trunc_penalty"], batch=a["mcts_batch"])

    def choose(obs, mask, prefix, expect, step):
        counts = mcts.search(seed, prefix, expect, noise=True)
        act = (int(rng.choice(len(counts), p=counts / counts.sum())) if step < a["explore_steps"]
               else int(np.argmax(counts)))
        # policy target: visit counts sharpened by --pi-temp, so the net learns
        # the search's choice, not just its spread (1 = plain counts)
        c = counts ** (1.0 / a["pi_temp"])
        return act, c / c.sum(), mcts.root_value

    with torch.no_grad():
        return run_episode(env, seed, choose, a["gamma"], a["reward_scale"], a["trunc_penalty"], a["n_step"])


def greedy_score(env, net, seeds, device, trunc_penalty):
    """Mean return of the net alone (argmax, no search) over fixed seeds,
    cut-off penalty included, so a policy that starves jobs scores badly."""
    net.eval()
    total = 0.0
    with torch.no_grad():
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            done = False
            while not done:
                mask = env.action_masks()
                p, _ = evaluate_net(net, obs, mask, device)
                obs, r, term, trunc, _ = env.step(int(np.argmax(np.where(mask, p, -1.0))))
                total += r
                if trunc and not term and trunc_penalty:
                    total += truncation_penalty(env, trunc_penalty)
                done = term or trunc
    return total / max(1, len(seeds))


def fit(net, opt, buffer, args, device, rng, anchor=()):
    """`anchor` (teacher samples) supplies a fixed --bc-mix share of every batch,
    so self-play cannot wash the teacher out of the buffer."""
    if len(buffer) < args.batch:
        return float("nan"), float("nan")
    data = list(buffer)
    n_anchor = int(round(args.bc_mix * args.batch)) if len(anchor) else 0
    pls, vls = [], []
    for _ in range(args.updates):
        idx = rng.choice(len(data), args.batch - n_anchor, replace=False)
        batch = [data[i] for i in idx]
        if n_anchor:
            batch += [anchor[i] for i in rng.choice(len(anchor), n_anchor, replace=len(anchor) < n_anchor)]
        pl, vl = train_step(net, opt, batch, device, args.value_coef)
        pls.append(pl)
        vls.append(vl)
    return float(np.mean(pls)), float(np.mean(vls))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=20, help="self-play / train iterations")
    ap.add_argument("--episodes", type=int, default=4, help="self-play episodes per iteration")
    ap.add_argument("--sims", type=int, default=32, help="MCTS simulations per move")
    ap.add_argument("--explore-steps", type=int, default=30, help="sample from π for the first N moves")
    ap.add_argument("--bc-teacher", default="sjf", help="built-in policy to imitate first ('' = none)")
    ap.add_argument("--bc-episodes", type=int, default=40)
    ap.add_argument("--bc-epochs", type=int, default=5)
    ap.add_argument("--updates", type=int, default=50, help="gradient steps per iteration")
    ap.add_argument("--bc-mix", type=float, default=0.25,
                    help="share of every training batch drawn from the teacher's samples (0 = off)")
    ap.add_argument("--gate-episodes", type=int, default=8,
                    help="keep a new net only if its greedy return on this many fixed seeds beats the best (0 = off)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--buffer", type=int, default=20_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.999,
                    help="discount; close to 1 so a job left waiting stays expensive (0.98 taught starvation)")
    ap.add_argument("--trunc-penalty", type=float, default=1.0,
                    help="multiplier of the max_ticks cut-off penalty: every job still waiting is charged "
                         "(sim length + dur)/dur + 10, as if placed at the end; 0 disables")
    ap.add_argument("--reward-scale", type=float, default=10.0)
    ap.add_argument("--n-step", type=int, default=20,
                    help="bootstrapped value targets from the search value n moves later (0 = full return)")
    ap.add_argument("--pi-temp", type=float, default=0.5,
                    help="policy target = visit counts^(1/T); < 1 trains on the search's choice")
    ap.add_argument("--mcts-batch", type=int, default=8, help="leaves per network call in MCTS (1 = unbatched)")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1, help="transformer encoder layers")
    ap.add_argument("--psplib", default=None,
                    help="take job profiles from PSPLIB instances (.sm file or directory) instead of DeepRM")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel self-play processes (0 = all cores but one); results do not depend on it")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="paper/out/ptr.pt")
    args = ap.parse_args()
    if args.sims < 1:
        ap.error("--sims must be >= 1")
    if args.pi_temp <= 0:
        ap.error("--pi-temp must be > 0")
    if not 0 <= args.bc_mix < 1:
        ap.error("--bc-mix must be in [0, 1)")
    all_seeds = list(train_seed_range(args.seed, args.bc_episodes + args.iters * args.episodes))
    # gate on already-used training seeds (the first ones), never on evaluation seeds
    gate_seeds = all_seeds[:args.gate_episodes]
    seeds = iter(all_seeds)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    env_kw = env_config(args.psplib)
    env = SchedEnv(**env_kw)
    net_kw = {"d": args.d_model, "layers": args.layers}
    net = PointerNet.for_env(env, **net_kw).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    buffer = collections.deque(maxlen=args.buffer)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = []
    anchor = []

    # 1. behaviour cloning ------------------------------------------------------
    if args.bc_teacher and args.bc_episodes:
        n_act = env.dims.n_actions

        def teacher(obs, mask, prefix, expect, step):
            a = env.policy_action(args.bc_teacher)
            pi = np.zeros(n_act, dtype=np.float32)
            pi[a] = 1
            return a, pi

        data = []
        for _ in range(args.bc_episodes):
            d, _, _ = run_episode(env, next(seeds), teacher, args.gamma, args.reward_scale, args.trunc_penalty)
            data += d
        for ep in range(args.bc_epochs):
            perm = rng.permutation(len(data))
            pls, vls = [], []
            for i in range(0, len(perm) - args.batch + 1, args.batch):
                pl, vl = train_step(net, opt, [data[k] for k in perm[i:i + args.batch]], device, args.value_coef)
                pls.append(pl)
                vls.append(vl)
            print(f"bc epoch {ep}: policy loss {np.mean(pls):.3f}  value loss {np.mean(vls):.3f}  ({len(data)} samples)")
        buffer.extend(data)
        anchor = data

    # 2-3. self-play + training ------------------------------------------------
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 2)
    workers = args.workers or max(1, cpus - 1)  # affinity: only the CPUs SLURM/taskset gave us
    pool = (mp.get_context("spawn").Pool(workers, _worker_init, (env_kw, net_kw, 1)) if workers > 1 else None)
    if pool is None:
        _worker_init(env_kw, net_kw, torch.get_num_threads())
    best_state = copy.deepcopy(net.state_dict())
    best_score = greedy_score(env, net, gate_seeds, device, args.trunc_penalty) if gate_seeds else -float("inf")
    if gate_seeds:
        print(f"gate: initial greedy return {best_score:.2f} on {len(gate_seeds)} seeds")
    print(f"self-play: {workers} worker(s), training on {device}")
    for it in range(args.iters):
        t0 = time.time()
        state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        jobs = [(state, next(seeds), vars(args)) for _ in range(args.episodes)]
        results = pool.map(self_play_episode, jobs) if pool else [self_play_episode(j) for j in jobs]
        rets, slow = [], []
        for d, stats, ret in results:
            buffer.extend(d)
            rets.append(ret)
            slow.append(-ret / max(1, stats["placed"]))
        net.train()
        pl, vl = fit(net, opt, buffer, args, device, rng, anchor)
        rec = {"iter": it, "return": float(np.mean(rets)), "slowdown_per_job": float(np.mean(slow)),
               "policy_loss": pl, "value_loss": vl, "buffer": len(buffer)}
        keep = True
        if gate_seeds:
            score = greedy_score(env, net, gate_seeds, device, args.trunc_penalty)
            keep = score > best_score
            rec.update(gate=round(score, 2), best=round(max(score, best_score), 2), accepted=keep)
            if keep:
                best_score, best_state = score, copy.deepcopy(net.state_dict())
            else:  # reject: self-play next iteration from the best net again
                net.load_state_dict(best_state)
        rec["sec"] = round(time.time() - t0, 1)
        log.append(rec)
        print(json.dumps(rec))
        if keep:  # --out always holds the best net so far
            ck = {"model": net.state_dict(), "d_model": args.d_model, "net": net_kw, "env": env_kw,
                  "args": vars(args), "log": log}
            atomic_write(out, lambda p: torch.save(ck, p))
    if pool:
        pool.close()
        pool.join()
    print("saved", out)


if __name__ == "__main__":
    main()
