"""Evaluate the pointer network (greedy and with MCTS) against the built-in
heuristics on the *same* seeded workloads.

    PYTHONPATH=python:paper python paper/evaluate.py paper/out/ptr.pt --episodes 20 --sims 32

Fairness / reproducibility
  * Every policy runs on seeds EVAL_SEED_START + 0..N-1 (disjoint from the
    training seeds). The seed fixes the arrival process and every job's
    demand profile, so all policies see the same workload.
  * This is checked, not assumed: for every seed a reference run with a
    pool large enough to admit every job records the full workload (job id
    -> arrival tick, duration, demand grid). Each policy's run is recorded
    with the trace observer and must have the same jobs at the same ticks,
    and every job it admitted must have the reference profile -- including
    jobs another policy rejected for a full pool.
  * Training and evaluation seeds must not overlap (checked from the
    checkpoint's training arguments).
  * Engine determinism is checked too: the first policy is run twice per
    seed and the two trace files must be byte-identical.

Metrics per episode (from the trace): average slowdown (wait + dur) / dur,
average wait, makespan, total reward, jobs rejected (full pool, impossible)
admitted jobs left unplaced, and episodes cut off at --max-ticks.
Writes paper/out/eval.json and prints a mean ± std table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from ptrmcts import (ENV_KW, EVAL_SEED_START, MCTS, PointerNet, act_greedy, atomic_write, fingerprint,
                     train_seed_range)
from sched_env import SchedEnv

BUILTIN = ["fifo", "sjf", "best_fit", "tetris", "tetris_srpt", "packer", "random"]


def read_trace(path: Path) -> tuple[dict, dict]:
    """-> (metrics, workload) of the single episode in `path`."""
    work: dict[int, tuple] = {}
    places = []
    done = None
    for line in path.read_text().splitlines():
        e = json.loads(line)
        if e["type"] == "arrive":
            work[e["id"]] = (e["t"], e["dur"], json.dumps(e["demand"]))
        elif e["type"] == "reject":
            work.setdefault(e["id"], (e["t"], None, None, e["reason"]))  # profile not in the trace
        elif e["type"] == "place":
            places.append(e)
        elif e["type"] == "done":
            done = e
    sd = [(p["t"] - p["arrival"] + p["dur"]) / p["dur"] for p in places]
    wait = [p["t"] - p["arrival"] for p in places]
    m = {"slowdown": float(np.mean(sd)) if sd else float("nan"),
         "wait": float(np.mean(wait)) if wait else float("nan"),
         "makespan": done["makespan"] if done else None,
         "unplaced": sum(1 for v in work.values() if v[1] is not None) - len(places),  # admitted, never placed
         "rejected": sum(1 for v in work.values() if v[1] is None),
         "truncated": int(bool(done and done.get("truncated")))}
    return m, work


def same_workload(ref: dict, run: dict) -> bool:
    """`run` has exactly the reference jobs at the same ticks, and every job
    it admitted has the reference duration and demand. (`ref` admitted every
    job, so this also covers jobs this or another policy rejected.)"""
    if ref.keys() != run.keys():
        return False
    for k, v in run.items():
        if v[0] != ref[k][0]:
            return False
        if v[1] is not None and v[1:3] != ref[k][1:3]:
            return False
    return True


def reference_workload(env_kw: dict, seed: int, path: Path) -> dict:
    """Full workload of `seed`: a run with a pool big enough to admit every job."""
    env = SchedEnv(**{**env_kw, "max_jobs": 1024})
    _, work = run(env, seed, lambda e, o, s, p: e.policy_action("fifo"), path)
    env.close()
    full = [k for k, v in work.items() if v[1] is None and v[3] == "pool_full"]
    if full:
        raise SystemExit(f"reference run for seed {seed} still rejected {len(full)} jobs for a full pool")
    return work


def run(env, seed, policy, trace: Path) -> dict:
    """One episode with `policy(env, obs, seed, prefix) -> action`; returns metrics."""
    env.start_trace(str(trace))
    obs, _ = env.reset(seed=seed)
    total, prefix, done = 0.0, [], False
    t0 = time.time()
    while not done:
        a = policy(env, obs, seed, prefix)
        obs, r, term, trunc, info = env.step(a)
        total += r
        prefix.append(a)
        done = term or trunc
    env.stop_trace()
    m, work = read_trace(trace)
    m.update(reward=total, sec=time.time() - t0, illegal=info["stats"]["illegal"])
    return m, work


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", nargs="?", help="model from train.py (omit: heuristics only)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=EVAL_SEED_START)
    ap.add_argument("--sims", type=int, default=32, help="MCTS simulations per move (0 = skip ptr+mcts)")
    ap.add_argument("--policies", default=",".join(BUILTIN))
    ap.add_argument("--max-ticks", type=int, default=2000,
                    help="end (truncate) an episode after this many ticks; counted in the 'truncated' column")
    ap.add_argument("--ppo", default=None, help="also evaluate a MaskablePPO model from ppo.py (.zip)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="paper/out/eval.json")
    args = ap.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device(args.device)
    seeds = [args.seed_start + i for i in range(args.episodes)]
    ck = None
    if args.checkpoint:
        # weights_only: plain tensors/containers only, no arbitrary code on load
        ck = torch.load(args.checkpoint, map_location=device, weights_only=True)
        missing = {"model", "d_model", "env", "args"} - set(ck)
        if missing:
            raise SystemExit(f"{args.checkpoint}: not a train.py checkpoint (missing {sorted(missing)})")
        tr = ck["args"]
        used = train_seed_range(tr["seed"], tr["bc_episodes"] + tr["iters"] * tr["episodes"])
        overlap = [s for s in seeds if s in used]
        if overlap:
            raise SystemExit(f"evaluation seeds {overlap[0]}.. were used in training ({used.start}..{used.stop - 1})")
    ppo_meta = None
    if args.ppo:
        ppo_meta = json.loads(Path(args.ppo).with_suffix(".json").read_text())
        lo, hi = ppo_meta["seeds"]
        overlap = [s for s in seeds if lo <= s < hi]
        if overlap:
            raise SystemExit(f"evaluation seeds {overlap[0]}.. were used in PPO training ({lo}..{hi - 1})")
        if ck and ppo_meta["env"] != ck["env"]:
            raise SystemExit("ptr checkpoint and PPO model were trained on different envs")
    env_kw = dict(ck["env"] if ck else ppo_meta["env"] if ppo_meta else ENV_KW)
    # a policy that keeps choosing "wait" would otherwise never end the episode;
    # every job has arrived long before this, so the cap does not change the workload
    env_kw["max_ticks"] = env_kw.get("max_ticks") or args.max_ticks
    env = SchedEnv(**env_kw)

    policies = {name: (lambda e, o, s, p, n=name: e.policy_action(n)) for name in args.policies.split(",") if name}
    if ck:
        net = PointerNet.for_env(env, **ck.get("net", {"d": ck["d_model"], "layers": 2})).to(device)
        net.load_state_dict(ck["model"])
        net.eval()
        policies["ptr"] = lambda e, o, s, p: act_greedy(net, o, e.action_masks(), device)
        if args.sims:
            mcts = MCTS(net, env_kw, device, sims=args.sims, gamma=tr["gamma"], reward_scale=tr["reward_scale"],
                        trunc_penalty=tr.get("trunc_penalty", 0.0))
            policies[f"ptr+mcts{args.sims}"] = (
                lambda e, o, s, p: int(np.argmax(mcts.search(s, p, fingerprint(e), noise=False))))

    if args.ppo:
        from sb3_contrib import MaskablePPO
        ppo = MaskablePPO.load(args.ppo, device=args.device)
        policies["ppo"] = lambda e, o, s, p: int(ppo.predict(o, action_masks=e.action_masks(), deterministic=True)[0])

    results = {name: [] for name in policies}
    checks = {"workload_mismatch": [], "nondeterministic": []}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for seed in seeds:
            ref_work = reference_workload(env_kw, seed, tmp / "ref.jsonl")
            for k, (name, pol) in enumerate(policies.items()):
                m, work = run(env, seed, pol, tmp / f"{k}.jsonl")
                results[name].append({"seed": seed, **m})
                if not same_workload(ref_work, work):
                    checks["workload_mismatch"].append((seed, name))
                if k == 0:
                    # determinism: same policy, same seed, again -> identical trace bytes
                    run(env, seed, pol, tmp / "again.jsonl")
                    h = [hashlib.sha256((tmp / f).read_bytes()).hexdigest() for f in (f"{k}.jsonl", "again.jsonl")]
                    if h[0] != h[1]:
                        checks["nondeterministic"].append(seed)
            print(f"seed {seed}: " + "  ".join(f"{n} {results[n][-1]['slowdown']:.2f}" for n in policies))

    print(f"\n{len(seeds)} seeds from {args.seed_start}; same workload for every policy: "
          f"{'yes' if not checks['workload_mismatch'] else 'NO ' + str(checks['workload_mismatch'])}; "
          f"engine deterministic: {'yes' if not checks['nondeterministic'] else 'NO ' + str(checks['nondeterministic'])}\n")
    cols = ["slowdown", "wait", "makespan", "reward", "rejected", "unplaced", "truncated", "sec"]
    print(f"{'policy':<14}" + "".join(f"{c:>18}" for c in cols))
    summary = {}
    for name, rows in results.items():
        summary[name] = {}
        line = f"{name:<14}"
        for c in cols:
            x = np.array([r[c] for r in rows], dtype=float)
            summary[name][c] = {"mean": float(np.nanmean(x)), "std": float(np.nanstd(x))}
            line += f"{np.nanmean(x):>11.2f} ±{np.nanstd(x):>5.2f}"
        print(line)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps({"seeds": seeds, "env": env_kw, "checks": checks, "summary": summary,
                       "episodes": results, "checkpoint": args.checkpoint}, indent=1)
    atomic_write(out, lambda p: Path(p).write_text(text))
    print("\nsaved", out)
    if checks["workload_mismatch"] or checks["nondeterministic"]:
        raise SystemExit("seed checks FAILED")


if __name__ == "__main__":
    main()
