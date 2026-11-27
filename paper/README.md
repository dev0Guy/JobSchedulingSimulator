# Paper experiments: pointer network + MCTS

| File | What |
|---|---|
| `ptrmcts.py` | Environment config, pointer network, MCTS, replay-based state restore |
| `train.py` | Behaviour-cloning warm start, then AlphaZero-style MCTS self-play |
| `evaluate.py` | Seeded evaluation of the learned policy and every built-in heuristic |

```bash
zig build lib
pip install torch numpy gymnasium
export PYTHONPATH=python:paper
python paper/train.py --iters 15 --episodes 2 --sims 24 --out paper/out/ptr.pt
python paper/evaluate.py paper/out/ptr.pt --episodes 10 --sims 24
```

Outputs go to `paper/out/`, which git ignores: `ptr.pt` holds the weights, the config and the training log, and `eval.json` holds the per-seed results.

## Environment

`ENV_KW` in `ptrmcts.py` defines the environment:
- 1 DeepRM machine;
- 80 Poisson jobs at 0.8 per tick;
- 2 resources;
- a 20-tick horizon;
- a waiting pool with one slot per job (80), so it is never full: no job is rejected and the model always sees every waiting job;
- the DeepRM slowdown reward.

Its sum over an episode is minus the total slowdown.

## Model

The network sees the waiting jobs and the machines as one set:
- each job carries its demand grid, its wait time and a valid flag;
- each machine carries its free-capacity grid.

A transformer encoder and a glimpse produce a context vector q. As in a pointer network, the output points at input elements:
- a pointer score for each job, `vᵀ tanh(W1 h_j + W2 q)`;
- plus a bilinear job–machine score, giving one logit per (job, machine) pair;
- a wait logit;
- a value head on q.

Illegal actions are masked out. The action order matches `SchedEnv`.

## Search

MCTS is PUCT in the AlphaZero style:
- the network gives the prior and evaluates leaves;
- Q values are min-max normalized (as in MuZero);
- Dirichlet noise is added at the root during training;
- leaves are evaluated in batches (`--mcts-batch`, default 8): virtual loss
  sends the simulations of one batch down different paths, and the network
  runs once per batch, on the occupied job slots only (~5x faster than one
  full call per leaf).

The engine cannot be cloned. Each simulation therefore restores the root state with `reset(seed)` and replays the actions taken so far. Every restore checks that the replayed state matches the recorded one, so this relies on, and continuously verifies, that the engine is deterministic for a given seed.

## Training

1. Imitate `sjf` for 40 episodes (`--bc-teacher`).
2. Repeat:
   - play self-play episodes, sampling from the visit distribution for the first 30 moves and choosing greedily after that;
   - policy target π = visit counts^(1/T), `--pi-temp` T = 0.5: sharper than the counts, so the network learns the search's choice;
   - value target: n-step bootstrapped return (`--n-step` 20) from the search value 20 moves later, divided by 10;
   - the value head predicts h(G) = sign(G)(√(|G|+1) − 1) + 0.001·G (MuZero's value transform), which keeps the loss well scaled with gamma 0.999 and cut-off penalties;
   - train on `CE(π) + 0.5·Huber(value, h(target))`.

All randomness is seeded from `--seed`. Training episode e uses seed `seed·10⁶ + e`.

## PPO baseline

`paper/ppo.py` trains sb3-contrib's MaskablePPO with the same PointerNet as
actor and critic (`--policy mlp` for sb3's flat MLP), illegal actions masked,
rewards normalized by VecNormalize, same env and cut-off penalty.
Evaluate it next to everything else with
`python paper/evaluate.py paper/out/ptr.pt --ppo paper/out/ppo.zip`; its
training seeds (block `--seed`, default 1) are checked against the eval seeds.

Tournament (evaluation only): train on few slots (`ppo.py --jobs 10`), then
`evaluate.py --ppo ... --jobs 25` runs the same weights on 25 jobs: placeable
jobs are split into groups of 10 (padded, masked), each group's best job
advances, and a final round over the winners picks the action (`ppo_tour10`).

## Running fast (GPU / many-core machine)

Self-play MCTS evaluates one state at a time and replays the engine on the
CPU, so it is CPU-bound; `--workers N` runs self-play episodes in N parallel
processes (`0` = all cores but one). Gradient updates run on `--device`.
Each episode's search randomness comes from its own seed, so the result is
identical for any number of workers (checked: 1 vs 3 workers give the same log).

    zig build lib -Doptimize=ReleaseFast
    PYTHONPATH=python:paper python paper/train.py --device cuda --workers 0 \
        --episodes 32 --sims 64 --batch 512 --updates 200 --iters 50 --out paper/out/ptr.pt

Use `--episodes` that is a multiple of the worker count. Bigger `--batch` /
`--updates` / `--d-model` use the GPU more; more `--episodes` / `--sims` use
the CPU cores more.

## PSPLIB workload

Job profiles can come from PSPLIB RCPSP instances (Kolisch & Sprecher 1997)
instead of DeepRM. Download a set (e.g. j30.sm) from the PSPLIB site and
unpack it, then:

    python paper/train.py --psplib data/psplib/j30 ...
    python paper/evaluate.py paper/out/ptr.pt ...      # uses the env stored in the checkpoint

Every activity with duration > 0 becomes a job template: its duration and
constant demand on the 4 renewable resources, divided by the instance's
availability and scaled to capacity 100 (so instances can be mixed). Each
job draws a template with the episode seed; precedence relations are
ignored. Arrivals stay `poisson:n=80,rate=0.8`. PSPLIB activities are larger
than DeepRM jobs, so lower the rate if the queue grows without bound.

From the CLI: `sim --machines psplib:path=data/psplib/j30,n=2 --res 4 ...`
(`order=sequential` takes activities in file order).

## Evaluation and seeds

Every policy runs on the same seeds: `10¹²`, `10¹² + 1`, … These never overlap with training seeds. Two things are checked, not assumed:

- **Same workload for every policy.** For each seed, a reference run with a pool large enough to admit every job records the full workload. Every policy's run, recorded with the trace observer, must have the same jobs at the same ticks. Every job it admitted must have the reference duration and demand, including jobs that another policy rejected for a full pool.
- **Deterministic engine.** The first policy runs twice per seed, and the two trace files must be byte-identical.

- **No seed overlap.** The training seeds stored in the checkpoint must not overlap the evaluation seeds.

If any check fails, the script exits with an error.

Other safeguards:
- Checkpoints are loaded with `weights_only=True`, so loading one cannot run code.
- Outputs are written atomically.
- During search, every replay is checked against a hash of the full state: each machine's free capacity, plus each waiting job's grid, duration and arrival tick.

## Seed bug found and fixed

The first evaluation failed the workload check. When the waiting pool was full, the engine rejected a job without generating its profile. The generator's random stream then shifted, so later jobs got different demands depending on the policy. Policies were not being compared on the same workload whenever jobs were rejected for a full pool.

The engine now always draws the profile, writing it into a scratch grid, before rejecting the job. Two Zig tests cover this:
- one checks that the same seed reproduces an episode on fresh or reused engines, for every arrival process × generator;
- one checks that a job's profile does not depend on the policy when the pool rejects jobs.

`python/tests/test_env.py` checks the same through the Python binding.

## First results

10 seeds, after 15 short iterations (about 4 minutes on an M1 Pro CPU):

| policy | slowdown | wait | reward |
|---|---|---|---|
| fifo | 2.66 ± 0.86 | 3.31 | −208 |
| sjf | 1.73 ± 0.32 | 2.15 | −138 |
| best_fit | 2.75 ± 0.91 | 3.33 | −212 |
| tetris | 3.85 ± 1.41 | 4.22 | −283 |
| tetris_srpt | 1.77 ± 0.38 | 2.27 | −141 |
| random | 2.69 ± 1.13 | 3.18 | −210 |
| ptr (greedy) | 1.73 ± 0.31 | 2.31 | −137 |
| ptr + MCTS (24 sims) | 1.73 ± 0.32 | 2.20 | −137 |

These results are preliminary:
- **Short run.** It is far too short to beat SJF. The network has mostly learned its `tetris_srpt` teacher, and is already slightly better than it.
- **For the paper,** use more iterations and simulations, and 100 or more evaluation seeds.

The model was trained before the seed fix, but evaluated after it.
