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
- a pool of 10 waiting slots;
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
- Dirichlet noise is added at the root during training.

The engine cannot be cloned. Each simulation therefore restores the root state with `reset(seed)` and replays the actions taken so far. Every restore checks that the replayed state matches the recorded one, so this relies on, and continuously verifies, that the engine is deterministic for a given seed.

## Training

1. Imitate `tetris_srpt` for 40 episodes.
2. Repeat:
   - play self-play episodes, sampling from the visit distribution π for the first 30 moves and choosing greedily after that;
   - train on `CE(π) + 0.5·Huber(value, discounted return / 10)`.

All randomness is seeded from `--seed`. Training episode e uses seed `seed·10⁶ + e`.

## Evaluation and seeds

Every policy runs on the same seeds: `10¹²`, `10¹² + 1`, … These never overlap with training seeds. Two things are checked, not assumed:

- **Same workload for every policy.** Each run is recorded with the trace observer. For each seed, every job's arrival tick and demand must match across policies.
- **Deterministic engine.** The first policy runs twice per seed, and the two trace files must be byte-identical.

If either check fails, the script exits with an error.

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
