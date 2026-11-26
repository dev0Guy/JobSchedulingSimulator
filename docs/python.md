# Python, Gymnasium and Stable-Baselines3

The Zig engine is exposed to Python through a small C ABI
([src/app/capi.zig](../src/app/capi.zig)) compiled into a shared library
`libsched`. A pure-Python package, [`python/sched_env`](../python/sched_env),
loads it with `ctypes` and wraps it as a `gymnasium.Env`. No compiler is
needed on the Python side and no data is copied for the grids.

```
 Python                                   Zig (libsched)
 ───────────────────────────────          ─────────────────────────────────
 SB3 / MaskablePPO                         
   │ step(a) / action_masks()             
 sched_env.SchedEnv (gymnasium.Env) ──ctypes──► sim_step / sim_action_mask
   │ obs = NumPy views  ◄──── zero-copy ─────  machine pool, job pool
                                           Engine + registry adapters
```

## 1. Build the library

```bash
zig build lib                 # → zig-out/lib/libsched.dylib (.so on Linux, .dll on Windows)
```

`zig build lib` builds ReleaseFast unless you pass another `-Doptimize`.
The Python package finds the library in `zig-out/lib` automatically; to use
one elsewhere set `SCHED_LIB=/path/to/libsched.dylib`.

## 2. Install the Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt     # numpy, gymnasium, stable-baselines3, sb3-contrib
export PYTHONPATH=$PWD/python
```

## 3. Use the environment

```python
from sched_env import SchedEnv

env = SchedEnv(
    machines="deeprm:n=1",              # same spec as  sim --machines
    jobs="poisson:n=60,rate=0.7",       # same spec as  sim --jobs
    n_res=2, n_ticks=20, max_jobs=10,
    reward="neg_in_system",             # | neg_waiting | slowdown | zero (same as --reward)
    max_ticks=2000,                     # truncation (None = never)
)
obs, info = env.reset(seed=0)
mask = env.action_masks()               # bool[n_actions]
obs, reward, terminated, truncated, info = env.step(env.action_space.n - 1)  # wait
```

Or through the registry: `gymnasium.make("Sched-v0", machines=..., ...)`.

Any adapter listed by `zig build run -- --list` works in `machines=` /
`jobs=`; a bad spec raises `ValueError` and prints the reason on stderr.

### Constructor arguments

| Argument | Meaning | Default |
|---|---|---|
| `machines` | usage-generator spec (`random:…`, `deeprm:…`) | `deeprm:n=2` |
| `jobs` | arrival spec (`poisson:…`, `batch:…`) | `poisson:n=100,rate=0.5` |
| `n_res` | resources per machine | 2 |
| `n_ticks` | horizon of every grid (also max job duration) | 20 |
| `max_jobs` | waiting-pool slots (observation & action size) | 10 |
| `reward` | reward spec: `neg_in_system`, `neg_waiting`, `slowdown[:min_dur=k]`, `zero` | `neg_in_system` |
| `max_ticks` | truncate the episode at this tick | `None` |
| `illegal_penalty` | reward added for an illegal action | -1.0 |
| `skip_empty` | fast-forward time while the pool is empty | `True` |
| `drop_impossible` | drop jobs no machine can ever run | `True` |
| `work_conserving` | mask "wait" whenever some job fits now (the agent cannot idle while work fits) | `False` |

### Observation — `Dict`, all `float32` in [0, 1]

| Key | Shape | Content |
|---|---|---|
| `machines` | `(n_machines, n_res, n_ticks)` | free capacity / max capacity, tick 0 = now |
| `jobs` | `(max_jobs, n_res, n_ticks)` | demand profile / max capacity, 0 for empty slots |
| `valid` | `(max_jobs,)` | 1 if the slot holds a waiting job |
| `wait` | `(max_jobs,)` | ticks waited / n_ticks, clipped to 1 |
| `age` | `(max_jobs,)` | log(1 + ticks waited) / log(1 + n_ticks), not clipped: 1 at n_ticks, keeps growing |

This is the DeepRM "image" representation: a job fits on a machine iff its
`jobs[s]` grid is ≤ `machines[m]` everywhere on `[0, duration)`.

### Action — `Discrete(max_jobs · n_machines + 1)`

| Index | Meaning |
|---|---|
| `a < max_jobs·n_machines` | place job in slot `a // n_machines` on machine `a % n_machines`, now |
| `a == max_jobs·n_machines` | wait: time advances one tick, finished jobs leave, new jobs arrive |

Several placements can happen in one tick; only *wait* moves time, and the
per-tick reward is paid on *wait*. An illegal action gets `illegal_penalty`
and is treated as *wait* (`info["illegal"] = True`). Use
`env.action_masks()` to avoid illegal actions altogether.

### Episode end

`terminated` when every job has arrived, been placed and finished;
`truncated` at `max_ticks`. The final `info["stats"]` holds `avg_wait`,
`max_wait`, `makespan`, `avg_util`, `placed`, `arrived`, `rejected`,
`illegal`, `total_reward`, …

### Baselines at native speed

```python
env.run_policy("sjf", seed=1000)    # fifo | fifo_strict | sjf | best_fit | tetris | tetris_srpt[:weight=2] | packer | random
```

runs a whole episode inside Zig with a built-in heuristic and returns the
same stats dict — use the same seeds as your agent's evaluation episodes for
a fair comparison.

### Stepping a built-in policy

`env.policy_action("sjf")` returns the action a built-in Zig policy would
take in the current state, so you can step heuristics from Python (stepping
SJF this way reproduces `run_policy("sjf", seed)` exactly). `env.raw()`
returns the unnormalized state and `env.capacity()` the machine capacities.

### Visualization

```bash
PYTHONPATH=python python python/examples/visualize.py --policy sjf --out sjf.gif
PYTHONPATH=python python python/examples/visualize.py --policy fifo        # live window
PYTHONPATH=python python python/examples/visualize.py --model ppo_sched.zip --out ppo.gif
```

One frame per tick, at the start of the tick (after arrivals, before
placements):

- **machines**: used / capacity heatmap per machine, resource × ticks ahead
  (column 0 = now);
- **waiting pool**: one row per resource of each slot (`slot s / r0, r1, …`),
  demand over the job's duration, with how long it has waited;
- **time series**: waiting jobs, jobs placed that tick, utilization.

From code: `frames = sched_env.viz.record(env, "sjf", seed=0)` then
`sched_env.viz.animate(env, frames, "out.gif")`. `record` also accepts a
callable `(env, obs) -> action`, e.g. a trained model.

### Recording and replay

The `trace` observer writes everything needed to rebuild an episode —
machine capacities, every arrival with its full demand profile, every
placement, rejection, illegal action and tick — to a JSON Lines file. The
file is self-contained: copy or download it and replay it anywhere with
Python, without the simulator or `libsched`.

```bash
# record (command line): one file, all episodes
zig build run -- --machines deeprm:n=1 --jobs poisson:n=60,rate=0.8 --res 2 --ticks 20 \
    --pool 10 --policy sjf --episodes 3 --observe trace:path=run.jsonl

# replay
PYTHONPATH=python python python/examples/replay.py run.jsonl --list
PYTHONPATH=python python python/examples/replay.py run.jsonl --episode 1 --out ep1.gif
PYTHONPATH=python python python/examples/replay.py run.jsonl --from-tick 40 --to-tick 80   # live window
```

From Python (e.g. to record a trained agent):

```python
env.start_trace("ppo.jsonl")          # before reset: whole episodes
obs, _ = env.reset(seed=0)
...                                   # step with any policy or model
env.stop_trace()

from sched_env.replay import load_trace
from sched_env.viz import animate
ep = load_trace("ppo.jsonl")[0]
print(ep.stats)                       # placed, arrived, avg_wait, max_wait, makespan, avg_util, ...
print(ep.placements[:3])              # raw place events (id, machine, t, arrival, dur)
animate(ep, ep.frames, "ppo.gif")
```

`load_trace` rebuilds one frame per tick (machine usage, waiting pool, waits)
— identical to a live `viz.record` run (checked by a test). Format: one
JSON object per line, `type` ∈ `episode`, `arrive`, `reject`, `place`,
`illegal`, `tick`, `done`; documented at the top of
[src/adapters/observers/trace.zig](../src/adapters/observers/trace.zig).
Size: roughly 100 bytes per event plus the demand profiles (a 60-job,
3-episode run ≈ 35 KB); `ticks=0` drops tick lines but then replay cannot
rebuild frames.

## 4. Train with Stable-Baselines3

Use `MaskablePPO` from `sb3-contrib` (plain PPO works too, but wastes most of
its exploration on illegal actions):

```python
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from sched_env import SchedEnv

KW = dict(machines="deeprm:n=1", jobs="poisson:n=60,rate=0.7",
          n_res=2, n_ticks=20, max_jobs=10, max_ticks=2000)

def make():
    return ActionMasker(SchedEnv(**KW), lambda e: e.action_masks())

venv = make_vec_env(make, n_envs=8, seed=0)
model = MaskablePPO("MultiInputPolicy", venv, n_steps=256, batch_size=256, verbose=1)
model.learn(total_timesteps=100_000)

env = SchedEnv(**KW)
obs, _ = env.reset(seed=1000)
done = False
while not done:
    a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
    obs, r, term, trunc, info = env.step(a)
    done = term or trunc
print(info["stats"]["avg_wait"], env.run_policy("sjf", 1000)["avg_wait"])
```

The full script, including a comparison against every heuristic on 20
seeds, is [python/examples/train_sb3.py](../python/examples/train_sb3.py):

```bash
PYTHONPATH=python python python/examples/train_sb3.py --steps 100000
PYTHONPATH=python python python/examples/train_sb3.py --device mps    # Apple GPU (cuda for NVIDIA)
```

`--device` only moves the neural network; the simulator always runs on the
CPU. For this small policy the GPU is often no faster than the CPU.

Sample output after 100k steps (8 envs, ≈4 400 env steps/s on an M-series
laptop, almost all of it in PyTorch):

```
ppo          avg_wait    2.06  makespan    98.6  reward    -356.1
fifo         avg_wait    2.07  makespan    98.5  reward    -355.0
sjf          avg_wait    1.49  makespan    99.9  reward    -322.4
best_fit     avg_wait    2.06  makespan    99.0  reward    -354.8
random       avg_wait    2.12  makespan    98.2  reward    -357.9
```

100k steps only reaches FIFO level; train longer (1–10 M steps) to approach
or beat SJF.

### Tips

- **Load matters.** With few jobs or many machines every policy is optimal
  and the comparison is meaningless; raise `rate` or lower `n` until
  `fifo` and `sjf` differ.
- **Keep the action space small.** It is `max_jobs·n_machines + 1`; DeepRM
  used 10 visible slots. Jobs beyond `max_jobs` are rejected (`rejected`
  in stats) — increase it if that happens.
- **Flat observations.** For `MlpPolicy` instead of `MultiInputPolicy`, wrap
  with `gymnasium.wrappers.FlattenObservation`.
- **Seeds.** `reset(seed=s)` makes arrivals and profiles reproducible.

## 5. Tests

```bash
zig build test                              # includes the C ABI test
PYTHONPATH=python pytest python/tests       # gymnasium check_env + episodes
```

## C ABI reference

All functions are in [src/app/capi.zig](../src/app/capi.zig); the ctypes
declarations mirror them in [python/sched_env/_lib.py](../python/sched_env/_lib.py).

| Function | Purpose |
|---|---|
| `sim_create(*SimConfig) → handle` | build generator, arrivals and engine; NULL on error |
| `sim_destroy(h)` | free everything |
| `sim_reset(h, seed) → int` | start an episode |
| `sim_step(h, action, *SimStep) → int` | one action; -1 if no episode is running |
| `sim_action_mask(h, u8*)` | `n_actions` bytes, 1 = legal |
| `sim_slot_valid(h, u8*)` | `max_jobs` bytes, 1 = waiting job |
| `sim_dims(h, *SimDims)` | sizes, job `stride`/`block`, machine `m_stride`/`m_block`, `max_cap` |
| `sim_jobs_ptr` | zero-copy job pool `[slot][res][stride]` (u16) |
| `sim_machines_ptr`, `sim_machine_head` | machine pool `[machine][res][m_stride]`; tick 0 is at `head` (moves with time) |
| `sim_capacity_ptr` | `[n_machines][n_res]` capacities |
| `sim_duration_ptr / sim_arrival_ptr` | per-slot metadata |
| `sim_now(h)`, `sim_stats(h, *SimStats)` | clock and episode statistics |
| `sim_trace_open(h, path)`, `sim_trace_close(h)` | record events to a JSON Lines trace |
| `sim_run_policy(h, name, seed, *SimStats)` | full episode with a built-in policy |

Pointers stay valid until `sim_destroy` and always reflect the current state.
Each job grid occupies `block` elements, each row `stride` elements, and only the
first `n_ticks` of a row are real. Machine rows are a sliding window
(`m_stride`, `m_block`, offset `sim_machine_head()`), so a machine view must be
rebuilt after every step — `SchedEnv` does this (see [data-layout.md](data-layout.md));
`_lib.grid_view` builds a strided NumPy view that hides the padding. One
handle is not thread-safe; use one per environment (SB3's vector envs do).
