# sched

A SIMD job-to-machine scheduling simulator in Zig, built for comparing
scheduling policies (heuristics now, reinforcement learning via Gymnasium
later).

Jobs and machines are `[resource][tick]` grids. Placing a job subtracts its
profile from a machine's free capacity; the fit test is a SIMD comparison of
two rows. When nothing can be placed, time advances: machine grids shift one
tick left.

Requires **Zig 0.17**.

## Quick start

```bash
zig build test                         # unit + integration tests
zig build run -- --observe viz         # animated terminal view
zig build run -Doptimize=ReleaseFast -- --observe metrics
zig build bench                        # SIMD + engine benchmarks
zig build example                      # custom policy example
zig build lib                          # libsched for Python / Gymnasium
```

## Python / Stable-Baselines3

```bash
zig build lib
pip install -r python/requirements.txt
PYTHONPATH=python python python/examples/train_sb3.py --steps 100000
PYTHONPATH=python python python/examples/visualize.py --policy sjf --out sjf.gif
```

```python
from sched_env import SchedEnv
env = SchedEnv(machines="deeprm:n=1", jobs="poisson:n=60,rate=0.7",
               n_res=2, n_ticks=20, max_jobs=10)
obs, _ = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.n - 1)  # wait
```

Dict observation (machine and job grids, zero-copy from Zig),
`Discrete(max_jobs·n_machines + 1)` actions, `env.action_masks()` for
`MaskablePPO`. Details in [docs/python.md](docs/python.md).

## Command line

```
sim [options]
  --machines <spec>   usage generator      (default random)
  --jobs <spec>       arrival source       (default poisson)
  --policy <name>     scheduling policy    (default fifo)
  --observe <spec>    observer, repeatable (viz, metrics, log)
  --res <n> --ticks <n> --pool <n> --seed <n> --episodes <n>
  --reward neg_in_system|neg_waiting|zero  --max-ticks <n>
  --list              all adapters and their options
```

A spec is `name:key=value,key=value`, for example:

```bash
zig build run -- --machines random:n=8,caps=100/50,dur=1-10,varying=1 \
                 --jobs poisson:n=1000,rate=1.5 --policy best_fit \
                 --observe viz:delay=30,per_resource=1 --observe metrics
zig build run -- --machines deeprm:n=2 --res 2 --ticks 30 --policy sjf
```

| Kind | Adapters |
|---|---|
| Machines / profiles | `random`, `deeprm` |
| Job arrivals | `poisson`, `batch` |
| Policies | `fifo`, `fifo_strict`, `sjf`, `best_fit`, `random` |
| Observers | `viz`, `metrics`, `log` |

## Build options

| Option | Meaning | Default |
|---|---|---|
| `-Dlanes=<n>` | CPU SIMD lanes | target CPU |
| `-Dcache-line=<bytes>` | CPU cache line | target arch |
| `-Dgpu-warp=<n>` | GPU threads per warp (padding rule) | 32 |
| `-Dgpu-align=<bytes>` | GPU grid alignment | 256 |

## Layout

```
src/
  root.zig       library API (module "sched")
  grid/          aligned grids, SIMD, CPU/GPU padding rules
  model/         machines, jobs, State / Action / Event
  ports/         Policy, Observer, UsageGenerator, ArrivalSource
  engine/        rules (legality, reward, action encoding) and the step loop
  adapters/      policies, arrivals, generators, observers
  app/           command line, registry, main, capi (C ABI for Python)
tests/           integration tests
bench/           simd_bench, engine_bench
examples/        custom_policy
python/          sched_env (Gymnasium env), examples/train_sb3.py, tests
docs/            guide, python, architecture, data-layout
```

## Documentation

- [docs/guide.md](docs/guide.md) — model, command line, metrics, writing policies and adapters
- [docs/python.md](docs/python.md) — Python binding, Gymnasium env, Stable-Baselines3
- [docs/architecture.md](docs/architecture.md) — layers, ports, step semantics
- [docs/data-layout.md](docs/data-layout.md) — grids, padding, SIMD
