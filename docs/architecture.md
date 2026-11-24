# Architecture

Hexagonal ("ports and adapters"): the core holds only the simulation rules;
everything that varies between experiments plugs in through ports.

```
            driving side:  app/main.zig (CLI)   tests   bench   capi (C ABI → Gymnasium)
                                     │ reset(seed) / step(action) / state()
                                     ▼
   ┌───────────────────────────── core ─────────────────────────────┐
   │ grid    Layout · Usage · SIMD          (no domain knowledge)    │
   │ model   Machines · Jobs · State/Action/Event                    │
   │ engine  rules · Engine (validate, place, advance, arrivals)     │
   └──────┬──────────────────┬───────────────────┬──────────────────┘
          │ ArrivalSource    │ UsageGenerator    │ Policy           │ Observer
          ▼                  ▼                   ▼                  ▼
     poisson, batch     random, deeprm     fifo, sjf, …       viz, metrics, log
```

## Layers and dependency rule

Each layer imports only layers above it in this list:

1. **grid** — `[resource][tick]` grids of `Elem` (u16), padding rules for CPU,
   GPU and unified memory, SIMD operations (`Usage`). No jobs or machines.
2. **model** — `Machines` (free capacity, fit test, sliding-window time), `Jobs`
   (slot pool, metadata, bitmask), and the plain types `State`, `Action`,
   `Event`, `JobMeta`.
3. **ports** — the four interfaces below.
4. **engine** — `rules` (legality, default reward, action encoding/masks) and
   `Engine` (applies actions, enforces rules, emits events).
5. **adapters** — implementations of the ports.
6. **app** — composition root: parses the command line, builds adapters by
   name (registry), wires them into the engine.

## Ports

| Port | Question | Called | Mechanism |
|---|---|---|---|
| `ArrivalSource` | which jobs arrive, when? | every tick | runtime (vtable) |
| `UsageGenerator` | machine capacities; a job's profile | init; per arrival | runtime (vtable) |
| `Policy` | what to do now? | every decision | comptime (inlined) or runtime |
| `Observer` | something happened | every event | runtime, list of up to 8 |
| `Reward` | how good was this tick? | every elapsed tick | runtime (vtable), optional |

Arrival time and profile are separate ports, so any arrival process combines
with any profile distribution.

Policies are chosen at run time by name, but each runs in its own
comptime-specialized loop (`registry.runPolicy`), so `decide` is inlined.

## Step semantics

One `step(action)`:

- `.place{slot, machine}` — must be legal (`rules.isLegal`): slot holds a
  waiting job, the job fits on the machine now. The profile is subtracted
  from the machine starting at tick 0; `job_placed` is emitted.
- `.wait` — time advances one tick: the machine window slides one tick, finished jobs
  leave, new arrivals enter the pool, `tick_advanced` is emitted, the tick
  reward is returned. With `skip_empty`, empty periods are fast-forwarded.
- illegal action — `illegal_action` is emitted, the penalty is added, and it
  is treated as `.wait`.

Several placements can happen in the same tick (repeated `.place` steps).
The episode terminates when all jobs arrived, none wait and none run; it is
truncated at `max_ticks`.

Rewards come from the `Reward` port (`engine.setReward`): `neg_in_system`
= −(waiting + running) (the default when none is set), `neg_waiting` =
−waiting, `slowdown` = −Σ 1/T_j over jobs in system (DeepRM), `zero`.

## Adding an adapter

1. Write a type implementing the port's methods in `src/adapters/<kind>/`.
2. Export it in `src/adapters.zig`.
3. Add one entry in `src/app/registry.zig` (and the help text there).

The engine and the other adapters do not change.

## Gymnasium

`src/app/capi.zig` is a second driving adapter next to the CLI: it exports a
C ABI (`sim_create/reset/step/action_mask/...`), built as `libsched` by
`zig build lib`. The Python `gymnasium.Env` in `python/sched_env` loads it
with ctypes, reads the job and machine pools through zero-copy NumPy views
(strides hide the padding), uses `Discrete(max_jobs·n_machines + 1)` actions
(`rules.encode/decode`) and `rules.actionMask` for masking. Adapters are
built by name through the same registry as the CLI. See
[python.md](python.md).
