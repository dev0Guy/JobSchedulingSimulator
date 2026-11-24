# User guide

How to run experiments, read the results, and extend the simulator.
For the internals see [architecture.md](architecture.md) and
[data-layout.md](data-layout.md); for reinforcement learning see
[python.md](python.md).

## 1. The model in one page

- **Machines** have `n_res` resources (CPU, memory, …), each with a capacity.
  A machine is a `[resource][tick]` grid of *free* capacity over a sliding
  horizon of `n_ticks` ticks; tick 0 is "now".
- **Jobs** arrive over time and wait in a pool of `max_jobs` slots. Each job
  is a `[resource][tick]` grid of *demand* relative to its start, with a
  duration `1 ≤ dur ≤ n_ticks`. Demand may vary over the job's lifetime.
- A job **fits** on a machine if its demand is ≤ the machine's free capacity
  on every resource and every tick of its duration. Placing it subtracts its
  grid from the machine's.
- A **policy** is asked, repeatedly, what to do: *place job s on machine m
  now*, or *wait*. Waiting advances time one tick: the machine horizon slides
  one tick (finished work disappears, a fresh empty tick enters at the end), new jobs
  arrive.
- The **reward** is paid per elapsed tick: `neg_in_system` = −(waiting +
  running jobs) (minimizes slowdown, as in DeepRM), `neg_waiting` =
  −(waiting jobs), or `zero`.
- An episode **terminates** when all jobs have arrived, been placed and
  finished, and is **truncated** at `--max-ticks`.

## 2. Running experiments from the command line

```bash
zig build run -Doptimize=ReleaseFast -- [options]
```

Always use ReleaseFast for real measurements; Debug is ~10–50× slower.

| Option | Meaning | Default |
|---|---|---|
| `--machines <spec>` | machine capacities + job profiles | `random` |
| `--jobs <spec>` | arrival process | `poisson` |
| `--policy <name>` | `fifo`, `fifo_strict`, `sjf`, `best_fit`, `random` | `fifo` |
| `--observe <spec>` | `viz`, `metrics`, `log` (repeatable) | none |
| `--res`, `--ticks`, `--pool` | resources, horizon, waiting slots | 10, 100, 128 |
| `--reward` | `neg_in_system`, `neg_waiting`, `zero` | `neg_in_system` |
| `--max-ticks <n>` | truncate episodes | never |
| `--seed`, `--episodes` | first seed; episode *i* uses `seed + i` | 42, 1 |
| `--list` | every adapter with its options | |

A *spec* is `name:key=value,key=value`. Values can be integers, floats,
booleans (`1/0/yes/no`), ranges `lo-hi` and lists `a/b/c`. Unknown keys are
errors, so typos are caught.

### Workloads

| Spec | Options | Description |
|---|---|---|
| `random` | `n=4 caps=100/80/60/40 dur=1-20 demand=0-30 varying=0` | machine *i* gets `caps[i % len]` on every resource; uniform duration and demand; `varying=1` changes demand at random points during a job |
| `deeprm` | `n=1 cap=100 short=3 long=10-15 short_prob=0.8` | DeepRM workload: identical machines, 80 % short jobs, one dominant resource per job (25–50 % of cap), others 5–10 % |
| `poisson` | `n=500 rate=1.0` | `n` jobs, exponential gaps, `rate` jobs per tick on average |
| `batch` | `n=200` | all jobs at tick 0 |

Arrivals and profiles are independent: any `--jobs` works with any `--machines`.

### Observers

| Spec | Output |
|---|---|
| `viz:delay=80,per_resource=0,on_place=0,jobs=20,cols=100,color=1` | animated terminal view of machines and the waiting pool |
| `metrics` | wait percentiles, utilization, jobs per machine at episode end |
| `log:ticks=0` | one line per event on stderr |

### Example: compare policies

```bash
for p in fifo sjf best_fit random; do
  zig build run -Doptimize=ReleaseFast -- --machines deeprm:n=2 \
      --jobs poisson:n=2000,rate=1.2 --res 2 --ticks 30 \
      --policy $p --episodes 10
done
```

Each episode prints `placed`, `makespan`, `avg wait`, `util`, `reward`,
`steps` and wall time. Use the same `--seed` for every policy so they see
identical workloads.

### Reading the numbers

| Metric | Meaning |
|---|---|
| `placed / arrived` | jobs started / jobs admitted to the pool |
| `rejected` | jobs dropped: pool full, impossible on every machine, or invalid profile |
| `avg wait`, `max wait` | ticks between arrival and start |
| `makespan` | tick when the last job finishes |
| `util` | mean over ticks of used/capacity at tick 0, averaged over machines and resources |
| `reward` | sum of per-tick rewards (plus illegal-action penalties) |

If `rejected > 0` with reason *pool full*, increase `--pool`.

## 3. Writing a policy in Zig

A policy is any type with `decide`:

```zig
const sched = @import("sched");
const rules = sched.engine.rules;

const LongestFirst = struct {
    pub const name = "longest";           // needed only for the registry
    pub fn decide(self: *LongestFirst, st: *const sched.State) sched.Action {
        _ = self;
        var best: ?usize = null;
        var best_m: usize = 0;
        var it = st.jobs.iter();           // waiting jobs, by slot
        while (it.next()) |s| {
            if (best) |b| if (st.jobs.duration[s] <= st.jobs.duration[b]) continue;
            best_m = rules.firstMachine(st, s) orelse continue;
            best = s;
        }
        const s = best orelse return .wait;
        return .{ .place = .{ .slot = @intCast(s), .machine = @intCast(best_m) } };
    }
};
```

Useful queries on `State`:

| Query | Returns |
|---|---|
| `st.now` | current tick |
| `st.jobs.iter()`, `st.jobs.count`, `st.jobs.oldest()` | waiting jobs |
| `st.jobs.duration[s]`, `arrival[s]`, `id[s]`, `peakOf(s)` | job metadata |
| `st.jobs.get(s)` | the job's demand grid (`Usage`) |
| `st.machines.n`, `st.machines.get(m)` | machines |
| `machine.freeNow(r)`, `.cap`, `.earliestFit(job, dur)`, `.utilizationNow()` | machine queries |
| `rules.fitsNow(st, s, m)`, `rules.firstMachine(st, s)` | legality, same test the engine uses |

Return `.wait` when you want time to move. Never mutate the state.

Run it directly (see [examples/custom_policy.zig](../examples/custom_policy.zig),
`zig build example`):

```zig
var eng = try sched.Engine.init(gpa, .{ .n_res = 2, .n_ticks = 30 }, .init(&arrivals), .init(&profiles));
defer eng.deinit();
try eng.reset(1);
var p = LongestFirst{};
const stats = try eng.run(&p, 10_000_000);
std.debug.print("avg wait {d:.2}\n", .{stats.avgWait()});
```

To make it available as `--policy longest` (and from Python's
`run_policy`): put it in `src/adapters/policies/`, export it in
`src/adapters.zig`, add it to `policy_types` and `list_text` in
`src/app/registry.zig`, and add a test.

## 4. Other adapters

The same three steps (implement, export, register) apply to every port.
Contracts are documented at the top of each port file:

| Port | File | Implement |
|---|---|---|
| `ArrivalSource` | [src/ports/arrival_source.zig](../src/ports/arrival_source.zig) | `next(now) ?JobMeta`, `done()`, optional `reset(seed)` |
| `UsageGenerator` | [src/ports/usage_generator.zig](../src/ports/usage_generator.zig) | `machineCount()`, `machine(i, cap)`, `job(meta, out) u16`, optional `reset(seed)` |
| `Observer` | [src/ports/observer.zig](../src/ports/observer.zig) | `onEvent(event, state)` |
| `Policy` | [src/ports/policy.zig](../src/ports/policy.zig) | `decide(state) Action`, optional `reset(seed)` |

Rules that keep experiments valid:

- Be deterministic for a given seed.
- Arrival times are non-decreasing and never in the future.
- Generators write only ticks `[0, dur)` of the job grid and return `dur`.
- Observers never change state or results.

## 5. Reinforcement learning

Build `libsched` and use the Gymnasium environment — see
[python.md](python.md) for setup, observation/action spaces and a
Stable-Baselines3 `MaskablePPO` example.

## 6. Testing and benchmarks

```bash
zig build test                  # unit, integration (tests/), C ABI
PYTHONPATH=python pytest python/tests
zig build bench                 # SIMD + engine throughput (ReleaseFast)
zig build fitmask               # CPU side of bench/compare.py (CPU vs GPU)
```

Build options `-Dlanes`, `-Dcache-line`, `-Dgpu-warp`, `-Dgpu-align` change
SIMD width and padding; results must not depend on them.
