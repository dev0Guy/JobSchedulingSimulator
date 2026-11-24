# Data layout

## Grids

Every job and every machine is a grid of `Elem` (`u16`):

```
row r (resource r):  tick 0 … tick n_ticks-1 | padding |   ← stride elements
```

- **Ticks innermost.** Consecutive ticks are consecutive in memory, so one
  SIMD instruction compares or subtracts `L` ticks.
- **Job grid** = requirement profile relative to the job's start.
  **Machine grid** = *free* capacity, tick 0 = now. The fit test is
  `job[r][k] <= free[r][t+k]`; placing is `free -= job`.
- **Padding.** `stride` is a multiple of the vector width with at least one
  vector of slack, so loops process whole vectors with no tail and offset
  reads stay inside the row. Job padding is 0; machine padding is full
  capacity, so neither changes any result.
- **Alignment.** Each grid starts on a cache line (CPU) or a 256-byte
  boundary (GPU / unified), so grids stored back to back stay aligned.

`Layout.init(n_res, n_ticks, padding)` computes `stride` and `block` from a
`Padding` rule: `grid.simd.padding` (CPU), `grid.simd_gpu.padding` (GPU
only), or `grid.simd_gpu.unified_padding` (shared CPU+GPU memory).

## Pools

```
machines.buf:  [machine][res][m_stride]   one allocation, aligned  (sliding window, see Time)
jobs.buf:      [slot][res][stride]        one allocation, aligned  (= observation)
```

Job metadata is struct-of-arrays (`duration[]`, `arrival[]`, `id[]`,
`peak[]`) so scans read only what they need. Live slots are a bitmask; the
iterator uses count-trailing-zeros to skip 64 empty slots per step. Removing
a job clears its bit and zeroes its grid; nothing moves.

## Time

Machine rows are a **sliding window**. Each row stores `n_ticks + slack`
ticks (`Machines.ring`, slack = max(n_ticks, 4 vectors)); the horizon
[now, now + n_ticks) starts at the shared offset `Machines.head`:

```
row r:  | stale … | head: tick 0 … tick n_ticks-1 | cap cap … cap | padding |
```

- **Advance k ticks** = `head += k`. Nothing is copied: every slot at or
  after `head + n_ticks` is always full capacity (placements never write
  past the horizon), so the ticks that become visible are already right.
- **Compaction.** When `head + n_ticks` would pass `n_ticks + slack`, each
  row's live window is copied back to index 0 and the rest refilled with
  capacity, and `head = 0`. That happens once every `slack` ticks, so the
  cost per tick is O(machines × resources) amortized instead of
  O(machines × resources × n_ticks) for a shift.
- **Contiguous windows.** Unlike a wrap-around ring, the horizon is never
  split, so the SIMD fit test and `subAt` work on `row[head + t ..]`
  unchanged. The price is 2× machine memory.

Running jobs live inside the machine grids, so they age as the window moves;
waiting jobs are relative to their own start and are never touched.
Always access machine ticks through `Machine` (`resource(r)`, `freeNow`,
`canPlace`, …), which add `head`; never index `free.row(r)` directly.

On the 300-machine × 4-resource × 1000-tick FIFO workload this took an
episode from ~415 ms (shifting 2.4 MB every tick) to ~6 ms; the small
engine benchmark (4 machines, 100 ticks) became ~2× faster.

## Fast paths

- **Quick reject** (`Machine.mayFit`): peak above capacity, or first tick
  above free capacity now — no grid scan.
- **SIMD fit test** (`Usage.covers`): `L` ticks per compare, early exit.
- **Arrival-ordered list** (`Jobs.byArrival`, `Jobs.oldest`): waiting jobs
  are also linked in (arrival, slot) order, so `oldest()` is O(1) and FIFO
  stops at the first job that fits instead of scanning the whole pool.
- **"Fits nowhere" memo** (`Jobs.blocked`, `rules.firstMachine`,
  `rules.knownBlocked/markBlocked`): one bit per job meaning "fits on no
  machine this tick". Free capacity only shrinks within a tick, so the bit
  stays valid until the engine clears it when time advances. A hit skips a
  scan over every machine.

Measured on an M1 Pro (ms per episode, 4 resources, 100 ticks, identical
results before and after):

| machines / pool | fifo | fifo_strict | sjf | best_fit | tetris_srpt |
|---|---|---|---|---|---|
| 4 / 128 | 2.4 → 1.6 | 0.5 → 0.3 | 2.8 → 2.7 | 4.5 → 4.6 | 9.2 → 9.0 |
| 32 / 128 | 31.6 → 12.3 | 3.4 → 2.3 | 28.0 → 21.1 | 172 → 132 | 263 → 199 |
| 300 / 128 | 95 → 24 | 16 → 13 | 69 → 33 | 2196 → 2107 | 6828 → 6514 |
| 300 / 1024 | 1683 → 141 | 36 → 15 | 1118 → 218 | 22698 → 21968 | — |

`best_fit` and `tetris` still score every (job, machine) pair that fits,
which dominates on large clusters; making that incremental is the next step.

Tried and rejected: a per-(job, machine) fit cache. The fit test is only a
quick reject plus a few SIMD compares, about as cheap as a scattered cache
lookup, and every miss adds a write — it made FIFO ~2× slower and best_fit
only ~5–10 % faster, so it was removed.

## Sizes (10 resources, 100 ticks, u16)

| Data | Size |
|---|---|
| one grid (L = 16, stride 128) | 2.5 KiB |
| 4 machines (2× for the sliding-window slack) | 20 KiB |
| 128 job slots | 320 KiB |
| metadata (128 slots) | ~5 KiB |
