# Project notes for Claude Code

SIMD job-to-machine scheduling simulator in **Zig 0.17**, for a thesis
comparing scheduling policies (later RL via Gymnasium).

- Commands: `zig build test`, `zig build run -- --help`, `zig build bench`,
  `zig build example`, `zig build lib` (libsched for Python),
  `PYTHONPATH=python pytest python/tests`. Benchmarks always build ReleaseFast.
- Python binding: src/app/capi.zig (C ABI) ↔ python/sched_env/_lib.py (ctypes
  structs must match field-for-field). Docs: docs/python.md.
- Architecture: hexagonal, layers grid → model → ports → engine → adapters →
  app; each imports only layers before it. See docs/architecture.md.
- Grid cells are `grid.Elem` (u16); never hard-code i32/u16 elsewhere.
- Machines are a sliding window (`Machines.head`, `ring` layout): access ticks
  only via `Machine` methods (`resource`, `freeNow`, `canPlace`), never
  `free.row(r)[t]` directly. Everything past head + n_ticks must stay = cap.
- `Jobs.blocked` is a per-tick "fits on no machine" memo: valid only because
  capacity never grows within a tick; the engine clears it on tick advance.
  Pair-scanning policies should use `rules.knownBlocked/markBlocked`, or
  `PairCache` (adapters/policies/pair_cache.zig), which caches scores across
  decisions using `Machines.epoch`/`stamp`. Anything that changes a machine
  outside time advance must call `Machines.touch(m)` (the engine does on place).
  Policies with caches have `deinit`; callers use `ports.policy.deinitIfAny`.
- Invariants: job padding = 0, machine padding = capacity, `t + dur <=
  n_ticks`, never subtract below 0 (check `canPlace` first).
- Ports: Policy, Observer, UsageGenerator, ArrivalSource, Reward.
- New adapters: implement the port, export in src/adapters.zig, register in
  src/app/registry.zig (+ help text).
- Every behavior change needs a test; integration tests in tests/. Engine
  semantics: add a hand-worked case to tests/scenario_test.zig; the
  `property:` test there checks against a naive reference simulator.
