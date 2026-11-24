"""CPU (Zig SIMD) vs GPU comparison of the fit-mask workload.

Runs both sides on the same sizes and prints one table:
    envs | Zig CPU µs | <each Python backend> µs | speedup vs Zig
and the crossover: the smallest batch at which a GPU backend beats Zig.
With matplotlib installed it also saves bench/compare.png.

Run from the project root (THREADS=8 to choose the thread count):
    pip install mlx            # Apple GPU (or: pip install torch)
    python3 bench/compare.py
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
import fitmask_gpu as g  # noqa: E402


def run_zig():
    """Build and run the Zig benchmark in CSV mode →
    ({envs: µs, 1 thread}, {envs: µs, N threads}, N)."""
    zig = os.environ.get("ZIG", "zig")
    cmd = [zig, "build", "fitmask", "--", "--csv"]
    if os.environ.get("THREADS"):
        cmd += ["--threads", os.environ["THREADS"]]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    one, many, n = {}, {}, 1
    for line in (proc.stdout + proc.stderr).splitlines():
        if line.startswith("CSV,"):
            _, envs, us1, usn, threads = line.split(",")
            one[int(envs)] = float(us1)
            many[int(envs)] = float(usn)
            n = int(threads)
    if not one:
        sys.exit("zig benchmark failed:\n" + proc.stderr)
    return one, many, n


def run_python(backend):
    """Run one Python backend → {envs: µs}, checked against numpy."""
    import time
    import numpy as np

    name, prep, run, to_np, _ = backend
    out = {}
    for n_env in g.ENVS:
        jobs, free = g.make_data(n_env)
        ref = g.numpy_backend()[2](jobs, free)
        dj, df = prep(jobs, free)
        run(dj, df)  # warm-up
        best = float("inf")
        for _ in range(g.REPS):
            t0 = time.perf_counter()
            res = run(dj, df)
            best = min(best, time.perf_counter() - t0)
        if not np.array_equal(to_np(res), ref):
            sys.exit(f"{name}: WRONG result at envs={n_env}")
        out[n_env] = best * 1e6
    return out


def main():
    print("running Zig CPU benchmark …")
    one, many, n = run_zig()
    results = {"zig cpu": one, f"zig cpu x{n}": many}
    for b in g.available_backends():
        print(f"running {b[0]} …")
        results[b[0]] = run_python(b)

    names = list(results)
    zig = results["zig cpu"]

    # ---- table: time per batch and speedup relative to Zig ----
    print()
    head = f"{'envs':>6} | " + " | ".join(f"{n:>18}" for n in names)
    print("time per batch (µs), speedup vs Zig in brackets (>1 = faster than Zig)")
    print(head)
    print("-" * len(head))
    for e in g.ENVS:
        cells = []
        for n in names:
            us = results[n][e]
            cells.append(f"{us:>9.1f}" + ("         " if n == "zig cpu" else f" ({zig[e]/us:>5.2f}x)"))
        print(f"{e:>6} | " + " | ".join(f"{c:>18}" for c in cells))

    # ---- crossover ----
    print()
    for n in names[2:]:
        wins = [e for e in g.ENVS if results[n][e] < zig[e]]
        if wins:
            print(f"{n}: faster than 1-thread Zig from {wins[0]} envs "
                  f"(at {g.ENVS[-1]} envs: {zig[g.ENVS[-1]] / results[n][g.ENVS[-1]]:.1f}x)")
        else:
            print(f"{n}: never faster than 1-thread Zig up to {g.ENVS[-1]} envs")

    best_cpu = {e: min(one[e], many[e]) for e in g.ENVS}
    for n in names[2:]:
        wins = [e for e in g.ENVS if results[n][e] < best_cpu[e]]
        print(f"{n}: " + (f"faster than the best Zig CPU result from {wins[0]} envs" if wins
                          else f"never faster than the best Zig CPU result up to {g.ENVS[-1]} envs"))

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(pip install matplotlib for bench/compare.png)")
        return
    tests = {e: e * g.J * g.M for e in g.ENVS}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in names:
        ax.plot(g.ENVS, [tests[e] / (results[n][e] * 1e-6) for e in g.ENVS], marker="o", label=n)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("environments per batch")
    ax.set_ylabel("fit tests per second")
    ax.set_title(f"Fit mask: CPU vs GPU  (J={g.J}, M={g.M}, R={g.R}, T={g.T})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    path = os.path.join(os.path.dirname(__file__), "compare.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nplot saved to {path}")


if __name__ == "__main__":
    main()
