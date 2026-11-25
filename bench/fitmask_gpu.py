"""Fit-mask benchmark, GPU side of the CPU-vs-GPU comparison.

Same workload and sizes as bench/fitmask_bench.zig:
    mask[e, j, m] = all(job[e, j, r, k] <= free[e, m, r, k])  for all r, k

Backends, whichever are installed:
    numpy        CPU reference (vectorized), also used to check correctness
    mlx          Apple GPU (pip install mlx)               ← use this on a Mac
    torch-mps    Apple GPU via PyTorch (pip install torch)
    torch-cuda   NVIDIA GPU via PyTorch

Run:  python3 bench/fitmask_gpu.py
Every GPU result is checked against numpy before its time is reported.
GPU timings synchronize before stopping the clock and include no host copies
(data is uploaded once), i.e. the "data stays on the GPU" scenario.
"""

import time
import numpy as np

J, M, R, T = 64, 4, 10, 100
ENVS = [1, 4, 16, 64, 256, 1024]
REPS = 7


def make_data(n_env, seed=42):
    rng = np.random.default_rng(seed)
    free = rng.integers(20, 101, size=(n_env, M, R, T), dtype=np.uint16)
    dur = rng.integers(1, 21, size=(n_env, J))
    demand = rng.integers(0, 31, size=(n_env, J, R, 1), dtype=np.uint16)
    ticks = np.arange(T)[None, None, None, :]
    jobs = np.where(ticks < dur[:, :, None, None], demand, 0).astype(np.uint16)
    return jobs, free


def chunks(n_env, max_cells=64_000_000):
    """Split envs so the (E, J, M, R, T) comparison fits in memory."""
    per_env = J * M * R * T
    step = max(1, max_cells // per_env)
    return [(i, min(n_env, i + step)) for i in range(0, n_env, step)]


# ---------------------------------------------------------------- backends

def numpy_backend():
    def prep(jobs, free):
        return jobs, free

    def run(jobs, free):
        out = []
        for a, b in chunks(jobs.shape[0]):
            ok = jobs[a:b, :, None] <= free[a:b, None, :]      # (E, J, M, R, T)
            out.append(ok.all(axis=(-1, -2)))
        return np.concatenate(out)

    return "numpy (CPU)", prep, run, lambda x: x, lambda: None


def mlx_backend():
    import mlx.core as mx

    def prep(jobs, free):
        j, f = mx.array(jobs), mx.array(free)
        mx.eval(j, f)
        return j, f

    def run(jobs, free):
        outs = []
        for a, b in chunks(jobs.shape[0]):
            ok = jobs[a:b, :, None] <= free[a:b, None, :]
            outs.append(mx.all(ok, axis=(-1, -2)))
        out = mx.concatenate(outs)
        mx.eval(out)
        return out

    return "mlx (Apple GPU)", prep, run, lambda x: np.array(x), lambda: None


def torch_backend(device):
    import torch

    if device == "mps" and not torch.backends.mps.is_available():
        raise ImportError
    if device == "cuda" and not torch.cuda.is_available():
        raise ImportError
    sync = torch.mps.synchronize if device == "mps" else torch.cuda.synchronize

    def prep(jobs, free):
        # torch has limited uint16 support; int32 is safe for comparisons
        j = torch.from_numpy(jobs.astype(np.int32)).to(device)
        f = torch.from_numpy(free.astype(np.int32)).to(device)
        sync()
        return j, f

    def run(jobs, free):
        outs = []
        for a, b in chunks(jobs.shape[0]):
            ok = jobs[a:b, :, None] <= free[a:b, None, :]
            outs.append(ok.flatten(-2).all(dim=-1))
        out = torch.cat(outs)
        sync()
        return out

    return f"torch-{device}", prep, run, lambda x: x.cpu().numpy(), sync


def available_backends():
    found = [numpy_backend()]
    for make in (mlx_backend, lambda: torch_backend("mps"), lambda: torch_backend("cuda")):
        try:
            found.append(make())
        except Exception:
            pass
    return found


# ---------------------------------------------------------------- bench

def bench(backend):
    name, prep, run, to_np, _ = backend
    print(f"\n{name}")
    print(f"{'envs':>6} | {'µs / batch':>12} | {'fit tests / s':>14} | {'check':>6}")
    print(f"{'-'*6}-+-{'-'*12}-+-{'-'*14}-+-{'-'*6}")
    for n_env in ENVS:
        jobs, free = make_data(n_env)
        ref = numpy_backend()[2](jobs, free)
        dj, df = prep(jobs, free)
        run(dj, df)  # warm-up (kernel compilation, allocation)
        best = float("inf")
        for _ in range(REPS):
            t0 = time.perf_counter()
            out = run(dj, df)
            best = min(best, time.perf_counter() - t0)
        ok = np.array_equal(to_np(out), ref)
        tests = n_env * J * M
        print(f"{n_env:>6} | {best*1e6:>12.1f} | {tests/best:>14.3e} | {'ok' if ok else 'WRONG':>6}")


if __name__ == "__main__":
    print(f"fit-mask bench   J={J} M={M} R={R} T={T}")
    backs = available_backends()
    if len(backs) == 1:
        print("(no GPU backend found: pip install mlx  or  pip install torch)")
    for b in backs:
        bench(b)
