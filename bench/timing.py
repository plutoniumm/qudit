import time

MAX_RUN_MS = 600_000   # 10m per individual run — stop collecting for this framework/size


def timed(fn, N, warmup, mps=False):
    """
    Run fn() warmup+N times, returning list of ms timings.
    Raises RuntimeError if any single run exceeds MAX_RUN_MS — caller records None
    and skips remaining sizes for this framework.
    """
    import torch as _torch

    for _ in range(warmup):
        t0 = time.perf_counter()
        fn()
        if mps:
            _torch.mps.synchronize()
        if (time.perf_counter() - t0) * 1e3 > MAX_RUN_MS:
            raise RuntimeError(f"warmup exceeded {MAX_RUN_MS}ms limit")

    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        fn()
        if mps:
            _torch.mps.synchronize()
        elapsed = (time.perf_counter() - t0) * 1e3
        if elapsed > MAX_RUN_MS:
            raise RuntimeError(f"run exceeded {MAX_RUN_MS}ms limit")
        times.append(elapsed)
    return times
