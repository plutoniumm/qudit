"""
QEC benchmark: recovery map construction + application for built-in codes.

Compares Petz / Leung / Cafaro recovery methods on standard codes.
No cross-framework comparison (qudit is unique here); benchmarks internal methods.
"""
import sys
import time
import numpy as np

sys.path.insert(0, "..")

import torch

from qudit.qec import Recovery
from qudit.qec.lib import Dutta3, Leung, Perfect
from qudit.noise import Process

C64 = torch.complex64
WARMUP = 5
N = 50


def _rho(psi):
    n = psi.numel()
    return (psi.view(n, 1) @ psi.view(1, n).conj()).to(C64)


def _time(fn):
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return times


# Codes to benchmark over
CODES = {
    "Dutta3 (3q)": Dutta3,
    "Leung (4q)": Leung,
    "Perfect (5q)": Perfect,
}

# Noise level
_Y = 0.05

RECOVERY_METHODS = {
    "petz": Recovery.petz,
    "leung": Recovery.leung,
    "cafaro": Recovery.cafaro,
}


def bench_construct(code_fn, method_fn):
    code = code_fn()
    n = code.dits
    noise = Process.AD(d=2, n=n, Y=_Y, order=n - 1)
    codewords = [c.to(C64) for c in code.toTensor()]

    def run():
        method_fn(noise, codewords)

    return _time(run)


def bench_apply(code_fn, method_fn):
    code = code_fn()
    n = code.dits
    noise = Process.AD(d=2, n=n, Y=_Y, order=n - 1)
    codewords = [c.to(C64) for c in code.toTensor()]
    rec = method_fn(noise, codewords)
    rho0 = _rho(codewords[0])
    noisy = noise.run(rho0)

    def run():
        rec.run(noisy)

    return _time(run)


# Registry for run.py
# Returns: { "Dutta3 (3q)": {mean_ms, std_ms}, ... } per method per operation
CONFIGS = list(CODES.keys())


def frameworks_construct():
    out = {}
    for method_name, method_fn in RECOVERY_METHODS.items():
        means, stds = [], []
        for code_name, code_fn in CODES.items():
            try:
                t = bench_construct(code_fn, method_fn)
                means.append(round(float(np.mean(t)), 4))
                stds.append(round(float(np.std(t)), 4))
            except Exception as e:
                means.append(None)
                stds.append(None)
                print(f"  [qec/construct/{method_name}/{code_name}] ERR: {e}")
        out[method_name] = {"mean_ms": means, "std_ms": stds}
    return out


def frameworks_apply():
    out = {}
    for method_name, method_fn in RECOVERY_METHODS.items():
        means, stds = [], []
        for code_name, code_fn in CODES.items():
            try:
                t = bench_apply(code_fn, method_fn)
                means.append(round(float(np.mean(t)), 4))
                stds.append(round(float(np.std(t)), 4))
            except Exception as e:
                means.append(None)
                stds.append(None)
                print(f"  [qec/apply/{method_name}/{code_name}] ERR: {e}")
        out[method_name] = {"mean_ms": means, "std_ms": stds}
    return out
