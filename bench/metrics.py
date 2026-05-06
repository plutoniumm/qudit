"""
Metrics benchmark — Fidelity, Entropy, Info, Distance on random density matrices.

No cross-framework comparison: these are qudit-specific implementations.
Benchmarks show scaling with system size (d^n matrix).
"""
import sys
import time
import numpy as np

sys.path.insert(0, "..")

from qudit.tools.metrics import Fidelity, Entropy, Info, Distance

from timing import timed

WARMUP = 10
N = 100

SIZES = [2, 4, 8, 16, 32, 64]   # density matrix dimension (not qubits)


def _rho(d):
    A = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    A = A @ A.conj().T
    return A / np.trace(A)


def _rho_pair(d):
    return _rho(d), _rho(d)


def _time(fn):
    return timed(fn, N, WARMUP)


# ── Fidelity ──────────────────────────────────────────────────────────────────

def bench_fidelity(d):
    rho, sigma = _rho_pair(d)
    return _time(lambda: Fidelity.default(rho, sigma))


# ── Von Neumann entropy ───────────────────────────────────────────────────────

def bench_entropy(d):
    rho = _rho(d)
    return _time(lambda: Entropy.neumann(rho))


# ── Mutual information ────────────────────────────────────────────────────────

def bench_mutual_info(d):
    rho = _rho(d * d)   # bipartite system
    return _time(lambda: Info.mutual(rho, d, d))


# ── Trace distance ────────────────────────────────────────────────────────────

def bench_trace_distance(d):
    rho, sigma = _rho_pair(d)
    return _time(lambda: Distance.trace(rho, sigma))


# ── Bures distance ────────────────────────────────────────────────────────────

def bench_bures(d):
    rho, sigma = _rho_pair(d)
    return _time(lambda: Distance.bures(rho, sigma))


# ── Registry ──────────────────────────────────────────────────────────────────

METRICS = {
    "fidelity": {
        "label": "Uhlmann Fidelity",
        "fn": bench_fidelity,
    },
    "entropy": {
        "label": "Von Neumann Entropy",
        "fn": bench_entropy,
    },
    "mutual_info": {
        "label": "Mutual Information",
        "fn": bench_mutual_info,
    },
    "trace_dist": {
        "label": "Trace Distance",
        "fn": bench_trace_distance,
    },
    "bures": {
        "label": "Bures Distance",
        "fn": bench_bures,
    },
}


def run_all():
    import numpy as np_
    out = {}
    for key, spec in METRICS.items():
        means, stds = [], []
        for d in SIZES:
            try:
                t = spec["fn"](d)
                means.append(round(float(np_.mean(t)), 4))
                stds.append(round(float(np_.std(t)), 4))
            except Exception as e:
                means.append(None)
                stds.append(None)
                print(f"  [metrics/{key}/d={d}] ERR: {e}")
        out[key] = {"label": spec["label"], "mean_ms": means, "std_ms": stds}
        print(f"  metrics/{key}: done")
    return {"meta": {"sizes": SIZES, "N": N, "warmup": WARMUP}, "results": out}
