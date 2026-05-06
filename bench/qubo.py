"""
QUBO / QAOA benchmark.

Sections:
  qubo_convert  — QUBO.toHamiltonian() scaling with problem size
  qaoa_forward  — QAOA forward pass (state construction) per layer/qubit config
  clock_solve   — ClockSolver full optimisation loop
"""
import sys
import time
import numpy as np

sys.path.insert(0, "..")

import torch

from qudit.algo.qaoa import QUBO, QAOA, ClockSolver

from timing import timed

WARMUP = 3
N = 10


def _random_qubo(n):
    Q = {}
    for i in range(n):
        Q[(i, i)] = float(np.random.uniform(-2, 2))
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() < 0.5:
                Q[(i, j)] = float(np.random.uniform(-1, 1))
    return Q


def _time(fn):
    return timed(fn, N, WARMUP)


# ── QUBO conversion ───────────────────────────────────────────────────────────

CONVERT_SIZES = [4, 8, 12, 16, 20, 24]

def bench_convert(n):
    Q = _random_qubo(n)
    return _time(lambda: QUBO.toHamiltonian(Q))


# ── QAOA forward pass ─────────────────────────────────────────────────────────

QAOA_CONFIGS = [
    {"label": "4q 1-layer",  "n": 4,  "layers": 1},
    {"label": "6q 1-layer",  "n": 6,  "layers": 1},
    {"label": "8q 1-layer",  "n": 8,  "layers": 1},
    {"label": "8q 3-layer",  "n": 8,  "layers": 3},
    {"label": "10q 2-layer", "n": 10, "layers": 2},
]

def bench_qaoa_forward(n, layers):
    Q = _random_qubo(n)
    model = QAOA(qubo=Q, d=2, wires=n, layers=layers)
    return _time(lambda: model())


# ── ClockSolver ───────────────────────────────────────────────────────────────

CLOCK_CONFIGS = [
    {"label": "4q 50 steps",  "n": 4,  "steps": 50},
    {"label": "6q 50 steps",  "n": 6,  "steps": 50},
    {"label": "8q 50 steps",  "n": 8,  "steps": 50},
    {"label": "10q 30 steps", "n": 10, "steps": 30},
]

def bench_clock(n, steps):
    Q = _random_qubo(n)
    def run():
        solver = ClockSolver(qubo=Q, d=2, wires=n, layers=2)
        solver.solve(steps=steps, lr=0.05)
    return _time(run)


# ── Registry ──────────────────────────────────────────────────────────────────

def run_all():
    import numpy as np_
    out = {}

    # QUBO conversion
    means, stds = [], []
    for n in CONVERT_SIZES:
        try:
            t = bench_convert(n)
            means.append(round(float(np_.mean(t)), 4))
            stds.append(round(float(np_.std(t)), 4))
        except Exception as e:
            means.append(None); stds.append(None)
            print(f"  [qubo/convert/n={n}] ERR: {e}")
    out["convert"] = {
        "meta": {"configs": [f"{n}var" for n in CONVERT_SIZES], "N": N, "warmup": WARMUP},
        "frameworks": {"qudit": {"mean_ms": means, "std_ms": stds}},
    }
    print("  qubo/convert: done")

    # QAOA forward
    labels = [c["label"] for c in QAOA_CONFIGS]

    means, stds = [], []
    for cfg in QAOA_CONFIGS:
        try:
            t = bench_qaoa_forward(cfg["n"], cfg["layers"])
            means.append(round(float(np_.mean(t)), 4))
            stds.append(round(float(np_.std(t)), 4))
        except Exception as e:
            means.append(None); stds.append(None)
            print(f"  [qubo/qaoa/{cfg['label']}] ERR: {e}")
    out["qaoa"] = {
        "meta": {"configs": labels, "N": N, "warmup": WARMUP},
        "frameworks": {"qudit": {"mean_ms": means, "std_ms": stds}},
    }
    print("  qubo/qaoa: done")

    # ClockSolver
    labels = [c["label"] for c in CLOCK_CONFIGS]
    means, stds = [], []
    for cfg in CLOCK_CONFIGS:
        try:
            t = bench_clock(cfg["n"], cfg["steps"])
            means.append(round(float(np_.mean(t)), 4))
            stds.append(round(float(np_.std(t)), 4))
        except Exception as e:
            means.append(None); stds.append(None)
            print(f"  [qubo/clock/{cfg['label']}] ERR: {e}")
    out["clock"] = {
        "meta": {"configs": labels, "N": N, "warmup": WARMUP},
        "frameworks": {"qudit": {"mean_ms": means, "std_ms": stds}},
    }
    print("  qubo/clock: done")

    return out
