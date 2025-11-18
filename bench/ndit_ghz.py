import sys, json

sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
# Add early-exit support similar to the GPU benchmark
import os

def plot_bench(data, output_png="bench_nn.png"):
    n_range = range(2, 7)
    LOG_THRESHOLD = 3
    for name, times in data.items():
        if times:
            plt.plot(list(n_range)[: len(times)], times, label=name, marker=".")
    plt.xticks(list(n_range))
    plt.xlim(min(n_range), max(n_range))
    plt.xlabel("dim=$n^n$")
    plt.ylabel("log (avg ms/run)")
    plt.axhline(LOG_THRESHOLD, color="red", linestyle="--", label="Log Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)

if os.path.exists("bench_nn.json"):
    with open("bench_nn.json") as f:
        data = json.load(f)
    plot_bench(data)
    sys.exit(0)

from time import perf_counter as bench
import torch
import quforge.quforge as qf
from qudit.circuit import Circuit
from qudiet.core.quantum_circuit import QuantumCircuit as Qudietc
from qudiet.core.backend.NumpyBackend import NumpyBackend


def b_qudit(n, repeats):
    circuit = Circuit(n, dim=n, device="cpu")
    G = circuit.gates[n]

    circuit.gate(G.H, [0])
    for i in range(n - 1):
        circuit.gate(G.CX, [i, i + 1])

    state = torch.zeros(n**n, dtype=torch.complex64)
    state[0] = 1

    start = bench()
    for _ in range(repeats):
        _ = circuit(state)
    return (bench() - start) / repeats


def b_qudiet(n, repeats):
    qc = Qudietc(qregs=[n] * n, backend=NumpyBackend)
    qc.h(0)
    for i in range(n - 1):
        qc.cx([i, i + 1], n)
    qc.measure_all()

    start = bench()
    for _ in range(repeats):
        _ = qc.run()
    return (bench() - start) / repeats


def b_quforge(n, repeats):
    circ = qf.Circuit(dim=n, wires=n)
    state = torch.zeros(n**n, dtype=torch.complex64)
    state[0] = 1  # |0...0>

    circ.H(index=[0])
    for i in range(n - 1):
        circ.CNOT(index=[i, i + 1])

    start = bench()
    for _ in range(repeats):
        circ(state)
    return (bench() - start) / repeats

n_range = range(2, 7)
LOG_THRESHOLD = 3
repeats = 20
ms = 1e3

backends = {
    "Qudit": b_qudit,
    "Qudiet": b_qudiet,
    "QuForge": b_quforge,
}

results = {name: [] for name in backends}

for n in n_range:
    print(f"{n}/{max(n_range)}")
    for name in list(backends.keys()):
        bench_fn = backends[name]
        if bench_fn is None:
            continue

        t = bench_fn(n, repeats) * ms
        log_t = np.log(t)
        print(f"\t{name}: {t:.3f} ms")
        if log_t > LOG_THRESHOLD:
            backends[name] = None
            continue
        results[name].append(log_t)

for name in list(backends.keys()):
    if backends[name] is None:
        del backends[name]

for name, times in results.items():
    if times:
        plt.plot(n_range[: len(times)], times, label=name, marker=".")

plt.xticks(list(n_range))
plt.xlim(min(n_range), max(n_range))
data = {name: times for name, times in results.items() if times}
with open("bench_nn.json", "w") as f:
    json.dump(data, f, indent=4)

plt.xlabel("dim=$n^n$")
plt.ylabel("log (avg ms/run)")
plt.axhline(LOG_THRESHOLD, color="red", linestyle="--", label="Log Threshold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("bench_nn.png", dpi=300)
