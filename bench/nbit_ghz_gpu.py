import matplotlib.pyplot as plt
import os, sys, json


def plot_bench(data, output_png="bench_n2_gpu.png"):
    n_range = range(3, 25)
    LOG_THRESHOLD = 6
    for name, times in data.items():
        if times:
            plt.plot(list(n_range)[: len(times)], times, label=name, marker=".")
    plt.xlabel("Num Qubits (n)")
    plt.ylabel("log (avg ms/run)")
    plt.title("GHZ Circuit Benchmark")
    plt.xticks(list(n_range))
    plt.legend()
    plt.axhline(LOG_THRESHOLD, color="red", linestyle="--", label="Log Threshold")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)


if os.path.exists("bench_n2_gpu.json"):
    with open("bench_n2_gpu.json") as f:
        data = json.load(f)
    plot_bench(data)
    sys.exit(0)

sys.path.append("..")
import numpy as np
from time import perf_counter as bench
import torch
import cudaq
import pennylane as qml
import quforge.quforge as qf
from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit as Qiskitc
from qudit.circuit import Circuit as QuditCircuit
from qudiet.core.quantum_circuit import QuantumCircuit as Qudietc
from qudiet.core.backend import CudaBackend


def b_qiskit(n, repeats):
    backend = AerSimulator(method="statevector", device="GPU")
    qc = Qiskitc(n)

    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)

    backend.run(qc).result()
    start = bench()
    for _ in range(repeats):
        _ = backend.run(qc).result().get_statevector()
    return (bench() - start) / repeats


def b_pennylane(n, repeats):
    dev = qml.device("lightning.gpu", wires=n)

    @qml.qnode(dev, diff_method=None)
    def ghz_circuit():
        qml.Hadamard(wires=0)
        for i in range(n - 1):
            qml.CNOT(wires=[i, i + 1])
        return qml.state()

    ghz_circuit()
    start = bench()
    for _ in range(repeats):
        _ = ghz_circuit()
    return (bench() - start) / repeats


def b_qudit(n, repeats):
    device = "cuda"

    circuit = QuditCircuit(n, dim=2, device=device)
    CX = circuit.make([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dim=2)
    H = circuit.make([[1, 1], [1, -1]], dim=2) / np.sqrt(2)

    circuit.gate(H, [0])
    for i in range(n - 1):
        circuit.gate(CX, [i, i + 1])

    x0 = torch.zeros(2**n, dtype=torch.complex64, device=device)
    x0[0] = 1.0

    torch.cuda.synchronize()
    start = bench()
    for _ in range(repeats):
        _ = circuit(x0)
    torch.cuda.synchronize()

    return (bench() - start) / repeats


def b_quforge(n, repeats):
    device = "cuda"

    circ = qf.Circuit(dim=2, wires=n)
    circ.H(index=[0])

    for i in range(n - 1):
        circ.CNOT(index=[i, i + 1])

    x0 = torch.zeros(2**n, dtype=torch.complex64, device=device)
    x0[0] = 1.0

    torch.cuda.synchronize()
    start = bench()
    for _ in range(repeats):
        _ = circ(x0)
    torch.cuda.synchronize()

    return (bench() - start) / repeats


def b_qudiet(n, repeats):
    qc = Qudietc(qregs=[2] * n, backend=CudaBackend)

    qc.h(0)
    for i in range(n - 1):
        qc.cx([i, i + 1], 2)
    qc.measure_all()

    qc.run()
    start = bench()

    for _ in range(repeats):
        _ = qc.run()
    return (bench() - start) / repeats


def b_cudaq(n, repeats):
    cudaq.set_target("nvidia")
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(n)
    kernel.h(qubits[0])

    for i in range(n - 1):
        kernel.cx(qubits[i], qubits[i + 1])
    _ = cudaq.get_state(kernel)
    start = bench()

    for _ in range(repeats):
        _ = cudaq.get_state(kernel)

    return (bench() - start) / repeats


n_range = range(3, 25)
LOG_THRESHOLD = 6
repeats = 100
ms = 1e3
backends = {
    "CUDAQ": b_cudaq,
    "Qiskit-Aer": b_qiskit,
    "PennyLane": b_pennylane,
    "Qudiet": b_qudiet,
    "Qudit": b_qudit,
    "QuForge": b_quforge,
}

results = {name: [] for name in backends}

for n in n_range:
    print(f"{n}/{len(n_range) + 2}")
    for name in list(backends.keys()):
        bench_fn = backends[name]
        if bench_fn is None:
            continue

        t = bench_fn(n, repeats) * ms
        log_t = np.log(t)
        print(f"\t{name}: {t:.3f} ms")
        if log_t > LOG_THRESHOLD:
            backends[name] = None  # Mark as stopped
            continue
        results[name].append(log_t)

for name in list(backends.keys()):
    if backends[name] is None:
        del backends[name]

data = {name: times for name, times in results.items() if times}
with open("bench_n2_gpu.json", "w") as f:
    json.dump(data, f, indent=4)

plot_bench(data)
