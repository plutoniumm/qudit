import sys, json, os

sys.path.append("..")

import matplotlib.pyplot as plt


def _early_plot_and_exit():
    if os.path.exists("bench_sgd.json"):
        with open("bench_sgd.json") as f:
            data = json.load(f)

        n_range = range(3, 21)
        LOG_THRESH = 8

        for k, ts in data.items():
            if ts:
                plt.plot(list(n_range)[: len(ts)], ts, label=k, marker=".")
        plt.axhline(LOG_THRESH, color="red", linestyle="--", label="Cutoff Threshold")

        plt.xlabel("Num Qubits (n)")
        plt.ylabel("log (avg ms/run)")
        plt.xticks(range(3, 18))
        plt.yticks(range(0, LOG_THRESH))
        plt.ylim(0, LOG_THRESH + 1)
        plt.title("Qubit GD Runtime")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig("bench_sgd.png", dpi=300)
        sys.exit(0)


_early_plot_and_exit()

import pennylane as qml
from torch import nn
import torch
import numpy as np
import sympy
import cirq
from qiskit.circuit import QuantumCircuit as Qiskitc, ParameterVector
from qiskit.quantum_info import Statevector
from qudit.circuit import Circuit as Quditc
import qudit.circuit.gates as G
import matplotlib.pyplot as plt
from time import time

C64 = torch.complex64
lr = 0.05
steps = 20
eps = 1e-3
ms = 1e3
repeats = 5


class HybridQubit(nn.Module):
    def __init__(self, n, device):
        super().__init__()
        self.wires, self.dim = n, 2
        self.circuit = C = Quditc(n, dim=2, device=device)

        rx = torch.randn(n, requires_grad=True)
        ry = torch.randn(n, requires_grad=True)
        self.opt = torch.optim.Adam([rx, ry], lr=lr)

        for i in range(n):
            C.gate(G.RX, i, angle=rx[i])
            C.gate(G.RY, i, angle=ry[i])

        for i in range(n - 1):
            C.gate(G.CX, [i, i + 1])

    def forward(self, x):
        return self.circuit(x).reshape(-1, 1)


def b_qudit(n, r):
    dev = "cpu"
    M = HybridQubit(n, device=dev)
    d = G.tensorise([1] + [0] * (2**n - 1), device=dev)
    tgt = G.tensorise([1] + [0] * (2**n - 2) + [1], device=dev) / np.sqrt(2)

    t = 0
    for _ in range(r):
        t0 = time()
        for _ in range(steps):
            out = M(d).view(-1)
            l = -torch.abs(torch.dot(out, tgt))

            M.opt.zero_grad()
            l.backward(retain_graph=False)
            M.opt.step()
        t += time() - t0

    return t / r


def b_qiskit(n, r):
    rx = ParameterVector("rx", n)
    ry = ParameterVector("ry", n)

    qc = Qiskitc(n)
    for i in range(n):
        qc.rx(rx[i], i)
        qc.ry(ry[i], i)
    for i in range(n - 1):
        qc.cx(i, i + 1)

    tgt = np.zeros(2**n, dtype=np.complex64)
    tgt[0] = tgt[-1] = 1 / np.sqrt(2)

    def cost(p):
        b = {rx[i]: p[i] for i in range(n)}
        b.update({ry[i]: p[i + n] for i in range(n)})
        return 1 - np.abs(np.vdot(Statevector(qc.assign_parameters(b)).data, tgt)) ** 2

    t = 0
    for _ in range(r):
        p = np.random.randn(2 * n)
        t0 = time()

        for _ in range(steps):
            g = np.zeros(2 * n)
            for i in range(2 * n):
                d = np.zeros_like(p)
                d[i] = eps
                g[i] = (cost(p + d) - cost(p - d)) / (2 * eps)
            p -= lr * g
        t += time() - t0

    return t / r


def b_cirq(n, r):
    q = [cirq.LineQubit(i) for i in range(n)]
    rx = sympy.symbols(f"rx0:{n}")
    ry = sympy.symbols(f"ry0:{n}")
    C = cirq.Circuit()

    for i in range(n):
        C.append(cirq.rx(rx[i])(q[i]))
        C.append(cirq.ry(ry[i])(q[i]))
    for i in range(n - 1):
        C.append(cirq.CNOT(q[i], q[i + 1]))

    sim = cirq.Simulator()
    tgt = np.zeros(2**n, dtype=np.complex64)
    tgt[0] = tgt[-1] = 1 / np.sqrt(2)

    def cost(p):
        b = {rx[i]: p[i] for i in range(n)}
        b.update({ry[i]: p[i + n] for i in range(n)})
        psi = sim.simulate(C, param_resolver=b).final_state_vector
        return 1 - np.abs(np.vdot(psi, tgt)) ** 2

    t = 0
    for _ in range(r):
        p = np.random.randn(2 * n)
        t0 = time()

        for _ in range(steps):
            g = np.zeros_like(p)
            for i in range(2 * n):
                d = np.zeros_like(p)
                d[i] = eps
                g[i] = (cost(p + d) - cost(p - d)) / (2 * eps)
            p -= lr * g
        t += time() - t0
    return t / r


def b_pennylane(n, r):
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, interface="torch")
    def circuit(rx, ry):
        for i in range(n):
            qml.RX(rx[i], wires=i)
            qml.RY(ry[i], wires=i)
        for i in range(n - 1):
            qml.CNOT(wires=[i, i + 1])
        return qml.state()

    tgt = torch.zeros(2**n, dtype=C64)
    tgt[0] = tgt[-1] = 1 / np.sqrt(2)

    t = 0
    for _ in range(r):
        rx = torch.randn(n, requires_grad=True)
        ry = torch.randn(n, requires_grad=True)
        opt = torch.optim.Adam([rx, ry], lr=lr)

        t0 = time()
        for _ in range(steps):
            opt.zero_grad()
            out = circuit(rx, ry).type(C64)
            l = 1 - torch.abs(torch.vdot(out, tgt)) ** 2
            l.backward()
            opt.step()
        t += time() - t0

    return t / r


backends = {
    "Qiskit": b_qiskit,
    "Cirq": b_cirq,
    "PennyLane": b_pennylane,
    "Qudit": b_qudit,
}

n_range = range(3, 21)
results = {k: [] for k in backends}
log_thresh = 8

for n in n_range:
    print(f"{n}/{max(n_range)}")
    for k in list(backends.keys()):
        fn = backends[k]
        if fn is None:
            continue
        t = fn(n, repeats) * ms
        log_t = np.log(t)
        print(f"\t{k}: {t:.2f} ms")
        if log_t > log_thresh:
            backends[k] = None
        else:
            results[k].append(log_t)

backends = {k: v for k, v in backends.items() if v is not None}

for k, ts in results.items():
    if ts:
        plt.plot(n_range[: len(ts)], ts, label=k, marker=".")

with open("bench_sgd.json", "w") as f:
    json.dump(results, f, indent=2)

plt.axhline(log_thresh, color="red", linestyle="--", label="Cutoff Threshold")
plt.xlabel("n Qubits")
plt.ylabel("Avg log(ms)")
plt.xticks(range(3, 18))
plt.yticks(range(0, log_thresh))
plt.ylim(0, log_thresh + 1)
plt.title("Qubit GD Runtime")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("bench_sgd.png", dpi=300)
plt.show()
