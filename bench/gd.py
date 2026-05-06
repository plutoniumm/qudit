"""
Gradient descent / VQE benchmark.

All frameworks optimise the same problem:
    minimise <Z⊗I⊗...> over a layered RY-CX ansatz.

Gradient method per framework:
  qudit       torch autograd (AD)
  pennylane   torch autograd (AD)
  qiskit      parameter-shift rule via StatevectorEstimator batch API (PS, 2L evals/step)
  cirq        parameter-shift rule, manual (PS, 2L evals/step)
  braket      parameter-shift rule, manual (PS, 2L evals/step)
  qutip       parameter-shift rule, manual (PS, 2L evals/step)

AD and PS are both O(L) evals/step — autograd is typically 1-2x forward cost,
param-shift is exactly 2L forward costs. The comparison is framework overhead,
not algorithmic complexity, so both gradient methods are noted in the output.
"""

import sys
import time
import numpy as np

sys.path.insert(0, "..")

import torch
import torch.nn as nn
import cirq
import sympy
import pennylane as qml
import qutip as qt
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from braket.circuits import Circuit as BCircuit, observables as bobs
from braket.devices import LocalSimulator
from qutip_qip.circuit import QubitCircuit, CircuitSimulator

from qudit.circuit import Circuit

C64 = torch.complex64
WARMUP = 3
N = 20

CONFIGS = [
    {"label": "2q 1-layer  50 steps", "wires": 2, "layers": 1, "steps": 50},
    {"label": "4q 2-layer  50 steps", "wires": 4, "layers": 2, "steps": 50},
    {"label": "6q 3-layer 100 steps", "wires": 6, "layers": 3, "steps": 100},
    {"label": "8q 3-layer 100 steps", "wires": 8, "layers": 3, "steps": 100},
]


def _ket0(n):
    x = torch.zeros(2**n, dtype=C64)
    x[0] = 1.0
    return x


def _zi_obs(wires):
    Z = torch.tensor([[1.0, 0], [0, -1.0]], dtype=C64)
    I = torch.eye(2, dtype=C64)
    obs = Z
    for _ in range(wires - 1):
        obs = torch.kron(obs, I)
    return obs


def bench_qudit(wires, layers, steps):
    def run():
        c = Circuit(wires=wires, dim=2)
        G = c.gates[2]
        params = []
        for _ in range(layers):
            for w in range(wires):
                p = nn.Parameter(torch.rand(()))
                params.append(p)
                c.gate(G.RY, [w], angle=p)
            for w in range(wires - 1):
                c.gate(G.CX, [w, w + 1])
        obs = _zi_obs(wires)
        ket = _ket0(wires)
        opt = torch.optim.Adam(params, lr=0.1)
        for _ in range(steps):
            opt.zero_grad()
            c.expectation(obs, ket).backward()
            opt.step()

    for _ in range(WARMUP):
        run()
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        run()
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def bench_pennylane(wires, layers, steps):
    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev, interface="torch")
    def circuit(angles):
        idx = 0
        for _ in range(layers):
            for w in range(wires):
                qml.RY(angles[idx], wires=w)
                idx += 1
            for w in range(wires - 1):
                qml.CNOT(wires=[w, w + 1])
        return qml.expval(qml.PauliZ(0))

    def run():
        angles = torch.rand(wires * layers, requires_grad=True)
        opt = torch.optim.Adam([angles], lr=0.1)
        for _ in range(steps):
            opt.zero_grad()
            circuit(angles).backward()
            opt.step()

    for _ in range(WARMUP):
        run()
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        run()
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def bench_qiskit(wires, layers, steps):
    nparams = wires * layers
    theta = ParameterVector("t", nparams)
    qc = QuantumCircuit(wires)
    idx = 0
    for _ in range(layers):
        for w in range(wires):
            qc.ry(theta[idx], w)
            idx += 1
        for w in range(wires - 1):
            qc.cx(w, w + 1)

    obs = SparsePauliOp("I" * (wires - 1) + "Z")
    estimator = StatevectorEstimator()

    def expectation(params):
        pub = (qc, obs, [params])
        return float(estimator.run([pub]).result()[0].data.evs)

    def run():
        p = np.random.uniform(0, np.pi, nparams)
        lr = 0.1
        shifts = [np.zeros(nparams) for _ in range(nparams)]
        for i in range(nparams):
            shifts[i][i] = np.pi / 2
        for _ in range(steps):
            pubs = [(qc, obs, [p + s]) for s in shifts] + [
                (qc, obs, [p - s]) for s in shifts
            ]
            evs = [float(r.data.evs[0]) for r in estimator.run(pubs).result()]
            grad = np.array([(evs[i] - evs[nparams + i]) / 2 for i in range(nparams)])
            p -= lr * grad

    for _ in range(WARMUP):
        run()
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        run()
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def bench_cirq(wires, layers, steps):
    q = cirq.LineQubit.range(wires)
    nparams = wires * layers
    syms = [sympy.Symbol(f"t{i}") for i in range(nparams)]
    ops = []
    idx = 0
    for _ in range(layers):
        for w in range(wires):
            ops.append(cirq.ry(syms[idx])(q[w]))
            idx += 1
        for w in range(wires - 1):
            ops.append(cirq.CNOT(q[w], q[w + 1]))
    circuit = cirq.Circuit(ops)
    sim = cirq.Simulator()
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)
    I2 = np.eye(2, dtype=np.complex64)
    obs_mat = Z
    for _ in range(wires - 1):
        obs_mat = np.kron(obs_mat, I2)

    def expectation(params):
        resolver = cirq.ParamResolver({str(syms[i]): params[i] for i in range(nparams)})
        sv = np.array(sim.simulate(circuit, param_resolver=resolver).final_state_vector)
        return float((sv.conj() @ obs_mat @ sv).real)

    def run():
        p = np.random.uniform(0, np.pi, nparams)
        lr = 0.1
        for _ in range(steps):
            grad = np.zeros(nparams)
            for i in range(nparams):
                shift = np.zeros(nparams)
                shift[i] = np.pi / 2
                grad[i] = (expectation(p + shift) - expectation(p - shift)) / 2
            p -= lr * grad

    for _ in range(WARMUP):
        run()
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        run()
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def bench_braket(wires, layers, steps):
    dev = LocalSimulator()
    nparams = wires * layers

    def expectation(params):
        c = BCircuit()
        idx = 0
        for _ in range(layers):
            for w in range(wires):
                c.ry(w, float(params[idx]))
                idx += 1
            for w in range(wires - 1):
                c.cnot(w, w + 1)
        c.expectation(bobs.Z(), [0])
        return float(dev.run(c, shots=0).result().values[0])

    def run():
        p = np.random.uniform(0, np.pi, nparams)
        lr = 0.1
        for _ in range(steps):
            grad = np.zeros(nparams)
            for i in range(nparams):
                shift = np.zeros(nparams)
                shift[i] = np.pi / 2
                grad[i] = (expectation(p + shift) - expectation(p - shift)) / 2
            p -= lr * grad

    for _ in range(WARMUP):
        run()
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        run()
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def bench_qutip(wires, layers, steps):
    nparams = wires * layers
    Z_op = qt.sigmaz()
    I_op = qt.qeye(2)
    obs = qt.tensor([Z_op] + [I_op] * (wires - 1))
    psi0 = qt.tensor([qt.basis(2, 0)] * wires)

    def expectation(params):
        qc = QubitCircuit(wires)
        idx = 0
        for _ in range(layers):
            for w in range(wires):
                qc.add_gate("RY", targets=[w], arg_value=float(params[idx]))
                idx += 1
            for w in range(wires - 1):
                qc.add_gate("CNOT", controls=[w], targets=[w + 1])
        return float(qt.expect(obs, CircuitSimulator(qc).run(psi0).final_states[0]))

    def run():
        p = np.random.uniform(0, np.pi, nparams)
        lr = 0.1
        for _ in range(steps):
            grad = np.zeros(nparams)
            for i in range(nparams):
                shift = np.zeros(nparams)
                shift[i] = np.pi / 2
                grad[i] = (expectation(p + shift) - expectation(p - shift)) / 2
            p -= lr * grad

    for _ in range(WARMUP):
        run()
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        run()
        times.append((time.perf_counter() - t0) * 1e3)
    return times


FRAMEWORKS = {
    "qudit (AD)": bench_qudit,
    "pennylane (AD)": bench_pennylane,
    "qiskit (PS)": bench_qiskit,
    "cirq (PS)": bench_cirq,
    "braket (PS)": bench_braket,
    "qutip (PS)": bench_qutip,
}

if __name__ == "__main__":
    print(__doc__.strip())
    print(f"\nN={N} runs, warmup={WARMUP}\n")

    for cfg in CONFIGS:
        w, l, s = cfg["wires"], cfg["layers"], cfg["steps"]
        print(f"  {cfg['label']}")
        print(f"  {'framework':<20} {'mean_ms':>10} {'std_ms':>8}")
        print("  " + "-" * 40)
        for name, fn in FRAMEWORKS.items():
            try:
                times = fn(w, l, s)
                print(f"  {name:<20} {np.mean(times):>10.2f} {np.std(times):>8.2f}")
            except Exception as e:
                print(f"  {name:<20} {'ERR':>10}  {e}")
        print()
