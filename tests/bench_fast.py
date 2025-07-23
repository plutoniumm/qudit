import sys

sys.path.append("..")

from time import perf_counter as bench
from qudit.circuit import Circuit
import numpy as np

# from qiskit.quantum_info import Operator
# from qiskit import QuantumCircuit
# import cirq as CQ

ms = 1e3


def benchmark_custom(n):
    C = Circuit(n, dim=2)

    C.gate("H", dits=[0])
    for i in range(n - 1):
        C.gate("CX", dits=[i, i + 1])
    start = bench()
    _ = C.run()
    return bench() - start


# def benchmark_qiskit(n):
#     qc = QuantumCircuit(n)
#     qc.h(0)
#     for i in range(n - 1):
#         qc.cx(i, i + 1)
#     start = bench()
#     _ = Operator.from_circuit(qc)
#     return bench() - start


# def benchmark_cirq(n):
#     q = CQ.LineQubit.range(n)
#     ops = [CQ.H(q[0])] + [CQ.CX(q[i], q[i + 1]) for i in range(n - 1)]
#     circuit = CQ.Circuit(ops)
#     sim = CQ.Simulator()
#     start = bench()
#     _ = sim.simulate(circuit)
#     return bench() - start

n_range = range(11, 12)
for n in n_range:
    print(n)
    # t_qiskit = ms * benchmark_qiskit(n)
    # print(f"\tQiskit: {t_qiskit:.3f} ms")
    # t_cirq = ms * benchmark_cirq(n)
    # print(f"\tCirq: {t_cirq:.3f} ms")
    t_custom = ms * benchmark_custom(n)
    print(f"\tQudit: {t_custom:.3f} ms")
