import sys

sys.path.append("..")
from neal import SimulatedAnnealingSampler
from qudit.algo import QAOA
import torch as pt


def evaluate(Q: dict, solution: list) -> float:
    energy = 0.0
    sol_dict = {i: bit for i, bit in enumerate(solution)}
    for (i, j), val in Q.items():
        energy += val * sol_dict.get(i, 0) * sol_dict.get(j, 0)

    return energy


def vsDwave():
    problem = {
        (0, 0): -2,
        (1, 1): -2,
        (2, 2): -2,
        (3, 3): -2,
        (0, 1): 2,
        (1, 2): 2,
        (2, 3): 2,
        (0, 3): 2,
    }

    sampleset = SimulatedAnnealingSampler().sample_qubo(problem)
    best_solution = sampleset.first
    print(f"   Lowest energy: {best_solution.energy}")
    print(f"   Configuration: {best_solution.sample}\n")

    qaoa = QAOA(
        d=2,
        wires=4,
        qubo=problem,
        layers=5,
        device="cpu",
    )
    qaoa_result = qaoa.solve(evaluate, steps=50, lr=0.05)


def clock():
    neighbors = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
    J, g = 1.0, 0.5  # coupling strength, transverse field

    ham = []

    for i, j in neighbors:
        ham.append((-J, "ZZ", [i, j]))
    for i in range(6):
        ham.append((-g, "Z", [i]))
        ham.append((-g, "Z", [i]))

    qaoa = QAOA(
        d=3,
        wires=6,
        hamiltonian=ham,
        offset=0.0,
        layers=3,
        device="cpu",
    )

    def eval_clock(qubo, solution: list) -> float:
        E = 0.0
        for i, j in neighbors:
            D = solution[i] - solution[j] % 3
            E += -J * pt.cos(2 * pt.pi * D / 3)
        for i, s in enumerate(solution):
            E += -g * 2 * pt.cos(2 * pt.pi * s / 3)

        return float(E)

    def hook(loss, step):
        print(f"Step {step}: Loss = {loss:.4f}")

    res = qaoa.solve(eval_clock, steps=10, lr=0.1, hook=hook)
    print(res)


if __name__ == "__main__":
    vsDwave()
    clock()
