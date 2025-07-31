import torch.nn as nn
import torch as pt

from ..circuit import gates as GG
from typing import Callable

devnull = lambda *args, **kwargs: None

C64 = pt.complex64


class QUBO:
    @staticmethod
    def toHamiltonian(Q: dict):
        ising = {}
        offset = 0.0

        for (i, j), val in Q.items():
            if i == j:
                ising[i] = ising.get(i, 0) - val / 2
                offset += val / 2
            else:
                key = tuple(sorted((i, j)))
                ising[key] = ising.get(key, 0) + val / 4
                ising[i] = ising.get(i, 0) - val / 4
                ising[j] = ising.get(j, 0) - val / 4
                offset += val / 4

        ham = []
        for keys, coeff in ising.items():
            if coeff == 0:
                continue
            if isinstance(keys, int):
                ham.append((coeff, "Z", [keys]))
            else:
                ham.append((coeff, "ZZ", list(keys)))

        return ham, offset


Energy = Callable[[dict, list], float]


class QAOA(nn.Module):
    def __init__(
        self,
        d: int,
        wires: int,
        qubo: dict = None,
        hamiltonian: list = None,
        offset: float = 0.0,
        layers=1,
        device="cpu",
    ):
        super().__init__()
        self.width = d**wires
        self.wires = wires
        self.d = d
        self.device = device
        self.layers = layers
        self.qubo = qubo

        if hamiltonian is not None:
            self.hamiltonian = hamiltonian
            self.offset = offset
        elif qubo is not None:
            self.hamiltonian, self.offset = QUBO.toHamiltonian(qubo)
        else:
            raise ValueError("Either 'hamiltonian' or 'qubo' must be provided.")

        self._H_P_matrix = self.getMat()
        self.gammas = nn.Parameter(pt.rand(layers, device=device) * (2 * pt.pi))
        self.betas = nn.Parameter(pt.rand(layers, device=device) * pt.pi)
        self.OpMap = {
            "Z": self._RZ,
            "ZZ": self._RZZ,
        }

    def gate(self, x, gate_class, index, **kwargs):
        gate = gate_class(
            dim=self.d, wires=self.wires, index=index, device=self.device, **kwargs
        )
        return gate.forward(x)

    def _RZ(self, x, angle, indices):
        return self.gate(x, GG.RZ, index=indices, angle=angle)

    def _RZZ(self, x, angle, indices):
        x = self.gate(x, GG.CX, index=indices)
        x = self.gate(x, GG.RZ, index=[indices[1]], angle=angle)
        x = self.gate(x, GG.CX, index=indices)

        return x

    def forward(self):
        state = pt.zeros((self.width, 1), dtype=C64, device=self.device)
        state[0, 0] = 1.0
        h_all = GG.H(
            dim=self.d,
            wires=self.wires,
            index=list(range(self.wires)),
            device=self.device,
        )

        state = h_all.forward(state)
        for i in range(self.layers):
            for coeff, gtype, indices in self.hamiltonian:
                angle = 2 * self.gammas[i] * coeff
                if gtype in self.OpMap:
                    state = self.OpMap[gtype](state, angle, indices)
                else:
                    raise NotImplementedError(f"Gate type '{gtype}' not supported.")

            for j in range(self.wires):
                state = self.gate(state, GG.RX, index=[j], angle=2 * self.betas[i])

        return state

    def getMat(self):
        H_P = pt.zeros((self.width, self.width), dtype=C64, device=self.device)
        gg = GG.Gategen(dim=self.d, device=self.device)
        opmat = {
            "I": gg.I,
            "Z": gg.Z,
            "X": gg.X,
        }
        for coeff, gtype, indices in self.hamiltonian:
            oplist = []
            if gtype == "Z":
                oplist = [
                    opmat["Z"] if i == indices[0] else opmat["I"]
                    for i in range(self.wires)
                ]
            elif gtype == "ZZ":
                oplist = [
                    opmat["Z"] if i in indices else opmat["I"]
                    for i in range(self.wires)
                ]
            else:
                raise NotImplementedError(f"Matrix for '{gtype}' not defined.")
            term = oplist[0]
            for k in range(1, len(oplist)):
                term = pt.kron(term, oplist[k])
            H_P += coeff * term

        return H_P

    def expectation(self):
        final_state = self.forward()
        exp_val = pt.vdot(
            final_state.squeeze(), (self._H_P_matrix @ final_state).squeeze()
        ).real

        return exp_val + self.offset

    def solve(self, func: Energy, optimizer=None, steps=100, lr=0.1, hook=devnull):
        if optimizer is None:
            optimizer = pt.optim.Adam(self.parameters(), lr=lr)

        for step in range(steps):
            optimizer.zero_grad()
            loss = self.expectation()
            loss.backward()
            optimizer.step()
            hook(loss, step)

        with pt.no_grad():
            final_state = self.forward()
            Pi = (pt.abs(final_state) ** 2).squeeze()
            maxP = pt.argmax(Pi).item()
            solution = format(maxP, f"0{self.wires}b")

            solution = [int(bit) for bit in solution]
            soltensr = [pt.tensor(bit) for bit in solution]

            min_energy = func(self.qubo, soltensr)

        return {
            "solution": solution,
            "value": min_energy,
            "probabilities": Pi.cpu().flatten(),
        }
