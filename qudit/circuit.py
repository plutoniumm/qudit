from sympy import SparseMatrix as Matrix
from scipy.sparse import csr_matrix
from .index import Gate, VarGate
from typing import List, Union
from .utils import CTensor, ID
from .gates import Gategen
import numpy as np

BARRIER = "─|─"


class Layer:
    vqc: bool = False
    data: np.ndarray
    counter: List[int]
    gates: List[Gate]
    gategen: Gategen
    span: int
    id: str
    d: int

    def __init__(self, size: int, dim: int, gategen: Gategen):
        assert size > 0, f"Size must be a >0, got: {size}"
        assert dim >= 0, f"Dimension must be int>=0, got: {dim}"
        assert isinstance(gategen, Gategen), f"Expected Gategen, got {type(gategen)}"

        self.id = ID()
        self.span = size
        self.counter = list(range(size))
        self.gates = []
        self.d = dim
        self.gategen = gategen

    def add(self, gate: Union[Gate, VarGate], dits: List[int]):
        gate.dits = dits
        self.vqc = self.vqc or gate.vqc

        for d in dits:
            if d in self.counter:
                self.counter.remove(d)

        self.gates.append(gate)
        return self

    def open(self, *args: List[int]) -> bool:
        avl = [d for d in range(self.span) if d in self.counter]
        return all(d in avl for d in args)

    def finalise(self):
        sublayer = self.getMat(self.gates)
        prod = sublayer[0]
        for sub in sublayer[1:]:
            prod = prod @ sub

        self.data = prod

    # return list of equal sized matrices
    def getMat(self, in_gates) -> List[np.ndarray]:
        sublayer = [[]]
        l_gates, s_gates = [], []

        for gate in in_gates:
            if gate.span == 2:
                l_gates.append(gate)
            elif gate.span == 1:
                s_gates.append(gate)
            else:
                raise ValueError(f"Expected span: 1/2, got {gate.span}")
        # endfor

        G = self.gategen

        sublayer[0] = [G.I] * self.span
        for gate in s_gates:
            idx = gate.dits[0]
            sublayer[0][idx] = gate
        # endfor
        sublayer[0] = CTensor(*sublayer[0])
        if len(l_gates) == 0:
            return sublayer

        for gate in l_gates:
            sublayer.append(G.swapper.widen(gate))

        return sublayer

    def __repr__(self):
        names = [gate.name for gate in self.gates]
        return f"Layer({', '.join(names)})"

    def __getitem__(self, index):
        return self.gates[index]

    def __iter__(self):
        return iter(self.gates)


class cfn:
    @staticmethod
    def balance(strings: List[str]) -> List[str]:
        lmax = max(len(s) for s in strings)
        return [s.ljust(lmax, "─") for s in strings]

    @staticmethod
    def cx(strings: List[str], dits: List[int], name: str = "U") -> List[str]:
        ctrl, targ = dits
        name = name[1:] if name.startswith("C") else name

        if ctrl > targ:
            strings[targ] += f"╭{name}─"
            strings[ctrl] += "╰●─"
            scan = range(targ + 1, ctrl)
        else:
            strings[ctrl] += "╭●─"
            strings[targ] += f"╰{name}─"
            scan = range(ctrl + 1, targ)

        for i in scan:
            strings[i] += "│─"

        return strings


class Circuit:
    layers: List[Layer]
    gates: Gategen
    vqc: bool = False
    span: int
    d: int

    def __init__(self, size: int, dim: int):
        assert size >= 0, "Size must be int>=0"
        assert dim >= 0, "Dimension must be int>=0"

        self.gates = Gategen(dim, width=size)
        self.layers = [Layer(size=size, dim=dim, gategen=self.gates)]
        self.d = dim
        self.span = size

    def gate(self, gate: Union[Gate, VarGate], dits: List[int]):
        layer = self.layers[-1]
        if not layer.open(*dits):
            layer.finalise()
            layer = Layer(size=self.span, dim=self.d, gategen=self.gates)
            self.layers.append(layer)

        layer.add(gate, dits)
        return self

    def solve(self) -> np.ndarray:
        self._refresh()
        for layer in self.layers:
            if not hasattr(layer, "data"):
                layer.finalise()

        for i in range(len(self.layers)):
            if self.vqc:
                self.layers[i].data = Matrix(self.layers[i].data)
            else:
                self.layers[i].data = csr_matrix(self.layers[i].data)

        prod = self.layers[0].data
        for m in self.layers[1:]:
            prod = m.data @ prod

        return prod.todense()

    def draw(self):
        qudits = self.layers[0].span
        strings = ["─"] * qudits
        for l, layer in enumerate(self.layers):
            qctr = 0
            if layer[0].name == BARRIER:
                strings = cfn.balance(strings)
                strings = [s + BARRIER for s in strings]
                continue

            for gate in layer:
                if gate.span == 2:
                    strings = cfn.balance(strings)
                    strings = cfn.cx(strings, gate.dits, gate.name)
                    qctr += 2
                else:
                    g = gate.dits[0]
                    if gate.name == "I" or gate.name == "_":
                        strings[g] += "──"
                    else:
                        strings[g] += f"{gate.name}─"
                    qctr += 1
            # endfor
            strings = cfn.balance(strings)
        # endfor

        return "\n".join(strings)

    def __repr__(self):
        return self.draw()

    def __getitem__(self, index):
        return self.layers[index]

    def __setitem__(self, index: int, value: Layer):
        if not isinstance(value, Layer):
            raise TypeError(f"Expected Layer, got {type(value)}")
        if index < 0 or index >= len(self.layers):
            raise IndexError(
                f"Expected index in [0, {len(self.layers) - 1}], got {index}"
            )
        self.layers[index] = value
        self._refresh()

    def __iter__(self):
        return iter(self.layers)

    def _refresh(self):
        self.vqc = any(layer.vqc for layer in self.layers)

    def barrier(self):
        if len(self.layers) < 1:
            raise ValueError("Add at least 1 layer for a barrier")
        assert self.span > 0, "Span Unknown, add a layer first"

        d = self.d
        layer = Layer(size=self.span, dim=self.d, gategen=self.gates).add(
            Gate(d, np.eye(d), BARRIER), dits=list(range(self.span))
        )
        self.layers.append(layer)

        return self
