from typing import List
from enum import Enum

class Marker(Enum):
    CONTROL = "●"
    L = "╰"
    L_ = "╭"
    BAR = "│"
    PIPE = "─"
    START = "┤"

class Table:
    def __init__(self, circ):
        drawing: List[str] = [
            f"|0> [{d}] {Marker.START.value}"
                for d in circ.dims_
        ]
        self.fig = drawing
        self.wires = len(drawing)

    def balance(self, start: int, end: int) -> None:
        strings = [self.fig[i] for i in range(start, end + 1)]

        lmax = max(len(s) for s in strings)
        for i in range(start, end + 1):
            self.fig[i] = self.fig[i].ljust(lmax, "─")

    def draw(self, circuit) -> str:
        for op in circuit.operations:
            size = len(op.index)
            if size == 0:
                continue
            if size == 1:
                idx = op.index[0]
                self.fig[idx] += f"─{str(op)}─"
                continue

            idxs = op.index
            a, b = min(idxs), max(idxs)
            self.balance(a, b)
            self.fig = Table.cont(self.fig, op.index, str(op))

        self.balance(0, self.wires - 1)
        diagram = []
        for row in self.fig:
            diagram.append(row + "┤")

        return "\n".join(diagram)

    @staticmethod
    def cont(strings: List[str], dits: List[int], name: str = "U") -> List[str]:
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