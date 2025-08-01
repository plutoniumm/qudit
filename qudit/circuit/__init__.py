# from .index import *
# from .gates import *

from .index import *
from .gates import *

if __name__ == "__main__":
    circuit = Circuit(2, dim=2)
    G = circuit.gates

    my_cnot = G.make(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], name="MyCNOT"
    )

    circuit.gate(G.X, [0])
    circuit.gate("H", [0])
    circuit.gate(my_cnot, [0, 1])

    state = np.array([1, 0, 0, 0], dtype=np.complex64)

    print(circuit(state))
