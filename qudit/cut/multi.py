from .utils import matrix_of, _run_vector, stitch
import numpy as np


def run_multi(cuts, wires, dims_, circuit, device):
    """
    Multi-cut amplitude reconstruction via tensor contraction.

    For N cuts, produces N+1 partitions. Each partition's amplitude tensor
    is computed by running sub-circuits for all combinations of adjacent
    decomposition terms, then contracting via einsum.

    cuts: list of (coupler, [cA, cB], split, cut_pos) in left-to-right wire order.
    """
    from ..circuit.index import Circuit

    n = len(cuts)

    splits = [s for _, _, s, _ in cuts]
    for i in range(1, n):
        if splits[i] <= splits[i - 1]:
            raise ValueError(
                f"Cut {i} split={splits[i]} <= cut {i-1} split={splits[i-1]}. "
                "Cuts must be added in left-to-right wire order."
            )

    cut_pos = [cp for _, _, _, cp in cuts]
    if cut_pos != sorted(cut_pos):
        raise ValueError(
            "Cuts must be added in left-to-right wire order "
            "(call cut() for leftmost partition boundary first)."
        )

    boundaries = [0] + splits + [wires]
    partitions = [set(range(boundaries[i], boundaries[i + 1])) for i in range(n + 1)]
    offsets = boundaries[:-1]
    dim_lists = [
        [dims_[w] for w in range(boundaries[i], boundaries[i + 1])]
        for i in range(n + 1)
    ]
    widths = [int(np.prod(d)) if d else 1 for d in dim_lists]

    couplers = [c for c, _, _, _ in cuts]
    c_As = [cuts[i][1][0] for i in range(n)]
    c_Bs = [cuts[i][1][1] for i in range(n)]

    for i, (coupler, [cA, cB], _, _) in enumerate(cuts):
        if cA not in partitions[i]:
            raise ValueError(
                f"Cut {i}: c_A={cA} not in partition {i} (wires {sorted(partitions[i])})."
            )
        if cB not in partitions[i + 1]:
            raise ValueError(
                f"Cut {i}: c_B={cB} not in partition {i+1} (wires {sorted(partitions[i+1])})."
            )

    # build gate segments per partition
    # end partitions: [pre, post]
    # middle partitions: [seg0, seg1, seg2]
    segs = []
    for p in range(n + 1):
        segs.append([[], [], []] if 0 < p < n else [[], []])

    for j, gate in enumerate(circuit.children()):
        gate_parts = {
            p for p, part in enumerate(partitions) if any(w in part for w in gate.index)
        }
        if len(gate_parts) > 1:
            raise ValueError(
                f"Gate on wires {gate.index} spans multiple partitions. "
                "Only cut gates may cross partition boundaries."
            )
        if not gate_parts:
            continue

        p = gate_parts.pop()
        entry = (matrix_of(gate), [w - offsets[p] for w in gate.index])

        if p == 0:
            segs[p][0 if j < cut_pos[0] else 1].append(entry)
        elif p == n:
            segs[p][0 if j < cut_pos[-1] else 1].append(entry)
        else:
            if j < cut_pos[p - 1]:
                segs[p][0].append(entry)
            elif j < cut_pos[p]:
                segs[p][1].append(entry)
            else:
                segs[p][2].append(entry)

    def make_sub(p, *ops):
        qc = Circuit(len(partitions[p]), dim=dim_lists[p], device=device)
        if p == 0:
            for U, idx in segs[p][0]:
                qc.gate(U, idx)
            qc.gate(ops[0], [c_As[0] - offsets[p]])
            for U, idx in segs[p][1]:
                qc.gate(U, idx)
        elif p == n:
            for U, idx in segs[p][0]:
                qc.gate(U, idx)
            qc.gate(ops[0], [c_Bs[-1] - offsets[p]])
            for U, idx in segs[p][1]:
                qc.gate(U, idx)
        else:
            op_left, op_right = ops
            for U, idx in segs[p][0]:
                qc.gate(U, idx)
            qc.gate(op_left, [c_Bs[p - 1] - offsets[p]])
            for U, idx in segs[p][1]:
                qc.gate(U, idx)
            qc.gate(op_right, [c_As[p] - offsets[p]])
            for U, idx in segs[p][2]:
                qc.gate(U, idx)

        return qc

    ns = [len(c.terms) for c in couplers]
    c_vecs = [
        np.array([coeff for _, _, coeff in c.terms], dtype=np.complex64)
        for c in couplers
    ]

    # leftmost partition: φ0[iA, k]
    phi0 = np.zeros((widths[0], ns[0]), dtype=np.complex64)
    for k, (opA, _, _) in enumerate(couplers[0].terms):
        phi0[:, k] = _run_vector(make_sub(0, opA))

    # rightmost partition: φN[iN, k_{N-1}]
    phiN = np.zeros((widths[n], ns[-1]), dtype=np.complex64)
    for k, (_, opB, _) in enumerate(couplers[-1].terms):
        phiN[:, k] = _run_vector(make_sub(n, opB))

    # middle partitions: φp[ip, k_{p-1}, k_p]
    phi_mid = []
    for p in range(1, n):
        phi_p = np.zeros((widths[p], ns[p - 1], ns[p]), dtype=np.complex64)
        for k, (_, opL, _) in enumerate(couplers[p - 1].terms):
            for l, (opR, _, _) in enumerate(couplers[p].terms):
                phi_p[:, k, l] = _run_vector(make_sub(p, opL, opR))
        phi_mid.append(phi_p)

    amp = _contract(c_vecs, phi0, phi_mid, phiN, n)

    all_dims = []
    for dl in dim_lists:
        all_dims.extend(dl)

    return stitch(amp.ravel(), all_dims)


def _contract(c_vecs, phi0, phi_mid, phiN, n):
    """
    Compute amp[i0,...,iN] = Σ_{k0,...,k(N-1)} Πj c_j[kj] · Πp φp[ip,...]
    """
    # letters for state indices: a, b, c, ...
    # letters for term indices:  k, l, m, ...
    sl = [chr(ord("a") + i) for i in range(n + 1)]
    tl = [chr(ord("k") + i) for i in range(n)]

    operands = list(c_vecs)
    subs = list(tl)

    subs.append(sl[0] + tl[0])
    operands.append(phi0)

    for i, phi_p in enumerate(phi_mid):
        p = i + 1
        subs.append(sl[p] + tl[p - 1] + tl[p])
        operands.append(phi_p)

    subs.append(sl[n] + tl[n - 1])
    operands.append(phiN)

    expr = ",".join(subs) + "->" + "".join(sl)

    return np.einsum(expr, *operands)
