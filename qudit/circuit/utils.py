import numpy as np
import torch


def kron(m1, m2):
    D1, D2 = m1.shape[0], m2.shape[0]
    dev = m1.device
    dim = D1 * D2

    sm1 = m1.coalesce()
    sm2 = m2.coalesce()

    vals1, idx1 = sm1.values(), sm1.indices().unsqueeze(2)
    vals2, idx2 = sm2.values(), sm2.indices().unsqueeze(2)

    pos = (idx1 * D2 + idx2.permute(0, 2, 1)).view(2, -1)
    val = (vals1.unsqueeze(1) * vals2.unsqueeze(0)).view(-1)

    return torch.sparse_coo_tensor(pos, val, size=(dim, dim)).to(dev)


def dec2den(j, N, d):
    den = [0 for _ in range(N)]
    jv = j
    if isinstance(d, int):
        for k in range(N):
            base = d ** (N - 1 - k)
            if jv >= base:
                den[k] = jv // base
                jv = jv - den[k] * base
    else:
        for k in range(N):
            prod = 1
            for dim in d[k + 1 :]:
                prod *= dim
            if prod == 0:
                prod = 1
            den[k] = jv // prod
            jv = jv % prod
    return den


def den2dec(local, d):
    N = len(local)
    j = 0
    if isinstance(d, int):
        for k in range(N):
            j += local[k] * (d ** (N - 1 - k))
    else:
        for k in range(N):
            prod = 1
            for dim in d[k + 1 :]:
                prod *= dim
            j += local[k] * prod
    return j


def cx_qudits_Position(c, t, n, d, device="cpu"):
    if isinstance(d, int):
        dims = [d] * n
    else:
        dims = d
        if len(dims) != n:
            raise ValueError(
                "Length of dimension list must equal the number of qudits (n)."
            )
    grid = [torch.arange(dim, dtype=torch.float, device=device) for dim in dims]
    meshes = torch.meshgrid(*grid, indexing="ij")
    L = torch.stack(meshes, dim=-1).reshape(-1, n)
    L[:, t] = (L[:, t] + L[:, c]) % dims[t]
    place = []
    prod = 1
    for dim in dims[::-1]:
        place.insert(0, prod)
        prod *= dim
    tt = torch.tensor(place, dtype=torch.float, device=device).reshape(n, 1)
    lin = torch.matmul(L, tt)
    D = int(prod)
    col = torch.arange(D, dtype=torch.float, device=device).reshape(D, 1)
    return torch.cat((lin, col), dim=1)


def CX_sparse(c, t, d, n, device="cpu"):
    if isinstance(d, int):
        dims = [d] * n
    else:
        dims = d
        if len(dims) != n:
            raise ValueError(
                "Length of dimension list must equal the number of qudits (n)."
            )
    D = int(np.prod(dims))
    indices = cx_qudits_Position(c, t, n, dims, device=device)
    values = torch.ones(D, device=device)
    eye_sparse = torch.sparse_coo_tensor(
        indices.t(), values, (D, D), dtype=torch.complex64, device=device
    )
    return eye_sparse


def sparse_index_put(M, indices, values, device):
    assert M.is_sparse, "Input matrix must be a sparse tensor"
    M = M.coalesce()
    old_indices = M.indices()
    old_values = M.values()
    all_indices = torch.cat([old_indices, indices], dim=1)
    all_values = torch.cat([old_values, values])
    flat_indices = all_indices[0] * M.shape[1] + all_indices[1]
    unique, inverse = torch.unique(flat_indices, return_inverse=True, sorted=False)
    last_occurrences = torch.zeros_like(unique, dtype=torch.long)
    last_occurrences[inverse] = torch.arange(len(flat_indices), device=device)
    new_indices = all_indices[:, last_occurrences]
    new_values = all_values[last_occurrences]
    return torch.sparse_coo_tensor(new_indices, new_values, M.shape, device=device)
