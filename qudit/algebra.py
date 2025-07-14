import numpy as np


Unity = lambda d: np.exp(2j * np.pi / d)

"""
Ordering followed same as:
  - https://github.com/CQuIC/pysme
  - https://pypi.org/project/liepy/

In general:
Symm(j, k) = |j⟩⟨k| + |k⟩⟨j|
Anti(j, j) = -i|j⟩⟨k| + i|k⟩⟨j|

A = sqrt(2/(j*(j+1)))
Diag(l) = A (sum_{j=0}^l |j⟩⟨j| - l|l+1⟩⟨l+1|)
- https://arxiv.org/pdf/0806.1174
"""

f = 2


def gellmann(j, k, d):
    mat = np.zeros((d, d), dtype=np.complex128)

    if j > k:
        mat[j - 1, k - 1] = 1
        mat[k - 1, j - 1] = 1
    elif k > j:
        mat[j - 1, k - 1] = -1j
        mat[k - 1, j - 1] = 1j
    elif j == k and j < d:
        norm = np.sqrt(2 / (j * (j + 1)))
        for m in range(j):
            mat[m, m] = norm
        mat[j, j] = -j * norm
    else:
        np.fill_diagonal(mat, 1)

    return mat


def dGellMann(d):
    arr = [gellmann(j, k, d) for j in range(1, d + 1) for k in range(1, d + 1)]
    arr.reverse()

    return arr
