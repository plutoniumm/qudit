from numpy import linalg as LA
import numpy as np


def GramSchmidt(vectors):
    ortho = []
    for v in vectors:
        w = v - sum(np.dot(v, np.conj(u)) * u for u in ortho)
        if LA.norm(w) > 1e-8:
            ortho.append(w / LA.norm(w))

    return np.array(ortho)
