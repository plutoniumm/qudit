import sys

sys.path.append("..")
from unittest import TestCase, main
from qudit.noise import Recovery, Channel, Process
from qudit.tools import Fidelity
from numpy import linalg as LA
import numpy as np

Y = 0.02


def _leung():
    leung_0 = np.zeros(16)
    leung_1 = np.zeros(16)

    leung_0[0] = 1
    leung_0[-1] = 1
    leung_1[3] = 1
    leung_1[12] = 1

    return np.array(
        [leung_0 / np.linalg.norm(leung_0), leung_1 / np.linalg.norm(leung_1)]
    )


code = _leung()

Ak = Process.GAD(2, 4, Y=Y, p=0.01)
Ek = Ak.correctable()

print(Ak)
print(Ek)
Rks = Recovery.petz(Ak, code)

fid = Fidelity.entanglement(Rks, Ak, code)
print("Fidelity:", fid)
