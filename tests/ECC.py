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

    return np.array([
      leung_0 / np.linalg.norm(leung_0),
      leung_1 / np.linalg.norm(leung_1)
    ])

code = _leung()

Ak = Process.GAD_full(4, Y=Y, p=0.01)
# Ek = Process.GAD(4, 1, Y=Y, p=0.01)
Ek = [e for e in Ak if e.correctable]

Rks = Recovery.petz(Ak, code)

fid = Fidelity.entanglement(Rks, Ak, code)
print("Fidelity:", fid)
