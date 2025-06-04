import numpy as np

# Peres-Horodecki criterion for separability of density matrices
def PPT(rho: np.ndarray, sub: int) -> bool:
  side = rho.shape[0]
  sub = 3
  if side % sub != 0:
    raise ValueError(f"Matrix side ({side}) not divisible by sub ({sub})")

  mat0 = rho.copy()
  for i in range(0, mat0.shape[0], sub):
    for j in range(0, mat0.shape[1], sub):
      mat0[i:i+sub, j:j+sub] = mat0[i:i+sub, j:j+sub].T

  return np.all(np.linalg.eigvals(mat0) >= 0)