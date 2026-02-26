<img src="https://raw.githubusercontent.com/plutoniumm/qudit/refs/heads/main/docs/public/icons/favi.svg" alt="icon" width="125" height="125" align="right" name="icon"/>

### `qudit`

High performance simulations for qudit systems. To make qudit machine learning, qudit error correction, and qudit circuit simulation easier. Qudit is made fully around `pytorch` to make it easy to mix and match tools without worrying about type errors.

[![PyPI version](https://badge.fury.io/py/qudit.svg)](https://pypi.org/project/qudit/)

```bash
pip install qudit
```

> WARNING: Error correction is currently broken and is being worked on. Currently anything inside `qudit.noise` or `qudit.qec` should be considered volatile.

## Quickstart

In most cases it should not matter if you mix and match `pytorch` with `qudit` since most abstractions are built on top of `pytorch` tensors. The following is two examples to do the same thing, one using the `Circuit` class and the other manually using the matrices.

**Using the `Circuit` class:**

```python
from qudit import Circuit
import torch as pt

C = Circuit(2, dim=2)  # 2 qBits with d=2
G = C.gates[2]

C.gate(G.H, dits=[0])
C.gate(G.CX, dits=[0, 1])

ket0 = pt.zeros(2**2)
ket0[0] = 1.0  # |00>

print(C(ket0))  # [1. 0. 0. 1.]/rt2
```

## Contributing
Qudit has a small shell scripts used to generate documentation and run tests. If you want to contribute, please fork the repo and make a pull request with your changes. The running script is `do`.

### `./do` helper script

The repository includes a small `./do` helper for common workflows (build, tests, docs, etc.).

```bash
./do help
```

### Caveats
- `qudit` does not, and will not support openQASM in the near future since hardware is not a goal. There are plans for some amount of interoperability with other qudit simulators, however that will be a long term goal and not a near term one. We will first make qudit stable.
-

#### Commands

- `./do build`
  - Builds `sdist` + `wheel` and runs `twine check dist/*`.
- `./do deploy`
  - Uploads `dist/*` to PyPI via `twine upload` using an API token.
  - Token is read from `FILE` (defaults to `.vscode/token.env`).
- `./do test`
  - Runs the Python test scripts in `./tests`.
- `./do prof`
  - Profiles `./tests/bench_fast.py` with `cProfile` and opens `snakeviz`.
- `./do head`
  - Runs `python ./view/headers.py` (used by the docs pipeline).
- `./do docs dev`
  - Runs the docs dev server (`npm run dev`).
- `./do docs build`
  - Builds the docs (`npm run build`).

#### Common workflows

```bash
# run test scripts
./do test

# build packages locally
./do build

# build docs
./do docs build

# docs dev server
./do docs dev
```

#### PyPI token setup (for `deploy`)

Create a file containing only your PyPI API token (no `export`, no quotes):

- default location: `.vscode/token.env`
- or specify a different file via `FILE=path/to/token.env`

Example:

```bash
FILE=.vscode/token.env ./do deploy
```

## Acknowledgements
Many many thanks to [Sai Sakunthala](https://github.com/Sai-sakunthala) and [R-Phoenix](https://github.com/R-Phoenix) for testing and fixing bugs!

This library is built on top of pytorch, and takes ideas from Qiskit, Cirq, and QuDiet.