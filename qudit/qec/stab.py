def Stabilisers(
    M: int, N: int, edge: str = "even", start: str = "X"
) -> list[list[str]]:
    edge = edge.lower()
    start = start.upper()

    if N == 1:
        return _linear(M, edge, start)

    return _grid(M, N, edge, start)


def _linear(N: int, edge: str, start: str) -> list[list[str]]:
    evens: list[list[str]] = []
    odds: list[list[str]] = []

    for i in range(N - 1):
        stab = ["I"] * N
        stab[i] = start
        stab[i + 1] = start

        if i % 2 == 0:
            evens.append(stab)
        else:
            odds.append(stab)

    if edge == "even":
        return evens + list(reversed(odds))

    return odds + list(reversed(evens))


def _grid(M: int, N: int, edge: str, start: str) -> list[list[str]]:
    mode = 0 if edge == "even" else 1
    total = M * N

    def c2i(x: int, y: int) -> int:
        return y * M + x

    stabs: list[list[str]] = []

    for i in range(M - 1):
        for j in range(N - 1):
            corners = {(i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1)}
            parity = (i + j) % 2
            sym = "A" if parity == 0 else "B"
            s = ["I"] * total
            for x, y in corners:
                s[c2i(x, y)] = sym
            stabs.append(s)

    border: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for j in range(N - 1):
        border.append(((0, j), (0, j + 1)))
    for i in range(M - 1):
        border.append(((i, N - 1), (i + 1, N - 1)))
    for j in range(N - 1, 0, -1):
        border.append(((M - 1, j), (M - 1, j - 1)))
    for i in range(M - 1, 0, -1):
        border.append(((i, 0), (i - 1, 0)))

    for idx, (p1, p2) in enumerate(border):
        if idx % 2 == mode:
            continue

        box_parity = _find_box(M, N, p1, p2)
        sym = "B" if box_parity == 0 else "A"
        s = ["I"] * total
        s[c2i(*p1)] = sym
        s[c2i(*p2)] = sym
        stabs.append(s)

    if start == "X":
        sub = {
            "A": "X",
            "B": "Z",
            "I": "I",
        }
    else:
        sub = {
            "A": "Z",
            "B": "X",
            "I": "I",
        }

    return [[sub[c] for c in row] for row in stabs]


def _find_box(M: int, N: int, p1: tuple[int, int], p2: tuple[int, int]) -> int:
    for i in range(M - 1):
        for j in range(N - 1):
            corners = {(i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1)}
            if p1 in corners and p2 in corners:
                return (i + j) % 2

    raise ValueError(f"No box found for edge {p1}-{p2}")
