"""
Stage 2a: the math a transformer-shaped bigram needs, before any context mixing.

Pure Python. Lists and math.exp only. No numpy, no torch.
Each function is the source of truth that the FastAPI visualizer calls.
"""


def token_embed(E: list[list[float]], idx: int) -> list[float]:
    """Look up row `idx` in embedding table E. Returns a copy."""
    return list(E[idx])


def position_embed(P: list[list[float]], t: int) -> list[float]:
    """Look up row `t` in position table P. Returns a copy."""
    return list(P[t])


def combined_input(
    E: list[list[float]],
    P: list[list[float]],
    idx: int,
    t: int,
) -> list[float]:
    """h = E[idx] + P[t], element-wise. Raises ValueError on dim mismatch."""
    e = E[idx]
    p = P[t]
    if len(e) != len(p):
        raise ValueError(f"E row width ({len(e)}) != P row width ({len(p)})")
    return [a + b for a, b in zip(e, p)]


def lower_tri_mask(T: int) -> list[list[float]]:
    """T×T lower-triangular mask. M[i][j] = 1.0 if j <= i else 0.0."""
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    return [[1.0 if j <= i else 0.0 for j in range(T)] for i in range(T)]


def normalize_rows(M: list[list[float]]) -> list[list[float]]:
    """Row-stochastic version of M: each row divided by its sum.

    Raises ValueError if any row sums to zero (can't normalize).
    """
    out = []
    for i, row in enumerate(M):
        s = sum(row)
        if s == 0:
            raise ValueError(f"row {i} sums to zero, cannot normalize")
        out.append([x / s for x in row])
    return out


def unembed_head(
    h: list[float],
    W_head: list[list[float]],
    b: list[float],
) -> list[float]:
    """logits = h @ W_head + b. W_head shape: (n_embd, V). h shape: (n_embd,). b shape: (V,)."""
    if len(W_head) != len(h):
        raise ValueError(
            f"W_head row count ({len(W_head)}) must equal len(h) ({len(h)})"
        )
    V = len(W_head[0]) if W_head else 0
    if any(len(r) != V for r in W_head):
        raise ValueError("W_head must be rectangular")
    if len(b) != V:
        raise ValueError(f"b length ({len(b)}) must equal V ({V})")
    return [sum(h[k] * W_head[k][j] for k in range(len(h))) + b[j] for j in range(V)]
