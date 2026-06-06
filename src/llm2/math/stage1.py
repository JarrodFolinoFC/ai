"""
Stage 1: bigram-stage formulas, pure Python.

Source of truth for both the bigram trainer (llm2.gpt_pure) and the
visualizer. No PyTorch, no globals, no logging. Each function is total
and pure: same inputs → same outputs, no side effects.
"""

import math


def softmax(logits: list[float]) -> list[float]:
    """e^x / sum(e^x). Subtract max for numerical stability (e^big = inf)."""
    if not logits:
        raise ValueError("logits must be non-empty")
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def cross_entropy_loss(probs: list[float], target_id: int) -> float:
    """-log(probs[target_id] + 1e-12). The +1e-12 keeps log finite when prob → 0."""
    if not probs:
        raise ValueError("probs must be non-empty")
    if not (0 <= target_id < len(probs)):
        raise ValueError(
            f"target_id={target_id} out of range [0, {len(probs)})"
        )
    return -math.log(probs[target_id] + 1e-12)


def bigram_backward(probs: list[float], target_id: int) -> list[float]:
    """Gradient of cross-entropy w.r.t. logits for one position: probs - one_hot(target).

    Wrong-class slots get pushed DOWN by their predicted prob; the correct
    slot gets pushed UP by (1 - prob_correct). Bigger mistake → bigger push.
    """
    if not probs:
        raise ValueError("probs must be non-empty")
    if not (0 <= target_id < len(probs)):
        raise ValueError(
            f"target_id={target_id} out of range [0, {len(probs)})"
        )
    grad = list(probs)
    grad[target_id] -= 1.0
    return grad


def sgd_step(
    weight_row: list[float],
    grad_row: list[float],
    lr: float,
    n_examples: int,
) -> list[float]:
    """Returns weight_row - lr * (grad_row / n_examples). Pure: input is not mutated."""
    if len(weight_row) != len(grad_row):
        raise ValueError(
            f"weight_row length ({len(weight_row)}) != grad_row length ({len(grad_row)})"
        )
    if n_examples <= 0:
        raise ValueError(f"n_examples must be > 0, got {n_examples}")
    scale = lr / n_examples
    return [w - scale * g for w, g in zip(weight_row, grad_row)]
