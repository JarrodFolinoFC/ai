"""Pydantic request/response models for stage2a formula endpoints.

Every response carries (result, formula_latex, steps) so the frontend
data-flow plumbing is the same shape across all six pages.
"""

from pydantic import BaseModel, model_validator


def _is_rectangular(rows: list[list[float]]) -> bool:
    return bool(rows) and all(len(r) == len(rows[0]) for r in rows)


# ---------- Common response shape ----------


Vector = list[float]
Matrix = list[list[float]]
Scalar = float
ResultValue = Scalar | Vector | Matrix


class FormulaResponse(BaseModel):
    """Common response shape across every endpoint."""

    result: ResultValue
    formula_latex: str
    steps: list[tuple[str, ResultValue]]
    source_code: str


# ---------- 1. token_embed ----------


class TokenEmbedRequest(BaseModel):
    E: Matrix
    idx: int

    @model_validator(mode="after")
    def _check(self):
        if not _is_rectangular(self.E):
            raise ValueError("E must be non-empty and rectangular")
        if not (0 <= self.idx < len(self.E)):
            raise ValueError(f"idx={self.idx} out of range [0, {len(self.E)})")
        return self


# ---------- 2. position_embed ----------


class PositionEmbedRequest(BaseModel):
    P: Matrix
    t: int

    @model_validator(mode="after")
    def _check(self):
        if not _is_rectangular(self.P):
            raise ValueError("P must be non-empty and rectangular")
        if not (0 <= self.t < len(self.P)):
            raise ValueError(f"t={self.t} out of range [0, {len(self.P)})")
        return self


# ---------- 3. combined_input ----------


class CombinedInputRequest(BaseModel):
    E: Matrix
    P: Matrix
    idx: int
    t: int

    @model_validator(mode="after")
    def _check(self):
        if not _is_rectangular(self.E):
            raise ValueError("E must be non-empty and rectangular")
        if not _is_rectangular(self.P):
            raise ValueError("P must be non-empty and rectangular")
        if not (0 <= self.idx < len(self.E)):
            raise ValueError(f"idx={self.idx} out of range [0, {len(self.E)})")
        if not (0 <= self.t < len(self.P)):
            raise ValueError(f"t={self.t} out of range [0, {len(self.P)})")
        if len(self.E[0]) != len(self.P[0]):
            raise ValueError(
                f"E row width ({len(self.E[0])}) != P row width ({len(self.P[0])})"
            )
        return self


# ---------- 4. lower_tri_mask ----------


class LowerTriMaskRequest(BaseModel):
    T: int

    @model_validator(mode="after")
    def _check(self):
        if self.T < 1:
            raise ValueError(f"T must be >= 1, got {self.T}")
        if self.T > 32:
            raise ValueError(f"T must be <= 32 for the visualizer, got {self.T}")
        return self


# ---------- 5. normalize_rows ----------


class NormalizeRowsRequest(BaseModel):
    M: Matrix

    @model_validator(mode="after")
    def _check(self):
        if not _is_rectangular(self.M):
            raise ValueError("M must be non-empty and rectangular")
        return self


# ---------- 6. unembed_head ----------


class UnembedHeadRequest(BaseModel):
    h: Vector
    W_head: Matrix
    b: Vector

    @model_validator(mode="after")
    def _check(self):
        if not _is_rectangular(self.W_head):
            raise ValueError("W_head must be non-empty and rectangular")
        if len(self.W_head) != len(self.h):
            raise ValueError(
                f"W_head row count ({len(self.W_head)}) must equal len(h) ({len(self.h)})"
            )
        V = len(self.W_head[0])
        if len(self.b) != V:
            raise ValueError(f"b length ({len(self.b)}) must equal V ({V})")
        return self


# ---------- stage 1: softmax ----------


class SoftmaxRequest(BaseModel):
    logits: Vector

    @model_validator(mode="after")
    def _check(self):
        if not self.logits:
            raise ValueError("logits must be non-empty")
        return self


# ---------- stage 1: cross_entropy_loss ----------


class CrossEntropyLossRequest(BaseModel):
    probs: Vector
    target_id: int

    @model_validator(mode="after")
    def _check(self):
        if not self.probs:
            raise ValueError("probs must be non-empty")
        if not (0 <= self.target_id < len(self.probs)):
            raise ValueError(
                f"target_id={self.target_id} out of range [0, {len(self.probs)})"
            )
        return self


# ---------- stage 1: bigram_backward ----------


class BigramBackwardRequest(BaseModel):
    probs: Vector
    target_id: int

    @model_validator(mode="after")
    def _check(self):
        if not self.probs:
            raise ValueError("probs must be non-empty")
        if not (0 <= self.target_id < len(self.probs)):
            raise ValueError(
                f"target_id={self.target_id} out of range [0, {len(self.probs)})"
            )
        return self


# ---------- stage 1: sgd_step ----------


class SgdStepRequest(BaseModel):
    weight_row: Vector
    grad_row: Vector
    lr: float
    n_examples: int

    @model_validator(mode="after")
    def _check(self):
        if not self.weight_row:
            raise ValueError("weight_row must be non-empty")
        if len(self.weight_row) != len(self.grad_row):
            raise ValueError(
                f"weight_row length ({len(self.weight_row)}) != grad_row length ({len(self.grad_row)})"
            )
        if self.n_examples <= 0:
            raise ValueError(f"n_examples must be > 0, got {self.n_examples}")
        return self
