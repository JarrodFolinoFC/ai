"""FastAPI visualizer backend. One endpoint per formula.

Every endpoint follows the same shape:
  - Pydantic validates the request at the HTTP boundary (422 on bad input).
  - The math layer (`llm2.math.*`) performs the computation.
  - ValueError from the math layer becomes 400 (e.g. degenerate inputs).
  - Successful responses always return {result, formula_latex, steps, source_code}.
"""

import inspect

from fastapi import FastAPI, HTTPException

from llm2.math import stage1, stage2a

from schemas import (
    BigramBackwardRequest,
    CombinedInputRequest,
    CrossEntropyLossRequest,
    FormulaResponse,
    LowerTriMaskRequest,
    NormalizeRowsRequest,
    PositionEmbedRequest,
    SgdStepRequest,
    SoftmaxRequest,
    TokenEmbedRequest,
    UnembedHeadRequest,
)

app = FastAPI(title="llm2 visualizer", version="0.1.0")


# Cache source code at import time — one read of the .py files, not per request.
_SOURCE: dict[str, str] = {
    "token_embed": inspect.getsource(stage2a.token_embed),
    "position_embed": inspect.getsource(stage2a.position_embed),
    "combined_input": inspect.getsource(stage2a.combined_input),
    "lower_tri_mask": inspect.getsource(stage2a.lower_tri_mask),
    "normalize_rows": inspect.getsource(stage2a.normalize_rows),
    "unembed_head": inspect.getsource(stage2a.unembed_head),
    "softmax": inspect.getsource(stage1.softmax),
    "cross_entropy_loss": inspect.getsource(stage1.cross_entropy_loss),
    "bigram_backward": inspect.getsource(stage1.bigram_backward),
    "sgd_step": inspect.getsource(stage1.sgd_step),
}


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


def _math_call(fn, *args):
    """Call into the math layer, converting ValueError to HTTP 400."""
    try:
        return fn(*args)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


# ---------- stage 2a ----------


@app.post("/api/stage2a/token-embed", response_model=FormulaResponse)
def token_embed(req: TokenEmbedRequest) -> FormulaResponse:
    out = _math_call(stage2a.token_embed, req.E, req.idx)
    return FormulaResponse(
        result=out,
        formula_latex=r"x = E[\text{idx}]",
        steps=[(f"row lookup at idx={req.idx}", out)],
        source_code=_SOURCE["token_embed"],
    )


@app.post("/api/stage2a/position-embed", response_model=FormulaResponse)
def position_embed(req: PositionEmbedRequest) -> FormulaResponse:
    out = _math_call(stage2a.position_embed, req.P, req.t)
    return FormulaResponse(
        result=out,
        formula_latex=r"p = P[t]",
        steps=[(f"row lookup at t={req.t}", out)],
        source_code=_SOURCE["position_embed"],
    )


@app.post("/api/stage2a/combined-input", response_model=FormulaResponse)
def combined_input(req: CombinedInputRequest) -> FormulaResponse:
    e = _math_call(stage2a.token_embed, req.E, req.idx)
    p = _math_call(stage2a.position_embed, req.P, req.t)
    out = _math_call(stage2a.combined_input, req.E, req.P, req.idx, req.t)
    return FormulaResponse(
        result=out,
        formula_latex=r"h = E[\text{idx}] + P[t]",
        steps=[
            (f"E[idx={req.idx}]", e),
            (f"P[t={req.t}]", p),
            ("h = E[idx] + P[t]", out),
        ],
        source_code=_SOURCE["combined_input"],
    )


@app.post("/api/stage2a/lower-tri-mask", response_model=FormulaResponse)
def lower_tri_mask(req: LowerTriMaskRequest) -> FormulaResponse:
    out = _math_call(stage2a.lower_tri_mask, req.T)
    return FormulaResponse(
        result=out,
        formula_latex=r"M_{i,j} = \begin{cases} 1 & j \le i \\ 0 & \text{otherwise} \end{cases}",
        steps=[(f"M (T={req.T})", out)],
        source_code=_SOURCE["lower_tri_mask"],
    )


@app.post("/api/stage2a/normalize-rows", response_model=FormulaResponse)
def normalize_rows(req: NormalizeRowsRequest) -> FormulaResponse:
    out = _math_call(stage2a.normalize_rows, req.M)
    return FormulaResponse(
        result=out,
        formula_latex=r"W_{i,j} = \frac{M_{i,j}}{\sum_j M_{i,j}}",
        steps=[("W (row-stochastic)", out)],
        source_code=_SOURCE["normalize_rows"],
    )


@app.post("/api/stage2a/unembed-head", response_model=FormulaResponse)
def unembed_head(req: UnembedHeadRequest) -> FormulaResponse:
    out = _math_call(stage2a.unembed_head, req.h, req.W_head, req.b)
    return FormulaResponse(
        result=out,
        formula_latex=r"\text{logits} = h W_{\text{head}} + b",
        steps=[("logits", out)],
        source_code=_SOURCE["unembed_head"],
    )


# ---------- stage 1 ----------


@app.post("/api/stage1/softmax", response_model=FormulaResponse)
def softmax(req: SoftmaxRequest) -> FormulaResponse:
    out = _math_call(stage1.softmax, req.logits)
    return FormulaResponse(
        result=out,
        formula_latex=r"p_i = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}",
        steps=[("logits", req.logits), ("probs", out)],
        source_code=_SOURCE["softmax"],
    )


@app.post("/api/stage1/cross-entropy-loss", response_model=FormulaResponse)
def cross_entropy_loss(req: CrossEntropyLossRequest) -> FormulaResponse:
    out = _math_call(stage1.cross_entropy_loss, req.probs, req.target_id)
    return FormulaResponse(
        result=out,
        formula_latex=r"\mathcal{L} = -\log(p_{\text{target}} + 10^{-12})",
        steps=[
            (f"probs[target={req.target_id}]", req.probs[req.target_id]),
            ("loss", out),
        ],
        source_code=_SOURCE["cross_entropy_loss"],
    )


@app.post("/api/stage1/bigram-backward", response_model=FormulaResponse)
def bigram_backward(req: BigramBackwardRequest) -> FormulaResponse:
    out = _math_call(stage1.bigram_backward, req.probs, req.target_id)
    return FormulaResponse(
        result=out,
        formula_latex=r"\nabla_z \mathcal{L} = p - \text{onehot}(\text{target})",
        steps=[
            ("probs", req.probs),
            (f"grad (target={req.target_id})", out),
        ],
        source_code=_SOURCE["bigram_backward"],
    )


@app.post("/api/stage1/sgd-step", response_model=FormulaResponse)
def sgd_step(req: SgdStepRequest) -> FormulaResponse:
    out = _math_call(
        stage1.sgd_step, req.weight_row, req.grad_row, req.lr, req.n_examples
    )
    return FormulaResponse(
        result=out,
        formula_latex=r"w' = w - \frac{\eta}{N} \, g",
        steps=[
            ("weight_row (before)", req.weight_row),
            ("grad_row", req.grad_row),
            ("weight_row (after)", out),
        ],
        source_code=_SOURCE["sgd_step"],
    )
