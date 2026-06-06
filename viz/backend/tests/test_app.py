"""Tests for the FastAPI visualizer backend."""

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


import pytest
from pydantic import ValidationError


def test_token_embed_request_rejects_ragged_matrix():
    from schemas import TokenEmbedRequest
    with pytest.raises(ValidationError):
        TokenEmbedRequest(E=[[0.0, 1.0], [2.0]], idx=0)


def test_token_embed_request_rejects_out_of_range_idx():
    from schemas import TokenEmbedRequest
    with pytest.raises(ValidationError):
        TokenEmbedRequest(E=[[0.0], [1.0]], idx=5)


def test_token_embed_request_accepts_valid():
    from schemas import TokenEmbedRequest
    req = TokenEmbedRequest(E=[[0.0, 1.0], [2.0, 3.0]], idx=1)
    assert req.idx == 1


def test_token_embed_endpoint_happy():
    payload = {"E": [[0.1, 0.2], [1.0, 2.0]], "idx": 1}
    r = client.post("/api/stage2a/token-embed", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["result"] == [1.0, 2.0]
    assert "E" in body["formula_latex"] or "E" in body["formula_latex"].lower() or body["formula_latex"]
    assert isinstance(body["steps"], list)


def test_token_embed_endpoint_invalid_returns_422():
    r = client.post("/api/stage2a/token-embed", json={"E": [[0.1]], "idx": 5})
    assert r.status_code == 422


def test_position_embed_endpoint_happy():
    payload = {"P": [[0.0, 0.0], [0.5, -0.5]], "t": 1}
    r = client.post("/api/stage2a/position-embed", json=payload)
    assert r.status_code == 200
    assert r.json()["result"] == [0.5, -0.5]


def test_combined_input_endpoint_happy():
    payload = {
        "E": [[0.1, 0.2], [1.0, 2.0]],
        "P": [[0.0, 0.0], [0.5, -0.5]],
        "idx": 1,
        "t": 1,
    }
    r = client.post("/api/stage2a/combined-input", json=payload)
    assert r.status_code == 200
    assert r.json()["result"] == [1.5, 1.5]


def test_lower_tri_mask_endpoint_happy():
    r = client.post("/api/stage2a/lower-tri-mask", json={"T": 3})
    assert r.status_code == 200
    assert r.json()["result"] == [
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ]


def test_lower_tri_mask_endpoint_zero_returns_422():
    r = client.post("/api/stage2a/lower-tri-mask", json={"T": 0})
    assert r.status_code == 422


def test_normalize_rows_endpoint_happy():
    r = client.post("/api/stage2a/normalize-rows", json={"M": [[1.0, 0.0], [1.0, 1.0]]})
    assert r.status_code == 200
    body = r.json()
    assert body["result"][0] == [1.0, 0.0]
    assert body["result"][1] == pytest.approx([0.5, 0.5])


def test_normalize_rows_endpoint_zero_row_returns_400():
    r = client.post("/api/stage2a/normalize-rows", json={"M": [[0.0, 0.0]]})
    assert r.status_code == 400
    assert "zero" in r.json()["detail"].lower()


def test_unembed_head_endpoint_happy():
    payload = {
        "h": [1.0, 2.0],
        "W_head": [[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]],
        "b": [0.5, 0.5, 0.5],
    }
    r = client.post("/api/stage2a/unembed-head", json=payload)
    assert r.status_code == 200
    assert r.json()["result"] == pytest.approx([1.5, 2.5, 8.5])


# ---------- source_code field ----------


def test_stage2a_response_includes_source_code():
    r = client.post("/api/stage2a/token-embed", json={"E": [[0.0, 1.0]], "idx": 0})
    assert r.status_code == 200
    body = r.json()
    assert "source_code" in body
    assert "def token_embed" in body["source_code"]


# ---------- stage 1: softmax ----------


def test_softmax_endpoint_happy():
    r = client.post("/api/stage1/softmax", json={"logits": [0.0, 0.0, 0.0, 0.0]})
    assert r.status_code == 200
    body = r.json()
    assert body["result"] == pytest.approx([0.25, 0.25, 0.25, 0.25])
    assert "def softmax" in body["source_code"]


def test_softmax_endpoint_empty_returns_422():
    r = client.post("/api/stage1/softmax", json={"logits": []})
    assert r.status_code == 422


# ---------- stage 1: cross-entropy loss ----------


def test_cross_entropy_loss_endpoint_happy():
    r = client.post(
        "/api/stage1/cross-entropy-loss",
        json={"probs": [0.1, 0.6, 0.3], "target_id": 1},
    )
    assert r.status_code == 200
    body = r.json()
    import math
    assert body["result"] == pytest.approx(-math.log(0.6 + 1e-12))
    assert "def cross_entropy_loss" in body["source_code"]


def test_cross_entropy_loss_endpoint_out_of_range_returns_422():
    r = client.post(
        "/api/stage1/cross-entropy-loss",
        json={"probs": [0.5, 0.5], "target_id": 9},
    )
    assert r.status_code == 422


# ---------- stage 1: bigram backward ----------


def test_bigram_backward_endpoint_happy():
    r = client.post(
        "/api/stage1/bigram-backward",
        json={"probs": [0.1, 0.6, 0.3], "target_id": 1},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["result"] == pytest.approx([0.1, -0.4, 0.3])
    assert "def bigram_backward" in body["source_code"]


def test_bigram_backward_endpoint_out_of_range_returns_422():
    r = client.post(
        "/api/stage1/bigram-backward",
        json={"probs": [0.5, 0.5], "target_id": 9},
    )
    assert r.status_code == 422


# ---------- stage 1: sgd step ----------


def test_sgd_step_endpoint_happy():
    r = client.post(
        "/api/stage1/sgd-step",
        json={
            "weight_row": [1.0, 2.0, 3.0],
            "grad_row": [0.5, -1.0, 0.0],
            "lr": 0.1,
            "n_examples": 2,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["result"] == pytest.approx([0.975, 2.05, 3.0])
    assert "def sgd_step" in body["source_code"]


def test_sgd_step_endpoint_zero_n_returns_422():
    r = client.post(
        "/api/stage1/sgd-step",
        json={
            "weight_row": [1.0],
            "grad_row": [1.0],
            "lr": 0.1,
            "n_examples": 0,
        },
    )
    assert r.status_code == 422


def test_sgd_step_endpoint_length_mismatch_returns_422():
    r = client.post(
        "/api/stage1/sgd-step",
        json={
            "weight_row": [1.0, 2.0],
            "grad_row": [1.0],
            "lr": 0.1,
            "n_examples": 1,
        },
    )
    assert r.status_code == 422
