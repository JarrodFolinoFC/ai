"""Tests for llm2.math.stage2a — one test per formula, hand-computed expected outputs."""

import math
import pytest


def test_module_imports():
    from llm2.math import stage2a  # noqa: F401


def test_token_embed_returns_correct_row():
    from llm2.math.stage2a import token_embed
    E = [
        [0.1, 0.2, 0.3],
        [1.0, 1.1, 1.2],
        [2.0, 2.1, 2.2],
    ]
    assert token_embed(E, 0) == [0.1, 0.2, 0.3]
    assert token_embed(E, 2) == [2.0, 2.1, 2.2]


def test_token_embed_returns_a_copy():
    from llm2.math.stage2a import token_embed
    E = [[0.0, 1.0], [2.0, 3.0]]
    out = token_embed(E, 0)
    out[0] = 999.0
    assert E[0][0] == 0.0


def test_position_embed_returns_correct_row():
    from llm2.math.stage2a import position_embed
    P = [
        [0.0, 0.0],
        [0.5, -0.5],
        [1.0, -1.0],
    ]
    assert position_embed(P, 1) == [0.5, -0.5]


def test_position_embed_returns_a_copy():
    from llm2.math.stage2a import position_embed
    P = [[7.0, 8.0]]
    out = position_embed(P, 0)
    out[1] = 999.0
    assert P[0][1] == 8.0


def test_combined_input_elementwise_sum():
    from llm2.math.stage2a import combined_input
    E = [[0.1, 0.2], [1.0, 2.0]]
    P = [[0.0, 0.0], [0.5, -0.5]]
    out = combined_input(E, P, idx=1, t=1)
    assert out == pytest.approx([1.5, 1.5])


def test_combined_input_raises_on_dim_mismatch():
    from llm2.math.stage2a import combined_input
    E = [[0.0, 0.0]]
    P = [[0.0, 0.0, 0.0]]
    with pytest.raises(ValueError):
        combined_input(E, P, idx=0, t=0)


def test_lower_tri_mask_t4():
    from llm2.math.stage2a import lower_tri_mask
    assert lower_tri_mask(4) == [
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ]


def test_lower_tri_mask_t1():
    from llm2.math.stage2a import lower_tri_mask
    assert lower_tri_mask(1) == [[1.0]]


def test_lower_tri_mask_raises_on_zero():
    from llm2.math.stage2a import lower_tri_mask
    with pytest.raises(ValueError):
        lower_tri_mask(0)


def test_normalize_rows_of_lower_tri():
    from llm2.math.stage2a import lower_tri_mask, normalize_rows
    M = lower_tri_mask(3)
    W = normalize_rows(M)
    assert W[0] == pytest.approx([1.0, 0.0, 0.0])
    assert W[1] == pytest.approx([0.5, 0.5, 0.0])
    assert W[2] == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_normalize_rows_each_row_sums_to_one():
    from llm2.math.stage2a import normalize_rows
    M = [[2.0, 1.0, 1.0], [0.0, 4.0, 0.0]]
    W = normalize_rows(M)
    for row in W:
        assert math.isclose(sum(row), 1.0, abs_tol=1e-9)


def test_normalize_rows_raises_on_zero_row():
    from llm2.math.stage2a import normalize_rows
    with pytest.raises(ValueError):
        normalize_rows([[0.0, 0.0, 0.0]])


def test_unembed_head_simple():
    from llm2.math.stage2a import unembed_head
    # logits[j] = sum(h[k] * W_head[k][j]) + b[j]
    h = [1.0, 2.0]
    W_head = [
        [1.0, 0.0, 2.0],   # k=0
        [0.0, 1.0, 3.0],   # k=1
    ]
    b = [0.5, 0.5, 0.5]
    # logits[0] = 1*1 + 2*0 + 0.5 = 1.5
    # logits[1] = 1*0 + 2*1 + 0.5 = 2.5
    # logits[2] = 1*2 + 2*3 + 0.5 = 8.5
    assert unembed_head(h, W_head, b) == pytest.approx([1.5, 2.5, 8.5])


def test_unembed_head_raises_on_dim_mismatch():
    from llm2.math.stage2a import unembed_head
    h = [1.0, 2.0]
    W_head = [[1.0, 0.0]]  # only 1 row, but h has 2 elements
    b = [0.0, 0.0]
    with pytest.raises(ValueError):
        unembed_head(h, W_head, b)


def test_unembed_head_raises_on_bias_length_mismatch():
    from llm2.math.stage2a import unembed_head
    h = [1.0]
    W_head = [[1.0, 2.0]]
    b = [0.0]  # should be length 2
    with pytest.raises(ValueError):
        unembed_head(h, W_head, b)
