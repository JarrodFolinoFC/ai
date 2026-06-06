"""Tests for llm2.math.stage1 — one test per formula, hand-computed expected outputs."""

import math
import pytest


def test_module_imports():
    from llm2.math import stage1  # noqa: F401


def test_softmax_uniform_logits_uniform_probs():
    from llm2.math.stage1 import softmax
    out = softmax([0.0, 0.0, 0.0, 0.0])
    assert out == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_softmax_sums_to_one():
    from llm2.math.stage1 import softmax
    out = softmax([1.0, 2.0, 3.0, -1.0])
    assert sum(out) == pytest.approx(1.0)


def test_softmax_numerically_stable_for_large_logits():
    from llm2.math.stage1 import softmax
    out = softmax([1000.0, 1001.0, 1002.0])
    assert sum(out) == pytest.approx(1.0)
    assert out[2] > out[1] > out[0]


def test_softmax_known_values():
    from llm2.math.stage1 import softmax
    out = softmax([1.0, 2.0, 3.0])
    expected = [math.exp(1) / (math.exp(1) + math.exp(2) + math.exp(3)),
                math.exp(2) / (math.exp(1) + math.exp(2) + math.exp(3)),
                math.exp(3) / (math.exp(1) + math.exp(2) + math.exp(3))]
    assert out == pytest.approx(expected)


def test_softmax_raises_on_empty():
    from llm2.math.stage1 import softmax
    with pytest.raises(ValueError):
        softmax([])


def test_cross_entropy_loss_matches_formula():
    from llm2.math.stage1 import cross_entropy_loss
    probs = [0.1, 0.6, 0.3]
    assert cross_entropy_loss(probs, 1) == pytest.approx(-math.log(0.6 + 1e-12))


def test_cross_entropy_loss_finite_for_zero_prob():
    from llm2.math.stage1 import cross_entropy_loss
    probs = [1.0, 0.0]
    out = cross_entropy_loss(probs, 1)
    assert math.isfinite(out)
    assert out == pytest.approx(-math.log(1e-12))


def test_cross_entropy_loss_raises_on_out_of_range():
    from llm2.math.stage1 import cross_entropy_loss
    with pytest.raises(ValueError):
        cross_entropy_loss([0.5, 0.5], 5)


def test_cross_entropy_loss_raises_on_empty():
    from llm2.math.stage1 import cross_entropy_loss
    with pytest.raises(ValueError):
        cross_entropy_loss([], 0)


def test_bigram_backward_returns_probs_minus_one_hot():
    from llm2.math.stage1 import bigram_backward
    probs = [0.1, 0.6, 0.3]
    assert bigram_backward(probs, 1) == pytest.approx([0.1, -0.4, 0.3])


def test_bigram_backward_does_not_mutate_input():
    from llm2.math.stage1 import bigram_backward
    probs = [0.1, 0.6, 0.3]
    _ = bigram_backward(probs, 1)
    assert probs == [0.1, 0.6, 0.3]


def test_bigram_backward_length_preserved():
    from llm2.math.stage1 import bigram_backward
    out = bigram_backward([0.5, 0.5], 0)
    assert len(out) == 2


def test_bigram_backward_raises_on_out_of_range():
    from llm2.math.stage1 import bigram_backward
    with pytest.raises(ValueError):
        bigram_backward([0.5, 0.5], 9)


def test_sgd_step_known_update():
    from llm2.math.stage1 import sgd_step
    out = sgd_step(weight_row=[1.0, 2.0, 3.0],
                   grad_row=[0.5, -1.0, 0.0],
                   lr=0.1,
                   n_examples=2)
    # scale = 0.1 / 2 = 0.05
    # 1.0 - 0.05*0.5 = 0.975
    # 2.0 - 0.05*-1.0 = 2.05
    # 3.0 - 0.05*0.0 = 3.0
    assert out == pytest.approx([0.975, 2.05, 3.0])


def test_sgd_step_does_not_mutate_input():
    from llm2.math.stage1 import sgd_step
    w = [1.0, 2.0]
    g = [1.0, 1.0]
    _ = sgd_step(w, g, 0.5, 1)
    assert w == [1.0, 2.0]
    assert g == [1.0, 1.0]


def test_sgd_step_raises_on_length_mismatch():
    from llm2.math.stage1 import sgd_step
    with pytest.raises(ValueError):
        sgd_step([1.0, 2.0], [1.0], 0.1, 1)


def test_sgd_step_raises_on_zero_n_examples():
    from llm2.math.stage1 import sgd_step
    with pytest.raises(ValueError):
        sgd_step([1.0], [1.0], 0.1, 0)


def test_sgd_step_raises_on_negative_n_examples():
    from llm2.math.stage1 import sgd_step
    with pytest.raises(ValueError):
        sgd_step([1.0], [1.0], 0.1, -1)
