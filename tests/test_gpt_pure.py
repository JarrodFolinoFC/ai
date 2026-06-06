"""
Tests for gpt_pure.py and gpt_pure_log.py.

Run with:  pytest test_gpt_pure.py -v

These tests double as worked examples — read them top-to-bottom to see
what each function does on tiny inputs.
"""

import logging
import math
import random

import pytest

from llm2 import gpt_pure_log
from llm2.gpt_pure import (
    decode,
    forward_one,
    generate,
    get_batch,
    init_weights,
    softmax,
    split_data,
    train,
)


# ---------- softmax ----------


def test_softmax_sums_to_one():
    out = softmax([1.0, 2.0, 3.0])
    assert math.isclose(sum(out), 1.0, abs_tol=1e-9)


def test_softmax_all_positive():
    out = softmax([-5.0, 0.0, 5.0])
    assert all(p > 0 for p in out)


def test_softmax_argmax_preserved():
    logits = [0.1, 5.0, -2.0, 1.5]
    out = softmax(logits)
    assert out.index(max(out)) == logits.index(max(logits))


def test_softmax_handles_huge_logits():
    # Without the max-subtract, exp(1000) overflows. This must not raise.
    out = softmax([1000.0, 1001.0, 999.0])
    assert math.isclose(sum(out), 1.0, abs_tol=1e-9)
    assert out.index(max(out)) == 1


def test_softmax_uniform_when_logits_equal():
    out = softmax([2.0, 2.0, 2.0, 2.0])
    for p in out:
        assert math.isclose(p, 0.25, abs_tol=1e-9)


# ---------- forward_one ----------


def test_forward_one_returns_probs_and_positive_loss():
    weight = [[0.0, 0.0], [0.0, 0.0]]
    probs, loss = forward_one(weight, token_id=0, target_id=1)
    assert math.isclose(sum(probs), 1.0, abs_tol=1e-9)
    assert loss > 0


def test_forward_one_loss_smaller_when_target_logit_is_higher():
    # Row 0 prefers token 1 strongly → lower loss when target is 1.
    weight_good = [[0.0, 5.0], [0.0, 0.0]]
    weight_bad = [[5.0, 0.0], [0.0, 0.0]]
    _, loss_good = forward_one(weight_good, 0, 1)
    _, loss_bad = forward_one(weight_bad, 0, 1)
    assert loss_good < loss_bad


# ---------- get_batch ----------


def test_get_batch_shapes():
    random.seed(0)
    data = list(range(100))
    x, y = get_batch(data, batch_size=4, block_size=8)
    assert len(x) == 4 and len(y) == 4
    assert all(len(row) == 8 for row in x)
    assert all(len(row) == 8 for row in y)


def test_get_batch_targets_are_inputs_shifted_by_one():
    random.seed(0)
    data = list(range(100))
    x, y = get_batch(data, batch_size=3, block_size=5)
    for x_row, y_row in zip(x, y):
        for i in range(len(x_row) - 1):
            # y[t] should be the token that follows x[t] in the source.
            assert y_row[i] == x_row[i] + 1
        # And y[-1] should follow x[-1].
        assert y_row[-1] == x_row[-1] + 1


# ---------- init_weights ----------


def test_init_weights_is_v_by_v():
    random.seed(0)
    w = init_weights(7)
    assert len(w) == 7
    assert all(len(row) == 7 for row in w)
    assert all(isinstance(v, float) for row in w for v in row)


# ---------- split_data ----------


def test_split_data_default_fraction():
    data = list(range(100))
    train, val = split_data(data)
    assert len(train) == 90 and len(val) == 10
    assert train + val == data


# ---------- decode ----------


def test_decode_joins_tokens_with_spaces():
    itos = {0: "the", 1: "cat", 2: "sat"}
    assert decode(itos, [0, 1, 2]) == "the cat sat"


# ---------- train (integration) ----------


def test_train_reduces_loss_on_tiny_corpus():
    random.seed(42)
    # Two tokens that always alternate: 0 1 0 1 0 1 ...
    # A bigram model should learn this almost perfectly.
    data = [0, 1] * 200
    train_data, val_data = split_data(data, train_frac=0.9)
    vocab_size = 2
    weight = init_weights(vocab_size)

    def avg_loss():
        total, n = 0.0, 0
        for inp, tgt in zip(train_data, train_data[1:]):
            _, l = forward_one(weight, inp, tgt)
            total += l
            n += 1
        return total / n

    before = avg_loss()
    train(
        weight=weight,
        train_data=train_data,
        val_data=val_data,
        vocab_size=vocab_size,
        batch_size=4,
        block_size=4,
        max_iters=20,
        eval_interval=1000,  # skip the noisy estimate_loss prints
        eval_iters=1,
        learning_rate=1.0,
    )
    after = avg_loss()
    assert after < before


# ---------- generate ----------


def test_generate_length_and_valid_ids():
    random.seed(0)
    vocab_size = 5
    weight = [[0.0] * vocab_size for _ in range(vocab_size)]
    out = generate(weight, start_id=0, max_new_tokens=10)
    assert len(out) == 11  # start + 10 new
    assert all(0 <= i < vocab_size for i in out)


def test_generate_deterministic_when_one_logit_dominates():
    random.seed(0)
    # Whatever the previous token, the next is forced to id 2.
    vocab_size = 4
    weight = [[0.0, 0.0, 1000.0, 0.0] for _ in range(vocab_size)]
    out = generate(weight, start_id=0, max_new_tokens=5)
    assert out[1:] == [2, 2, 2, 2, 2]


# ---------- logging ----------


class _CapturingHandler(logging.Handler):
    """Test double for OpenObserveHandler — keeps records in a list."""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def capture_logs():
    """Enable logging with a capturing handler. Tears down after the test."""
    handler = _CapturingHandler()
    gpt_pure_log.enable_logging(sample_rate=1.0, handler=handler)
    yield handler
    gpt_pure_log.disable_logging()


def test_logger_silent_when_disabled():
    gpt_pure_log.disable_logging()
    captured = []

    class Cap(logging.Handler):
        def emit(self, record):
            captured.append(record)

    # Attach a handler manually so we'd see records IF anything was emitted.
    logger = logging.getLogger("gpt_pure")
    h = Cap()
    logger.addHandler(h)
    try:
        weight = [[0.0, 0.0], [0.0, 0.0]]
        for _ in range(50):
            forward_one(weight, 0, 1)
    finally:
        logger.removeHandler(h)
    assert captured == []


def test_logger_emits_when_enabled_at_full_sample(capture_logs):
    weight = [[0.0, 0.0], [0.0, 0.0]]
    forward_one(weight, 0, 1)
    assert any(getattr(r, "fields", {}).get("event") == "forward_one" for r in capture_logs.records)


def test_sample_rate_zero_emits_nothing_from_sampled_sites():
    handler = _CapturingHandler()
    gpt_pure_log.enable_logging(sample_rate=0.0, handler=handler)
    try:
        weight = [[0.0, 0.0], [0.0, 0.0]]
        for _ in range(100):
            forward_one(weight, 0, 1)
        # forward_one is a sampled call site — at rate 0 we expect zero.
        sampled = [r for r in handler.records if getattr(r, "fields", {}).get("event") == "forward_one"]
        assert sampled == []
    finally:
        gpt_pure_log.disable_logging()


def test_sample_rate_is_approximately_honored():
    random.seed(0)
    handler = _CapturingHandler()
    gpt_pure_log.enable_logging(sample_rate=0.1, handler=handler)
    try:
        weight = [[0.0, 0.0], [0.0, 0.0]]
        n = 2000
        for _ in range(n):
            forward_one(weight, 0, 1)
        emitted = sum(
            1 for r in handler.records if getattr(r, "fields", {}).get("event") == "forward_one"
        )
        # Expect ~200; allow a generous band so this isn't flaky.
        assert 100 < emitted < 350, f"got {emitted} samples out of {n} at rate 0.1"
    finally:
        gpt_pure_log.disable_logging()


def test_log_event_payload_contains_event_and_fields(capture_logs):
    gpt_pure_log.log_event("custom", foo=1, bar="x")
    rec = capture_logs.records[-1]
    assert rec.fields["event"] == "custom"
    assert rec.fields["foo"] == 1
    assert rec.fields["bar"] == "x"
