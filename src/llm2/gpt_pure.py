"""
Stage 1: Bigram baseline — pure Python version.

Same model as gpt.py but with ZERO PyTorch. Every operation is written by
hand so you can see exactly what's hiding inside nn.Embedding, F.softmax,
F.cross_entropy, loss.backward(), and optimizer.step().

What maps to what:
  nn.Embedding(V, V)        →  weight (a list of V lists, each V floats)
  embedding(idx)            →  weight[idx]  (literal list lookup)
  F.softmax                 →  softmax() function below
  F.cross_entropy           →  forward_one() returns -log(probs[target])
  loss.backward()           →  manual: gradient = probs - one_hot(target)
  optimizer.step()  (SGD)   →  weight[i][j] -= lr * grad[i][j]

Trade-offs vs gpt.py:
  - SGD instead of AdamW (simpler, fewer moving parts).
  - Smaller batches and fewer iters because pure Python is ~1000x slower.
  - No GPU. No vectorization. Just for-loops.

"""

import random

from llm2.gpt_pure_log import log_event, should_sample
from llm2.math.stage1 import (
    bigram_backward,
    cross_entropy_loss,
    sgd_step,
    softmax,
)


def forward_one(weight, token_id, target_id):
    """One position: pull row, softmax, return (probs, loss).

    The 'embedding lookup' is literally `weight[token_id]` — no magic.
    """
    logits = weight[token_id]
    probs = softmax(logits)
    loss = cross_entropy_loss(probs, target_id)
    if should_sample():
        top3 = sorted(range(len(probs)), key=lambda j: probs[j], reverse=True)[:3]
        log_event(
            "forward_one",
            inp=token_id,
            tgt=target_id,
            loss=loss,
            top3=[(j, probs[j]) for j in top3],
        )
    return probs, loss


def decode(itos, ids):
    return " ".join(itos[i] for i in ids)


def load_corpus(path):
    """Read corpus, build vocab, return everything you need to train."""
    with open(path, "r") as f:
        text = f.read()
    tokens = text.replace(".", " .").split()
    vocab = sorted(set(tokens))
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for w, i in stoi.items()}
    data = [stoi[t] for t in tokens]
    return data, vocab, stoi, itos


def split_data(data, train_frac=0.9):
    n = int(train_frac * len(data))
    return data[:n], data[n:]


def get_batch(d, batch_size, block_size):
    """Sample batch_size random windows of block_size tokens from sequence d."""
    ix = [random.randint(0, len(d) - block_size - 1) for _ in range(batch_size)]
    x = [[d[i + t] for t in range(block_size)] for i in ix]
    y = [[d[i + t + 1] for t in range(block_size)] for i in ix]
    return x, y


def estimate_loss(weight, train_data, val_data, batch_size, block_size, eval_iters):
    """Average loss across eval_iters batches, for both train and val splits."""
    out = {}
    for split_name, d in [("train", train_data), ("val", val_data)]:
        total = 0.0
        count = 0
        for _ in range(eval_iters):
            x_batch, y_batch = get_batch(d, batch_size, block_size)
            for x_seq, y_seq in zip(x_batch, y_batch):
                for inp, tgt in zip(x_seq, y_seq):
                    _, loss = forward_one(weight, inp, tgt)
                    total += loss
                    count += 1
        out[split_name] = total / count
    return out


def init_weights(vocab_size):
    """V×V table of Gaussian(0, 1) floats. Same default as nn.Embedding."""
    return [[random.gauss(0, 1) for _ in range(vocab_size)] for _ in range(vocab_size)]


def train(
    weight,
    train_data,
    val_data,
    vocab_size,
    batch_size,
    block_size,
    max_iters,
    eval_interval,
    eval_iters,
    learning_rate,
):
    """Mutates `weight` in place. Each iteration: forward → manual grads → SGD step."""
    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(
                weight, train_data, val_data, batch_size, block_size, eval_iters
            )
            print(f"step {it:4d}  train {losses['train']:.4f}  val {losses['val']:.4f}")
            log_event(
                "train_step",
                iter=it,
                train_loss=losses["train"],
                val_loss=losses["val"],
            )

        x_batch, y_batch = get_batch(train_data, batch_size, block_size)

        # grads[i][j] accumulates the gradient on weight[i][j] across the batch.
        grads = [[0.0] * vocab_size for _ in range(vocab_size)]
        n_examples = 0

        for x_seq, y_seq in zip(x_batch, y_batch):
            for inp, tgt in zip(x_seq, y_seq):
                probs, _ = forward_one(weight, inp, tgt)
                grad_row = bigram_backward(probs, tgt)
                for j in range(vocab_size):
                    grads[inp][j] += grad_row[j]
                n_examples += 1

        # optimizer.step() for SGD: weight -= lr * (mean grad).
        for i in range(vocab_size):
            if any(g != 0.0 for g in grads[i]):
                weight[i] = sgd_step(weight[i], grads[i], learning_rate, n_examples)


def generate(weight, start_id, max_new_tokens):
    """Greedy-with-randomness: softmax → weighted random draw → repeat."""
    ids = [start_id]
    for step in range(max_new_tokens):
        probs = softmax(weight[ids[-1]])
        r = random.random()
        cum = 0.0
        for j, p in enumerate(probs):
            cum += p
            if r < cum:
                ids.append(j)
                log_event("generate_token", step=step, prev_id=ids[-2], sampled_id=j)
                break
    return ids


def main():
    # ---------- Hyperparameters ----------
    batch_size = 32
    block_size = 8
    max_iters = 5100
    eval_interval = 50
    eval_iters = 20
    # SGD without Adam's adaptive scaling needs a much higher lr than 1e-2.
    learning_rate = 1.0
    random.seed(1337)

    # ---------- Data ----------
    data, vocab, _, itos = load_corpus("universe_50_data.txt")
    vocab_size = len(vocab)
    print(f"vocab size: {vocab_size}")
    print(f"total tokens: {len(data):,}")
    train_data, val_data = split_data(data)

    # ---------- Model ----------
    weight = init_weights(vocab_size)
    print(f"params: {vocab_size * vocab_size:,}")

    # ---------- Train ----------
    train(
        weight,
        train_data,
        val_data,
        vocab_size,
        batch_size,
        block_size,
        max_iters,
        eval_interval,
        eval_iters,
        learning_rate,
    )

    # ---------- Sample ----------
    print("\n--- sample ---")
    print(decode(itos, generate(weight, 0, 40)))


if __name__ == "__main__":
    main()
