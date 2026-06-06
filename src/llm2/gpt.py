"""
Stage 1: Bigram base.

Predicts the next token using ONLY the current token. No attention, no context.
This is the loss floor every later stage must beat.
"""

# torch: the tensor + autograd + GPU library. Everything we do hangs off this.
import torch

# nn: pre-built building blocks (Embedding, Linear, etc.) that register their
# weights as trainable parameters automatically.
import torch.nn as nn

# F: the *functional* form of the same ops (no state). We use F.cross_entropy
# and F.softmax — they don't have weights, so they don't need to be modules.
from torch.nn import functional as F

# ---------- Hyperparameters ----------
# How many independent sequences we process in parallel each step. Bigger = more stable gradients but more memory. 32 is fine on MPS for this tiny model.
batch_size = 32
# Context length: how many tokens each sequence in the batch contains. The bigram model only ever looks at the LAST token, so block_size doesn't really matter for stage 1 — but later stages will use it heavily.
block_size = 8
# Total training steps. Each step = one forward + backward + optimizer update.
max_iters = 3000
# How often to print train/val loss. Every 300 steps = 10 prints total.
eval_interval = 300
# When estimating loss, average over this many random batches to smooth noise.
eval_iters = 100
# Learning rate: how big a step the optimizer takes per gradient. 1e-2 is
# aggressive; works for tiny models, would explode on big ones.
learning_rate = 1e-2
# Use Apple Silicon GPU if available, else fall back to CPU. MPS = Metal Performance Shaders, Apple's CUDA equivalent.
device = "mps" if torch.backends.mps.is_available() else "cpu"
# Seed the RNG so runs are reproducible. Same seed → same weights → same loss.
torch.manual_seed(1337)

# ---------- Load + tokenize ----------
# Read the entire 2MB corpus into memory as one big string.
with open("universe_50_data.txt", "r") as f:
    text = f.read()
# Word-level tokenizer. Period is its own token. We replace "." with " ." so split() treats the period as a separate word. Without this, "rug." and "rug" would be different tokens — wasteful.
tokens = text.replace(".", " .").split()
# Build the vocabulary: every UNIQUE token, sorted alphabetically for stability.
vocab = sorted(set(tokens))
# Number of distinct tokens. The model's input/output dimensions depend on this.
vocab_size = len(vocab)
print(f"vocab size: {vocab_size}")
print(f"total tokens: {len(tokens):,}")
# stoi = "string to integer". Maps each word to a unique ID (0 to vocab_size-1).
stoi = {w: i for i, w in enumerate(vocab)}
# itos = "integer to string". The inverse, used to decode model output back to text.
itos = {i: w for w, i in stoi.items()}


# Helper: turn a Python string into a list of token IDs.
def encode(s):
    return [stoi[w] for w in s.replace(".", " .").split()]


# Helper: turn a list of token IDs back into a readable string.
def decode(ids):
    return " ".join(itos[i] for i in ids)


# Convert the entire corpus to a 1-D tensor of token IDs. dtype=long because embeddings expect integer indices, and Python ints default to int64 = "long".
data = torch.tensor([stoi[t] for t in tokens], dtype=torch.long)
# 90/10 train/val split. Validation loss tells us if we're overfitting.
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# Sample a random batch of (input, target) pairs from the chosen split.
def get_batch(split):
    # Pick the right tensor.
    d = train_data if split == "train" else val_data
    # Random starting indices — one per sequence in the batch.
    # We subtract block_size so we don't run off the end.
    ix = torch.randint(len(d) - block_size, (batch_size,))
    # x = input sequences. Each row is `block_size` consecutive tokens.
    x = torch.stack([d[i : i + block_size] for i in ix])
    # y = target sequences. Same as x but shifted right by 1 — the model's job
    # is to predict y[t] given x[:t+1]. For position t, target is the next token.
    y = torch.stack([d[i + 1 : i + 1 + block_size] for i in ix])
    # Move both tensors onto the GPU/MPS so the model can use them.
    return x.to(device), y.to(device)


# ---------- Model ----------
class BigramLM(nn.Module):
    """Each token directly indexes a row of logits over the vocab. That's it."""

    def __init__(self, vocab_size):
        # Required so nn.Module can track our parameters.
        super().__init__()
        # The ENTIRE model: a (vocab_size, vocab_size) lookup table. Row i = "logits over the next token, given that the current token is i." No weights are shared, no context, no attention — pure bigram statistics learned via backprop. After training, row i ≈ log P(next | current=i).
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx is (B, T) — batch of B sequences, each T tokens long. Looking up each token gives us its row of logits → shape (B, T, vocab_size).
        logits = self.token_embedding(idx)
        # If no targets passed (e.g., during generation), just return raw logits.
        if targets is None:
            return logits, None
        # Reshape for cross_entropy: it expects (N, C) predictions and (N,) targets.
        # We flatten batch + time into one dimension. Each (token, next-token) pair
        # is treated as an independent classification problem.
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    # @torch.no_grad disables gradient tracking — saves memory + speed during
    # generation since we're not training, just sampling.
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # Generate one token at a time, max_new_tokens times.
        for _ in range(max_new_tokens):
            # Forward pass on the current sequence so far.
            logits, _ = self(idx)
            # We only care about predictions for the LAST time step — that's
            # the next-token distribution. Shape: (B, vocab_size).
            logits = logits[:, -1, :]
            # Convert logits to a proper probability distribution (sums to 1).
            probs = F.softmax(logits, dim=-1)
            # Sample one token according to those probabilities. Multinomial =
            # weighted random draw. NOT argmax — that would make output deterministic.
            next_id = torch.multinomial(probs, num_samples=1)
            # Append the new token to our running sequence and loop again.
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ---------- Train ----------
# Decorator: this whole function runs without building the autograd graph.
# Estimating loss doesn't need gradients, and skipping them is a big speedup.
@torch.no_grad()
def estimate_loss(model):
    # eval() disables dropout / changes BatchNorm behavior. Doesn't matter for
    # this model, but it's a habit worth keeping.
    model.eval()
    out = {}
    # Compute mean loss on both train and val splits separately.
    for split in ["train", "val"]:
        # Pre-allocate a tensor to hold each batch's loss.
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    # Switch back to training mode before returning.
    model.train()
    return out


# Instantiate the model and move it to the GPU/MPS.
model = BigramLM(vocab_size).to(device)
# Sanity check: how many trainable parameters? Should be vocab_size² for bigram.
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

# AdamW: Adam optimizer with decoupled weight decay. The standard choice for
# transformers. It adapts per-parameter learning rates based on gradient history.
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Main training loop.
for it in range(max_iters):
    # Periodically print loss so we can watch training progress.
    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss(model)
        print(f"step {it:4d}  train {losses['train']:.4f}  val {losses['val']:.4f}")
    # Sample a fresh batch of training data.
    x, y = get_batch("train")
    # Forward pass: compute predictions and loss.
    _, loss = model(x, y)
    # Clear stale gradients from the previous step. set_to_none=True is slightly
    # faster than zeroing because it deallocates the tensors.
    optimizer.zero_grad(set_to_none=True)
    # Backward pass: compute ∂loss/∂param for every parameter via autograd.
    loss.backward()
    # Apply the gradients: param ← param − lr · grad (with Adam's adaptive scaling).
    optimizer.step()

# ---------- Sample ----------
print("\n--- sample ---")
# Start with a single token (id 0, whatever that maps to alphabetically).
# Shape (1, 1) = batch of 1, sequence length 1.
start = torch.zeros((1, 1), dtype=torch.long, device=device)
# Generate 40 new tokens and decode the whole sequence to text.
print(decode(model.generate(start, 40)[0].tolist()))
