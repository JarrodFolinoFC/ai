# llm2

Building a GPT in seven progressive stages. Stage 1 is a bigram baseline implemented twice — once in PyTorch (`gpt.py`) and once in pure Python (`gpt_pure.py`) so every operation hidden inside `nn.Embedding`, `F.softmax`, `F.cross_entropy`, `loss.backward()`, and `optimizer.step()` is visible.

## Setup

```bash
uv sync --extra dev
```

## Run

```bash
uv run python -m llm2.gpt_pure
uv run python -m llm2.gpt
```

## Test

```bash
uv run pytest
```

## Logging

`gpt_pure.py` ships sampled events to OpenObserve on `localhost:5080`.

```bash
GPT_PURE_LOG=1 GPT_PURE_LOG_SAMPLE=0.05 uv run python -m llm2.gpt_pure
```
