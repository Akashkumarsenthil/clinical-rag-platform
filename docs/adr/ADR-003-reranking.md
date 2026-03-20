# ADR-003: Reranking Strategy

**Status**: Accepted  
**Date**: 2024-06-10  
**Deciders**: Platform team

---

## Context

After hybrid retrieval returns 20 candidate chunks, a reranker scores
query-document relevance more precisely than the initial retrieval scores.
Two main approaches were considered: **CrossEncoder** models and **ColBERT**
(late interaction).

## Decision

**Use `cross-encoder/ms-marco-MiniLM-L-6-v2`** via sentence-transformers.

## Alternatives Considered

| Criterion | CrossEncoder (MiniLM-L-6-v2) | ColBERT (v2) | BM25 re-sort |
|---|---|---|---|
| Deployment complexity | Low — single `.predict()` call | High — requires token-level indices | None |
| Latency (top-20 → top-5) | ~80 ms on CPU | ~300 ms (index lookup) | <5 ms |
| Accuracy (MS MARCO MRR@10) | 0.390 | 0.397 | baseline |
| Memory footprint | ~120 MB | ~2 GB (token index) | negligible |
| GPU required | No | Strongly recommended | No |
| Open-source | ✅ | ✅ | ✅ |

## Rationale

1. **Simpler deployment**: CrossEncoder requires only a standard PyTorch model load.
   ColBERT's late-interaction mechanism requires a separate token-level inverted index
   (PLAID engine) which adds significant operational complexity.
2. **Sufficient quality delta**: The MiniLM-L-6-v2 model achieves MRR@10 of 0.390
   vs ColBERT's 0.397 — a 1.8 % gap that does not justify 10× the infrastructure.
3. **CPU-friendly**: The model scores 20 pairs in ~80 ms on a single CPU core,
   well within the 2-second P99 latency budget.
4. **Top-20 → top-5 funnel**: The coarse retrieval already eliminates poor candidates;
   the reranker only needs to correctly order 20 pre-filtered chunks, a task where
   CrossEncoder is highly effective.

## Consequences

- The model is downloaded from HuggingFace Hub on first startup (~120 MB) and
  cached at `~/.cache/huggingface/`.
- The `CrossEncoderReranker` class is injected via dependency injection for easy
  swap in tests and future model upgrades.
- If latency becomes critical, the model can be served via ONNX Runtime with
  ~3× speedup at the cost of a quantization step.
