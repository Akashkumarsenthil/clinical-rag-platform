# ADR-002: Default Chunking Strategy

**Status**: Accepted  
**Date**: 2024-06-05  
**Deciders**: Platform team

---

## Context

Chunking determines the granularity of information stored in the vector store.
The wrong strategy causes either context fragmentation (chunks too small, clinical
facts split across boundaries) or context dilution (chunks too large, irrelevant
text drowning out the relevant signal).

Three strategies were evaluated: recursive character splitting, semantic splitting,
and sliding window.

## Decision

**Use `RecursiveCharacterChunker` as the default strategy** (512 tokens, 50-token overlap).

## Strategy Comparison

| Strategy | Faithfulness | Answer Relevancy | Context Precision | Latency |
|---|---|---|---|---|
| **Recursive (default)** | **0.872** | 0.791 | **0.844** | 1 240 ms |
| Semantic | 0.831 | **0.834** | 0.797 | 2 870 ms |
| Sliding window | 0.798 | 0.762 | 0.771 | 1 105 ms |

Full details in [chunking_benchmarks.md](../chunking_benchmarks.md).

## Rationale

1. **Highest faithfulness (0.872)**: Clinical facts are mostly contained within
   single paragraphs.  Recursive splitting on `\n\n` → `\n` → `. ` → ` ` naturally
   respects these boundaries without requiring inference.
2. **Best context precision (0.844)**: Fewer spurious chunks in the context window
   reduce hallucination risk, which is critical in a clinical setting.
3. **Low latency (1 240 ms)**: No embedding inference at chunk time; only tokenization.
4. **Configurability**: The strategy is exposed via `CHUNK_STRATEGY` env var and the
   `/ingest` API parameter, so operators can override per-document.

## Consequences

- Semantic and sliding-window strategies are implemented and accessible via the
  factory `get_chunker(strategy)` for use in experiments and benchmarks.
- Chunk size (512 tokens) and overlap (50 tokens) are tunable; larger sizes
  may improve recall for long-form clinical guidelines.
- The 50-token overlap reduces boundary-cut information loss at marginal storage cost.
