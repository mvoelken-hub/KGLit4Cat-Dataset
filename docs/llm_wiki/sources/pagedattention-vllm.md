---
type: source
title: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
created: 2026-06-26
updated: 2026-06-26
sources: [pagedattention-vllm]
tags: [llm, serving, infrastructure]
---

# PagedAttention / vLLM

**Authors:** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, et al.
**Year:** 2023
**Venue:** SOSP
**File:** `docs/literature/2309.06180v1.pdf`

## Abstract
Presents PagedAttention, an attention algorithm inspired by operating-system virtual memory, and vLLM, a serving system that reduces key-value cache waste for batched LLM inference. The system improves throughput by managing dynamic request memory more efficiently.

## Key Findings
- KV cache memory is a major bottleneck in high-throughput LLM serving.
- Paging-style memory management reduces fragmentation and duplication.
- vLLM improves throughput by roughly 2-4x at similar latency in reported evaluations.
- Gains are especially important for long sequences, large models, and complex decoding.

## Relevance
Infrastructure background for local or hosted LLM extraction services. Relevant when deploying SIMONE-style workflows with long documents, batched extraction, or local open models. See [[llm-foundations]].
