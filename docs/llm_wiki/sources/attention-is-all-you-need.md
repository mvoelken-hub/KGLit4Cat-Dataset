---
type: source
title: "Attention Is All You Need"
created: 2026-06-26
updated: 2026-06-26
sources: [attention-is-all-you-need]
tags: [llm, transformer, foundational]
---

# Attention Is All You Need

**Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, et al.
**Year:** 2017
**Venue:** NeurIPS 30
**File:** `docs/literature/NIPS-2017-attention-is-all-you-need-Paper.pdf`

## Abstract
Introduces the Transformer architecture, replacing recurrent and convolutional sequence models with attention-only encoder-decoder layers. The paper establishes self-attention as a parallelizable way to model token dependencies and becomes the architectural basis for later BERT-style encoders, GPT-style decoders, and contemporary LLMs.

## Key Findings
- Multi-head self-attention can replace recurrence for sequence transduction.
- Positional encodings supply order information without recurrent state.
- The architecture improves machine translation quality while reducing training cost.
- Provides the core attention mechanism used by later scientific NLP and LLM systems.

## Relevance
Foundational architecture for almost every LLM-based extraction, RAG, and agentic workflow in this wiki. Relevant to [[llm-foundations]] and downstream systems such as [[bert]], [[scibert]], and [[gpt3-few-shot-learners]].
