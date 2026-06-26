---
type: source
title: "SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer"
created: 2026-06-26
updated: 2026-06-26
sources: [sentencepiece]
tags: [llm, tokenization, foundational]
---

# SentencePiece

**Authors:** Taku Kudo and John Richardson
**Year:** 2018
**Venue:** EMNLP System Demonstrations
**File:** `docs/literature/D18-2012.pdf`

## Abstract
Describes a language-independent tokenizer and detokenizer for neural text processing. SentencePiece trains subword models directly from raw text and supports BPE and unigram language-model segmentation, reducing dependence on language-specific preprocessing.

## Key Findings
- Trains subword tokenizers directly from raw sentences.
- Supports language-independent tokenization and detokenization.
- Enables end-to-end neural NLP pipelines without external word segmenters.
- Became a common tokenizer component for multilingual and LLM systems.

## Relevance
Background for how LLMs represent scientific terms, chemical names, and identifiers. Tokenization affects extraction quality, schema labels, rare material names, and chunking behavior in SIMONE-like workflows. Relevant to [[llm-foundations]].
