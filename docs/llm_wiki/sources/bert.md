---
type: source
title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
created: 2026-06-26
updated: 2026-06-26
sources: [bert]
tags: [llm, transformer, text-mining]
---

# BERT

**Authors:** Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
**Year:** 2019
**Venue:** NAACL-HLT
**File:** `docs/literature/N19-1423.pdf`

## Abstract
Introduces BERT, a bidirectional Transformer encoder pre-trained with masked language modeling and next-sentence prediction. BERT can be fine-tuned with minimal task-specific layers for tasks such as question answering, language inference, and named entity recognition.

## Key Findings
- Deep bidirectional pre-training improves language understanding tasks.
- Fine-tuning reuses a common encoder across many downstream NLP tasks.
- Provides a template for domain-specific encoders such as SciBERT and MatSciBERT.
- Strongly influenced later scientific text-mining pipelines.

## Relevance
Foundation for encoder-based scientific NLP and entity extraction. Relevant to [[llm-foundations]], [[text-mining-for-catalysis]], [[scibert]], and [[gupta2022matscibert]].
