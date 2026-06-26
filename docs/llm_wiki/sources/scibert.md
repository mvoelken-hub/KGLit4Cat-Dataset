---
type: source
title: "SciBERT: A Pretrained Language Model for Scientific Text"
created: 2026-06-26
updated: 2026-06-26
sources: [scibert]
tags: [llm, text-mining, scientific-nlp]
---

# SciBERT

**Authors:** Iz Beltagy, Kyle Lo, Arman Cohan
**Year:** 2019
**Venue:** EMNLP-IJCNLP
**File:** `docs/literature/D19-1371.pdf`

## Abstract
Presents SciBERT, a BERT-based model pre-trained on a large corpus of scientific publications. The model improves performance on scientific sequence tagging, sentence classification, and parsing tasks, addressing the shortage of labeled scientific NLP data.

## Key Findings
- Domain pre-training on scientific text improves scientific NLP performance.
- Evaluates fine-tuning and frozen-embedding uses across multiple scientific tasks.
- Introduces an in-domain vocabulary and public model weights.
- Serves as a bridge from general BERT to materials-specific models such as MatSciBERT.

## Relevance
Important baseline for scientific text extraction before instruction-tuned LLMs. Relevant to [[text-mining-for-catalysis]], [[chemical-llms]], and [[gupta2022matscibert]].
