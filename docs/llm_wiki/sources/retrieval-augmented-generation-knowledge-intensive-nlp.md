---
type: source
title: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
created: 2026-06-26
updated: 2026-06-26
sources: [retrieval-augmented-generation-knowledge-intensive-nlp]
tags: [llm, rag, information-retrieval]
---

# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

**Authors:** Patrick Lewis, Ethan Perez, Aleksandra Piktus, et al.
**Year:** 2020
**Venue:** NeurIPS 33
**File:** `docs/literature/NeurIPS-2020-retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks-Paper.pdf`

## Abstract
Introduces RAG models that combine a pre-trained seq2seq generator with a non-parametric dense vector index. Retrieved passages are used as external memory for knowledge-intensive generation, improving factuality, specificity, and provenance relative to parametric-only generation.

## Key Findings
- Combines parametric memory with an editable retrieval index.
- Evaluates variants that condition on fixed retrieved passages or different passages per generated token.
- Improves open-domain QA and produces more factual, specific generations.
- Makes retrieved evidence inspectable, unlike purely parametric memory.

## Relevance
Foundational RAG paper for source-grounded literature extraction, ontology assistance, and thesis support workflows. Relevant to [[retrieval-augmented-generation]], [[knowledge-graphs-for-catalysis]], and [[text-mining-for-catalysis]].
