---
type: source
title: "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
created: 2026-06-26
updated: 2026-06-26
sources: [self-rag]
tags: [llm, rag, self-reflection]
---

# Self-RAG

**Authors:** Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi
**Year:** 2024
**Venue:** ICLR
**File:** `docs/literature/6283_Self_RAG_Learning_to_Retr.pdf`

## Abstract
Introduces Self-Reflective Retrieval-Augmented Generation, where a language model learns to retrieve on demand and critique retrieved passages and its own generations through special reflection tokens. The approach improves factuality, citation quality, and task performance relative to standard RAG baselines.

## Key Findings
- Retrieval is adaptive rather than always-on.
- Reflection tokens let the model decide when evidence is needed and whether passages are useful.
- Critique behavior helps select better generations and improve factuality.
- Strong gains are reported for open-domain QA, reasoning, fact verification, and long-form generation.

## Relevance
Important for evidence-aware extraction workflows: retrieval should be invoked when source grounding is useful, and outputs should be judged against retrieved evidence. Relevant to [[retrieval-augmented-generation]] and [[structured-output-reliability]].
