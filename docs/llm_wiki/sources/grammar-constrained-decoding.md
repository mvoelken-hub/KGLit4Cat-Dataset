---
type: source
title: "Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning"
created: 2026-06-26
updated: 2026-06-26
sources: [grammar-constrained-decoding]
tags: [llm, structured-output, text-mining]
---

# Grammar-Constrained Decoding

**Authors:** Saibo Geng, Martin Josifoski, Maxime Peyrard, Robert West
**Year:** 2023
**Venue:** EMNLP
**File:** `docs/literature/2023.emnlp-main.674.pdf`

## Abstract
Shows that formal grammars can constrain LLM generation for a broad set of structured NLP tasks without task-specific fine-tuning. Input-dependent grammars allow output spaces to vary with the input, supporting information extraction, entity disambiguation, and parsing.

## Key Findings
- Grammar constraints guarantee valid output structures.
- Input-dependent grammars support flexible task-specific output spaces.
- Constrained LMs outperform unconstrained LMs and can match or beat task-specific models.
- Especially useful when training data is scarce or fine-tuning is expensive.

## Relevance
Directly relevant to schema-conforming extraction and ontology-grounded output. For SIMONE-like systems, this supports validating generated entities, relations, and metadata patches against formal grammars or schemas. See [[structured-output-reliability]] and [[text-mining-for-catalysis]].
