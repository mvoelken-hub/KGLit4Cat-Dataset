---
type: source
title: "Toolformer: Language Models Can Teach Themselves to Use Tools"
created: 2026-06-26
updated: 2026-06-26
sources: [toolformer]
tags: [llm, agent, tool-use]
---

# Toolformer

**Authors:** Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, et al.
**Year:** 2023
**Venue:** NeurIPS 36
**File:** `docs/literature/2302.04761v1.pdf`

## Abstract
Introduces a self-supervised method for teaching language models to call external APIs. The model learns when to call tools, which arguments to pass, and how to incorporate tool outputs into future token prediction, using only a handful of demonstrations per API.

## Key Findings
- LMs can learn tool-use behavior from self-supervised API-call annotations.
- Tools include calculator, QA, search, translation, and calendar APIs.
- Tool use improves zero-shot performance without sacrificing language modeling ability.
- Provides an early template for agentic LLM systems.

## Relevance
Background for LLM agents in scientific workflows, including retrieval calls, ontology lookup, validation tools, and extraction pipeline orchestration. Relevant to [[automated-scientific-discovery]] and [[llm-guided-catalyst-discovery]].
