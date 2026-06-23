---
type: source
title: "ChemDFM-R: A Chemical Reasoning LLM Enhanced with Atomized Chemical Knowledge"
created: 2025-06-23
updated: 2025-06-23
sources: [chemdfm-r]
tags: [llm, chemical-reasoning, materials-discovery]
---

# ChemDFM-R: Chemical Reasoning LLM

**Authors:** Zihan Zhao, Bo Chen, Ziping Wan, Lu Chen, et al. (Shanghai Jiao Tong University / Suzhou Laboratory)
**Year:** 2025
**Venue:** arXiv
**File:** `docs/literature/2507.21990v3.pdf`

## Abstract
Develops a chemical reasoning LLM (ChemDFM-R) by constructing a dataset of atomized chemical knowledge (functional groups in molecules and their changes during reactions). Proposes mixed-source distillation combining atomized knowledge with general reasoning skills, followed by domain-specific reinforcement learning.

## Key Findings
- ChemFG corpus: 101 billion tokens from 12M literature, 30M molecules, 7M reactions
- Atomized knowledge = functional group presence and changes during reactions
- Mixed-source distillation: combines domain expertise with general reasoning LLM capabilities
- Domain-specific RL after distillation enhances chemical reasoning
- Cutting-edge performance on chemical benchmarks with interpretable, rationale-driven outputs

## Methodology
- Toolkit to identify functional groups and their changes in reactions
- Pre-training corpus enrichment with atomized knowledge
- Mixed-source distillation (domain expertise + general reasoning)
- Domain-specific reinforcement learning

## Relevance
Key contribution to [[chemical-llms]] and [[chemical-reasoning-with-llms]]. Addresses the shallow domain understanding problem in chemical LLMs.