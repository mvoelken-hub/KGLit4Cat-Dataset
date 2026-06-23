---
type: source
title: "Shattering the Shortcut: A Topology-Regularized Benchmark for Multi-hop Medical Reasoning in LLMs"
created: 2025-06-23
updated: 2025-06-23
sources: [zi2026shattering]
tags: [llm, knowledge-graph]
---

# Shattering the Shortcut (ShatterMed-QA)

**Authors:** Xing Zi, Xinying Zhou, Jinghao Xiao, Catarina Moreira, Mukesh Prasad (University of Technology Sydney)
**Year:** 2026
**Venue:** arXiv:2603.12458
**File:** `docs/literature/2603.12458v1.pdf`

## Abstract
Introduces ShatterMed-QA, a bilingual benchmark of 10,558 multi-hop clinical questions for evaluating deep diagnostic reasoning in LLMs. Uses novel k-Shattering algorithm to prune generic hub nodes in medical KGs, severing shortcut learning pathways. Evaluates 21 LLMs — exposes systemic vulnerability to topology-driven distractors (53% Hard Negative Error rate vs 33% random baseline). RAG recovery up to 70%.

## Key Findings
- LLMs exploit highly connected hub nodes (e.g., "inflammation") to bypass authentic reasoning
- k-Shattering algorithm physically prunes generic hubs in KGs
- Implicit bridge entity masking + topology-driven hard negative sampling
- 21 LLMs evaluated; frontier models fall into distractor traps at 53% rate
- RAG recovery validates structural fidelity — proves models have knowledge gaps, not reasoning failures

## Relevance
Relevant to understanding LLM reasoning limitations with KGs. Connected to [[knowledge-graphs-for-catalysis]] (KG topology effects on LLM reasoning).