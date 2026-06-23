---
type: source
title: "Bayesian Optimization of Catalysis with In-Context Learning"
created: 2025-06-23
updated: 2025-06-23
sources: [bo-icl-bayesian-optimization-catalysis]
tags: [catalysis, llm, bayesian-optimization, materials-discovery]
---

# Bayesian Optimization of Catalysis with In-Context Learning (BO-ICL)

**Authors:** Mayk Caldas Ramos, Shane S. Michtavy, Marc D. Porosoff, Andrew D. White
**Year:** 2023 (v2: May 2025)
**Venue:** arXiv / FutureHouse Inc.
**File:** `docs/literature/bayesian-optimization-of-catalysis-with-in-context-learning-2304.05341v2.pdf`

## Abstract
LLMs can perform accurate classification with zero/few examples through in-context learning. This work extends that to **regression with uncertainty estimation** using frozen LLMs (GPT-3.5, Gemini), enabling Bayesian optimization in natural language without explicit model training or feature engineering. Applied to catalyst synthesis and testing procedures represented as natural language prompts.

## Key Findings
- BO-ICL matches or outperforms Gaussian processes on benchmarks (aqueous solubility, oxidative coupling of methane)
- In live experiments on reverse water-gas shift (RWGS) reaction, BO-ICL identifies near-optimal multi-metallic catalysts within **6 iterations** from 3,700 candidates
- Operates directly in language space — no structural or electronic descriptors needed
- Task-agnostic: same workflow applies across catalysis, materials science, and AI

## Methodology
- Represents experimental catalyst synthesis and testing procedures as natural language prompts
- Uses frozen LLMs for regression with uncertainty via in-context learning
- Standard BO loop with LLM as surrogate model

## Relevance
Pioneering work showing LLMs can serve as surrogate models in Bayesian optimization for catalysis without feature engineering. Directly relevant to [[llm-guided-catalyst-discovery]] and [[bayesian-optimization-for-catalysis]].