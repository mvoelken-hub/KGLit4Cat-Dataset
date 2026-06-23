---
type: source
title: "GOLLuM: Large Language Models as Uncertainty-Calibrated Optimizers for Experimental Discovery"
created: 2025-06-23
updated: 2025-06-23
sources: [gollum-uncertainty-calibrated-llm]
tags: [catalysis, llm, bayesian-optimization, materials-discovery]
---

# GOLLuM: Gaussian Process Optimized LLMs

**Authors:** Bojana Ranković, Ryan-Rhys Griffiths, Philippe Schwaller
**Year:** 2025
**Venue:** arXiv (EPFL / NCCR Catalysis)
**File:** `docs/literature/large-language-models-as-uncertainty-calibrated-optimizers-for-experimental-discovery-2504.06265v3.pdf`

## Abstract
First framework to train language models through the same probabilistic objective as Bayesian optimization. LLM embeddings serve as GP inputs; learning signals flow back from GP to update LLM. Transforms LLM overconfidence into a precise calibration mechanism.

## Key Findings
- Nearly doubles discovery rate of high-yielding conditions: 24% → 43% in 50 iterations (Buchwald-Hartwig reactions)
- Across 19 diverse optimization problems (organic synthesis, materials science, catalysis, process chemistry, molecular design), ranks first on average
- Uses fixed hyperparameters tuned once on a single dataset
- LLM latent space self-organizes: groups reactions by key chemical components (iodides in high-yield regions, bromides/chlorides in low-yield)

## Methodology
- Joint optimization: LLM embeddings → GP surrogate → learning signals back to LLM
- Bidirectional mechanism: LLM learns to encode experimental performance rather than textual similarity
- Transfers across domains without re-engineering features
- Natural language descriptions of experiments as input

## Relevance
State-of-the-art LLM+BO integration. Extends [[bo-icl-bayesian-optimization-catalysis]] with uncertainty calibration. Relevant to [[llm-guided-catalyst-discovery]], [[bayesian-optimization-for-catalysis]], and [[chemical-llms]].