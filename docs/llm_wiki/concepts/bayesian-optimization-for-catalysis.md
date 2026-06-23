---
type: concept
title: "Bayesian Optimization for Catalysis"
created: 2025-06-23
updated: 2025-06-23
sources: [bo-icl-bayesian-optimization-catalysis, gollum-uncertainty-calibrated-llm, ifac-optimization]
tags: [bayesian-optimization, catalysis, llm, materials-discovery]
---

# Bayesian Optimization for Catalysis

## Overview
Bayesian optimization (BO) is the standard approach for sample-efficient optimization of expensive black-box functions — ideal for catalyst discovery where each experiment is costly. Recent work integrates LLMs to eliminate feature engineering and leverage chemical knowledge.

## Traditional BO
- Surrogate model: Gaussian Process (GP) provides predictions + uncertainty
- Acquisition function: balances exploration vs. exploitation
- Challenge: requires domain-specific feature representations (descriptors)
- Problem: descriptors don't transfer across domains; each new domain starts from scratch

## LLM-Enhanced BO

### BO-ICL ([[bo-icl-bayesian-optimization-catalysis]])
- Frozen LLMs as surrogate models via in-context learning
- Natural language prompts represent experiments (synthesis + reaction parameters)
- No feature engineering, no model training
- Matches GPs on aqueous solubility and OCM benchmarks
- Live experiment: near-optimal RWGS catalysts in 6/3,700 candidates

### GOLLuM ([[gollum-uncertainty-calibrated-llm]])
- LLM embeddings → GP surrogate → learning signals back to LLM
- Bidirectional: LLM learns to encode experimental performance
- Calibrated uncertainty (not overconfident)
- 24% → 43% discovery rate (Buchwald-Hartwig, 50 iterations, 10 starts)
- Ranks first on average across 19 diverse problems
- Fixed hyperparameters (tuned once on single dataset)
- LLM latent space self-organizes by chemical patterns

## Key Advances
1. **Natural language as representation space**: Eliminates descriptor engineering
2. **Uncertainty calibration**: GOLLuM transforms LLM overconfidence into calibration mechanism
3. **Transferability**: GOLLuM transfers across organic synthesis, materials, catalysis, process chemistry, molecular design
4. **Interpretability**: LLM latent space reveals chemical patterns (iodides in high-yield, bromides/chlorides in low-yield)

## Comparison
| Aspect | Traditional GP | BO-ICL | GOLLuM |
|--------|---------------|--------|--------|
| Features | Domain descriptors | NL prompts | NL embeddings |
| Training | GP fitting | None (frozen) | LLM+GP joint |
| Uncertainty | GP posterior | LLM-based | Calibrated (GP+LLM) |
| Transfer | No | Limited | Yes (19 domains) |
| Feature engineering | Required | None | None |

## Related
- [[ifac-optimization]]: Traditional optimization in chemical process engineering
- [[llm-guided-catalyst-discovery]]: Broader LLM applications in catalyst design