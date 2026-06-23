---
type: entity
title: "Catalysis Benchmarks and Datasets"
created: 2025-06-23
updated: 2025-06-23
sources: [alchemybench, knowledge-extraction-catalysis-literature, monte-carlo-thought-search, catminer-llm-catalysis-extraction, cattesthub, dagdelen2024structured]
tags: [benchmark, catalysis, llm]
---

# Catalysis Benchmarks and Datasets

## Overview
Benchmarks and datasets for evaluating AI/LLM approaches in catalysis and materials science.

## Datasets

### Open Materials Guide (OMG) / AlchemyBench ([[alchemybench]])
- 17K expert-verified synthesis recipes from open-access literature
- AlchemyBench: end-to-end benchmark for synthesis prediction
  - Raw materials and equipment prediction
  - Synthesis procedure generation
  - Characterization outcome forecasting
- LLM-as-a-Judge evaluation framework
- https://github.com/HeegyuKim/AlchemyBench

### Catalysis IE Benchmark ([[knowledge-extraction-catalysis-literature]])
- First benchmark dataset for catalysis information extraction
- Full-text journal articles
- Six entity types: catalyst, reaction, reactant, product, characterization, treatment
- 90% extraction accuracy

### BioFuelQR ([[monte-carlo-thought-search]])
- Complex reasoning questions and answers from catalysis researchers
- About novel chemical conversion processes
- Part of ChemReasoner project
- https://github.com/pnnl/chemreasoner

### Computational Chemistry Simulations ([[monte-carlo-thought-search]])
- Curation of computational chemistry simulations
- Companion dataset to BioFuelQR

## Notable Benchmarks Referenced
- **Aqueous solubility**: Used in [[bo-icl-bayesian-optimization-catalysis]]
- **Oxidative coupling of methane (OCM)**: Used in BO-ICL and [[catminer-llm-catalysis-extraction]]
- **Buchwald-Hartwig reactions**: Used in [[gollum-uncertainty-calibrated-llm]]
- **Suzuki-Miyaura cross couplings**: Referenced in GOLLuM
- **Reverse water-gas shift (RWGS)**: Live experiment in BO-ICL

## Gaps
- No comprehensive catalysis KG benchmark for LLM evaluation
- Need for standardized evaluation protocols across approaches
- Most datasets are domain-specific; transfer is not benchmarked