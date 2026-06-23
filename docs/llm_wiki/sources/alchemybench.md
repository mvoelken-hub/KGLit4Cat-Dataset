---
type: source
title: "AlchemyBench: Towards Fully-Automated Materials Discovery via Large-Scale Synthesis Dataset and Expert-Level LLM-as-a-Judge"
created: 2025-06-23
updated: 2025-06-23
sources: [alchemybench]
tags: [materials-discovery, llm, benchmark]
---

# AlchemyBench & Open Materials Guide (OMG)

**Authors:** Heegyu Kim, Taeyang Jeon, Seungtaek Choi, et al. (Ajou University / Hanyang University)
**Year:** 2025
**Venue:** arXiv
**File:** `docs/literature/towards-fully-automated-materials-discovery-via-large-scale-synthesis-dataset-and-expert-level-llm-as-a-judge-2502.16457v4.pdf`

## Abstract
Curates 17K expert-verified synthesis recipes from open-access literature (Open Materials Guide / OMG dataset). Develops AlchemyBench, an end-to-end benchmark for LLM-driven synthesis prediction. Proposes LLM-as-a-Judge framework showing strong statistical agreement with expert assessments.

## Key Findings
- OMG: largest materials synthesis dataset (17K high-quality, expert-verified recipes)
- AlchemyBench: end-to-end benchmark covering raw materials/equipment prediction, procedure generation, characterization outcome forecasting
- LLM-as-a-Judge: automated evaluation with strong agreement to expert judgments
- RAG with OMG data improves model performance
- Prior datasets had >92-98% missing essential synthesis parameters

## Methodology
- Expert-verified curation from open-access literature
- Benchmark tasks: synthesis prediction across multiple facets
- LLM-as-a-Judge for scalable automated evaluation
- RAG experiments validate dataset applicability

## Relevance
Important benchmark for [[materials-discovery-with-llms]] and [[catalysis-benchmarks-and-datasets]]. Open-source: https://github.com/HeegyuKim/AlchemyBench