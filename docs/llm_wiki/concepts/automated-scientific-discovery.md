---
type: concept
title: "Automated Scientific Discovery"
created: 2025-06-23
updated: 2026-06-26
sources: [bo-icl-bayesian-optimization-catalysis, gollum-uncertainty-calibrated-llm, monte-carlo-thought-search, adsorb-agent, autonomous-protein-engineering, alchemybench, kg-survey-ai-for-science, catalyst-informatics, toolformer, self-rag, llm-survey-zhao2026]
tags: [materials-discovery, llm, agent, catalysis]
---

# Automated Scientific Discovery

## Overview
The vision of AI-driven, closed-loop scientific experimentation — where AI proposes, executes, and learns from experiments autonomously. Catalysis and materials science are prime domains for this paradigm due to large design spaces and expensive experiments.

## Closed-Loop Optimization

### Bayesian Optimization with LLMs
- [[bo-icl-bayesian-optimization-catalysis]]: BO-ICL finds near-optimal catalysts in 6 iterations from 3,700 candidates
- [[gollum-uncertainty-calibrated-llm]]: GOLLuM doubles discovery rate (24%→43%) with calibrated uncertainty across 19 domains

### Structured Reasoning Search
- [[monte-carlo-thought-search]]: MCTS over prompt space for combinatorial catalyst reasoning. 25.8% improvement.
- [[adsorb-agent]]: LLM agent for adsorption configuration identification, reducing DFT computations

## Autonomous Platforms
[[autonomous-protein-engineering]] demonstrates the DBTL (Design-Build-Test-Learn) cycle in protein engineering:
- Zero-shot protein language models for mutation prediction
- Automated expression and assay pipelines
- ML-driven decision-making
- Engineering cycles: months → days
- Vision: distributed self-driving laboratories with FAIR data + KGs

## Catalyst Informatics Vision
[[catalyst-informatics]] frames the three pillars:
1. Experimental catalysts database
2. Knowledge extraction via data science
3. Catalysts informatics platform
Future: ontology-driven catalyst informatics

## Benchmarks and Evaluation
- [[alchemybench]]: AlchemyBench with 17K expert-verified synthesis recipes + LLM-as-a-Judge evaluation
- LLM-as-a-Judge shows strong statistical agreement with expert assessments

## Vision: AI Scientists
[[kg-survey-ai-for-science]] envisions:
- Self-updating SciKGs co-evolving with LLMs
- AI scientists as core infrastructure for autonomous discovery
- SciKGs as knowledge infrastructure; LLMs as dynamic semantic engines
- Self-evolving, auditable, interoperable knowledge graphs

## Tool Use and Retrieval
- [[toolformer]] provides an early general pattern for LMs learning when and how to call external tools.
- [[self-rag]] adds adaptive retrieval and critique, an important pattern for agents that must decide when source evidence is required.
- [[llm-survey-zhao2026]] frames agentic reasoning and tool use as part of modern LLM utilization.

## Key Enablers
1. **FAIR data** (see [[catalysis-data-infrastructure]]) — without data, no learning
2. **Knowledge graphs** (see [[knowledge-graphs-for-catalysis]]) — structured knowledge for reasoning
3. **Uncertainty quantification** — critical for real experimental decisions
4. **Domain knowledge** — chemical reasoning (see [[chemical-llms]])
