---
type: concept
title: "LLM-Guided Catalyst Discovery"
created: 2025-06-23
updated: 2025-06-23
sources: [bo-icl-bayesian-optimization-catalysis, catalm-catalyst-design-llm, monte-carlo-thought-search, adsorb-agent, gollum-uncertainty-calibrated-llm, catminer-llm-catalysis-extraction, knowledge-extraction-catalysis-literature, chemdfm-r, alchemybench, catalyst-informatics, tos-catalyst-reactivity-ai, schilling2025text, zhang2024chemicaltextmining, dagdelen2024structured]
tags: [catalysis, llm, materials-discovery, bayesian-optimization, agent]
---

# LLM-Guided Catalyst Discovery

## Overview
A rapidly emerging paradigm where large language models serve as reasoning engines, surrogate models, or agents to accelerate catalyst discovery and optimization. The literature shows multiple complementary approaches: LLMs as Bayesian optimization surrogates, LLM agents for configuration search, structured reasoning via tree search, and fine-tuned domain LLMs.

## Key Approaches

### LLMs as Surrogate Models in Bayesian Optimization
- **BO-ICL** ([[bo-icl-bayesian-optimization-catalysis]]): Frozen LLMs perform regression with uncertainty via in-context learning. Matches GPs on benchmarks, finds near-optimal catalysts in 6 iterations from 3,700 candidates (RWGS reaction).
- **GOLLuM** ([[gollum-uncertainty-calibrated-llm]]): Trains LLM embeddings through GP objective. Bidirectional LLM↔GP. Doubles discovery rate (24%→43% in 50 iterations). Ranks first on average across 19 diverse problems.

### LLM Agents for Scientific Search
- **ChemReasoner / Monte Carlo Thought Search** ([[monte-carlo-thought-search]]): MCTS over prompt space for combinatorial catalyst search. 25.8% improvement over best baseline. Tree-structured prompts add constraints (cost, efficiency, selectivity).
- **Adsorb-Agent** ([[adsorb-agent]]): LLM agent for adsorption configuration search. 84% comparable energies, 35% lower energies vs. exhaustive search. GPT-4o best reasoning engine.

### Domain-Specialized LLMs
- **CataLM** ([[catalm-catalyst-design-llm]]): Fine-tuned LLM for electrocatalytic materials. First domain-specific LLM for catalysis.
- **ChemDFM-R** ([[chemdfm-r]]): Chemical reasoning LLM with atomized functional-group knowledge. 101B-token corpus. Mixed-source distillation + domain RL.

### Data-Driven Catalyst Design
- **Catalyst Informatics** ([[catalyst-informatics]]): Vision for data-driven catalyst design with databases, knowledge extraction, and platforms.
- **Time-on-Stream Modeling** ([[tos-catalyst-reactivity-ai]]): Subgroup discovery + SISSO symbolic regression for Pd catalysts. Experimental validation.

## Synthesis
The field is converging on several key insights:
1. **Natural language as representation**: Eliminates feature engineering; LLMs encode chemical intuition from pre-training
2. **Uncertainty matters**: GOLLuM shows calibrated uncertainty is critical for real experimental campaigns
3. **Structured search beats direct prompting**: MCTS (ChemReasoner) and agent-based search (Adsorb-Agent) outperform naive prompts
4. **Domain specialization helps**: Fine-tuning on catalysis literature (CataLM, ChemDFM-R) improves over general LLMs
5. **Benchmarks are emerging**: AlchemyBench ([[alchemybench]]) provides end-to-end evaluation for materials synthesis

## Additional Methods

### Structured Extraction with Fine-Tuned LLMs
- [[dagdelen2024structured]]: Fine-tuned GPT-3/Llama-2 for joint NER + relation extraction in materials chemistry. Output as JSON or English sentences. Simple, flexible, no pipeline needed.
- [[zhang2024chemicaltextmining]]: Fine-tuned GPT-3.5/Mistral/Llama3 on five chemical text mining tasks. 69–95% accuracy with minimal annotated data. Outperforms task-adaptive pre-training.
- [[schilling2025text]]: Tutorial review of LLM-based chemical data extraction. End-to-end workflows, multimodal approaches, agentic systems, quality assurance.

## Open Questions
- How to combine domain-specialized LLMs with structured search (MCTS/agents)?
- Can LLM surrogates transfer across catalysis subdomains (heterogeneous → electro → photo)?
- What reporting standards are needed to make LLM-extracted data reliable? (See [[fair-data-for-catalysis]])