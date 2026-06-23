---
type: concept
title: "Chemical LLMs and Domain-Specialized Models"
created: 2025-06-23
updated: 2025-06-23
sources: [catalm-catalyst-design-llm, chemdfm-r, bo-icl-bayesian-optimization-catalysis, gollum-uncertainty-calibrated-llm, catminer-llm-catalysis-extraction, llama3-lora-qlora, monte-carlo-thought-search, adsorb-agent, schilling2025text, zhang2024chemicaltextmining, marconato2025reasoning]
tags: [llm, chemical-reasoning, catalysis]
---

# Chemical LLMs and Domain-Specialized Models

## Overview
General-purpose LLMs lack deep domain knowledge in chemistry and catalysis. Multiple strategies exist for adapting LLMs to chemical domains: fine-tuning, in-context learning, agent architectures, and reasoning-augmented training.

## Fine-Tuned Domain LLMs

### CataLM ([[catalm-catalyst-design-llm]])
- Fine-tuned open-source LLM for electrocatalytic materials
- First LLM dedicated to catalyst domain
- Enables conversational catalyst knowledge exploration and design

### ChemDFM-R ([[chemdfm-r]])
- Chemical reasoning LLM with atomized functional-group knowledge
- ChemFG corpus: 101B tokens, 12M literature, 30M molecules, 7M reactions
- Mixed-source distillation: domain expertise + general reasoning
- Domain-specific reinforcement learning
- Interpretable, rationale-driven outputs

## Frozen LLMs via In-Context Learning

### BO-ICL ([[bo-icl-bayesian-optimization-catalysis]])
- Frozen LLMs (GPT-3.5, Gemini) as regression models with uncertainty
- No fine-tuning needed — works via in-context learning
- Bayesian optimization in natural language

### GOLLuM ([[gollum-uncertainty-calibrated-llm]])
- LLM embeddings as GP inputs with bidirectional learning
- LLM learns to encode experimental performance (not just textual similarity)
- Transfers across 19 diverse domains with fixed hyperparameters

## LLM Agents

### ChemReasoner ([[monte-carlo-thought-search]])
- MCTS over prompt space for combinatorial catalyst search
- Tree-structured prompts with cascading constraints

### Adsorb-Agent ([[adsorb-agent]])
- LLM agent for adsorption configuration search
- Evaluates GPT-4o, GPT-4o-mini, Claude-3.7-Sonnet, DeepSeek-Chat

## Fine-Tuning Methodology
[[llama3-lora-qlora]] provides technical reference on LoRA/QLoRA parameter-efficient fine-tuning — applicable to building domain-specific chemical LLMs.

## Key Tensions
1. **Fine-tuning vs. in-context learning**: CataLM/ChemDFM-R invest in fine-tuning; BO-ICL/GOLLuM show frozen LLMs can work with clever prompting
2. **General vs. specialized**: Domain-specific models (CataLM) vs. general LLMs with domain prompting (CatMiner)
3. **Reasoning vs. retrieval**: ChemDFM-R builds reasoning; others rely on knowledge retrieval
4. **Open vs. proprietary**: Adsorb-Agent tests both; ChemDFM-R uses open-source; BO-ICL uses GPT-3.5/Gemini