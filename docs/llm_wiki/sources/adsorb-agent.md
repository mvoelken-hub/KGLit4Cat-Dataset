---
type: source
title: "Adsorb-Agent: Autonomous Identification of Stable Adsorption Configurations via LLM Agent"
created: 2025-06-23
updated: 2025-06-23
sources: [adsorb-agent]
tags: [catalysis, llm, agent]
---

# Adsorb-Agent

**Authors:** Janghoon Ock, Radheesh Sharma Meda, Tirtha Vinchurkar, Yayati Jadhav, Amir Barati Farimani
**Year:** 2024 (v4: July 2025)
**Venue:** arXiv (Carnegie Mellon University)
**File:** `docs/literature/adsorb-agent-autonomous-identification-of-stable-adsorption-configurations-via-large-language-model-agent-2410.16658v4.pdf`

## Abstract
LLM agent for efficiently identifying stable adsorption configurations corresponding to global minimum energy. Leverages built-in knowledge and reasoning to strategically explore configurations, reducing the number of initial setups while improving energy prediction accuracy.

## Key Findings
- Tested on 20 diverse systems; identifies comparable adsorption energies for 84% of cases, lower energies for 35%
- Excels in complex systems: 47% of intermetallic systems, 67% of systems with large adsorbates
- GPT-4o shows strongest overall performance as reasoning engine
- Compared against GPT-4o-mini, Claude-3.7-Sonnet, DeepSeek-Chat

## Methodology
- LLM agent strategically explores adsorbate-catalyst configurations
- Adsorption energy (Eads) = global minimum energy among all possible configurations
- Reduces computational cost vs. exhaustive DFT sampling

## Relevance
Demonstrates LLM agents for computational catalysis. Relevant to [[llm-agents-for-science]] and [[llm-guided-catalyst-discovery]].