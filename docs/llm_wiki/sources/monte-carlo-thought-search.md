---
type: source
title: "Monte Carlo Thought Search: LLM Querying for Complex Scientific Reasoning in Catalyst Design"
created: 2025-06-23
updated: 2025-06-23
sources: [monte-carlo-thought-search]
tags: [catalysis, llm, agent, materials-discovery]
---

# Monte Carlo Thought Search (ChemReasoner)

**Authors:** Henry W. Sprueill, Carl Edwards, Mariefel V. Olarte, Udishnu Sanyal, Heng Ji, Sutanay Choudhury
**Year:** 2023
**Venue:** arXiv (Pacific Northwest National Laboratory)
**File:** `docs/literature/monte-carlo-thought-search-large-language-model-querying-for-complex-scientific-reasoning-in-catalyst-design-2310.14420v1.pdf`

## Abstract
Uses Monte Carlo Tree Search (MCTS) to augment LLM reasoning for catalyst design. Introduces two reasoning datasets: computational chemistry simulations and catalysis researcher questions about novel chemical conversion processes. Improves over best baseline by 25.8%.

## Key Findings
- MCTS-based approach improves beyond chain-of-thought prompting for scientific reasoning
- Combinatorial search using LLMs for catalyst discovery (reaction + catalyst + conditions optimization)
- Two new datasets: BioFuelQR (reasoning questions) and computational chemistry simulations
- Tree-structured prompt design: root = generic query, children add criteria (low cost, high efficiency, etc.)

## Methodology
- Monte Carlo Tree Search over prompt space
- Each node represents a prompt with additional constraints
- LLM generates candidate catalysts at each node
- Search returns optimal answer via efficient exploration

## Relevance
Key work on structured LLM reasoning for catalyst design. Relevant to [[llm-guided-catalyst-discovery]] and [[llm-agents-for-science]]. Code/datasets at https://github.com/pnnl/chemreasoner.