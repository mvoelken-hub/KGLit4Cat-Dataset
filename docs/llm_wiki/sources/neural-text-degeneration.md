---
type: source
title: "The Curious Case of Neural Text Degeneration"
created: 2026-06-26
updated: 2026-06-26
sources: [neural-text-degeneration]
tags: [llm, decoding, generation]
---

# The Curious Case of Neural Text Degeneration

**Authors:** Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi
**Year:** 2020
**Venue:** ICLR
**File:** `docs/literature/1608_the_curious_case_of_neural_tex.pdf`

## Abstract
Analyzes why maximization-based decoding such as beam search can produce bland, repetitive, or incoherent open-ended text. Introduces nucleus sampling, which truncates the unreliable tail of the token distribution and samples from a dynamic high-probability nucleus.

## Key Findings
- Maximizing likelihood is often a poor objective for open-ended generation.
- Neural language models have unreliable probability tails.
- Nucleus sampling improves diversity and quality compared with beam search and pure sampling.
- Decoding choices strongly affect generation behavior even with the same model.

## Relevance
Useful background for controlling LLM outputs. For structured extraction, this source motivates deterministic or constrained decoding rather than free-form sampling when exact schemas and citations matter. Relevant to [[structured-output-reliability]].
