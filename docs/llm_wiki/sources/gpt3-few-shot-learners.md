---
type: source
title: "Language Models Are Few-Shot Learners"
created: 2026-06-26
updated: 2026-06-26
sources: [gpt3-few-shot-learners]
tags: [llm, in-context-learning, foundational]
---

# GPT-3: Language Models Are Few-Shot Learners

**Authors:** Tom B. Brown, Benjamin Mann, Nick Ryder, et al.
**Year:** 2020
**Venue:** NeurIPS 33
**File:** `docs/literature/NeurIPS-2020-language-models-are-few-shot-learners-Paper.pdf`

## Abstract
Demonstrates that scaling autoregressive language models produces strong zero-shot, one-shot, and few-shot task performance from natural-language prompts, without gradient updates. The work popularizes in-context learning as a practical interface for broad NLP task adaptation.

## Key Findings
- Large decoder-only language models can perform many tasks from prompt examples alone.
- Few-shot prompting often improves over zero-shot prompting.
- Performance scales with model size, though limitations remain for reasoning, factuality, and calibration.
- The paper helps establish prompting as a central method for using LLMs.

## Relevance
Conceptual basis for prompt-based extraction and optimization systems in the catalysis literature, including BO-ICL and schema-guided extraction. Relevant to [[llm-foundations]], [[chemical-llms]], and [[llm-guided-catalyst-discovery]].
