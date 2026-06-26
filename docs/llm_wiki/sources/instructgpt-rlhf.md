---
type: source
title: "Training Language Models to Follow Instructions with Human Feedback"
created: 2026-06-26
updated: 2026-06-26
sources: [instructgpt-rlhf]
tags: [llm, alignment, instruction-following]
---

# InstructGPT / RLHF

**Authors:** Long Ouyang, Jeff Wu, Xu Jiang, et al.
**Year:** 2022
**Venue:** NeurIPS 35
**File:** `docs/literature/2203.02155v1.pdf`

## Abstract
Shows that larger language models are not automatically better at following user intent, and introduces a training pipeline using supervised demonstrations and reinforcement learning from human feedback. The resulting instruction-following models are preferred over much larger base models.

## Key Findings
- Human feedback can align LLM behavior with user instructions.
- Supervised fine-tuning plus preference optimization improves helpfulness and reduces unwanted behavior.
- Smaller aligned models can be preferred over larger unaligned base models.
- Instruction tuning becomes central to practical LLM use.

## Relevance
Explains why modern chat/instruction models can be used for extraction workflows with natural-language task descriptions. Relevant to [[llm-foundations]], [[structured-output-reliability]], and [[llm-guided-catalyst-discovery]].
