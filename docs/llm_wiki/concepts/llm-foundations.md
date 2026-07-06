---
type: concept
title: "LLM Foundations"
created: 2026-06-26
updated: 2026-07-06
sources: [attention-is-all-you-need, sentencepiece, bert, scibert, gpt3-few-shot-learners, instructgpt-rlhf, pagedattention-vllm, llm-survey-zhao2026, llm-as-a-judge-mt-bench-chatbot-arena]
tags: [llm, transformer, foundational]
---

# LLM Foundations

This page tracks general LLM foundations needed to interpret the applied catalysis, ontology, and extraction papers in the wiki.

## Architecture and Representation
- [[attention-is-all-you-need]] establishes the Transformer and self-attention as the default architecture for modern LLMs.
- [[sentencepiece]] explains subword tokenization, which affects rare chemical names, identifiers, units, and schema labels.
- [[bert]] shows bidirectional Transformer pre-training for language understanding tasks.
- [[scibert]] adapts BERT to scientific text, motivating later domain-specific encoders such as [[gupta2022matscibert]].

## Prompting and Alignment
- [[gpt3-few-shot-learners]] establishes in-context learning as a practical interface for using large models without task-specific training.
- [[instructgpt-rlhf]] explains why instruction-tuned models follow extraction prompts better than base language models.
- [[llm-survey-zhao2026]] provides a broad map of pre-training, post-training, utilization, evaluation, and open issues.

## Evaluation
- [[llm-as-a-judge-mt-bench-chatbot-arena]] establishes LLM-as-a-judge as a scalable evaluation pattern for open-ended chatbot responses, while also documenting bias and reasoning limitations that matter for model-generated review artifacts.

## Serving and Deployment
- [[pagedattention-vllm]] matters when long-document extraction is deployed at scale: KV-cache memory management can dominate throughput and cost.

## Relevance to SIMONE
The SIMONE workflow depends on all layers: tokenization and embeddings affect source retrieval; instruction following affects schema completion; and serving constraints affect whether local open models can process uploaded archives and evidence windows economically.
