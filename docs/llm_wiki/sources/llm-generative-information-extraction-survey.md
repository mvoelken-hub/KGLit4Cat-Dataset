---
type: source
title: "Large Language Models for Generative Information Extraction: A Survey"
created: 2026-06-29
updated: 2026-06-29
sources: [llm-generative-information-extraction-survey]
tags: [llm, text-mining, information-extraction, structured-output, survey]
---

# Large Language Models for Generative Information Extraction

**Authors:** Derong Xu, Wei Chen, Wenjun Peng, Chao Zhang, Tong Xu, Xiangyu Zhao, Xian Wu, Yefeng Zheng, Yang Wang, Enhong Chen
**Year:** 2024
**Venue:** Frontiers of Computer Science 18(6):186357
**DOI:** 10.1007/s11704-024-40555-y
**File:** `docs/literature/s11704-024-40555-y.pdf`
**Repository:** LLM4IE repository

## Abstract
Survey of generative information extraction with large language models. It frames IE as structured knowledge extraction from natural-language text and reviews how LLMs are used for named entity recognition, relation extraction, event extraction, universal IE, low-resource settings, prompt design, data augmentation, retrieval, and domain-specific applications.

## Key Findings
- Distinguishes discriminative IE from generative IE, where an LLM produces target extraction sequences for entities, relations, events, or unified structures.
- Organizes the field through two taxonomies: IE subtasks and LLM-based IE techniques.
- Reviews task-specific and universal IE frameworks, including natural-language schema approaches and code/schema-generation approaches.
- Highlights that universal IE models can benefit complex relation/event extraction by learning dependencies across tasks, but performance varies by dataset and domain.
- Identifies persistent challenges: mismatch between natural language and structured output, hallucination, context dependence, high compute cost, and difficulty updating model knowledge.
- Emphasizes low-resource techniques such as in-context learning, supervised fine-tuning, data augmentation, retrieval, distillation, and self-improvement.

## Relevance
Useful background for SIMONE's extraction design because it situates metadata extraction within the broader move from task-specific IE models toward generative, schema-oriented, and retrieval-assisted workflows. It supports [[structured-output-reliability]], [[text-mining-for-catalysis]], and [[retrieval-augmented-generation]] by documenting recurring reliability concerns around hallucination, schema adherence, and low-resource extraction.

## Limitations
The survey is broad and mostly text-centered. It is not specific to catalysis, scientific data packages, ontology-grounded application profiles, or multimodal laboratory artifacts.
