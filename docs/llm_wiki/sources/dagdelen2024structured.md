---
type: source
title: "Structured Information Extraction from Scientific Text with Large Language Models"
created: 2025-06-23
updated: 2025-06-23
sources: [dagdelen2024structured]
tags: [llm, text-mining, materials-discovery]
---

# Structured IE from Scientific Text (Dagdelen et al.)

**Authors:** John Dagdelen, Alexander Dunn, Sanghoon Lee, et al. (Lawrence Berkeley National Laboratory)
**Year:** 2024
**Venue:** Nat. Commun. 15: 1418
**File:** `docs/literature/s41467-024-45563-x.pdf`

## Abstract
Simple approach to joint named entity recognition and relation extraction using fine-tuned LLMs (GPT-3, Llama-2). Tested on three materials chemistry tasks: linking dopants and host materials, cataloging MOFs, and general composition/phase/morphology/application extraction. Output as English sentences or structured JSON. Represents accessible route to large databases of structured scientific knowledge.

## Key Findings
- Fine-tuned LLMs (GPT-3, Llama-2) for joint NER + relation extraction
- Three representative tasks in materials chemistry
- Handles complex relations (compound entities, multiple entity types)
- Output as natural language sentences or JSON objects
- Simple, accessible, highly flexible — no pipeline architecture needed

## Relevance
Key method for LLM-based structured extraction from materials science text. Relevant to [[text-mining-for-catalysis]], [[knowledge-graphs-for-catalysis]], and [[llm-for-catalysis-data-extraction]].