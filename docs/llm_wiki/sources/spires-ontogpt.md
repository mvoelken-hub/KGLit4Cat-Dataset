---
type: source
title: "SPIRES: Structured Prompt Interrogation and Recursive Extraction of Semantics"
created: 2025-06-23
updated: 2025-06-23
sources: [spires-ontogpt]
tags: [llm, text-mining, ontology]
---

# SPIRES (OntoGPT)

**Authors:** J. Harry Caufield, Harshad Hegde, Vincent Emonet, et al. (Lawrence Berkeley National Laboratory)
**Year:** 2024
**Venue:** Bioinformatics 40(3): btae104
**File:** `docs/literature/btae104.pdf`

## Abstract
Knowledge extraction approach using LLM zero-shot learning to populate arbitrary knowledge schemas from text. SPIRES recursively performs prompt interrogation against an LLM to obtain schema-conforming responses. Uses existing ontologies for grounding entities with identifiers.

## Key Findings
- Zero-shot KB population from text — no training data needed
- Handles arbitrarily complex nested knowledge schemas
- Uses ontologies/vocabularies for entity grounding (bypasses LLM hallucination)
- Applications: food recipes, cellular signaling pathways, disease treatments, drug mechanisms, chemical-disease relationships
- Accuracy comparable to mid-range relation extraction methods

## Methodology
- Recursive prompt interrogation: LLM fills schema attributes, including nested complex classes
- Ontology-based grounding via OntoPortal, Gilda, OGER
- Part of open-source OntoGPT package

## Relevance
Key tool for LLM-driven KB construction. Relevant to [[llm-for-catalysis-data-extraction]] and [[ontology-development]]. https://github.com/monarch-initiative/ontogpt