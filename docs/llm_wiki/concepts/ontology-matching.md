---
type: concept
title: "Ontology Matching"
created: 2025-06-23
updated: 2025-06-23
sources: [complex-ontology-matching-llm, lakermap-ontology-matching, hgnn-ontology-matching]
tags: [ontology, ontology-matching]
---

# Ontology Matching

## Overview
Ontology matching (OM) identifies semantic correspondences between concepts across different ontologies — essential for data integration, knowledge sharing, and interoperability. Traditional methods focus on simple (1:1) correspondences; recent work addresses complex correspondences and structural information.

## Methods

### LLM-Based Embedding Matching
[[complex-ontology-matching-llm]]:
- Integrates LLM embeddings into CANARD approach for complex correspondences
- Four modifications: label similarity, SPARQL result embeddings, subgraph embeddings, instance embeddings
- 45% F-measure improvement over baseline
- Pre-trained models circumvent need for reference alignments

### Self-Supervised Structural Learning
[[lakermap-ontology-matching]] (LaKERMap):
- Two transformer encoders for contextual + structural information
- Self-supervised objectives: triplet contrastive learning, masked concept prediction
- Surpasses OAEI 2022 state-of-the-art in F-score, recall, and runtime
- Large ontologies aligned within minutes

### Heterogeneous Graph Neural Networks
[[hgnn-ontology-matching]]:
- BERT fine-tuning for semantic embeddings
- HGNN for structural information with iterative node/relation fusion
- Hungarian algorithm for optimal one-to-one alignment
- Higher F1 on 2/3 OAEI Bio-ML tasks

## Key Tensions
1. **Lexical vs. structural**: All three methods combine both, but weight them differently
2. **Simple vs. complex correspondences**: Most work focuses on simple; complex (logical constructors, transformation functions) remains challenging
3. **Supervised vs. self-supervised**: LaKERMap uses self-supervised learning; others rely on pre-trained models
4. **Speed vs. accuracy**: LaKERMap emphasizes inference time; others focus on F-score

## Relevance to Catalysis
Ontology matching is needed to align catalysis ontologies with broader chemistry ontologies (ChEBI, CHEMINF, CHMO) and to integrate across NFDI4Chem/NFDI4Cat (see [[ontology-development]] and [[catalysis-data-infrastructure]]).