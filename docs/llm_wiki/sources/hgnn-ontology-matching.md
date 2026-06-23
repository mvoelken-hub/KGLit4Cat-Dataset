---
type: source
title: "Ontology Matching with Heterogeneous Graph Neural Network"
created: 2025-06-23
updated: 2025-06-23
sources: [hgnn-ontology-matching]
tags: [ontology, ontology-matching]
---

# Ontology Matching with HGNN

**Authors:** Daoqu Geng, Shuai Zhang (Chongqing University of Posts and Telecommunications)
**Year:** 2026
**Venue:** ISBDAS 2026 (IEEE)
**File:** `docs/literature/Ontology_Matching_with_Heterogeneous_Graph_Neural_Network.pdf`

## Abstract
Relation-aware heterogeneous graph neural network framework for ontology matching. Fine-tunes BERT for node representations, extracts relation representations, constructs heterogeneous graph, iteratively updates and fuses node/relation embeddings. Uses Hungarian algorithm for optimal alignment.

## Key Findings
- BERT fine-tuning for semantic embeddings + HGNN for structural information
- Iterative fusion of node and relation features
- Hungarian algorithm for one-to-one alignment
- Higher F1 scores than existing methods on 2/3 OAEI Bio-ML tasks

## Relevance
Advances [[ontology-matching]] with graph neural networks.