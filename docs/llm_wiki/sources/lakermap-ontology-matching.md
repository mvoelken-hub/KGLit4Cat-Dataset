---
type: source
title: "LaKERMap: Contextualized Structural Self-supervised Learning for Ontology Matching"
created: 2025-06-23
updated: 2025-06-23
sources: [lakermap-ontology-matching]
tags: [ontology, ontology-matching]
---

# LaKERMap: Ontology Matching with Self-supervised Learning

**Authors:** Zhu Wang (University of Illinois at Chicago)
**Year:** 2023
**Venue:** OM 2023 Workshop (ISWC)
**File:** `docs/literature/Contextualized_Structural_Self-supervised_Learning_for_Ontology_Matching-om2023_LTpaper4.pdf`

## Abstract
Novel self-supervised learning OM framework that integrates contextual and structural information of concepts into transformers. Captures multiple structural contexts (local and global) using distinct training objectives. Surpasses state-of-the-art in alignment quality and inference time on Bio-ML datasets.

## Key Findings
- Two transformer encoders for contextual and structural information
- Self-supervised training: triplet contrastive learning, masked concept prediction
- Outperforms OAEI 2022 state-of-the-art systems in F-score and recall
- Generates alignments for large ontologies within minutes

## Relevance
Advances [[ontology-matching]]. https://github.com/ellenzhuwang/lakermap