---
type: source
title: "kNN-BioEL: Improving Biomedical Entity Linking with Retrieval-Enhanced Learning"
created: 2025-06-23
updated: 2025-06-23
sources: [lin2024knnbioel]
tags: [llm, text-mining]
---

# kNN-BioEL

**Authors:** Zhenxi Lin, Ziheng Zhang, Xian Wu, Yefeng Zheng (Tencent YouTu Lab)
**Year:** 2024
**Venue:** ICASSP 2024 (pp. 11461–11465)
**File:** `docs/literature/Improving_Biomedical_Entity_Linking_with_Retrieval-Enhanced_Learning.pdf`

## Abstract
Introduces kNN-BioEL, a retrieval-enhanced learning paradigm for biomedical entity linking that addresses long-tailed entity distribution. Constructs a datastore of mention embeddings and entity labels; during inference, retrieves top-k nearest neighbors as clues. Combines with contrastive learning objective using dynamic hard negative sampling (DHNS). Outperforms SOTA baselines.

## Key Findings
- Addresses long-tailed entity distribution in BioEL
- kNN retrieval from training corpus as clues for prediction
- Dynamic hard negative sampling (DHNS) improves retrieved neighbor quality
- Versatile: directly applicable to most existing BioEL models
- SOTA on several benchmark datasets

## Relevance
Entity linking methodology relevant to ontology grounding and KG population. Related to [[text-mining-for-catalysis]] (entity grounding). Code: https://github.com/lzxlin/kNN-BioEL