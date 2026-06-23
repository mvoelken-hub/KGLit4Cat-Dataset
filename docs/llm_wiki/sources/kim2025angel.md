---
type: source
title: "ANGEL: Learning from Negative Samples in Biomedical Generative Entity Linking"
created: 2025-06-23
updated: 2025-06-23
sources: [kim2025angel]
tags: [llm, text-mining]
---

# ANGEL: Learning from Negative Samples in Biomedical Entity Linking

**Authors:** Chanhwi Kim, Hyunjae Kim, Sihyeon Park, Jiwoo Lee, Mujeen Sung, Jaewoo Kang
**Year:** 2025
**Venue:** Findings of ACL 2025
**File:** `docs/literature/2025.findings-acl.558.pdf`

## Abstract
First framework that trains generative biomedical entity linking (BioEL) models using negative samples. Initially trained on positive samples, then updated via preference optimization using correct/incorrect model predictions. Outperforms previous best baselines by up to 1.4% top-1 accuracy on five benchmarks; 1.7% when incorporated into pre-training.

## Key Findings
- Generative BioEL models typically trained only with positive samples
- ANGEL introduces negative sample learning via preference optimization
- Top-k predictions gathered; model updated to prioritize correct predictions
- Effective in both pre-training and fine-tuning stages

## Relevance
Relevant to entity linking methodology applicable to ontology grounding and KG population. Related to [[text-mining-for-catalysis]] (entity grounding). Code: https://github.com/dmis-lab/ANGEL