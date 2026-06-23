---
type: source
title: "MaRDIFlow: A CSE Workflow Framework for Abstracting Meta-data from FAIR Computational Experiments"
created: 2025-06-23
updated: 2025-06-23
sources: [mardiflow]
tags: [data-infrastructure, fair-data]
---

# MaRDIFlow

**Authors:** Pavan L. Veluvali, Jan Heiland, Peter Benner (Max Planck Institute, Magdeburg)
**Year:** 2024
**Venue:** arXiv (cs.DC)
**File:** `docs/literature/2405.00028v1.pdf`

## Abstract
Computational framework for automating metadata abstraction from FAIR computational experiments. Multi-layered descriptions characterize workflow components through input-output relations, allowing interchangeable use of models, code, and data. Addresses execution and environmental dependencies.

## Key Findings
- CSE workflow = chain of interconnected models for simulations
- Components characterized through input/output: model, data, code used interchangeably
- Redundancy in representation of models, code, and data = positive feature (robustness, compatibility)
- Multi-layered descriptions: composition/abstraction, execution
- Working prototype demonstrated with example use cases

## Relevance
Workflow framework for FAIR computational experiments. See also [[mardiflow-voc4cat]] for Voc4Cat integration. Relevant to [[catalysis-data-infrastructure]].