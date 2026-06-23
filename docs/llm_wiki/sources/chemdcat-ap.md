---
type: source
title: "ChemDCAT-AP: Enabling Semantic Interoperability with a Contextual Extension of DCAT-AP"
created: 2025-06-23
updated: 2025-06-23
sources: [chemdcat-ap]
tags: [ontology, fair-data, data-infrastructure, nfdi]
---

# ChemDCAT-AP

**Authors:** Philip Strömert, Hendrik Borgelt, David Linke, Mark Doerr, Bhavin Katabathuni, Oliver Koepler, Norbert Kockmann
**Year:** 2026
**Venue:** arXiv (cs.DB) / MTSR 2025
**File:** `docs/literature/2602.01822v1.pdf` (also `docs/literature/MTSR_2025_paper_1431-1.pdf`)

## Abstract
Proposes DCAT-AP+ (domain-agnostic extension of W3C DCAT-AP) and ChemDCAT-AP (chemistry/catalysis-specific profile). Enables cross-domain data integration with comprehensive provenance/context representation. Uses LinkML for schema inheritance, validation, and format conversion.

## Key Findings
- DCAT-AP+ adds provenance pattern based on PROV-O, classification mechanism, generic attribute pattern
- ChemDCAT-AP imports DCAT-AP+ and adds chemistry/catalysis concepts aligned with ChEBI, CHEMINF, CHMO
- LinkML enables compilation to Python/Pydantic, SHACL shapes, JSON Schema
- Shared foundation for NFDI4Chem and NFDI4Cat

## Relevance
Core interoperability standard for [[catalysis-data-infrastructure]]. Relevant to [[nfdi4cat]], [[nfdi4chem]], and [[catalysis-ontologies]].