---
type: source
title: "OLS4: A New Ontology Lookup Service"
created: 2025-06-23
updated: 2025-06-23
sources: [ols4]
tags: [ontology, data-infrastructure]
---

# OLS4: Ontology Lookup Service

**Authors:** James McLaughlin, Josh Lagrimas, Haider Iqbal, Henriette Parkinson, Henriette Harmse
**Year:** 2025
**Venue:** Bioinformatics 41(5): btaf279
**File:** `docs/literature/btaf279.pdf`

## Abstract
New version of the Ontology Lookup Service (OLS) — open-source search engine for ontologies used extensively in bioinformatics and chemistry. Implements complete OWL2 specification, internationalization support, and new UI with UX enhancements.

## Key Findings
- Full OWL2 specification implementation
- Internationalization (multi-language ontology browsing)
- Cross-references between ontology terms
- Bioregistry integration for external database links
- Architecture: Neo4j (graph DB) + Solr (full-text search)
- Dec 2024: 266 ontologies, 8.68M classes indexed
- Replaced OLS3 in production at EMBL-EBI

## Relevance
Essential infrastructure for ontology access. Used by NFDI4Chem Terminology Service. Relevant to [[ontology-resources]]. https://www.ebi.ac.uk/ols4