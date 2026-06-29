---
type: entity
title: "Catalysis Ontologies and Vocabularies"
created: 2025-06-23
updated: 2026-06-28
sources: [ontologies4cat, chemdcat-ap, reac4cat-ontology, ontology-reaction-classification, ontocape, nfdi4cat-whitepaper]
tags: [ontology, catalysis]
---

# Catalysis Ontologies and Vocabularies

## Overview
The catalysis ontology landscape includes domain-specific ontologies, vocabularies, and application profiles developed primarily within the NFDI4Cat and NFDI4Chem initiatives.

## Domain-Specific

### Voc4Cat ([[voc4cat]])
- SKOS vocabulary for catalysis
- Community-driven, open

### Reac4Cat
- Ontology for reaction and catalysis knowledge graphs ([[reac4cat-ontology]])
- Uses OWL description logic and General Class Axioms to infer reaction roles and catalyst relations
- Extended by [[ontology-reaction-classification]] with automatic functional group decomposition

### ChemDCAT-AP ([[chemdcat-ap]])
- Application profile for chemistry/catalysis data catalogs
- Extends W3C DCAT-AP
- Uses ChEBI, CHEMINF, CHMO alignments

### OntoCAPE ([[ontocape]])
- Re-usable ontology for chemical process engineering
- Predecessor influence on catalysis ontology efforts

## External Ontologies Used in Catalysis
- **ChEBI**: Chemical Entities of Biological Interest
- **CHEMINF**: Chemical Information Ontology
- **CHMO**: Chemical Methods Ontology
- **PROV-O**: Provenance Ontology (used by DCAT-AP+)
- **IUPAC Goldbook**: Chemical terminology
- **NCIT**: National Cancer Institute Thesaurus

## Survey
[[ontologies4cat]] provides systematic collection and classification of ontologies for the catalysis research data value chain. GitHub: https://github.com/nfdi4cat/Ontology-Overview-of-NFDI4Cat

## Related
- [[ontology-development]] — How these ontologies are built
- [[ontology-matching]] — Aligning ontologies
- [[nfdi4cat]]
