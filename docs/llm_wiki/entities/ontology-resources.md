---
type: entity
title: "Ontology Resources and Infrastructure"
created: 2025-06-23
updated: 2025-06-23
sources: [ols4, kg-hub, spires-ontogpt, ontologies4cat]
tags: [ontology, data-infrastructure]
---

# Ontology Resources and Infrastructure

## Ontology Lookup Services

### OLS4 ([[ols4]])
- EMBL-EBI Ontology Lookup Service
- 266 ontologies, 8.68M classes (Dec 2024)
- Full OWL2 specification, internationalization
- Neo4j + Solr architecture
- Used by NFDI4Chem Terminology Service
- https://www.ebi.ac.uk/ols4

### BioPortal
- Ontology repository and search (Stanford)
- Lacks deep classification (per [[ontologies4cat]])

## Knowledge Graph Platforms

### KG-Hub ([[kg-hub]])
- Modular ETL for KG construction
- Biolink Model compliance
- OBO ontology integration
- Versioned, automatically updated
- Graph ML tools (embeddings, link prediction)
- https://kghub.org

## Extraction Tools

### SPIRES / OntoGPT ([[spires-ontogpt]])
- Zero-shot KB population via LLM prompt interrogation
- Ontology grounding via OntoPortal, Gilda, OGER
- https://github.com/monarch-initiative/ontogpt

## Ontology Standards
- **OBO Foundry**: Open Biological and Biomedical Ontologies
- **OWL2**: Web Ontology Language v2
- **SHACL**: Shapes Constraint Language
- **SKOS**: Simple Knowledge Organization System
- **LinkML**: YAML-based modeling framework (used by ChemDCAT-AP)

## Related
- [[ontology-development]]
- [[ontology-matching]]
- [[catalysis-ontologies]]