---
type: concept
title: "Ontology Development for Catalysis"
created: 2025-06-23
updated: 2025-06-23
sources: [bfo-textbook, guarino-fois98, ontocape, chemdcat-ap, ontologies4cat, ontology-reaction-classification, nfdi4cat-whitepaper, mardiflow-voc4cat, llm-kg-ontology-generation]
tags: [ontology, catalysis, nfdi, formal-ontology]
---

# Ontology Development for Catalysis

## Overview
Ontologies provide the formal vocabulary and logical structure for representing catalysis knowledge in machine-understandable form. They are essential for FAIR data management, semantic interoperability, and knowledge graph construction.

## Foundations
- **Formal ontology** ([[guarino-fois98]]): Ontology = vocabulary + explicit assumptions about intended meaning. Conceptualization vs. ontology distinction.
- **BFO** ([[bfo-textbook]]): Upper-level ontology providing generic categories (universals, classes, relations). Framework for building domain ontologies.
- **OntoCAPE** ([[ontocape]]): Re-usable ontology for chemical process engineering — early domain ontology influencing catalysis efforts.

## Catalysis-Specific Ontologies & Vocabularies

### ChemDCAT-AP ([[chemdcat-ap]])
- Extension of W3C DCAT-AP for chemistry/catalysis data catalogs
- Built on DCAT-AP+ (domain-agnostic provenance extension using PROV-O)
- Aligns with ChEBI, CHEMINF, CHMO ontologies
- Uses LinkML for schema inheritance, SHACL shapes, JSON Schema generation
- Shared foundation for NFDI4Chem and NFDI4Cat

### Ontologies4Cat ([[ontologies4cat]])
- Systematic collection of ontology metadata for catalysis research data value chain
- Classification by subdomains of catalysis
- Automatic mapping of ontology classes for relatedness analysis
- GitHub: https://github.com/nfdi4cat/Ontology-Overview-of-NFDI4Cat

### Reac4Cat / Reaction Classification ([[ontology-reaction-classification]])
- Extends Reac4Cat ontology with semantic reaction classification
- Automatic functional group decomposition via chemical database APIs (KEGG, PubChem)
- OWL + SHACL for reasoning and constraints
- GUI for KG interaction

### Voc4Cat
- Domain-specific SKOS vocabulary for catalysis
- Integrated into MaRDIFlow ([[mardiflow-voc4cat]]) for workflow metadata annotation
- Open, community-driven resource
- Used in NOMAD catalysis plugin ([[nomad-catalysis-plugin]])

## LLM-Assisted Ontology Construction
- [[llm-kg-ontology-generation]]: Zero-shot pipeline for vocabulary/taxonomy extraction from literature using open-source LLMs. First KG+ontology for Single Atom Catalysis.
- [[complex-ontology-matching-llm]]: LLM embeddings for complex ontology matching — 45% F-measure improvement.
- [[spires-ontogpt]]: Zero-shot KB population using LLM prompt interrogation with ontology grounding.

## NFDI Context
Ontology development is coordinated through [[nfdi4cat]] and [[nfdi4chem]]. The NFDI4Cat White Paper ([[nfdi4cat-whitepaper]]) provides practical guidance. See also [[catalysis-ontologies]] for entity pages.