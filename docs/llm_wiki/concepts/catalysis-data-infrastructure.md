---
type: concept
title: "FAIR Data Infrastructure for Catalysis"
created: 2025-06-23
updated: 2026-07-01
sources: [nfdi4cat-unified-infrastructure, data-key-resource-catalysis, nfdi4cat-whitepaper, repo4cat, nfdi4chem, nomad-catalysis-plugin, orchestrating-catalysis-data, chemdcat-ap, mardiflow, mardiflow-voc4cat, methane-reforming-reporting-framework, nfdi4chem, marshall2023digital, moshantaf2024advancing, huskova2025improvement, cattesthub, mendes2021opendata, steinbeck2020nfdi4chem, fair-cookbook, chemical-data-storage-architectures, fair-biopharma-rd, fair-digital-twins, fairmat-materials-research, enzymeml-fair-biocatalysis, fair-biocatalytic-science, electrocatalysis-research-database, catalysis-data-infrastructure-ukch, chemspectra]
tags: [fair-data, data-infrastructure, catalysis, nfdi]
---

# FAIR Data Infrastructure for Catalysis

## Overview
Catalysis research generates vast, heterogeneous data across subdomains (heterogeneous, homogeneous, electro-, photo-, bio-catalysis). Making this data FAIR (Findable, Accessible, Interoperable, Reusable) requires coordinated infrastructure, standards, ontologies, and cultural change.

## The Problem
- Catalysis dataset-to-article ratio ≈ 1/100 — data is typically not shared ([[data-key-resource-catalysis]])
- Even available data hindered by standardization and metadata deficiencies
- Most data not machine-readable, unsuitable for data science
- Progress in AI-accelerated catalysis is "modest" largely due to lack of standardized, machine-readable, openly shared data

## German NFDI Initiative

### NFDI4Cat ([[nfdi4cat-unified-infrastructure]], [[nfdi4cat-whitepaper]])
- National Research Data Infrastructure for Catalysis-related Sciences
- Vision: integrate all research data along catalysis value chain (molecule → process)
- Core topics: ontologies, metadata, infrastructure, IP, community embedding
- Tools: LARAsuite, ELN FURTHRmind, Digital Labs, NOMAD, ADACTA

### NFDI4Chem ([[nfdi4chem]])
- Sister consortium for chemistry
- Smart Lab concept: ELN (Chemotion) + instrument integration + data processing + repository transfer
- Chemotion ELN supports SMILES, InChI, jcamp.dx
- ChemSpectra ([[chemspectra]]) adds a web-based analytical-data layer for Chemotion-style workflows, centered on JCAMP-DX/mzML interoperability, sample-linked spectra handling, and signal transfer back into ELN/repository records
- Federated repository infrastructure

## Infrastructure Components

### Repositories
- **Repo4Cat** ([[repo4cat]]): NFDI4Cat central data repository. Built on existing software. Supports working spaces for IP-protected industry collaboration.
- **NOMAD catalysis plugin** ([[nomad-catalysis-plugin]]): Structured experimental data upload with Voc4Cat alignment. Search by reactions, materials, conditions, kinetic properties.
- **NOMAD / FAIRmat pattern** ([[fairmat-materials-research]]): Federated data with centralized metadata, local "oases", persistent identifiers/DOIs, domain-aware search, AI tooling, ELN/LIMS integration, and metadata/ontology development for materials science. Useful adjacent model for catalysis because FAIRmat explicitly includes heterogeneous catalysis among its demonstration areas.
- **UK Catalysis Data Infrastructure** ([[catalysis-data-infrastructure-ukch]]): Catalogue-centered infrastructure that links publications, datasets, authors, institutions, and themes rather than creating yet another repository. Highlights that data objects are often not consistently linked in publication metadata and that many supporting chemistry/catalysis data objects remain unstructured.
- **Catalysis-Hub Experimental** ([[electrocatalysis-research-database]]): Domain-specific experimental electrocatalysis database with web access, GraphQL and Python APIs, multimodal experimental data, catalyst material/matrix/testing tables, spectra/curve data, and metadata for reuse and ML workflows.

### Metadata Standards
- **ChemDCAT-AP** ([[chemdcat-ap]]): DCAT-AP extension for chemistry/catalysis. LinkML-based. Shared by NFDI4Chem and NFDI4Cat.
- **Voc4Cat**: Community SKOS vocabulary for catalysis
- **Reporting frameworks** ([[methane-reforming-reporting-framework]]): Standardized reporting for catalytic methane reforming data

### Workflow Tools
- **MaRDIFlow** ([[mardiflow]]): CSE workflow framework with metadata abstraction. FAIR computational experiments.
- **MaRDIFlow + Voc4Cat** ([[mardiflow-voc4cat]]): Ontology-integrated workflow tool for semantic metadata annotation.

### Practical FAIRification and Data Management
- **FAIR Cookbook** ([[fair-cookbook]]): Practical recipes for identifiers, metadata, provenance, licensing, secure transfer, terminology/ontology selection, FAIRness maturity, and FAIRification workflows. Important for translating FAIR from principles into operations.
- **Chemical database architectures** ([[chemical-data-storage-architectures]]): Chemistry-facing guide showing why spreadsheets/file systems are weak for FAIR data, and why schemas, ETL, parsers, APIs, and database management systems matter for search, automation, and reuse.
- **Industrial FAIR implementation** ([[fair-biopharma-rd]], [[fair-digital-twins]]): Adjacent evidence that FAIR is not equivalent to open access, and that FAIR requires governance, data stewardship, authentication/authorization, machine-actionable restrictions, knowledge representation, and provenance.
- **EnzymeML and biocatalysis FAIR practice** ([[enzymeml-fair-biocatalysis]], [[fair-biocatalytic-science]]): Concrete biocatalysis model for FAIR data/software integration, combining exchange formats, STRENDA-style minimum information, raw time-course data, ELNs/LIMSs, APIs, Dataverse/Zenodo-style repositories, DOIs, ontologies, and FAIR software practices.
- **ChemSpectra analytical-data tooling** ([[chemspectra]]): Shows that FAIR chemistry workflows also need browser-based tooling for open analytical formats, not only repositories and metadata profiles. It connects JCAMP-DX/mzML ingestion, analysis actions, persistent storage of edited/original spectra, PNG derivatives, and machine-readable signal transfer into ELN/repository systems.

## Path Forward
1. **Mandatory FAIR data depositing** before publication ([[data-key-resource-catalysis]])
2. **Top-down guidelines** on data sharing, powered by easy-to-use tools
3. **Community standards** for ontologies and metadata (see [[ontology-development]])
4. **Cultural change** through training and education
5. **Interoperability** via shared standards (ChemDCAT-AP, Voc4Cat)

## FAIR Principle Mapping

- **Findable**: Persistent identifiers and resource-level metadata are necessary, but domain-aware search requires richer metadata and repository indexing. FAIRmat/NOMAD emphasize centralized metadata and domain-specific search; Repo4Cat and NOMAD catalysis provide repository-level discovery.
- **Accessible**: FAIR access may be open or restricted. Repo4Cat working spaces, NOMAD embargoes/oases, and industrial FAIR sources show that authentication, authorization, and clear access protocols can be part of FAIR practice.
- **Interoperable**: Vocabularies alone are not enough. Interoperability also needs shared schemas, parsers/converters, units, data formats, APIs, RDF/OWL/JSON-LD-style representations, and mappings between models.
- **Reusable**: Reuse depends on provenance, licensing, sufficient experimental context, quality information, versioning, and structured underlying data. ChemDCAT-AP and vocabulary grounding provide important metadata context, while database/schema work addresses the actual data representation.
- **Package-level gap**: UKCH CDI and EnzymeML both show that publication metadata is insufficient for reuse. Useful records need links to data objects, raw measurements, software, processing context, experimental conditions, and provenance.

## Cross-Domain Parallels
- [[ontology-energy-systems]] applies similar FAIR/ontology principles to energy systems
- [[eclass-semantic-search]] shows ECLASS standard for industrial product data
- [[autonomous-protein-engineering]] envisions FAIR + KG orchestration for distributed self-driving labs
- [[fairmat-materials-research]] gives the closest adjacent materials-science infrastructure model, especially through NOMAD, FAIRmat, federated repositories, metadata schemas, and ontology-derived workflows.
- [[enzymeml-fair-biocatalysis]] and [[fair-biocatalytic-science]] provide strong biocatalysis examples for data exchange formats and FAIR software.
- [[electrocatalysis-research-database]] and [[catalysis-data-infrastructure-ukch]] provide catalysis-specific examples of domain databases/catalogues and the continuing problem of unstructured or inconsistently linked data objects.
