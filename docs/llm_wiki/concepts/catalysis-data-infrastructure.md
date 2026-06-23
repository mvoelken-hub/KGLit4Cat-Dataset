---
type: concept
title: "FAIR Data Infrastructure for Catalysis"
created: 2025-06-23
updated: 2025-06-23
sources: [nfdi4cat-unified-infrastructure, data-key-resource-catalysis, nfdi4cat-whitepaper, repo4cat, nfdi4chem, nomad-catalysis-plugin, orchestrating-catalysis-data, chemdcat-ap, mardiflow, mardiflow-voc4cat, methane-reforming-reporting-framework, nfdi4chem, marshall2023digital, moshantaf2024advancing, huskova2025improvement, cattesthub, mendes2021opendata, steinbeck2020nfdi4chem]
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
- Federated repository infrastructure

## Infrastructure Components

### Repositories
- **Repo4Cat** ([[repo4cat]]): NFDI4Cat central data repository. Built on existing software. Supports working spaces for IP-protected industry collaboration.
- **NOMAD catalysis plugin** ([[nomad-catalysis-plugin]]): Structured experimental data upload with Voc4Cat alignment. Search by reactions, materials, conditions, kinetic properties.

### Metadata Standards
- **ChemDCAT-AP** ([[chemdcat-ap]]): DCAT-AP extension for chemistry/catalysis. LinkML-based. Shared by NFDI4Chem and NFDI4Cat.
- **Voc4Cat**: Community SKOS vocabulary for catalysis
- **Reporting frameworks** ([[methane-reforming-reporting-framework]]): Standardized reporting for catalytic methane reforming data

### Workflow Tools
- **MaRDIFlow** ([[mardiflow]]): CSE workflow framework with metadata abstraction. FAIR computational experiments.
- **MaRDIFlow + Voc4Cat** ([[mardiflow-voc4cat]]): Ontology-integrated workflow tool for semantic metadata annotation.

## Path Forward
1. **Mandatory FAIR data depositing** before publication ([[data-key-resource-catalysis]])
2. **Top-down guidelines** on data sharing, powered by easy-to-use tools
3. **Community standards** for ontologies and metadata (see [[ontology-development]])
4. **Cultural change** through training and education
5. **Interoperability** via shared standards (ChemDCAT-AP, Voc4Cat)

## Cross-Domain Parallels
- [[ontology-energy-systems]] applies similar FAIR/ontology principles to energy systems
- [[eclass-semantic-search]] shows ECLASS standard for industrial product data
- [[autonomous-protein-engineering]] envisions FAIR + KG orchestration for distributed self-driving labs