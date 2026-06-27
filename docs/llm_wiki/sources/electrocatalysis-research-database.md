---
type: source
title: "A Research Database for Experimental Electrocatalysis: Advancing Data Sharing and Reusability"
created: 2026-06-27
updated: 2026-06-27
sources: [electrocatalysis-research-database]
tags: [catalysis, electrocatalysis, fair-data, database, machine-learning]
---

# Experimental Electrocatalysis Research Database

**Authors:** Ruchika Mahajan, Ashton M. Aleman, Colin F. Crago, Suman Bhasker-Ranganath, Melissa E. Kreider, Jose A. Zamora Zeledon, Johanna Schroder, Gaurav A. Kamat, McKenzie A. Hubert, Adam C. Nielander, Thomas F. Jaramillo, Michaela Burke Stevens, Johannes Voss, Kirsten T. Winther  
**Year:** 2025  
**Venue:** Journal of Chemical Physics 163:124704  
**File:** `docs/literature/124704_1_5.0280821.pdf`

## Abstract
Introduces an open research database for experimental electrocatalysis hosted through Catalysis-Hub. The database curates multimodal experimental data and metadata from electrocatalysis experiments, with 241 experimental entries covering reaction conditions, material properties, characterization, testing data, and performance metrics.

## Key Findings
- Experimental catalysis lacks consistent, well-structured, reliable datasets for machine learning compared with computational catalysis.
- The platform follows FAIR principles by making data searchable, accessible through a web interface, and programmatically available through a GraphQL API and CatHub Python API.
- The database contains web-readable and machine-readable structures for electrocatalysis data, including catalyst material, catalyst matrix, catalyst testing, spectra, cyclic voltammetry curves, and metadata.
- Reusability is tied to sufficient metadata and provenance: users need context on how data were produced, data quality, and whether data are suitable for another context.
- The paper argues that generalized all-purpose repositories may not be sufficient; domain-specific structures tailored to user communities are more likely to be adopted and useful.
- The platform keeps problematic entries with explicit labels/notes rather than deleting them, preserving reproducibility and trust.

## Relevance
Highly relevant catalysis-specific source for Chapter 3. It supports claims about domain-specific repository design, structured experimental catalysis metadata, machine-readable APIs, multimodal data, provenance, benchmarking, and ML-oriented reuse. It also complements CatTestHub and NOMAD as concrete examples of FAIR experimental catalysis infrastructure.
