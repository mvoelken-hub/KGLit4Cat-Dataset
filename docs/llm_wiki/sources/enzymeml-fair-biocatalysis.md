---
type: source
title: "Standardized Data, Scalable Documentation, Sustainable Storage: EnzymeML as a Basis for FAIR Data Management in Biocatalysis"
created: 2026-06-27
updated: 2026-06-27
sources: [enzymeml-fair-biocatalysis]
tags: [biocatalysis, fair-data, data-infrastructure, enzymeml]
---

# EnzymeML as FAIR Data Management in Biocatalysis

**Authors:** Jurgen Pleiss  
**Year:** 2021  
**Venue:** ChemCatChem 13:3909-3913  
**File:** `docs/literature/ChemCatChem - 2021 - Pleiss - Standardized Data  Scalable Documentation  Sustainable Storage   EnzymeML As A Basis For FAIR.pdf`

## Abstract
Concept article proposing EnzymeML-based infrastructure for FAIR enzymology and biocatalysis data. The paper argues that reproducibility problems in enzymology and biocatalysis are caused largely by incomplete reporting of reaction conditions, unavailable raw data, local file storage, missing data models, and poor interoperability between databases and tools.

## Key Findings
- EnzymeML is presented as a standardized data exchange format for enzymology and biocatalysis, aligned with STRENDA reporting guidelines.
- EnzymeML combines metadata, reaction conditions, kinetic models, kinetic parameters, and measured substrate/product time courses in an OMEX package with XML and CSV content.
- The EnzymeML API reads, writes, edits, validates, and exchanges EnzymeML documents and can be exposed as a RESTful web service.
- ELNs such as BioCatHub can use EnzymeML to connect experimental data acquisition with modelling tools and databases such as SABIO-RK and STRENDA DB.
- EnzymeML Dataverse entries can act as DOI-addressable machine-readable micropublications.
- The proposed stack combines standardized data, an API, ELN integration, and distributed Dataverse repositories to support reproducibility, scalability, accessibility, and long-term data security.

## Relevance
Important domain-adjacent source for SIMONE because it gives a concrete example of FAIR data practice in enzyme catalysis: minimum information guidelines, structured exchange formats, APIs, ELNs, repositories, DOI-based micropublications, validation, and database exchange. It strengthens Chapter 3 claims that reusability requires more than metadata records: raw time-course data, reaction conditions, models, parameters, and provenance must be structured together.
