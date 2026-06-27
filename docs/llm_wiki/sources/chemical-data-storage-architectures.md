---
type: source
title: "Data Storage Architectures to Accelerate Chemical Discovery"
created: 2026-06-27
updated: 2026-06-27
sources: [chemical-data-storage-architectures]
tags: [chemistry, fair-data, data-infrastructure, databases]
---

# Chemical Data Storage Architectures

**Authors:** Rebekah Duke, Vinayak Bhat, Chad Risko  
**Year:** 2022  
**Venue:** Chemical Science 13:13646-13656  
**File:** `docs/literature/d2sc05142g.pdf`

## Abstract
Introduces database fundamentals for chemists and argues that database management systems are better suited than spreadsheets or file systems for storing, managing, querying, sharing, and automating chemical data workflows. The paper links database architecture directly to FAIR principles and provides examples spanning SQL, NoSQL, extract-transform-load (ETL), schemas, parsing, APIs, and laboratory data management.

## Key Findings
- Spreadsheets and file-based systems struggle with scale, redundancy, consistency, multi-dimensional data, and FAIR adaptation.
- Databases provide scalable, searchable, shareable structures that support automated analysis and autonomous/robotic experimentation.
- Schema design is central: unintuitive or inconsistent schemas reduce interoperability and query efficiency.
- ETL pipelines are needed to extract data from heterogeneous raw files, transform them into schema-compatible structures, and load them into databases.
- Major challenges include lack of agreed domain schemas, need for parsers across instrument formats, missing metadata in instrument outputs, and the need for APIs/REST APIs for machine accessibility.

## Relevance
Useful for SIMONE's motivation around heterogeneous source packages, schema projection, and the need to move beyond opaque files. It is chemistry-focused rather than catalysis-specific, but it directly supports claims that structured databases, schemas, parsing, APIs, and metadata capture are prerequisites for FAIR chemical data and automated analysis.
