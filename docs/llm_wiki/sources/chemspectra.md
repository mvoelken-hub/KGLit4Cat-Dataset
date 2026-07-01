---
type: source
title: "ChemSpectra: a web-based spectra editor for analytical data"
created: 2026-07-01
updated: 2026-07-01
sources: [chemspectra]
tags: [fair-data, data-infrastructure, chemistry, spectroscopy]
---

# ChemSpectra: a Web-Based Spectra Editor for Analytical Data

**Authors:** Yu-Chieh Huang, Pierre Tremouilhac, An Nguyen, Nicole Jung, Stefan Braese  
**Year:** 2021  
**Venue:** Journal of Cheminformatics 13(1):8  
**File:** `docs/literature/s13321-020-00481-0.pdf`

## Abstract
Software paper describing ChemSpectra, a web-based editor for analytical spectra in synthetic chemistry. The system focuses on IR, MS, and one-dimensional 1H/13C NMR data, with explicit support for open formats such as JCAMP-DX and mzML so that spectra can be visualized, edited, persisted, and transferred inside FAIR-oriented digital research workflows.

## Key Findings
- ChemSpectra targets common analytical data types in chemistry research: IR, MS, and one-dimensional 1H/13C NMR.
- The tool is designed around open, interoperable spectral formats: JCAMP-DX for IR/MS/NMR and mzML for MS.
- Proprietary or instrument-native formats can be incorporated through conversion/decoding steps, demonstrated for Thermo RAW MS files and NMR FID files.
- The architecture combines a JavaScript/React frontend with a Python/Flask backend microservice, allowing both stand-alone use and embedding into larger web applications.
- Integration into the Chemotion ELN and Chemotion repository adds workflow-level capabilities beyond visualization: persistent file management, storage of original and edited spectra, automatic PNG generation, sample association, and transfer of selected or auto-detected signals back into the host system.
- Domain-specific analysis features are included rather than only passive viewing: peak picking, threshold control, NMR integration, reference-signal selection, coupling-constant calculation, multiplicity assignment, IR intensity annotation, and MS scan selection.
- The software is released under AGPL and explicitly positioned as reusable community infrastructure for extending FAIR analytical-data workflows.

## Relevance
Important adjacent source for SIMONE's FAIR-data framing because it moves the discussion from repositories and metadata profiles down to the analytical-data tooling layer. It supports claims that interoperable research data workflows need open formats, embedded viewers/editors, persistent links between raw files and derived annotations, and machine-readable handoff from instrument data into ELNs or repositories. While not catalysis-specific, it complements NFDI4Chem, Chemotion, and other FAIR chemistry infrastructure sources.
