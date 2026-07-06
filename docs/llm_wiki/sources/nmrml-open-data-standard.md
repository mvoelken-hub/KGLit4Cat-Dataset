---
type: source
title: "nmrML: A Community Supported Open Data Standard for the Description, Storage, and Exchange of NMR Data"
created: 2026-07-06
updated: 2026-07-06
sources: [nmrml-open-data-standard]
tags: [fair-data, data-standard, spectroscopy, nmr, chemistry]
---

# nmrML: A Community Supported Open Data Standard for NMR Data

**Authors:** Daniel Schober et al.  
**Year:** 2018  
**Venue:** Analytical Chemistry 90:649-656  
**File:** `docs/literature/nmrml-a-community-supported-open-data-standard-for-the-description-storage-and-exchange-of-nmr-data.pdf`

## Abstract
Perspective introducing nmrML, an open XML-based exchange and storage format for nuclear magnetic resonance data. The standard is intended to preserve raw NMR data, acquisition parameters, spectral metadata, assignments, and chemical-structure information in a vendor-agnostic, long-lived, and repository-compatible form.

## Key Findings
- NMR data reuse is hindered by proprietary vendor formats, long-term readability risks, inconsistent JCAMP-DX variants, and limited semantic validation.
- nmrML is designed as an XML-based standard analogous to mzML for mass spectrometry, with explicit support for controlled-vocabulary terms through nmrCV.
- The format can represent raw free-induction decay data, acquisition settings, processing metadata, spectral assignments, chemical structures, and NMR experiment metadata.
- It targets both pure-compound reference spectra and complex biomixture/metabolomics data.
- Converter tooling is provided for Bruker, JEOL, and Agilent/Varian formats, with parser libraries, web services, validators, and repository/workflow integrations.
- Validation is layered: XML schema validation checks structure, while controlled-vocabulary and rule-based validation support minimum-information and content-completeness checks.
- nmrML is used or supported by tools and repositories such as MetaboLights, nmrGlue, NMRProcFlow, Batman, Bayesil-related tools, and Galaxy-style workflow components.
- The paper frames nmrML as community infrastructure, endorsed by the Metabolomics Standards Initiative and intended to support long-term archiving, interoperability, quality assurance, and reproducible NMR workflows.

## Relevance
Important adjacent evidence for SIMONE's FAIR chemistry framing. It strengthens the point that reusable scientific metadata is not only bibliographic or dataset-level metadata: analytical instrument data need open exchange formats, controlled vocabularies, converters, validators, and repository handoff. It complements ChemSpectra, NFDI4Chem, EnzymeML, mzML/JCAMP-DX mentions, and FAIR software discussions by showing a concrete open standard for NMR data.
