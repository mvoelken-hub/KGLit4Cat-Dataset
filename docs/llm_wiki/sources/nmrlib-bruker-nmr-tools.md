---
type: source
title: "NMRlib: user-friendly pulse sequence tools for Bruker NMR spectrometers"
created: 2026-07-06
updated: 2026-07-06
sources: [nmrlib-bruker-nmr-tools]
tags: [software, spectroscopy, nmr, laboratory-workflows]
---

# NMRlib: User-Friendly Pulse Sequence Tools for Bruker NMR Spectrometers

**Authors:** Adrien Favier, Bernhard Brutscher  
**Year:** 2019  
**Venue:** Journal of Biomolecular NMR 73:199-211  
**File:** `docs/literature/s10858-019-00249-1.pdf`

## Abstract
Software communication presenting NMRlib, a suite of Jython-based tools for Bruker TopSpin spectrometers. NMRlib centralizes pulse-sequence setup, experiment management, processing, visualization, and analysis behind a GUI-oriented library that allows NMR experiments to be configured and exchanged with less manual parameter tuning.

## Key Findings
- NMRlib targets Bruker TopSpin versions 3.2-4.0 and is implemented as a set of Jython scripts, pulse sequences, shape files, lists, and GUI definitions.
- Experiments are organized in a directory tree and exposed through GUI windows, allowing users to set up experiments by selecting a library entry rather than manually editing all parameters.
- The library is magnetic-field independent, which is useful for laboratories operating multiple spectrometers at different field strengths.
- A central shared library can be mounted across spectrometers, simplifying maintenance and making updates immediately available across a facility.
- The system supports adding, deleting, reorganizing, and exchanging experiments, including creation of acquisition and processing scripts from an experiment configured in TopSpin.
- NMRlib includes polarization-enhanced fast-pulsing NMR experiments such as SOFAST, BEST, and HADAMAC, plus calibration, processing, display, and analysis utilities.
- The authors position NMRlib as free for academic users and as an extensible platform for sharing NMR experiment tools between laboratories.

## Relevance
Adjacent source for the practical tooling side of FAIR analytical workflows. Unlike nmrML, this is not primarily a data standard; it is laboratory software for standardizing how NMR experiments are set up, executed, shared, and maintained. It supports the broader thesis point that machine-readable data ecosystems depend on routine tools at the instrument and workflow layer, not only on repositories and ontology pages.
