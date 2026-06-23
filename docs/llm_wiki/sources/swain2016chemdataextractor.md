---
type: source
title: "ChemDataExtractor: A Toolkit for Automated Extraction of Chemical Information from the Scientific Literature"
created: 2025-06-23
updated: 2025-06-23
sources: [swain2016chemdataextractor]
tags: [text-mining, catalysis]
---

# ChemDataExtractor

**Authors:** Matthew C. Swain, Jacqueline M. Cole (University of Cambridge)
**Year:** 2016
**Venue:** J. Chem. Inf. Model. 56(10): 1894–1904
**File:** `docs/literature/chemdataextractor-a-toolkit-for-automated-extraction-of-chemical-information-from-the-scientific-literature.pdf`

## Abstract
Complete toolkit for automated extraction of chemical entities, properties, measurements, and relationships from scientific documents. Chemistry-aware NLP pipeline with tokenization, POS tagging, NER, and phrase parsing. Multiple rule-based grammars for different document domains (text, captions, tables). Document-level processing resolves data interdependencies.

## Key Findings
- F-scores: 93.4% (chemical identifiers), 86.8% (spectroscopic attributes), 91.5% (chemical property attributes)
- Competitive 87.8% F-score on CHEMDNER challenge
- Unsupervised word clustering on massive chemistry corpus for improved NER
- Rule-based grammars tailored for paragraphs, captions, and tables
- Document-level processing for resolving cross-references
- Open-source (MIT license): http://www.chemdataextractor.org

## Relevance
Foundational tool for chemical text mining. Precursor to LLM-based approaches like CatMiner ([[catminer-llm-catalysis-extraction]]) and SPIRES ([[spires-ontogpt]]). Relevant to [[text-mining-for-catalysis]].