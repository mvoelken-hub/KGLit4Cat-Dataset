---
type: source
title: "OpenChemIE: An Information Extraction Toolkit for Chemistry Literature"
created: 2026-06-29
updated: 2026-06-29
sources: [openchemie]
tags: [chemistry, text-mining, multimodal, reaction-extraction]
---

# OpenChemIE

**Authors:** Vincent Fan, Yujie Qian, Alex Wang, Amber Wang, Connor W. Coley, Regina Barzilay
**Year:** 2024
**Venue:** arXiv preprint / Journal of Chemical Information and Modeling entry in thesis bibliography
**File:** `docs/literature/2404.01462v1.pdf`
**Code:** https://github.com/CrystalEye42/OpenChemIE
**Web:** https://mit.openchemie.info

## Abstract
OpenChemIE is an open-source toolkit for document-level reaction extraction from chemistry literature. It addresses the multimodal nature of reaction data by extracting information from text, tables, and figures, then integrating the outputs into complete reaction records.

## Key Findings
- Treats reaction extraction as a document-level task rather than a single-modality text or figure task.
- Combines specialized modules for molecule recognition, reaction diagram parsing, chemical named entity recognition, reaction text extraction, molecule-label coreference, and R-group resolution.
- Uses chemistry-informed integration algorithms to align reaction templates, substrate scope tables, R-groups, conditions, yields, and molecular structures.
- Introduces/uses a manually annotated evaluation set of 1007 reactions from 78 substrate-scope figures across five organic chemistry journals.
- Reports 69.5% F1 on the R-group/substrate-scope reaction extraction evaluation and 64.3% soft-match accuracy against Reaxys entries.
- Error analysis identifies molecule recognition and OCR as major bottlenecks, while the R-group resolution algorithm is comparatively robust when upstream inputs are correct.

## Relevance
OpenChemIE broadens the wiki's chemistry IE coverage beyond prose-only extraction. For SIMONE, it is useful evidence that scientific extraction workflows often need to route evidence across modalities and document structures, not merely prompt over text chunks. It complements [[swain2016chemdataextractor]], [[knowledge-extraction-catalysis-literature]], [[schilling2025text]], and [[llm-kg-extraction-tables-materials]].

## Limitations
- The pipeline remains sensitive to PDF parsing, diagram segmentation, molecule recognition, and OCR errors.
- The public web portal is constrained by compute limits and processes only a limited number of pages per paper.
- The system focuses on chemistry reaction extraction and does not directly solve ontology grounding, application-profile validation, or catalysis-specific metadata completeness.
