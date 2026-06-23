---
type: source
title: "CatMiner: Use of Large Language Models for Extracting and Analyzing Data from Heterogeneous Catalysis Literature"
created: 2025-06-23
updated: 2025-06-23
sources: [catminer-llm-catalysis-extraction]
tags: [catalysis, llm, text-mining]
---

# CatMiner: LLM-Based Text Mining for Heterogeneous Catalysis

**Authors:** Benjamin W. Walls, Suljo Linic
**Year:** 2025
**Venue:** ACS Catalysis
**File:** `docs/literature/use-of-large-language-models-for-extracting-and-analyzing-data-from-heterogeneous-catalysis-literature.pdf`

## Abstract
Develops CatMiner, a text mining tool that extracts arbitrary user-specified structure-environment-property data from catalysis literature using LLMs. Agnostic to LLM choice (OpenAI GPT, open-source Llama, DeepSeek). Case study on oxidative coupling of methane.

## Key Findings
- CatMiner extracts structure-environment-property triples from heterogeneous catalysis papers
- LLM-agnostic: works with both proprietary (GPT) and open-source (Llama, DeepSeek) models without modification
- Critical capabilities: domain knowledge injection, iterative prompting, document-wide context handling
- Identifies situations where CatMiner struggles; suggests community reporting standards

## Methodology
- Multi-turn question-answering formulation for text mining
- Chat-like memory and follow-up prompting strategies
- User-specified extraction schema (not predefined)

## Relevance
Practical tool for building catalysis databases from literature. Relevant to [[text-mining-for-catalysis]] and [[llm-for-catalysis-data-extraction]].