# Thesis Structure Proposal

Title: Development of an LLM-supported workflow for semantic metadata extraction from catalytic experiments

## 1. Goal of this structure

This structure is designed for a master's thesis that combines:
- catalysis / experimental chemistry context
- semantic metadata and ontologies
- LLM-supported information extraction
- workflow design and evaluation

It aims to clearly separate:
1. domain background
2. technical foundations
3. workflow design
4. implementation
5. evaluation
6. discussion and outlook

---

## 2. Recommended high-level chapter structure

### 1. Introduction
Purpose:
- motivate the problem
- explain why metadata extraction from catalytic experiments matters
- show why LLMs are relevant
- define research gap, objective, and thesis contributions

Suggested subsections:
- 1.1 Motivation
- 1.2 Problem Statement
- 1.3 Research Questions
- 1.4 Objectives and Scope
- 1.5 Methodological Approach
- 1.6 Thesis Outline

Notes:
- Keep this chapter compact and clear.
- End with a short paragraph describing the remainder of the thesis.

---

### 2. Background and Foundations
Purpose:
- establish the scientific and technical basis needed for the thesis

Suggested subsections:
- 2.1 Catalytic Experiments and Their Data Landscape
- 2.2 Metadata in Experimental Chemistry
- 2.3 Semantic Metadata, Ontologies, and Knowledge Graphs
- 2.4 Large Language Models for Information Extraction
- 2.5 Challenges of Unstructured and Semi-structured Experimental Data

Notes:
- This chapter explains core concepts.
- Focus on definitions and concepts that are needed later.

---

### 3. State of the Art
Purpose:
- review existing literature and systems
- position your work in relation to prior approaches

Suggested subsections:
- 3.1 Existing Approaches to Metadata Extraction in Chemistry
- 3.2 Semantic Data Models and Standards for Catalysis / Chemistry
- 3.3 NLP and LLM-based Extraction Workflows
- 3.4 Human-in-the-loop and Validation Approaches
- 3.5 Research Gap

Notes:
- This chapter should lead naturally to your own approach.
- The final subsection should explicitly justify your workflow design.

---

### 4. Requirements and Research Design
Purpose:
- translate the problem into system and research requirements
- define evaluation dimensions before presenting the solution

Suggested subsections:
- 4.1 Use Case and Data Sources
- 4.2 Functional Requirements for Metadata Extraction
- 4.3 Requirements for Semantic Representation
- 4.4 Quality Criteria for the Workflow
- 4.5 Evaluation Strategy

Possible quality criteria:
- extraction completeness
- correctness / precision
- schema conformity
- reproducibility
- interpretability
- effort reduction compared to manual extraction

Notes:
- This chapter is very useful because it creates a bridge between theory and method.

---

### 5. Design of the LLM-supported Workflow
Purpose:
- present the conceptual workflow before implementation details

Suggested subsections:
- 5.1 Overall Workflow Architecture
- 5.2 Input Data and Preprocessing
- 5.3 Prompting / Extraction Strategy
- 5.4 Semantic Mapping to Metadata Schema or Ontology
- 5.5 Validation and Error Handling
- 5.6 Human Review and Correction Loop
- 5.7 Output Representation and Storage

Notes:
- This is one of the core chapters of the thesis.
- Use one or more workflow diagrams here.
- Clearly distinguish between conceptual design and actual implementation.

---

### 6. Implementation
Purpose:
- describe how the workflow was technically realized

Suggested subsections:
- 6.1 System Environment and Tooling
- 6.2 Data Model / Metadata Schema
- 6.3 LLM Integration
- 6.4 Prompt Templates and Output Formats
- 6.5 Parsing, Post-processing, and Semantic Annotation
- 6.6 Storage, Interfaces, or Knowledge Base Integration
- 6.7 Limitations of the Implemented Prototype

Notes:
- Keep this chapter concrete and reproducible.
- If relevant, include versions, frameworks, and model choices.

---

### 7. Evaluation and Results
Purpose:
- demonstrate how well the workflow performs
- report results in a structured and transparent way

Suggested subsections:
- 7.1 Evaluation Setup
- 7.2 Dataset / Experimental Sample
- 7.3 Metrics and Assessment Procedure
- 7.4 Quantitative Results
- 7.5 Qualitative Error Analysis
- 7.6 Comparison with Baseline or Manual Process

Possible metrics:
- field-level precision / recall / F1
- exact match for schema fields
- ontology mapping success rate
- hallucination/error categories
- time saved in manual curation

Notes:
- This chapter should be evidence-driven.
- Tables are especially important here.

---

### 8. Discussion
Purpose:
- interpret the results
- connect findings back to the research questions

Suggested subsections:
- 8.1 Interpretation of Main Findings
- 8.2 Strengths of the Proposed Workflow
- 8.3 Weaknesses and Failure Modes
- 8.4 Implications for Catalysis Data Management
- 8.5 Generalizability to Other Experimental Domains

Notes:
- Do not repeat the results section.
- Focus on meaning, implications, and limitations.

---

### 9. Conclusion and Outlook
Purpose:
- conclude the thesis clearly and compactly

Suggested subsections:
- 9.1 Summary of Contributions
- 9.2 Answers to the Research Questions
- 9.3 Future Work

Notes:
- This should be concise but strong.
- End with a realistic outlook on future development and adoption.

---

## 3. Recommended appendices

Possible appendix content:
- prompt templates
- metadata schema
- ontology fragments
- annotation guidelines
- additional result tables
- example extracted records
- error category catalog

---

## 4. Suggested LaTeX file structure

A clean structure for your `Simon` folder could be:

```text
Simon/
├── main.tex
├── thesis-structure.md
├── Frontpage/
│   └── frontpage.tex
├── packages/
│   └── packages.sty
├── sections/
│   ├── 01_introduction.tex
│   ├── 02_background_foundations.tex
│   ├── 03_state_of_the_art.tex
│   ├── 04_requirements_research_design.tex
│   ├── 05_workflow_design.tex
│   ├── 06_implementation.tex
│   ├── 07_evaluation_results.tex
│   ├── 08_discussion.tex
│   └── 09_conclusion_outlook.tex
├── figures/
├── tables/
├── bibliography/
│   └── references.bib
└── appendix/
    ├── appendix_a_prompts.tex
    ├── appendix_b_schema.tex
    └── appendix_c_examples.tex
```

---

## 5. Suggested chapter mapping for `main.tex`

```latex
\input{Frontpage/frontpage}

\tableofcontents

\chapter{Introduction}
\input{sections/01_introduction}

\chapter{Background and Foundations}
\input{sections/02_background_foundations}

\chapter{State of the Art}
\input{sections/03_state_of_the_art}

\chapter{Requirements and Research Design}
\input{sections/04_requirements_research_design}

\chapter{Design of the LLM-supported Workflow}
\input{sections/05_workflow_design}

\chapter{Implementation}
\input{sections/06_implementation}

\chapter{Evaluation and Results}
\input{sections/07_evaluation_results}

\chapter{Discussion}
\input{sections/08_discussion}

\chapter{Conclusion and Outlook}
\input{sections/09_conclusion_outlook}

\printbibliography

\appendix
\input{appendix/appendix_a_prompts}
\input{appendix/appendix_b_schema}
\input{appendix/appendix_c_examples}
```

---

## 6. Strong narrative logic for your specific thesis

A good thesis story for your topic is:

1. Catalytic experiments generate valuable but poorly structured information.
2. High-quality semantic metadata is necessary for reuse, integration, and FAIR data practices.
3. Manual extraction is slow and inconsistent.
4. LLMs may support extraction, but they require controlled workflow design, validation, and semantic mapping.
5. Therefore, this thesis develops and evaluates a structured LLM-supported workflow.

This narrative should stay visible throughout the thesis.

---

## 7. Recommendation for chapter emphasis

For your topic, the most important chapters will likely be:
- Chapter 3: State of the Art
- Chapter 5: Design of the LLM-supported Workflow
- Chapter 7: Evaluation and Results
- Chapter 8: Discussion

These chapters will probably carry the scientific contribution most strongly.

---

## 8. Minimum viable version if time gets tight

If you need a lean but strong thesis, preserve this order:
- Introduction
- Background and Foundations
- State of the Art
- Requirements and Research Design
- Workflow Design
- Evaluation and Results
- Discussion
- Conclusion

In that case, implementation details can be shortened and partially moved to the appendix.

---

## 9. Next recommended step

The next useful step is to convert this structure into:
1. a chapter-by-chapter writing plan
2. a LaTeX scaffold with section files
3. a set of research questions and objectives aligned to the structure
