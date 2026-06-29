---
type: source
title: "Reac4Cat-Ontology: Harnessing the Power of Ontological Description Logic in Catalysis Research as a Practical Approach to Knowledge Inferences"
created: 2026-06-28
updated: 2026-06-28
sources: [reac4cat-ontology]
tags: [ontology, catalysis, knowledge-graph, semantic-web, process-simulation, nfdi]
---

# Reac4Cat-Ontology

**Authors:** Alexander S. Behr, Hendrik Borgelt, Norbert Kockmann (TU Dortmund)
**Year:** 2024
**Venue:** Datenbank-Spektrum 24:139-150
**DOI:** 10.1007/s13222-024-00476-3
**File:** `docs/literature/s13222-024-00476-3.pdf`

## Abstract
Introduces Reac4Cat as an ontology-based knowledge graph approach for catalysis and reaction research. The paper shows how OWL description logic, especially general class axioms / left-hand-side logic, can classify reaction experiments, infer catalytic roles, and enrich laboratory/process-simulation data beyond what relational data models comfortably express.

## Key Findings
- Models reaction experiments through educt mixtures, product sets, reaction components, reaction roles, and catalyst relations so competency questions about products, side reactions, catalytic effects, and reaction classes can be answered.
- Reuses and connects established semantic resources including RXNO/MOP for reaction processes, ChEBI for substances, BFO/OBO relations, SHACL, SPARQL, and OWL reasoners.
- Uses General Class Axioms (GCAs) to infer reaction-role assignments and catalyst participation from component relations rather than manually asserting every derived fact.
- Demonstrates the ontology on a biocatalytic redox process connected to EnzymeML and DWSIM process-simulation data, with Python-generated GCAs and HermiT reasoning.
- Positions Reac4Cat as a practical automation layer for semantic annotation, data enrichment, digital agents, and process intensification in catalysis workflows.

## Limitations
- OWL's open-world assumption and mostly unary description logic make generalized reaction inference, mathematical conditions, concentrations, and environmental constraints hard to express.
- GCAs can be axiom-intensive and computationally costly; the paper reports a nontrivial reasoning time even for the demonstration graph.
- Complex reaction systems, intermediates, and ring-closure-like logical patterns remain difficult to model while preserving factuality.
- The authors argue against one monolithic catalysis knowledge graph; specialized domain graphs, for example for biocatalysis, are more computationally realistic.

## Relevance
Important bridge between ontology landscape work ([[ontologies4cat]]) and later reaction-classification pipelines ([[ontology-reaction-classification]]). It is especially relevant to [[ontology-development]], [[catalysis-ontologies]], [[knowledge-graphs-for-catalysis]], and SIMONE's distinction between extracting metadata from heterogeneous packages and grounding that metadata in formal catalysis semantics.

## Outlook
The paper points toward mergeable graphs via SPARQL, possible SHACL-rule inference, and LinkML-based data uptake as future mechanisms for mapping incoming metadata directly to ontology concepts.
