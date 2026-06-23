---
type: concept
title: "Formal Ontology: Theory and Foundations"
created: 2025-06-23
updated: 2025-06-23
sources: [guarino-fois98, bfo-textbook]
tags: [formal-ontology, ontology]
---

# Formal Ontology: Theory and Foundations

## Overview
Formal ontology provides the theoretical foundations for building domain ontologies. It draws on philosophy, linguistics, and logic to analyze the structure of reality and create rigorous vocabularies for knowledge representation.

## Core Definitions (Guarino, [[guarino-fois98]])

### Ontology (AI sense)
An engineering artifact: a specific vocabulary used to describe reality, plus a set of explicit assumptions regarding the intended meaning of vocabulary words. Typically formalized as a first-order logical theory where vocabulary words appear as concept (unary predicate) and relation (binary predicate) names.

### Conceptualization
The philosophical reading of ontology — a particular system of categories accounting for a vision of the world, independent of language used to describe it. Two ontologies can share the same conceptualization but differ in vocabulary.

### Ontological Commitment
Adopting a particular conceptualization when using a vocabulary.

## Ontology Structure (BFO, [[bfo-textbook]])

### Ontology
A representational artifact comprising a taxonomy as proper part, whose representations designate universals, defined classes, and relations between them.

### Taxonomy
A hierarchy of terms denoting types (universals) linked by subtype relations. Single root, unique parent per node.

### Key Terms
- **Entity**: anything that exists (objects, processes, qualities)
- **Representation**: an entity that refers to some other entity (term, idea, image, label)
- **Universal/type**: the entities referred to by nodes in a taxonomy

## Architectural Role
Guarino argues ontologies play a central role in information systems:
- **Information resources**: data models, schemas
- **User interfaces**: vocabulary for communication
- **Application programs**: domain knowledge for reasoning

This leads to the perspective of **ontology-driven information systems**.

## Relevance
These foundations underpin all catalysis ontology work (see [[ontology-development]]). BFO serves as upper-level ontology for domain ontologies. Guarino's definitions are the standard reference in the field.