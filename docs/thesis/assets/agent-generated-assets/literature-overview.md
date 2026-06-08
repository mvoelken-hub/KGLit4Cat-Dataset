## Literature overview of the Google Drive folder

The folder contains a coherent literature set around **FAIR catalysis data**, **ontology-based data management**, **knowledge graphs**, and **LLM/AI-assisted catalyst discovery**. The sources can be read as four connected strands: foundational ontology work, catalysis-specific research data infrastructure, knowledge graph construction/extraction, and AI/LLM methods for catalysis and chemical reasoning.

### 1. Ontologies as the semantic foundation

The conceptual foundation is built by classical ontology literature and engineering-domain ontologies. Guarino’s *Formal Ontology and Information Systems* frames ontologies as central architectural components for information systems, especially where knowledge engineering, database integration, information retrieval, and information extraction intersect.  BFO-oriented material provides a more philosophical and methodological basis: an ontology is described as a representational artifact with a taxonomy as a proper part, intended to represent universals, defined classes, and relations. 

For chemical and process engineering, *OntoCAPE* is the key domain reference. It is explicitly positioned as a reusable ontology for chemical process engineering and computer-aided process engineering, intended to provide shared conceptualization and reusable knowledge components for CAPE software applications.  OntoCAPE is therefore directly relevant when simulation data, process units, reaction systems, and engineering models need to be represented in a structured semantic layer.

Recent ontology-matching papers extend this foundation toward interoperability between independently developed semantic resources. The folder includes work on heterogeneous graph neural networks, LLM embeddings, and self-supervised structural ontology matching. These works address the same core problem: semantic systems are fragmented, and useful research data integration depends on aligning classes, properties, and relations across ontologies. For example, the LLM-embedding ontology matching paper reports that LLM integration improved its baseline F-measure by 45%.  The LaKERMap paper similarly combines contextual and structural information for ontology matching and reports improved alignment quality and runtime. 

### 2. FAIR catalysis data and research data infrastructure

A second major strand focuses on the catalysis community’s need for standardized, machine-readable, reusable data. Wulf et al. identify the central issue clearly: catalysis produces valuable data, but access to high-quality, reusable data remains limited; implementing FAIR principles would improve the situation substantially.  The paper also emphasizes that catalysis data must bridge molecule-scale information, material properties, reactor behavior, and process engineering, which makes metadata and ontology design unusually demanding. 

The NFDI4Cat white paper operationalizes this view. It argues that catalysis research generates heterogeneous data and that ontologies, scientific vocabularies, standard metadata, and knowledge graphs can structure this data in a FAIR way for later analysis.  It also gives a semantic spectrum from lists and thesauri to conceptual models and OWL ontologies, showing how stronger semantics improve interoperability and machine interpretability. 

The practical NFDI4Cat booklet, *Orchestrating Catalysis Data*, translates this into tools and services: Voc4Cat for catalysis terminology, CoreMeta4Cat for minimum catalysis metadata, DCAT-AP+/Chem-DCAT-AP for metadata structures, Repo4Cat for publication and discovery, KGLit4Cat for ontology-based knowledge graph creation from literature, and CaRMeN for reaction mechanisms, kinetic models, and reactor simulations.   

A newer metadata contribution, ChemDCAT-AP/DCAT-AP+, is especially relevant for implementation. It proposes a generic DCAT-AP extension that can be specialized by domains while preserving cross-domain compatibility, and it uses LinkML to support schema inheritance, validation, and format conversion.  This is important because it connects semantic modeling with practical software artifacts such as SHACL shapes, JSON Schema, Python/Pydantic classes, and RDF knowledge graphs. 

### 3. Knowledge graphs for catalysis and scientific discovery

The knowledge-graph literature in the folder provides the bridge between semantic standards and AI-ready data. A 2025 review on knowledge graphs in heterogeneous catalysis describes KGs as machine-readable frameworks for integrating catalytic materials, reaction conditions, mechanisms, and synthesis routes under FAIR principles.  It highlights ontology-guided text mining, graph population, maintenance, LLM-assisted querying, catalyst recommendation systems, and reaction mechanism discovery as the main application areas. 

A broader survey on scientific knowledge graphs frames SciKGs as infrastructures for AI-for-science. It emphasizes applications in drug development, omics, chemical reaction prediction, and materials design, and argues that SciKGs and LLMs are complementary: knowledge graphs provide structured grounding, while LLMs support extraction, semantic enrichment, and interaction.  

For catalysis-specific ontology discovery, *Ontologies4Cat* is central. It classifies ontologies relevant to the catalysis research data value chain and provides a reusable workflow and codebase for representing ontology metadata.  This fills an important practical gap: researchers need not only ontologies, but also criteria for selecting, comparing, and reusing them.

### 4. LLM-based extraction, ontology generation, and catalyst discovery

The folder also contains a strong group of recent LLM and AI-for-catalysis papers. These sources shift from data infrastructure toward automated knowledge acquisition and experimental discovery.

Walls and Linic’s CatMiner paper is a clear example. It uses LLMs to extract user-specified structure–environment–property data from heterogeneous catalysis literature, supports multiple model families, and demonstrates extraction of oxidative coupling of methane records.  The paper is useful because it exposes both the potential and the limitations of LLM extraction: performance depends on model choice, prompting strategy, domain knowledge, document-wide context, and reporting standards. 

Another paper uses open LLMs for end-to-end ontology and knowledge graph generation from scientific literature. It applies the method to single-atom catalysts and argues that LLM-generated ontologies and KGs can support retrieval and reasoning where manual curation is too slow.  This is particularly relevant for emerging catalysis topics where no mature ontology or curated database exists.

On the catalyst-design side, CataLM is introduced as a domain-adapted LLM for electrocatalytic materials, trained on catalyst literature and expert-annotated data, with tasks such as entity extraction and control-method recommendation.  ChemDFM-R similarly targets chemical reasoning by adding atomized chemical knowledge, especially functional-group information and reaction-level transformations, to improve interpretability and reasoning quality. 

The folder also contains more conventional AI-in-catalysis examples. The time-on-stream catalyst reactivity paper combines systematic experiments with AI methods such as subgroup discovery and symbolic regression to model time-dependent catalyst performance under industrially relevant acetylene hydrogenation conditions.  This complements the semantic-data literature by showing why structured experimental metadata is necessary: time-on-stream, catalyst history, material descriptors, and reaction environment all influence catalytic behavior.

### 5. Process engineering and simulation relevance

Several files connect the semantic-data topic back to process engineering and simulation. The IFAC paper on non-intrusive Time-POD for optimal control of a fixed-bed CO₂ methanation reactor is directly relevant to automated simulation workflows: it treats the simulation as a black-box input-output system and proposes low-dimensional temporal parametrizations for optimization.  This supports a thesis framing in which semantic data management is not only for archiving results, but also for enabling automated model setup, parameterization, optimization, and reuse.

The broader heterogeneous catalysis review provides contextual background on why this matters. It shows catalysis as a historically central industrial technology, emphasizes the importance of catalyst activity and selectivity, and identifies future challenges in energy, environment, CO₂ upgrading, biomass conversion, and hydrogen-related processes. 

## Synthesis for a thesis literature review

Taken together, the folder supports the following argument:

Current catalysis research suffers less from a lack of data generation than from a lack of **standardized, machine-readable, semantically interoperable, and reusable data structures**. FAIR principles define the target state, but implementation requires a layered infrastructure: controlled vocabularies such as Voc4Cat, metadata profiles such as CoreMeta4Cat and ChemDCAT-AP, persistent identifiers, repositories such as Repo4Cat, and ontologies capable of representing reactions, catalysts, processes, simulations, and provenance. Ontologies and knowledge graphs provide the semantic backbone for this infrastructure, while LinkML, SHACL, RDF, SPARQL, and DCAT-style profiles make the semantics operational in software.

The recent AI/LLM literature adds a second layer: once data and literature can be transformed into structured knowledge graphs, LLMs can support extraction, querying, ontology generation, catalyst recommendation, and experimental optimization. However, these methods remain dependent on data quality, reporting standards, ontology alignment, and provenance. Therefore, the most important research gap is not merely “using AI for catalysis,” but building trustworthy semantic workflows that connect literature, experimental data, simulation inputs, simulation outputs, and repository metadata into a reproducible knowledge infrastructure.

A suitable thesis-oriented formulation would be:

> The reviewed literature shows a convergence between FAIR research data management, ontology-based semantic modeling, knowledge graph construction, and AI-assisted catalysis research. Foundational ontologies such as BFO and OntoCAPE provide reusable conceptual structures, while NFDI4Cat-related work translates these principles into catalysis-specific metadata standards, vocabularies, repositories, and workflow tools. Recent knowledge graph and LLM studies demonstrate that structured semantic representations can support automated extraction, retrieval, reasoning, and catalyst discovery. Yet the literature also identifies persistent challenges: heterogeneous data formats, incomplete metadata, weak reporting standards, ontology alignment, and long-term graph maintenance. These limitations motivate ontology-based workflows that semantically integrate reaction, catalyst, process, and simulation data and make them reusable for automated process simulation and downstream AI-driven analysis.
