# Architecture and Dataset Workflow of the SIMONE Prototype

## Semantic Inference Module for Ontology-driven Node Extraction

## 1. Purpose and role of the prototype

SIMONE is implemented as a full-stack prototype for **LLM-supported semantic metadata extraction from heterogeneous scientific data packages**. Its purpose is to take an uploaded research dataset archive, extract textual content from the contained files, split that content into semantically meaningful chunks, extract structured experimental context with an LLM, normalize extracted attributes against semantic vocabularies, and finally project the accumulated extraction context into a registered metadata profile.

The repository README defines SIMONE as the **Semantic Inference Module for Ontology-driven Node Extraction** and describes the thesis contribution as an automated extraction approach that turns heterogeneous catalysis research data into structured, reusable, FAIR-compliant datasets.  The prototype objective is stated more concretely as transforming an uploaded dataset archive into a progressively refined metadata representation by extracting document content, deriving artifact context, generating an initial metadata draft, refining it iteratively, and enriching selected fields with vocabulary-backed semantic references. 

The system is therefore not a single extraction script. It is an application architecture around five interacting concerns:

1. **Datasource ingestion and text extraction**
2. **Semantic chunking**
3. **LLM-based extraction of experimental context**
4. **Vocabulary-based semantic normalization**
5. **Profile projection and schema validation**

---

## 2. High-level architecture

At runtime, SIMONE consists of a frontend, backend API, local or remote infrastructure services, persistent file-system repositories, and external or local model services.

```text
┌────────────────────────────────────────────────────────────────────┐
│                            User / Agent                             │
│ Uploads data package, selects profile, starts extraction, reviews UI │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                         React / Vite Frontend                       │
│ Dataset upload, chunk inspection, extraction status, vocabulary UI   │
└───────────────────────────────┬────────────────────────────────────┘
                                │ HTTP / JSON
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                         FastAPI Backend                             │
│ Datasource API | Extraction API | Semantic API | Profile API         │
└───────────────┬───────────────┬────────────────┬───────────────────┘
                │               │                │
                ▼               ▼                ▼
┌────────────────────┐ ┌────────────────┐ ┌──────────────────────────┐
│ File-system storage │ │ Ollama service │ │ Neo4j semantic graph      │
│ uploads, chunks,    │ │ chat + embed   │ │ RDF vocabularies, indexes │
│ profiles, outputs   │ │ models         │ │ vector/full-text search   │
└────────────────────┘ └────────────────┘ └──────────────────────────┘
```

The backend is a FastAPI application titled **“Semantic Metadata Extraction API.”** During startup it initializes logging, setup tasks, Neo4j, Ollama, the task registry, and the semantic service.  It registers separate API routers for system, datasources, extraction, profiles, and semantic functionality.  

The prototype is intended to run with Docker, `uv`, Neo4j, Ollama, a backend API, and a frontend. The README lists Docker Desktop and `uv` as required tools and Node/npm as optional for frontend hot reload.  The backend project is a Python package named `SIMONE`, version `0.1.0`, requiring Python `>=3.12`, with dependencies including FastAPI, Ollama, pandas, PyMuPDF, rdflib, rdflib-neo4j, scikit-learn, Pydantic, LinkML, jsonschema, and Typer.  The frontend is a React/Vite/TypeScript application. 

---

## 3. Runtime components

### 3.1 CLI and application lifecycle

SIMONE provides a command-line wrapper named `simone`. The wrapper delegates execution into the backend environment by calling `uv --directory "$SCRIPT_DIR/backend" run simone "$@"`.  The backend registers the CLI command as `simone = "app.cli:app"`. 

The CLI manages several runtime modes:

* **Production mode:** starts API and frontend in Docker, with Neo4j and Ollama also started locally unless configured as remote services.
* **Development mode:** runs the API locally with Uvicorn reload, while Neo4j and Ollama run in Docker unless configured remotely.
* **Infrastructure mode:** starts only Neo4j and/or Ollama, useful when another machine connects to these services remotely.

The README explicitly describes these modes and their commands: `simone up`, `simone dev`, and `simone host`.  

### 3.2 Backend API

The FastAPI backend is the central orchestration layer. It exposes domain-specific APIs:

```text
/api/v1/datasources  → upload, inspect, delete, chunk data packages
/api/v1/extraction   → run extraction, pause, track progress, rerun vocab queries
/api/v1/semantic     → import vocabularies, query vocabularies, generate embeddings
/api/v1/profiles     → register profiles, validate documents, export JSON-LD
/api/v1/system       → health, model/runtime configuration
```

The routers are registered centrally in `app/main.py`. 

### 3.3 Dependency wiring

The backend dependencies are assembled in `app/dependencies.py`. This file wires together the configured settings, task registry, Neo4j driver, Ollama client, datasource service, profile service, semantic service, and extraction service.  It uses file-system repositories for datasource blobs, profiles, and extraction outputs, while the semantic graph repository is backed by Neo4j.  The extraction service is constructed with access to the profile service, datasource service, Ollama client, output repository, task registry, and semantic service. 

This wiring is important because it shows that extraction is not isolated. Extraction depends on:

* stored data packages,
* chunked content,
* a registered profile,
* LLM calls,
* semantic vocabulary queries,
* persistent output storage,
* asynchronous task tracking.

---

## 4. Persistent storage architecture

SIMONE uses different storage layers for different concerns.

```text
.runtime / data directories
│
├── uploads / datasource blobs
│   └── uploaded ZIP packages, extracted package metadata, chunk outputs
│
├── profiles / profile artifacts
│   └── registered profile manifests, JSON Schemas, JSON-LD contexts
│
├── outputs / extraction outputs
│   └── extraction context, run state, warnings, final result, token usage
│
└── logs / runtime logs
```

The source code shows the following concrete repository construction:

* `FileSystemDataSourceBlobRepository(settings.uploads_dir)`
* `FileSystemProfileRepository(settings.dcat_profiles_dir)`
* `FileSystemExtractionOutputRepository(settings.output_dir)`
* `Neo4jSemanticGraphRepository(neo4j_driver, ollama_client)`

These are wired in `app/dependencies.py`. 

Neo4j stores RDF vocabulary resources and graph relations. The semantic graph repository imports RDF graphs, creates vocabulary scheme nodes, links them to resources, builds vector and full-text indexes, and performs graph expansion.  

---

## 5. Frontend architecture

The frontend is a React/Vite application named `simone-frontend`.  It imports API clients for datasources, extraction, profiles, system, and semantic services. 

The frontend supports the following current workflow interactions:

* uploading and selecting data packages,
* triggering chunking,
* inspecting chunks,
* registering and selecting profiles,
* starting extraction,
* monitoring extraction progress,
* pausing extraction,
* viewing extraction context objects,
* viewing trace/source-text evidence,
* configuring vocabulary query parameters,
* rerunning vocabulary queries,
* viewing token usage.

The UI contains dedicated rendering logic for structured extraction context. Extracted objects are grouped as resources, methods, activities, entities, and agents.  A modal shows the extracted object and the trace source text, making the extraction evidence visible to the user. 

Some older patch-review concepts remain in the frontend as compatibility stubs. For example, patch artifacts return empty arrays, saved drafts are returned directly without backend persistence, and `resolvePatchReview()` reports that manual patch review was removed from the extraction workflow.   This indicates that the current active workflow is the backend extraction pipeline rather than a separate manual patch-review pipeline.

---

## 6. Semantic vocabulary architecture

### 6.1 Vocabulary import

SIMONE implements a semantic vocabulary subsystem. Vocabularies can be imported through the `/semantic/vocabularies` API from either RDF URLs or uploaded RDF files.  The semantic service loads the RDF graph, constructs vocabulary metadata, imports it into the graph repository, creates indexes, and generates embeddings. 

### 6.2 Initial vocabularies

The prototype has a configured bootstrap vocabulary list. It currently includes:

| Identifier                                | Source              |
| ----------------------------------------- | ------------------- |
| `https://w3id.org/nfdi4cat/voc4cat`       | Voc4Cat TTL         |
| `http://qudt.org/vocab/quantitykind`      | QUDT quantity kinds |
| `http://qudt.org/vocab/unit`              | QUDT units          |
| `http://qudt.org/vocab/constant`          | QUDT constants      |
| `http://purl.obolibrary.org/obo/chmo.owl` | CHMO                |
| `http://nmrML.org/nmrCV`                  | nmrCV               |

These are defined in `INITIAL_VOCABS`. 

### 6.3 Vocabulary indexing

For each vocabulary term scheme, the Neo4j repository creates both vector and full-text indexes. Vector indexes are built on the `embedding` property, while full-text indexes are built over the available textual properties of the RDF resources. 

### 6.4 Vocabulary query flow

A vocabulary query has two retrieval branches:

1. **Vector branch:** the query text is embedded through Ollama and compared with stored vocabulary embeddings.
2. **Full-text branch:** the query text is passed to a Neo4j full-text index.

The semantic service fuses both result lists with reciprocal rank fusion and uses the best fused candidates as graph seeds.  Around those seeds, it expands graph context using configured allowed relationship types, traversal direction, maximum hops, and maximum statements per seed. 

The README describes the same flow as: query parameters arrive, each RDF type is processed separately, vector candidates are retrieved, full-text candidates are retrieved, candidates are fused with reciprocal rank fusion, top seeds become graph starting points, graph context is expanded, raw entities are cleaned, context objects are built, and the final result is compacted.   

---

## 7. Profile architecture

Profiles define the target metadata shape. They are separate from the generic extraction context.

A profile can be registered from either:

* a remote schema URL, or
* an uploaded schema file.

The profile API allows registering profiles, listing profiles, retrieving JSON Schema and JSON-LD context artifacts, validating documents, exporting JSON-LD, retrieving profile manifests, and deleting profiles.  

The profile service enforces that exactly one schema source is supplied, generates profile artifacts, validates documents against the profile schema, and exports documents to JSON-LD.  

This separation is central:

```text
ExtractionContext
    Generic intermediate representation:
    activities, methods, resources, entities, agents, quantities, qualitative attributes, traces

Profile document
    Final target representation:
    schema-conformant metadata document for a selected profile target class
```

The LLM first extracts generic experimental context. Only later is that context projected into a target profile schema.

---

# 8. Dataset path through the SIMONE workflow

The following section describes the complete lifecycle of a dataset inside SIMONE.

---

## Stage 0: Runtime preparation

Before a dataset can be processed, the runtime environment must be available.

```text
simone up / simone dev
        │
        ▼
FastAPI backend starts
        │
        ├── logging initialized
        ├── Neo4j driver initialized
        ├── Ollama client initialized
        ├── task registry initialized
        ├── initial vocab setup may run
        └── API routes mounted
```

The FastAPI lifespan startup performs setup through `start_setup()` with settings, logger, Ollama client, Neo4j driver, task registry, and semantic service. 

Production mode can pull configured Ollama models, import initial vocabularies, and generate missing embeddings. The README states that production pulls configured models, imports initial vocabularies, and generates missing embeddings unless overridden. 

---

## Stage 1: Dataset upload

The user uploads a dataset archive through the frontend. The frontend sends the archive to:

```text
POST /api/v1/datasources
```

The backend endpoint receives an uploaded file, checks that it has a filename, reads its bytes, and passes it to `DataSourceService.save_data_package()`. 

The expected upload unit is a ZIP-based data package. `DataPackage.from_bytes()` opens the uploaded bytes as a ZIP file. If the uploaded file is not a valid ZIP archive, it raises an invalid ZIP error. 

---

## Stage 2: ZIP expansion and file-entry construction

Once the archive is accepted, SIMONE traverses the ZIP file. Every non-directory entry becomes a `FileEntry`. If a file inside the ZIP is itself a ZIP archive, SIMONE recursively expands it and adds its contained files as nested entries. 

Each `FileEntry` contains:

* `file_path`
* `file_name`
* `file_extension`
* `raw_content`
* computed `file_type`

The `FileEntry` model derives the file type from the file extension and can extract textual content through `get_extracted_content()`. 

The resulting `DataPackage` receives a deterministic ID derived from the package filename and list of file paths. 

```text
uploaded archive.zip
        │
        ▼
DataPackage
        │
        ├── FileEntry(path="report.pdf", raw_content=...)
        ├── FileEntry(path="measurements.xlsx", raw_content=...)
        ├── FileEntry(path="instrument_export.txt", raw_content=...)
        └── FileEntry(path="nested/archive/file.csv", raw_content=...)
```

The datasource blob repository then stores the uploaded package under the package ID. 

---

## Stage 3: File-type specific text extraction

When content is needed, each file is converted into text according to its file type.

SIMONE currently recognizes:

| File type    | Extensions                                                        | Extraction behavior                              |
| ------------ | ----------------------------------------------------------------- | ------------------------------------------------ |
| Archive      | `.zip`                                                            | recursively unpacked during package construction |
| Table        | `.xls`, `.xlsx`, `.xlsm`, `.xlsb`, `.odf`, `.ods`, `.odt`, `.csv` | converted to CSV-like text                       |
| PDF          | `.pdf`                                                            | text extracted with PyMuPDF                      |
| Image        | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tiff`, `.tif`          | placeholder string, no OCR                       |
| Default text | all other extensions                                              | decoded with charset detection                   |

The file-type definitions and extraction dispatch are implemented in `file_types.py`.  

For spreadsheets, SIMONE reads sheets with pandas, filters some fully numeric interior rows, and serializes the remaining sheet content into CSV-like text with sheet headers.  For PDFs, it iterates over pages and concatenates extracted text.  Images are not OCR-processed; they return the placeholder `"[Image content cannot be extracted as text]"`. 

---

## Stage 4: Semantic chunking

Before extraction can run, the dataset must be chunked. The frontend calls:

```text
POST /api/v1/datasources/chunk
```

The chunk request includes:

* data package ID,
* buffer window size,
* semantic chunking threshold,
* whether existing chunks should be replaced,
* optional protected line indices,
* optional text-quality configuration.

The API forwards these parameters to `DataSourceService.chunk_file_entries_in_data_package()`. 

### 4.1 Chunking task management

Chunking runs asynchronously through the task registry. If chunks already exist and replacement is not requested, they are returned. If chunking is not yet running, a chunking task is created.  Existing chunking status can be queried through `/datasources/{id}/chunks/status`. 

### 4.2 Chunking algorithm

For each file entry, SIMONE:

1. extracts text content,
2. splits it into lines,
3. filters lines through a text-quality classifier,
4. reinserts protected lines if requested,
5. if too few lines remain, returns a single chunk,
6. combines each line with neighboring lines using a buffer window,
7. embeds each combined line window,
8. calculates cosine distances between adjacent windows,
9. computes a percentile threshold,
10. creates chunk boundaries where distances exceed that threshold,
11. stores chunks with original line indices.

This is implemented in `ContentChunk.create_chunks_for_file_entry()`.   

The line-window mechanism is implemented by `combine_lines()`, which combines each line with its neighboring lines according to the buffer size.  The semantic distance calculation uses cosine similarity between embeddings and converts it to distance with `1 - similarity`. 

The result is a list of `ContentChunk` objects:

```text
ContentChunk
    content: textual chunk
    data_package_id: package ID
    file_path: original file path
    start_idx: first original line index
    end_idx: last original line index
    filtered_line_indices: retained original line numbers
    summary: optional
    embedding: optional
```

The `ContentChunk` fields are defined in the chunking domain model. 

---

## Stage 5: Extraction run initialization

After chunking is complete, the user starts extraction through:

```text
POST /api/v1/extraction/run
```

The request includes:

* `data_package_id`
* `profile_identifier`
* optional qualitative vocabulary identifiers
* optional resume flag

The extraction endpoint calls `ExtractionService.run_extraction()`. 

Before starting, the service checks:

1. the data package exists,
2. the profile exists,
3. the profile JSON Schema can be loaded,
4. completed chunks exist.

If completed chunks do not exist, extraction fails with `ChunkingRequiredError`: “Extraction requires completed datasource chunking. Run chunking first.” 

If no run is active, the service creates an asynchronous workflow task. 

---

## Stage 6: Profile loading and validation schema preparation

Inside the workflow task, SIMONE loads:

* the data package,
* the selected profile manifest,
* the profile JSON Schema,
* a validation schema for the profile target class,
* completed content chunks.

This occurs at the start of `_run_extraction_task()`. 

The validation schema is later used for profile projection. This means the workflow knows the final target structure before it asks the LLM to produce the final metadata document.

---

## Stage 7: File ranking

Before processing chunks, SIMONE ranks files. The rationale is that a ZIP package may contain many files with different relevance. Instead of simply processing files in archive order, the workflow asks the LLM to rank file paths by likely metadata relevance.

The workflow creates a list of file contexts with file path and byte size, then calls the LLM using `FILE_RANKING_SYSTEM_PROMPT` and `build_file_ranking_prompt()`.  If the LLM ranking fails, the system records a warning and falls back to heuristic ranking. 

The resulting ranked files are stored in the extraction run state and progress object. 

---

## Stage 8: Chunk ordering

After ranking files, SIMONE flattens all chunk groups and orders the chunks by:

1. ranked file order,
2. file path,
3. start line index.

This is implemented by `_ordered_chunks()`. 

The result is a deterministic chunk sequence for extraction:

```text
ranked files
        │
        ▼
ordered chunks
        │
        ├── chunk 0: most relevant file, early lines
        ├── chunk 1: most relevant file, later lines
        ├── chunk 2: next file
        └── ...
```

---

## Stage 9: Chunk-level LLM extraction

Each ordered chunk is processed into an `ExtractionContext`.

### 9.1 Extraction prompt

The extraction prompt instructs the model to extract structured metadata about scientific experiments from unstructured text. It asks for:

* data-generating activities,
* evaluated entities,
* agentic entities,
* resources,
* methods,
* quantitative attributes,
* qualitative attributes,
* source-text evidence.

It also instructs the model not to create standalone objects for low-level parameter names or isolated technical metadata unless they clearly represent meaningful activities, methods, resources, agents, or targets. 

### 9.2 Chunk context sent to the model

Each model call receives:

* chunk content,
* file path,
* start and end line indices,
* data package name,
* optional accumulated extraction context from previous chunks of the same file.

The prompt builder serializes this as chunk metadata plus residual chunk content. 

### 9.3 Structured output

The model is called through `generate_structured()` with output type `ExtractionContext`.  The `ExtractionContext` consists of traced extraction objects. 

The main intermediate object types are:

| Object type              | Meaning                                                                      |
| ------------------------ | ---------------------------------------------------------------------------- |
| `DataGeneratingActivity` | measurement, acquisition, analysis, processing, or other data-generating run |
| `Method`                 | protocol, procedure, acquisition method, processing method                   |
| `EvaluatedEntity`        | sample, material, catalyst, specimen, or evaluated target                    |
| `AgenticEntity`          | person, organization, instrument, or software system                         |
| `Resource`               | file, dataset, spectrum, report, peak table, data artifact                   |
| `QuantitativeAttribute`  | measured/calculated/selected quantity with value, unit, quantity kind        |
| `QualitativeAttribute`   | label, mode, setting, or observed characteristic                             |

The model definitions for quantitative and qualitative attributes are defined in the extraction context domain.  The main extraction object classes carry quantitative and qualitative attributes. 

Each extracted object is wrapped in a `TracedExtractionObject`, which stores object type, extracted object, and the exact source-text substring from the chunk. 

---

## Stage 10: Structured-output repair

If a chunk extraction fails because the structured model output cannot be parsed or validated, SIMONE can queue the failed response for repair. In the first extraction pass, a `MaxRetriesExceeded` exception marks the chunk as failed but repairable if a failed response exists. 

After the first pass, queued chunk repairs are processed with `repair_structured_output()`, again targeting `ExtractionContext`. 

This gives the workflow a two-stage extraction strategy:

```text
first-pass structured extraction
        │
        ├── valid → completed chunk
        └── invalid but repairable → repair queue
                                  │
                                  ▼
                         structured-output repair
```

---

## Stage 11: Incremental context accumulation

After each successful chunk extraction, the workflow:

1. records token usage,
2. marks the chunk as completed,
3. stores the `ExtractionContext`,
4. schedules vocabulary candidate discovery,
5. merges completed chunk contexts,
6. saves interim extraction context,
7. updates progress.

This is visible in the extraction loop. 

Merged context is produced by concatenating and deduplicating extraction contexts. 

The workflow also uses previous completed chunk context as optional prompt context for later chunks of the same file. If the accumulated context would become too large, it is capped according to token/context thresholds.  

---

## Stage 12: Vocabulary candidate discovery

As chunks complete, SIMONE schedules vocabulary candidate discovery tasks for extracted attributes.

For every extracted quantitative attribute, the workflow creates two vocabulary query records:

1. one for the **quantity kind**,
2. one for the **unit**.

The quantity-kind query targets the configured QUDT quantity-kind vocabulary, and the unit query targets the configured QUDT unit vocabulary. 

For every qualitative attribute, the workflow queries configured qualitative vocabularies. For each selected vocabulary, it retrieves available term schemes and builds vocabulary queries per RDF type. 

The query records store:

* query ID,
* kind,
* source value,
* source context,
* vocabulary identifier,
* RDF type,
* query configuration,
* status,
* result,
* error,
* duration.

The frontend type definitions expose this record structure as `ExtractionVocabQueryRecord`. 

---

## Stage 13: Vocabulary query execution

Each vocabulary query is sent through the semantic service.

The semantic service:

1. loads the vocabulary,
2. verifies that the requested RDF type exists,
3. embeds the vector query if present,
4. retrieves vector candidates,
5. retrieves full-text candidates,
6. fuses candidates with reciprocal rank fusion,
7. expands graph context around seed nodes,
8. loads compact resource data,
9. returns a compact `VocabQueryResult`.

The implementation is in `SemanticService.query_vocabulary()`. 

The Neo4j implementation uses:

* `db.index.vector.queryNodes()` for vector candidate retrieval,
* `db.index.fulltext.queryNodes()` for full-text candidate retrieval,
* Cypher path expansion for graph context.   

The vocabulary query result contains seeds and graph statements suitable for LLM-based candidate selection.

---

## Stage 14: Vocabulary normalization

After all chunk extraction and candidate discovery tasks finish, the workflow enters the `vocabulary_normalization` stage. 

### 14.1 Quantitative normalization

Each quantitative attribute is normalized by selecting:

* a QUDT quantity-kind term,
* a QUDT unit term.

The selection process calls `_select_term_with_fallback_candidates()`, which first tries to select from the initial candidates. If no candidate is selected, it asks for a fallback query and then retries candidate retrieval and selection. 

If no semantic term can be selected, SIMONE preserves the raw quantity kind or raw unit and records a warning. 

### 14.2 Qualitative normalization

Qualitative attributes are normalized against candidate terms gathered from selected qualitative vocabularies. The workflow deduplicates candidates and invokes LLM-based candidate selection. If no semantic term can be selected, the raw value is retained with a warning. 

The normalization result is an `ExtractionNormalization` object containing:

* quantity normalizations,
* qualitative attribute normalizations.

---

## Stage 15: Profile projection

After vocabulary normalization, SIMONE enters the `profile_projection` stage. 

The profile projection call sends the following to the LLM:

* data package ID,
* profile identifier,
* profile target class,
* merged extraction context,
* vocabulary normalization result,
* warnings,
* profile validation schema.

The code calls `generate_structured()` with `PROFILE_PROJECTION_SYSTEM_PROMPT`, the profile projection prompt, and the validation schema as output type. 

This step converts the generic intermediate `ExtractionContext` into the final profile-specific metadata document.

---

## Stage 16: Schema validation and result persistence

After projection, SIMONE:

1. removes null values,
2. validates the document against the selected profile,
3. raises an extraction validation error if validation fails,
4. collects token usage,
5. creates an `ExtractionRunResult`,
6. saves warnings,
7. saves the final extraction result.

This is implemented in `_project_and_save_result()`. 

The final result contains:

```text
ExtractionRunResult
    document: final profile-conformant metadata document
    extraction_context: merged generic extraction context
    warnings: workflow warnings
    token_usage: aggregated usage information
```

The frontend type definition mirrors this result shape. 

---

## Stage 17: Result retrieval and downstream use

After completion, the frontend or another agent can retrieve:

```text
GET /api/v1/extraction/result/{data_package_id}
GET /api/v1/extraction/{data_package_id}/token-usage
```

The extraction API exposes both endpoints. 

The result can be used in several ways:

* displayed in the frontend as a metadata draft/document,
* inspected through the extraction context trace view,
* exported as JSON-LD through the profile API,
* used as a schema-valid metadata representation for FAIR data workflows,
* compared against manual reference annotations in thesis evaluation.

The profile API supports JSON-LD export through `/profiles/{identifier}/jsonld`. 

---

# 9. End-to-end dataset-flow diagram

```text
User uploads ZIP archive
        │
        ▼
POST /datasources
        │
        ▼
DataPackage.from_bytes()
        │
        ├── validate ZIP
        ├── recursively unpack nested ZIPs
        ├── create FileEntry objects
        └── compute package ID
        │
        ▼
File-system datasource storage
        │
        ▼
POST /datasources/chunk
        │
        ▼
For each FileEntry:
        │
        ├── extract text
        │     ├── PDF → PyMuPDF text
        │     ├── spreadsheet/CSV → pandas → CSV-like text
        │     ├── image → placeholder, no OCR
        │     └── text/default → charset decoding
        │
        ├── split into lines
        ├── apply text-quality filter
        ├── reinsert protected lines
        ├── create neighbor line windows
        ├── embed line windows with Ollama
        ├── calculate adjacent cosine distances
        ├── choose percentile breakpoint threshold
        └── persist ContentChunk objects
        │
        ▼
POST /extraction/run
        │
        ├── verify data package
        ├── verify profile
        ├── load profile JSON Schema
        ├── require completed chunks
        └── start async extraction task
        │
        ▼
File ranking
        │
        ├── LLM ranks files by metadata relevance
        └── fallback heuristic ranking if needed
        │
        ▼
Chunk ordering
        │
        └── order by file rank, file path, start line
        │
        ▼
Chunk extraction loop
        │
        ├── send chunk + metadata + prior file context to LLM
        ├── receive structured ExtractionContext
        ├── repair invalid structured output if possible
        ├── persist chunk result
        ├── merge interim context
        └── schedule vocabulary candidate discovery
        │
        ▼
Vocabulary candidate discovery
        │
        ├── quantitative attributes
        │     ├── query QUDT quantity kinds
        │     └── query QUDT units
        │
        └── qualitative attributes
              └── query configured vocabularies and RDF term schemes
        │
        ▼
SemanticService.query_vocabulary()
        │
        ├── vector retrieval from Neo4j vector index
        ├── full-text retrieval from Neo4j full-text index
        ├── reciprocal-rank fusion
        ├── graph expansion around seed nodes
        └── compact VocabQueryResult
        │
        ▼
Vocabulary normalization
        │
        ├── LLM selects quantity-kind and unit mappings
        ├── LLM selects qualitative mappings
        ├── fallback queries if needed
        └── raw values retained with warnings if unmapped
        │
        ▼
Profile projection
        │
        ├── merge extraction context
        ├── combine semantic normalizations
        ├── send profile schema + context to LLM
        └── receive profile-shaped document
        │
        ▼
Schema validation
        │
        ├── remove nulls
        ├── validate against registered profile
        ├── fail if invalid
        └── save result if valid
        │
        ▼
Final ExtractionRunResult
        │
        ├── document
        ├── extraction_context
        ├── warnings
        └── token_usage
```

---

# 10. Architectural interpretation for the thesis

The current SIMONE prototype implements a **layered semantic extraction architecture** rather than a monolithic LLM prompt. Its key design decision is to separate the workflow into intermediate representations and validation boundaries:

```text
Raw files
    ↓
Extracted text
    ↓
Semantic chunks
    ↓
Generic ExtractionContext
    ↓
Vocabulary-normalized attributes
    ↓
Profile-specific metadata document
    ↓
Validated final output
```

This has several implications for the thesis.

First, the LLM is not used as an unconstrained generator of final metadata. It is used at controlled points: file ranking, chunk-level context extraction, candidate selection/fallback query generation, and profile projection. The surrounding system constrains and records these calls through structured output models, vocabulary query records, validation schemas, and persistent workflow state.

Second, the workflow distinguishes **information extraction** from **semantic grounding**. The extraction model identifies activities, methods, resources, entities, agents, and attributes from text. A later vocabulary subsystem grounds quantities and qualitative attributes against RDF vocabularies through retrieval, graph expansion, and candidate selection.

Third, the architecture makes traceability a core object-level feature. Each extracted object carries a `source_text` field with a verbatim substring from the original chunk.  The frontend exposes this trace evidence through object modals. 

Fourth, the system is designed for heterogeneous datasets. The input is not a single text file but a ZIP-based data package that can contain PDFs, spreadsheets, CSV files, text files, nested archives, and images. The current image handling is limited because no OCR is implemented. 

Fifth, the final metadata document is profile-driven. The same generic extraction context can, in principle, be projected into different metadata profiles, as long as the profile is registered and its schema can be validated. The profile API supports registration, validation, and JSON-LD export.  

---

# 11. Current implementation boundaries

The source code indicates several boundaries that should be stated clearly in the thesis:

| Boundary                       | Current implementation status                                                                                                           |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| Image understanding            | Images are not OCR-processed; they become a placeholder string.                                                                         |
| Human-in-the-loop patch review | Older frontend patch-review functions are stubs; active backend workflow is direct extraction → normalization → projection.             |
| Evaluation pipeline            | The inspected code shows workflow and test infrastructure, but not a completed thesis-level benchmark/evaluation dataset.               |
| Ontology coverage              | Initial vocabularies are configured, but extraction quality still depends on the imported vocabularies and their term schemes.          |
| LLM dependency                 | Extraction, ranking, candidate selection, fallback query generation, and profile projection depend on the configured Ollama chat model. |
| Chunking dependency            | Extraction requires completed chunking; it does not automatically chunk inside the extraction endpoint.                                 |
| Profile dependency             | Final output requires a registered profile and successful schema validation.                                                            |

---

# 12. Concise thesis-ready summary

The current SIMONE prototype is a full-stack workflow system for semantic metadata extraction from heterogeneous research data packages. A user uploads a ZIP archive, which is recursively unpacked into file entries. File contents are converted into text through file-type-specific extractors for PDFs, spreadsheets, CSV files, and plain text, while images are currently not OCR-processed. The extracted text is filtered and semantically chunked using embedding-based line-window distances. Once chunking is complete, an asynchronous extraction workflow ranks files, orders chunks, and uses an Ollama-backed LLM to extract a generic `ExtractionContext` consisting of traced resources, methods, activities, entities, agents, quantitative attributes, and qualitative attributes. Each extracted object includes source-text evidence. Quantitative attributes are normalized against QUDT quantity-kind and unit vocabularies, while qualitative attributes are normalized against configurable RDF vocabularies such as Voc4Cat, CHMO, and nmrCV. Vocabulary grounding is implemented through a Neo4j-backed semantic graph service that imports RDF vocabularies, creates vector and full-text indexes, retrieves candidates through hybrid search, fuses them with reciprocal rank fusion, expands graph context around seed nodes, and returns compact candidate contexts for selection. Finally, the merged extraction context and normalization results are projected into a registered metadata profile, validated against the profile schema, and stored as a final extraction result containing the profile document, extraction context, warnings, and token usage.
