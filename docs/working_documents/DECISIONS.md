# SIMONE Decisions Ledger

> Snapshot note: This working document records implementation and thesis-facing decisions, with reasons and known tradeoffs. Update it when a decision changes or when a discussion settles a design direction.

## How To Use This File

Record decisions that affect workflow meaning, prototype behavior, evaluation interpretation, or thesis wording. Keep entries short enough to review before writing or changing code.

Each decision should include:

- Date
- Decision
- Reason
- Tradeoff
- Revisit trigger

## Decisions

### 2026-07-02: Use Natural-Language Initial File Summaries

Decision: Initial file summaries now use a lean schema with file path, data format, explicit purpose, and a short natural-language information summary. The previous status field and structured purpose evidence, metadata-signal, instrument/software/setting, and quantitative-signal buckets were removed.

Reason: The bucketed summary shape introduced provenance and settings labels too early in the workflow. Later prompts could treat origin, owner, vendor, host, tool, or numeric orientation fragments as stronger semantic hints than they were. Natural-language summaries keep the overview stage useful for scope and file-role orientation without pre-classifying claims before chunk evidence extraction.

Tradeoff: Deterministic ranking now relies on broad terms in purpose and summary text plus file-path, extension, and size heuristics. It has less pre-counted signal detail, but it also leaks fewer unguided role hints into profile construction.

Revisit trigger: Revisit if multi-dataset evaluation shows that natural-language file summaries are too vague for overview graph construction or file prioritization.

### 2026-07-02: Give Parent Attribute Prompts Core-Draft Context

Decision: Parent-scoped attribute construction now sends the whole core draft, the focus target path and class, selected parent-local evidence, task instructions, and the small attribute output schema to the LLM. The prompt no longer includes the backend's inferred parent-role label.

Reason: The LLM should judge the focused object in its draft context rather than inherit a deterministic role interpretation from the backend. This reduces over-attachment of attributes to unrelated parents while preserving backend ownership of target paths, evidence selection, validation, and duplicate checks.

Tradeoff: The prompt is slightly larger because it includes the core draft. This is acceptable because the core draft is compact and the selected evidence packet remains small.

Revisit trigger: Revisit if larger core drafts cause context pressure or if evaluation shows that removing the role label reduces attribute placement accuracy.

### 2026-07-02: Keep Initial Dataset Description Backend-Owned

Decision: The initial dataset-level projection prompt no longer asks the LLM to generate a Dataset `description`. The backend inserts the dataset-level summary as the default Dataset description. Later provenance-core prompts reuse that description through the current draft and do not duplicate it in the separate orientation context.

Reason: The dataset-level summary is already generated in the initial overview stage. Reusing it gives the draft one description source of truth and avoids asking a later LLM step to restate or drift from it.

Tradeoff: The initial draft cannot refine the description during shallow projection. Later requirement checks and manual curation remain the place to correct an insufficient summary.

Revisit trigger: Revisit if evaluation shows that the initial dataset summary is consistently too broad or too sparse for the Dataset description field.

### 2026-07-01: Flip Profile Draft Construction To Requirement-Scoped Parents

Decision: Replace note-driven measurement projection with a staged profile-draft flow: `generated_initial_draft.json`, `generated_core_draft.json`, `generated_attribute_draft.json`, and `generated_reconstructed_draft.json`. The initial draft is a dataset shell. The core draft creates one generic DataGeneratingActivity with evaluated targets, agents, plan/context, inputs, and outputs. Attribute construction iterates existing parents and asks narrow structured LLM questions; the backend owns the schema path. Dataset `is_about_entity` and `is_about_activity` are not actively constructed.

Reason: Evidence extraction is intentionally broad and noisy. Letting evidence notes materialize directly into the draft made low-level parameters dominate and confused Dataset subject matter with DataGeneratingActivity evaluation targets. Requirement-scoped questions reduce organizing overhead for the model and make provenance easy to record as "created by requirement X from packet Y" without forcing the LLM to reason about patch paths.

Tradeoff: Some valid facts may remain in evidence/context until a suitable parent exists and a scoped requirement asks for them. Final semantic reconstruction is still needed to triage duplicates, misplaced attributes, invalid ranges, and relation defects.

Revisit trigger: Revisit if parent-scoped questions consistently miss important attributes, or if evaluation shows Dataset aboutness is needed as a separate active construction target.

### 2026-06-21: Run Vocabulary Grounding As A Separate Stage

Decision: Vocabulary grounding is invoked through a dedicated POST /extraction/stages/grounding/{id}/run endpoint that grounds the persisted reconstructed profile draft in place, instead of driving the full evidence-resume pipeline up to the grounding stage.

Reason: Resuming the whole workflow to reach grounding re-ran profile projection and requirement enrichment every time, rebuilding the draft the user only wanted to ground. Grounding is the final enrichment step and should be runnable independently once a profile draft exists.

Tradeoff: The dedicated run cannot build a profile draft from scratch; it requires a prior reconstructed profile draft and completed evidence context. The general resume path remains available for end-to-end runs.

Revisit trigger: Revisit if grounding needs inputs only produced by re-running an earlier stage, or if first-time grounding should also discover fields incrementally.

### 2026-06-19: Default Prototype LLM Calls To Deterministic Generation

Decision: Prototype LLM calls use `OLLAMA_GENERATION_TEMPERATURE=0.0` by default and can enforce a context-derived `num_predict` output cap.

Reason: Evaluation runs should not drift because sampling changed the initial draft. Runaway/repetitive output is bounded by output-token limits instead of nonzero temperature.

Tradeoff: Deterministic decoding can reduce variety in borderline extraction cases.

Revisit trigger: Revisit if a model repeatedly fails structured JSON at temperature 0 despite output caps and repair retries.

### 2026-06-19: Prefer Denoising Over Over-Retention For Numeric Evidence

Decision: The quantitative evidence projection should prefer denoising when uncertain, even if this means some valid numeric facts remain only in evidence/context artifacts instead of becoming final profile attributes.

Reason: Noisy quantitative metadata is harder to remove after projection than missing-but-grounded evidence is to curate or add later. This also keeps the final generated profile more reviewable and reduces low-level parameter clutter. Qualitative labels, placeholder/default values, and encoded enum-like values should not become quantitative attributes merely because they contain numbers.

Tradeoff: Some real extraction targets, such as point counts, resolution values, acquisition dates, or scale factors, may be filtered out of the final draft until the selection rules improve.

Revisit trigger: Revisit when evaluation shows important numeric targets are consistently absent from final drafts, or when curation workload shifts from noise removal to missing-value recovery.

### 2026-06-19: Keep Projection Filters Domain-Agnostic

Decision: Deterministic evidence selection and patching must not use dataset-specific field names, vendor names, instrument names, file names, profile examples, or benchmark sample identifiers as filters.

Reason: Sample-specific filters make evaluation circular and thesis claims indefensible. Projection may use domain-agnostic evidence categories, schema shape, source structure, and broad semantic distinctions such as setting/configuration versus primary data summary.

Tradeoff: Some misclassified evidence may pass through until the category prompt, semantic evaluator, or later reconstruction flow improves.

Revisit trigger: Revisit only to add profile-declared rules or benchmark-independent structural rules, not ad hoc keys from failed examples.

### 2026-06-19: Keep Raw Units Before Vocabulary Normalization

Decision: Profile projection may keep a raw unit string when the source explicitly provides one, but should not pretend local regex cleanup is semantic unit normalization.

Reason: Raw units keep the draft schema-conformant and provide query text for later QUDT/vocabulary normalization. True unit normalization belongs in the grounding stage.

Tradeoff: Raw unit strings may be inconsistent until normalization runs.

Revisit trigger: Revisit when profile schemas support unresolved unit objects or when grounding can reliably update projected unit fields before persistence.

### 2026-06-19: Replace Data Quality Evidence Category With Typed Signals

Decision: Evidence categories are limited to `resource_signal`, `method_signal`, `measurement_signal`, `measurement_condition`, `software_signal`, `activity_signal`, `instrument_signal`, `surrounding_signal`, and `other`; `data_quality_signal`, `entity_signal`, `agent_signal`, and category-level `uncertainty` are removed. Evidence candidates also carry a required routing `role`: `qualitative_attribute`, `identity`, `descriptor`, `context`, `parameter`, or `other_metadata`.

Reason: `data_quality_signal` mixed raw data values, metadata, and quality-like notes. The split keeps `measurement_signal` available for later semantic routing into activity/entity attributes, routes measurement descriptors such as axis bounds, units, ranges, and point counts through `measurement_condition`, routes activities and methods separately, separates software from instrument/device signals, and uses role to make evidence easier to place in DCAT-AP+ object/attribute patterns without adding redundant parent hints.

Tradeoff: The evidence extraction schema is not backward-compatible with older artifacts or tests: candidates must provide both `category` and `role`.

Revisit trigger: Revisit when evaluation shows semantically routed measurement evidence is too noisy, when `measurement_condition` admits too many row-like observations, or when category + role is still insufficient for generic DCAT-AP+ parent routing.

### 2026-06-21: Route Measurement Evidence Once, Without Rescue

Decision: Each routable measurement note receives one structured LLM call that returns an activity/entity attribute path, stable semantic merge key, confidence, and reason. Confidence below `0.7`, unavailable routing, null decisions, and unsupported paths are recorded and skipped. Accepted notes merge by target path plus merge key; deterministic code only constructs, applies, deduplicates, and validates the selected attribute write.

Reason: Parent ownership and quantitative-versus-qualitative placement are semantic decisions. Re-running routing or rescuing uncertain decisions with path heuristics makes behavior inconsistent and weakens the audit trail.

Tradeoff: Useful attributes can be omitted when the router is unavailable or uncertain. This is preferred over silently attaching evidence to the wrong scientific parent.

Revisit trigger: Revisit the confidence threshold only with evaluation evidence showing a better precision/coverage tradeoff.

### 2026-06-19: Split Requirement Reporting Into Coverage, Semantics, And Trace

Decision: `requirement_report.json` no longer exposes one `metadata_completeness_score`. It reports deterministic filled-field coverage counts, LLM-assessed `semantic_requirements_score`, and deterministic `source_trace_score`.

Reason: One scalar mixed field presence, semantic adequacy, and source traceability. Filled-field counts are informative without pretending coverage is a quality percentage; semantic quality and trace quality remain separate audit concerns.

Tradeoff: Existing runtime requirement reports are not backward-compatible with the new shape.

Revisit trigger: Revisit when the UI or thesis evaluation needs a deliberately named composite score instead of the three separate metrics.

### 2026-06-19: Separate Coverage Patching From LLM Semantic Reconstruction

Decision: Coverage patching remains evidence-selected and slot-focused; semantic defects are handled afterward by an LLM reconstruction pass that returns constrained JSON Patch operations over the current draft and existing semantic requirement artifacts.

Reason: Semantic issues such as misplaced agents, bloated descriptions, or missing provenance should reorganize the draft rather than trigger another evidence-search patch loop.

Tradeoff: Reconstruction quality depends on the chat model, but deterministic code still enforces allowed paths, schema validation, rollback, and trace records.

Revisit trigger: Revisit when profile-declared reconstruction rules can replace part of the LLM edit workload. Final vocabulary grounding normalizes placed values and does not replace semantic reconstruction.

### 2026-06-20: Observe Description Mining Before Pipeline Integration

Decision: Keep description mining outside the extraction pipeline as a manual, observation-only two-pass probe. Pass one extracts atomic facts from non-distribution description texts only. Deterministic schema search retrieves candidate branches for each fact, and pass two returns concrete JSON Pointer target/value proposals from those candidates without applying them.

Reason: Description fields may contain structured facts that belong in dedicated schema fields, but real model proposals must be inspected before defining deduplication, acceptance, validation, or mutation rules. Description values remain unchanged.

Tradeoff: The probe produces inspectable artifacts but does not improve generated drafts. Routing quality now depends strongly on schema-branch retrieval, and proposed values may still violate candidate object shapes.

Revisit trigger: Integrate it after evidence patching and before semantic evaluation only after observed proposals support clear per-proposal acceptance and validation rules.

### 2026-06-20: Integrate Dataset-Description Facts Through Evidence Patching

Decision: Supersede the observation-only probe. Mine atomic facts only from top-level dataset descriptions after initial draft creation, validate each fact against its exact source text, and add valid facts to a local portable evidence context used by later requirement-scoped construction. Remove the independent schema-routing pass. Keep semantic evaluation and deterministic source-trace scoring on original source evidence.

Reason: The first mining pass produced compact useful facts, while independent schema routing produced weak targets and invalid value shapes. Later construction stages provide the controlled write path through scoped evidence selection, schema validation, collision handling, and rollback.

Tradeoff: Description-derived facts can fill profile fields but are generated secondary evidence, not direct source-file evidence. They are marked with `draft-description:` provenance and retained in `description_facts.json`, while source-trace scoring excludes them.

Revisit trigger: Revisit provenance linking if generated descriptions gain reliable links to the original evidence IDs that supported them, or when semantic reconstruction is redesigned.

### 2026-06-20: Constrain Patch LLMs With Sliced Target Schemas

Decision: Evidence enrichment, requirement patching, and semantic reconstruction now ask patch LLMs for schema-constrained write envelopes rather than free-form instances or JSON Patch operations. The backend constructs a small output JSON Schema from the allowed target paths and only the reachable `$defs`.

Reason: Profile validation was catching useful but structurally invalid patches too late, causing one invalid operation to consume or roll back useful writes. Target-specific schema slices make invalid value shapes fail at structured-output generation time without sending the full DCAT-AP+ schema.

Tradeoff: The first route supports upsert writes only. Removals, moves, and arbitrary nested JSON Patch edits are intentionally deferred until the schema-constrained route is reliable.

Revisit trigger: Revisit if semantic reconstruction needs validated remove/move operations or if schema slices become too large for specific profile targets.

### 2026-06-20: Simplify Write Envelopes And Separate Attribute Parent Semantics

Decision: Schema-constrained write envelopes use `{writes, reason}`; empty `writes` means no-op. Append writes use canonical array paths instead of `/-`. Semantic requirements include separate attribute parent semantics. The earlier active aboutness reconstruction behavior in this decision was superseded on 2026-07-01.

Reason: `should_apply` duplicated the meaning of empty writes, recursive relation schemas caused structured-output failures, and attribute presence needed to be separated from correct parent placement.

Tradeoff: Range splitting remains prompt-led, so a schema-valid collapsed range can still pass if the model emits one.

Revisit trigger: Revisit if evaluation shows prompt-led range handling remains unreliable or if richer profile-declared parent rules replace the current generic routing cues.

### 2026-06-22: Separate Dataset Subject From Activity Evaluation Target

Decision: Dataset `is_about_entity`/`is_about_activity` and DataGeneratingActivity `evaluated_entity`/`evaluated_activity` were treated as distinct claims. This active Dataset-aboutness construction was superseded on 2026-07-01; the current flow focuses on DataGeneratingActivity evaluated/input/output relations and does not create Dataset aboutness by default.

Implementation guard: Either evaluation-target family independently fulfills the activity requirement. Reconstruction may remove invalid or self-referential edges but must not synthesize a missing evaluation-target relation. Forced profile rebuilds clear stale grounded/report artifacts, and the requirement report is deterministically revalidated against the delivered grounded document.

Reason: Dataset aboutness answers what the Dataset is about; activity evaluation targets answer what one data-generating activity measured, observed, analysed, or studied. Conflating them loses the claim expressed by the owning relation and can turn generated outputs or central Dataset subjects into unsupported experiment targets.

Tradeoff: Relation support remains prompt-interpreted rather than encoded as a new evidence field. Missing target evidence therefore remains a visible semantic defect, and unsupported mirrored edges may be removed even when another requirement becomes missing.

Revisit trigger: Revisit if prompt-only relation interpretation is unstable enough to require explicit evidence relation claims or profile-declared target rules.

### 2026-06-21: Preserve Evidence Source Context For Parent Routing

Decision: Evidence candidates keep `evidence_text` as the exact atomic support span and add backend-derived `source_context` as a compact contiguous copied source window needed to preserve local scope for later routing. Prompts explicitly forbid paraphrasing, normalization, reordering, and stitched non-contiguous range evidence, but they tell the model not to populate `source_context`.

Reason: Attribute parent placement depends on local source context such as file section, block, resource, method, instrument/software, or activity scope. Claims alone are too paraphrased to distinguish facts such as raw-spectrum point counts from peak-table point counts.

Tradeoff: Artifacts get slightly larger, and source context remains provenance rather than a deterministic semantic-parent hint. The fixed line window may be broader or narrower than an ideal human-selected source block.

Revisit trigger: Revisit if source_context is still too broad/noisy for generic DCAT-AP+ parent routing or if profile-declared source-scope rules can replace part of this derivation rule.

### 2026-06-21: Run Vocabulary Grounding After Profile Finalization

Decision: Vocabulary grounding is the final workflow enrichment step. It runs after profile construction, semantic reconstruction, and validation, using fields already placed in the profile draft.

Reason: Vocabulary grounding should normalize the meaning of selected profile values after profile construction has established their semantic owners and schema paths. It must not influence or precede parent placement.

Tradeoff: Ungrounded raw labels and units remain in intermediate profile-draft artifacts until the final grounding stage.

Revisit trigger: Revisit only if a future profile explicitly requires grounded identifiers to make an earlier structural placement decision.

### 2026-06-21: Treat Reconstructed Draft As Completed Profile Stage

Decision: The profile-draft stage is completed at `generated_reconstructed_draft.json`. Vocabulary grounding reads that reconstructed draft and writes a separate grounded `generated_final_draft`.

Reason: Semantic reconstruction and grounding answer different questions. Reconstruction decides whether values are in the right profile objects and fields; grounding assigns controlled identifiers to already placed values. Keeping the documents separate makes stage review possible and avoids hiding raw reconstruction behavior behind vocabulary normalization.

Tradeoff: The runtime now carries two closely related profile documents, so artifact names and UI labels must stay explicit.

Revisit trigger: Revisit only if future profiles require grounded identifiers during reconstruction, or if artifact naming is migrated to make raw and grounded documents impossible to confuse.

### 2026-06-21: Separate Concept Type From Ontology RDF Type

Decision: Grounding policy treats `type` fields as vocabulary concept roles, normally SKOS concepts, and `rdf_type` fields as ontology-class roles. General `rdf_type` grounding requires an explicit ontology vocabulary and class policy; it does not fall back to fake SKOS semantics.

Reason: `type` and `rdf_type` carry different semantics. Concept vocabularies classify domain terms, while RDF type links instances or terms to ontology classes. Mixing them would create schema-valid but semantically weak metadata.

Tradeoff: Non-quantitative `rdf_type` fields may remain ungrounded until the user configures an ontology vocabulary and class.

Revisit trigger: Revisit when profiles declare their own ontology-class policies or when imported vocabularies expose reliable class scopes.

### 2026-06-21: Ground Quantitative DefinedTerms With QUDT Classes

Decision: Quantitative attributes receive deterministic QUDT class terms during final grounding: the attribute `rdf_type` is QUDT `Quantity`, grounded `has_quantity_type` terms receive QUDT `QuantityKind`, and grounded `unit` terms receive QUDT `Unit`.

Reason: These class assignments are known from the schema role before querying. Querying should normalize the specific quantity kind or unit identifier, not rediscover the ontology class of the field.

Tradeoff: Unit synonym normalization is still vocabulary-query dependent; this decision only fixes class typing and controlled-vocabulary source policy.

Revisit trigger: Revisit if profile schemas rename the quantity-kind role or if deterministic unit synonym mapping is introduced.

### 2026-06-21: Separate Evidence Extraction From Profile Draft Construction

Decision: Evidence extraction and generated profile draft construction use separate stage entrypoints and task names. The profile stage consumes persisted evidence context and fails if evidence has not been extracted.

Reason: Building or rebuilding the generated draft should not rerun chunk evidence extraction or merge token usage from a different workflow stage.

Tradeoff: Users must run evidence extraction before profile projection; missing evidence is reported as an explicit stage-order error instead of being repaired implicitly.

Revisit trigger: Revisit only if a future complete-workflow controller needs a combined orchestration endpoint with clearly separate subtask state.

### 2026-06-21: Split Semantic Reconstruction Into Diagnosis, Compilation, And Synthesis

Decision: Semantic reconstruction no longer asks the LLM for profile patch envelopes as its main repair mechanism. The profile stage now evaluates smaller semantic requirements, runs backend duplicate/range cleanup, asks the LLM for compact defect diagnoses, optionally asks for small synthesized target values, and lets backend code compile and validate all mutations.

Reason: The prior broad requirements and patch-envelope prompt produced unchanged reconstructed drafts and fragile structured output. Smaller requirements improve traceability; small diagnosis schemas give the model semantic headroom; backend compilation keeps JSON Pointer mutation, deduplication, validation, salvage, and artifact accounting deterministic.

Tradeoff: Some repairs remain unresolved unless a deterministic compiler or synthesis path exists for that defect type. This is preferred over silently accepting no-op reconstruction as success.

Revisit trigger: Revisit when evaluation shows a recurring unresolved defect type that should gain a deterministic compiler, a small synthesis schema, or a profile-declared rule.

### 2026-06-21: Keep Semantic Model Contracts Target-Specific And Auditable

Decision: Semantic diagnosis returns defects only, synthesis returns one value matching a schema derived from the exact target path, and backend artifacts persist both diagnosed defects and compiled actions. Requirement-specific evidence limits omit evidence from mechanical draft checks and bound semantic placement context. Evidence can establish applicability but cannot satisfy an empty draft target. Same-parent numeric deduplication uses canonical quantity and unit identity plus a small numeric tolerance.

Reason: A fresh profile run exposed conflicting diagnosis instructions, target-shape synthesis failures, oversized attribute prompts, false evidence-based fulfillment, and rounded duplicate values that bypassed exact signatures.

Safety follow-up: Mechanical merge/remove/move actions never synthesize replacement values, indexed attribute diagnoses cannot replace whole collections, range bounds may be recovered deterministically from the dataset description or sibling boundary attributes, and verified cross-parent measurement duplicates keep the semantically preferred owner.

Tradeoff: Tolerant deduplication and compact prompt policies add deterministic judgment before model reasoning. Numeric values near zero require exact nonzero agreement against zero, and unusual target schemas may still remain unresolved rather than receive a generic synthesized object.

Revisit trigger: Revisit the numeric tolerance or requirement packet limits when multi-dataset evaluation shows false merges, missing context, or recurring unresolved target types.
