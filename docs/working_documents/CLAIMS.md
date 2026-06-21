# SIMONE Thesis Claims Ledger

> Snapshot note: This working document tracks thesis claims that must stay aligned with the written thesis, prototype status, and evaluation evidence. Update it whenever thesis wording, implementation state, or evaluation evidence changes.

This is the quick-check ledger for thesis claims. `WORKFLOW.md` explains the conceptual workflow; `PROTOTYPE_STATUS.md` checks current code against that model.

## Claim Status Legend

- `Supported`: current implementation and evidence are enough for the claim as written.
- `Partly supported`: implementation exists, but wording needs constraints or evaluation is incomplete.
- `Not yet supported`: do not make the claim without more evidence.
- `Framing only`: acceptable as motivation or design rationale, not as an empirical result.

## Current Claims

| Claim | Current status | Evidence / constraint | Thesis wording guidance |
| --- | --- | --- | --- |
| SIMONE processes heterogeneous research dataset packages rather than single isolated documents. | Partly supported | ZIP ingestion, recursive archive expansion, text extraction for PDFs/tables/text, image placeholders. | Say "text-accessible heterogeneous packages"; do not imply full binary/image understanding. |
| SIMONE uses a staged workflow rather than a monolithic LLM prompt. | Supported | Initial context, chunking, evidence extraction, profile construction, semantic reconstruction, final vocabulary grounding, and validation. | Strong architecture claim is safe. |
| SIMONE improves traceability by grounding extracted observations in source text. | Partly supported | Evidence candidates carry copied source evidence and are routed/criticized. | Claim traceability of prototype artifacts; do not claim users always review them effectively. |
| SIMONE separates evidence extraction and profile construction from semantic vocabulary grounding. | Supported | Vocabulary candidate retrieval and selection operate on enrichable fields in the finalized profile draft as the last enrichment stage. | Safe as a design claim. |
| SIMONE grounds selected metadata elements against semantic vocabularies. | Partly supported | Neo4j vocabulary import/search, vector/full-text retrieval, graph expansion, candidate selection. | Say "selected fields/attributes"; avoid claiming comprehensive ontology coverage. |
| SIMONE projects a generic extraction context into a registered metadata profile. | Supported | Profile projection and schema validation are implemented. | Safe as an implementation/design claim. |
| SIMONE produces schema-valid metadata documents. | Partly supported | Final result validates against the selected profile when the workflow succeeds. | Say "can produce" or "requires successful validation"; do not imply every run succeeds. |
| SIMONE metadata is scientifically correct. | Not yet supported | Schema validation and source evidence do not prove scientific correctness. | Avoid unless expert evaluation supports it. |
| SIMONE improves extraction quality over baselines. | Not yet supported | Preliminary scoring is incomplete and currently weak for tested examples. | Do not claim improvement until benchmark results support it. |
| SIMONE improves semantic grounding quality over baselines. | Not yet supported | Vocabulary mapping F1 is not yet substantiated. | Keep as a research question or evaluation target. |
| SIMONE supports FAIR-oriented metadata construction. | Partly supported | Profile projection, JSON-LD/profile support, and vocabulary grounding support FAIR orientation. | Say "FAIR-oriented" or "supports"; avoid claiming FAIR compliance without assessment. |
| SIMONE reduces manual metadata construction burden. | Framing only | Workflow automation exists, but there is no user/time study yet. | Present as motivation or potential benefit, not as a measured result. |

## Claims To Recheck Before Thesis Submission

- Any claim using "accurate", "improves", "outperforms", or "reduces effort" needs evaluation evidence.
- Any claim using "automated" should mention dependency on profiles, services, vocabularies, model authorization, and validation success where relevant.
- Any claim using "heterogeneous" should clarify text-accessible sources and current image/binary limitations.
- Any claim using "FAIR-compliant" should be softened unless a FAIR assessment supports it.
- Any claim about semantic grounding should distinguish the implemented grounding workflow from measured grounding quality.
