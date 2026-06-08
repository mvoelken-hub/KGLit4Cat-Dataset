# NotebookLM And Literature Workflow

Use this reference for thesis prose, citation support, NotebookLM work, literature discovery, BibTeX synchronization, and source traceability.

Primary local sources to re-check:

- Fresh `C:\Users\simcl\Downloads\notebooklm-local.zip`
- `docs/thesis/bibliography/references.bib`
- `docs/thesis/sections/*.tex`

## Safety

Never print:

- Cookie values.
- Raw NotebookLM auth files.
- Browser storage state.
- API keys.
- Maton credentials.

It is fine to report high-level auth/search status.

## First Checks

Validate auth with a real network token fetch before relying on NotebookLM:

```powershell
.\scripts\notebooklm-local.ps1 auth check --test --json
```

Require both `"status": "ok"` and `"token_fetch": true`. A parse-only auth check can be a false positive.

If token fetch fails, try login yourself once before asking the user for help:

```powershell
$env:SIMONE_NOTEBOOKLM_ALLOW_MUTATION='1'
.\scripts\notebooklm-local.ps1 login
Remove-Item Env:\SIMONE_NOTEBOOKLM_ALLOW_MUTATION -ErrorAction SilentlyContinue
.\scripts\notebooklm-local.ps1 auth check --test --json
```

Only escalate to the user if login fails, requires interaction you cannot complete, or the follow-up token-fetch check still fails.

Then list notebooks:

```powershell
.\scripts\notebooklm-local.ps1 list --json
```

## Thesis Content Rule

Never bypass NotebookLM when editing thesis content. For any change that adds, rewrites, reorganizes, or substantially interprets thesis prose:

1. Validate NotebookLM auth.
2. Choose the thesis-relevant focused notebook.
3. Ask NotebookLM for source-grounded context.
4. Use the Citation Sync Flow before editing LaTeX.

When NotebookLM is blocked after this self-login recovery attempt, report the blocker and do not make thesis-content edits unless the user explicitly authorizes a no-NotebookLM fallback for that specific turn.

## Focused Thesis Notebooks

Use the broad thesis notebook for cross-chapter orientation, and focused notebooks for topic-specific synthesis.

```text
c1644a03-c921-48a5-96e8-3535ad62b5be
Development of an LLM supported workflow for semantic metadata extraction from catalytic experiments

e1ee729b-32f8-4a23-a17a-fb9ad07eac61
LLMs for Scientific Semantic Data Extraction

ecf48655-d9e6-4654-8dd8-69a8493674c3
Catalysis, Ontologies, and Semantic Metadata

75ad7b6d-1d53-4f43-a7c5-d1bf326411a7
LLM-Based Metadata Extraction from Scientific Resources
```

Notebook selection:

- LLM foundations, RAG, structured output, context behavior, scientific/chemical language models: `e1ee729b-32f8-4a23-a17a-fb9ad07eac61`.
- Catalysis, ontologies, FAIR data, semantic metadata, repositories, metadata profiles, catalysis data infrastructure: `ecf48655-d9e6-4654-8dd8-69a8493674c3`.
- LLM/NLP metadata extraction from scientific resources, entity/relation extraction, tables, knowledge graphs, structured data: `75ad7b6d-1d53-4f43-a7c5-d1bf326411a7`.
- Broad thesis orientation or cross-cutting synthesis: `c1644a03-c921-48a5-96e8-3535ad62b5be`.

## Citation Traceability

Optimize for traceability rather than mechanical citation after every sentence.

- Place citations close to the specific empirical result, definition, method, comparison, or technical claim they support.
- Prefer one strong, directly relevant source when possible.
- Use two citations only when one sentence combines distinct support needs.
- Avoid three or more citations after one sentence; split the sentence or pick the best source set.
- Grouped paragraph-end citations are acceptable only when the whole paragraph uses the same source or same source set.
- Treat NotebookLM citations as candidates to rank, not citation clusters to copy.

## Citation Sync Flow

When NotebookLM is used for thesis-writing questions:

1. Ask NotebookLM with `--json`.
2. List notebook sources with `source list -n <notebook-id> --json`.
3. Map returned `source_id` values to source titles.
4. Match source titles, DOI-like filenames, arXiv IDs, or fulltext metadata against `docs/thesis/bibliography/references.bib`.
5. For unmatched cited sources, inspect source fulltext to recover title, authors, venue, year, pages, DOI, or URL.
6. Add missing BibTeX entries only when the source is selected for thesis use.
7. Report which NotebookLM citation numbers matched which BibTeX keys and name unresolved sources.

Read-only audit:

```powershell
python .\scripts\audit-notebook-bib-sync.py --json
```

## Literature Source-Of-Truth Policy

Maintain these invariants where possible:

1. `docs/thesis/bibliography/references.bib` is the metadata superset for thesis-relevant sources.
2. The canonical Google Drive literature folder is the PDF superset when Drive sync is configured.
3. NotebookLM notebooks are topic-specific working subsets.

Use:

```powershell
python .\scripts\audit-literature-status.py --json
```

If Drive audit is unavailable because Maton/Drive is not configured for this repo, report that explicitly rather than fabricating status.

## Reference Discovery

Use SerpAPI Google Scholar when NotebookLM lacks a suitable source or a thesis claim needs stronger support:

```powershell
python .\scripts\serpapi-scholar.py "catalysis research data management FAIR metadata ontology" --num 5
python .\scripts\serpapi-scholar.py "heterogeneous catalysis knowledge graph ontology" --num 5 --year-from 2020
python .\scripts\serpapi-scholar.py "large language models scientific information extraction structured data" --num 8 --year-from 2022
```

Prefer peer-reviewed reviews, standards, community infrastructure papers, and high-relevance domain papers over generic or weakly related hits.

Do not add BibTeX entries merely because a source was discovered. Add BibTeX only when the source is selected for thesis use.

## PDF Build Check

Use the build helper before moving to a new thesis content block or when asked whether the thesis build is healthy:

```powershell
.\scripts\thesis-build-check.ps1
```

The helper targets `docs/thesis/main.tex`. It can refresh LaTeX build artifacts, so run it only when that is acceptable.
