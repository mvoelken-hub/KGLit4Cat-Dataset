#


## Vocab Query flow
**Vocab Query Flow Handout**

A Vocab Query turns free-text search input into a compact graph-centered result. The process has two main phases: first it finds the best matching vocabulary resources, then it expands those resources into nearby graph context and compacts the result.

**1. Query Parameters Arrive**

The query provides:

- RDF types to search, such as concepts, units, or classes
- relationship types that may be followed during graph expansion
- traversal direction: outgoing, incoming, or undirected
- separate text inputs for vector search and full-text search
- limits and weights for ranking and expansion

These parameters define both what kinds of resources are searchable and how much surrounding graph context should be returned.

**2. Each RDF Type Is Processed Separately**

For every requested RDF type, the system looks up the matching vector index and full-text index.

This means a query can search multiple kinds of vocabulary resources in one request, but each type is retrieved and ranked independently before the results are merged into the final response.

**3. Vector Candidates Are Retrieved**

The vector query text is converted into an embedding.

That embedding is compared against stored embeddings on vocabulary resources of the selected RDF type. The closest resources are returned as vector candidates.

Each vector candidate records:

- resource URI
- vector score
- vector rank
- source: `vector`

**4. Full-Text Candidates Are Retrieved**

The full-text query is run against the text-search index for the same RDF type.

This finds resources whose indexed labels, names, titles, symbols, or similar textual properties match the query text.

Each full-text candidate records:

- resource URI
- full-text score
- full-text rank
- source: `fulltext`

**5. Candidates Are Fused with Reciprocal Rank Fusion**

The vector and full-text candidate lists are merged by URI.

The ranking algorithm does not directly compare raw vector scores with raw full-text scores. Instead, it uses each candidate’s rank position:

```text
weighted score = source weight * (1 / (rrf_k + rank))
```

A resource found by both retrieval methods receives contributions from both. This favors resources that rank well in either search mode, especially resources that appear in both.

The result is a sorted seed list, limited by `seed_top_k`.

**6. Top Seeds Become Graph Starting Points**

The highest-ranked fused resources become seeds.

Each seed keeps its retrieval metadata:

- vector score and rank, if found by vector search
- full-text score and rank, if found by full-text search
- combined RRF score

These seeds are the central resources around which graph context is collected.

**7. Graph Context Is Expanded Around Seeds**

For each seed, the system searches nearby graph relationships.

The expansion is controlled by:

- `max_hops`: how far from the seed to walk
- `traversal_direction`: which relationship direction to follow
- `allowed_rel_types`: which predicates are allowed

The output of this step is a set of graph statements connected to the seed.

Each graph statement has:

```text
subject -- predicate --> object
```

**8. Raw Entities Are Cleaned**

Before building the result, raw graph entities are cleaned.

Large internal fields such as embeddings are removed. Remaining node data is normalized into simple resource-shaped objects.

A resource is represented as:

```text
uri
rdf_types
properties
```

**9. Context Objects Are Built**

Each seed is paired with:

- the cleaned seed resource
- its retrieval scores
- the graph statements found around it

These context objects are sorted by the seed’s combined retrieval score, so stronger matches appear first.

**10. Results Are Compacted**

The final compaction step turns all context objects into one clean `VocabQueryResult`.

It performs three main transformations:

- collects all seeds into a sorted seed list
- deduplicates repeated graph statements
- converts graph entities into compact resource objects

Graph statements are deduplicated by:

```text
subject URI + predicate type + object URI
```

This prevents the same relationship from appearing multiple times when it is reachable from multiple seeds.

**11. Final Compacted Result**

The final result contains:

```text
identifier
seeds
graph_statements
```

`seeds` are the best-matching vocabulary resources, including retrieval scores.

`graph_statements` are the compacted relationships around those seeds, suitable for showing or evaluating vocabulary context without returning the full raw graph.