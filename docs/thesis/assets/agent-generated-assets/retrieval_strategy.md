Below is a thesis-ready summary of the retrieval strategy you developed.

## Retrieval strategy summary

The implemented retrieval strategy combines **semantic retrieval**, **lexical retrieval**, **rank fusion**, and **graph-based contextualization** to identify candidate ontology terms for mapping non-annotated data to RDF vocabulary terms.

The overall pipeline consists of four main steps:

```text
1. Retrieve semantically similar candidate terms using a vector index.
2. Retrieve lexically matching candidate terms using a full-text index.
3. Combine both ranked result lists using Reciprocal Rank Fusion.
4. Expand the highest-ranked seed terms in the RDF graph to construct LLM context.
```

The goal is not only to find individual matching terms, but to provide an LLM with a compact semantic context that includes the candidate terms and their ontology neighborhood.

---

## Hybrid retrieval: vector search and full-text search

The first stage uses a **vector index** over embedded RDF term descriptions. The embedding text is generated from labels, definitions, synonyms, examples, and other descriptive RDF properties. This allows natural-language queries such as:

```text
acidity metric
```

to retrieve semantically related terms, even when the exact lexical form does not appear in the vocabulary.

The second stage uses a **full-text index** over RDF label-like properties such as:

```text
rdfs:label
skos:prefLabel
skos:altLabel
skos:example
dcterms:title
```

This captures exact or near-exact lexical evidence, for example matching the query:

```text
pH
```

against terms such as `pH`, `sample pH`, `buffer pH`, or `post buffer pH`.

The two retrieval methods serve complementary purposes:

```text
Vector retrieval:
  high semantic recall
  useful for paraphrases and conceptual similarity

Full-text retrieval:
  high lexical precision
  useful for exact labels, abbreviations, symbols, and known terms
```

---

## Reciprocal Rank Fusion

The results from vector search and full-text search are combined using **Reciprocal Rank Fusion** rather than by directly adding raw scores.

This is important because vector similarity scores and full-text scores are not directly comparable. Vector scores represent embedding-space similarity, while full-text scores represent lexical relevance. Therefore, the fusion is based on ranks:

```text
RRF(candidate) =
  vectorWeight   * 1 / (rrfK + vectorRank)
+ fulltextWeight * 1 / (rrfK + fulltextRank)
```

This makes the fusion robust across different scoring scales.

In the implemented setup, candidates may come from:

```text
vector search only
full-text search only
both vector and full-text search
```

A candidate retrieved by both methods receives evidence from both channels and is therefore ranked higher. This is useful because a good ontology mapping candidate should ideally be both semantically related and lexically plausible.

Example:

```text
Query:
  vectorQuery = "acidity metric"
  fulltextQuery = "pH"

High-ranking candidates:
  buffer pH
  post buffer pH
  sample pH
  pH
```

---

## Seed terms

After RRF ranking, the top candidates are selected as **seed terms**. These are the terms around which graph context is constructed.

The seed terms retain retrieval metadata:

```json
{
  "label": "buffer pH",
  "uri": "http://nmrML.org/nmrCV#NMR:1000165",
  "retrieval": {
    "vectorScore": 0.743,
    "vectorRank": 3,
    "fulltextScore": 2.684,
    "fulltextRank": 2,
    "rrfScore": 0.016
  }
}
```

This makes the retrieval process transparent and allows later analysis of whether a term was selected mainly because of semantic similarity, lexical match, or both.

---

## Graph-based context construction

The seed terms are then expanded in the RDF graph using selected relationship types, for example:

```text
rdfs__subClassOf
```

The traversal produces a graph context in the form of RDF-like triples:

```json
{
  "subject": {
    "label": "buffer pH",
    "uri": "http://nmrML.org/nmrCV#NMR:1000165"
  },
  "predicate": {
    "type": "rdfs__subClassOf"
  },
  "object": {
    "label": "pH",
    "uri": "http://nmrML.org/nmrCV#NMR:1002011"
  }
}
```

All triples from all seed terms are merged into one deduplicated graph object. This avoids repeating the same ontology statements for multiple seeds.

The final context object therefore contains:

```text
1. A ranked list of retrieved seed terms.
2. A deduplicated list of RDF-like triples connecting the seeds to nearby ontology concepts.
```

This structure is more useful for an LLM than a flat list of nodes because it preserves semantic relationships.

---

## Traversal directions

Three traversal modes were considered:

```text
outgoing
incoming
undirected
```

### Outgoing traversal

Outgoing traversal follows RDF hierarchy edges in their stored direction:

```text
specific term -> broader term
```

For `rdfs__subClassOf`, this produces superclass context.

Example:

```text
buffer pH -> pH -> sample attribute -> object attribute
```

This gives concise semantic grounding and is useful for mapping because it explains what kind of concept the candidate is.

Outgoing traversal is the best default for controlled LLM context.

### Incoming traversal

Incoming traversal follows edges from broader concepts to more specific subclasses:

```text
broader term <- specific term
```

This is useful when the seed term is very general and examples of its specializations are needed.

Example:

```text
pH <- sample pH
pH <- buffer pH
pH <- post buffer pH
```

Incoming traversal can help identify how a broad concept is specialized in the vocabulary.

### Undirected traversal

Undirected traversal ignores edge direction and retrieves a local neighborhood.

For subclass hierarchies, this can introduce sibling terms through shared parents:

```text
buffer pH -> pH -> sample attribute <- sample mass information
                                     <- sample volume
                                     <- sample concentration
```

This increases the amount of context and can introduce noise. However, it can also provide useful **contrastive context**.

---

## Contrastive context

A key observation is that undirected traversal can expose sibling terms under the same semantic parent. Although this may look noisy, it can help the LLM distinguish between nearby alternatives.

For example, when mapping a source datapoint related to acidity or pH, the LLM may benefit from seeing that the ontology also contains other sample attributes such as:

```text
sample mass information
sample volume
sample concentration
sample temperature information
NMR solvent information
```

This helps the model understand that `sample pH`, `buffer pH`, or `pH` are more appropriate mappings than other sibling terms.

This can be described as **contrastive graph context**:

```text
Directed superclass traversal provides positive semantic grounding.
Undirected traversal additionally provides nearby alternatives,
which may help the LLM discriminate between related ontology terms.
```

However, this must be controlled because undirected traversal can quickly expand into unrelated branches.

A good thesis formulation would be:

> Undirected traversal introduces additional sibling concepts through shared ancestors. While this increases the amount of potentially noisy context, it can also provide contrastive evidence by showing nearby alternatives in the same ontology branch. This may support LLM-based term mapping by helping the model distinguish the best candidate from semantically related but incorrect alternatives.

---

## Recommended retrieval modes for evaluation

A useful evaluation setup would compare several retrieval/context variants:

```text
A. Retrieval only
   Vector + full-text + RRF, without graph context.

B. Retrieval + outgoing traversal
   Adds superclass paths for semantic grounding.

C. Retrieval + undirected traversal
   Adds local graph neighborhood and sibling concepts.

D. Retrieval + outgoing traversal + limited contrastive siblings
   Combines clean hierarchy context with controlled alternatives.
```

The expected trade-off is:

```text
Retrieval only:
  compact but lacks ontology context

Outgoing traversal:
  precise and semantically grounded

Undirected traversal:
  richer but noisier

Outgoing + limited contrastive siblings:
  likely best balance between precision and discriminative context
```

---

## Final thesis-ready description

You could summarize the method like this:

> The retrieval strategy combines semantic and lexical evidence to identify candidate RDF vocabulary terms. First, a vector index is queried with a natural-language description to retrieve semantically similar terms. Second, a full-text index is queried with lexical terms to retrieve exact or near-exact label matches. The two ranked lists are combined using Reciprocal Rank Fusion, avoiding direct comparison of incompatible score scales. The highest-ranked candidates are then used as seed nodes for graph expansion over selected RDF relationships such as `rdfs:subClassOf`. The resulting statements are merged into a deduplicated RDF-like triple set that serves as structured context for an LLM. Different traversal directions provide different kinds of context: outgoing traversal yields concise superclass grounding, incoming traversal yields specialization examples, and undirected traversal yields contrastive neighborhood context. This allows the system to balance retrieval precision, semantic grounding, and discriminative evidence for term mapping.
