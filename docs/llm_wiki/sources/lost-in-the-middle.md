---
type: source
title: "Lost in the Middle: How Language Models Use Long Contexts"
created: 2026-06-26
updated: 2026-06-26
sources: [lost-in-the-middle]
tags: [llm, long-context, rag]
---

# Lost in the Middle

**Authors:** Nelson F. Liu, Kevin Lin, John Hewitt, Michele Bevilacqua, Fabio Petroni
**Year:** 2024
**Venue:** Transactions of the Association for Computational Linguistics
**File:** `docs/literature/2024.tacl-1.9.pdf`

## Abstract
Studies how LLMs use long input contexts in multi-document question answering and key-value retrieval. Performance often depends strongly on the position of relevant information, with models doing better when evidence appears near the beginning or end of the context and worse when it appears in the middle.

## Key Findings
- Long context size does not guarantee robust use of all included information.
- Relevant evidence position can significantly affect answer quality.
- Multi-document QA and key-value retrieval expose context-use failures.
- Context ordering and chunk selection matter in RAG pipelines.

## Relevance
Crucial caution for long-document scientific extraction: adding more PDF text to a prompt may reduce reliability if relevant evidence is buried. Relevant to [[retrieval-augmented-generation]], [[structured-output-reliability]], and [[text-mining-for-catalysis]].
