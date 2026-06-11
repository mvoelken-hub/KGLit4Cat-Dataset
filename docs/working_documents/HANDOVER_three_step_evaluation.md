Here's where things stand right now. I'll be concrete and point at the files.

**1. Is the flow technically working? Mostly yes, with two outstanding defects.**

- Chunking, repair, object-grounding, and quantity-vocab discovery all run. In the most recent run on `30089342` the chunker did 132/132 chunks, all completed (no `failed`, no `pending`). That's the part I had to fix this session: the 11949-line `.dx` spectrum chunk was triggering a 400 from Ollama because the chunk content pushed the body over Ollama's per-request body limit. I added a `max_lines_per_chunk=500` cap to `create_chunks_for_file_entry` in `backend/app/domain/datasources/chunking.py` that post-splits anything bigger, and re-chunking with `replace_existing_chunks=true` confirmed it works (now 132 well-sized chunks).
- Vocab queries ran to completion: 108 `quantity_kind` + 108 `unit` + 60 `skos__Concept` + 60 `skos__Collection` = 336 queries, all `completed`. Vocab results are persisted in the run state.
- Profile projection is still failing with a 400 from Ollama on this real run. I isolated the cause: it isn't a single threshold but a combination of Ollama `format=schema` + a 32K-token `num_ctx` + a 73 KB dcat-ap-plus schema + a 47 KB prompt text (the prompt still embeds the merged context for 60 objects). The total body around 122-176 KB trips the per-request limit. I made two changes that I have not yet confirmed end-to-end:
  1. `build_profile_projection_prompt` no longer embeds the schema dict into the prompt text — it just references the structured-output contract — in `backend/app/domain/extraction/profile_projection.py`.
  2. `_INPUT_CONTEXT_BUDGET_RATIO` dropped from `0.75` to `0.5` in `backend/app/ollama/completion.py` so the in-prompt schema and example additions are dropped sooner.
- That second change broke one existing test (`tests/test_ollama_completion.py::test_uses_example_shape_when_full_schema_exceeds_context_budget` expects the example to fit at `num_ctx=120`; with the new 0.5 ratio the example no longer fits and the test now fails). I have not fixed that test yet, and I have not yet re-run the workflow end-to-end after both changes, so the projection is still unverified for this dataset.

**2. Are the vocab results useful? Partially, with a real over-extraction problem.**

- **QUDT quantity matching:** The LLM is over-extracting quantities. The merged context has 60 unique object identifiers but the list of `has_quantitative_attributes` includes the actual NMR quantities (`frequency`, `fidres`, `n_points`, `nucleus`, `averages`, `min_intensity`, `max_intensity`, `x_units`, `y_units`, `cs_threshold`, `observe_frequency`, `file_size`, `shift_reference`) **plus** every Bruker parameter header — `shim_setting_1` through `shim_setting_40`, `O1` through `O8`, `SFO1` through `SFO8`, `STSI`, `STSR`, `STSI` twice, `TDeff`, `TDoff`, `XDIM`, `YMIN_p`, `YMAX_p`, `TM1`, `TM2`, etc. That's why 108 quantity queries. The same dedup logic in `_unique_quantities` keeps them all because they have distinct (identifier, value, unit, quantity_kind) tuples. Quantity grounding is therefore a candidate evaluation per quantity, but most of them are noise the LLM turned into objects because the chunks separated `##.SHIM_SETTINGS=` from the rest of the param block.
- **QUDT candidate results are useful for the legitimate quantities.** Of the 60 `skos__Concept` queries, every one returned between 10-20 voc4cat Concept candidates. But the selection step never ran because the projection crashed first. So the LLM never picked a QUDT term for any quantity — no `QuantityNormalization` records got selected into a final mapping. The artifacts on disk are raw candidate pools, not finalized mappings.
- **voc4cat object grounding:** Same shape: 60 unique object identifiers, all queried, all returned 10-20 candidates each, none selected. None of the 60 objects are real scientific objects in the voc4cat sense — most are parameter headers like `SSB`, `TILT`, `P`, `XFAC`, `Bruker Alpha-P ATR`, `zg30`, `1H_NMR_measurement`. A few are real (`spectrum`, `peak_table`, `dataset_description.txt`). The dedup at the chunk-merge level doesn't help because each chunk produced its own parameter-line object that the merge considers "distinct". Object grounding is therefore querying voc4cat with very low-precision identifiers, and the LLM selection would (correctly) reject most of them.

The core issue: chunks are split at semantic breakpoints in the Bruker parameter header region, and the chunked-extraction LLM is treating every `##.XYZ=` line as its own object. Either the chunker needs to be smarter about NMR parameter blocks, or the extraction prompt needs to be told to ignore bare parameter lines. That is a quality/precision problem on top of the flow problems.

**3. Is the extraction result ready for dcat-ap-plus projection? Not yet.**

- I have not produced a final projected document. No `extraction_result.json` was written.
- The `normalization.json` was never written either (no successful normalization completion since I changed the budget ratio).
- The 60-object merged context is too wide to feed into the dcat profile even if projection succeeded. Real dcat-ap-plus `Dataset` profiles expect ~10-30 facts, not 60 mostly-parameter objects.
- Before projection, we need: (a) the over-extraction fixed so the merged context is a sensible set of ~15-20 real objects, (b) the projection 400 fixed and verified end-to-end on this dataset, (c) a finalized normalization with QUDT and voc4cat mappings to feed into the profile projection.

**Next steps I'd take, in order:**

1. Fix the one failing test (`test_uses_example_shape_when_full_schema_exceeds_context_budget`) by making the test set `num_ctx` large enough to fit the example at the new 0.5 ratio, then re-run the full suite.
2. Re-run the workflow end-to-end to verify profile projection now succeeds with the projection-prompt schema-removal + tighter budget-ratio change.
3. Address the over-extraction: tighten the extraction system prompt or add a "no bare parameter lines" instruction, then re-run on `30089342` and verify the merged object count drops to a sensible ~15-25.
4. Once the merged context is clean, let the LLM selection step run on the candidate pools to produce real QUDT quantity mappings and voc4cat object terms, and persist the `normalization.json` and `extraction_result.json` artifacts.

