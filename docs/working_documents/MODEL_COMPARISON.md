# Model Comparison Tracking

## Purpose

SIMONE evaluation requires comparing extraction results for the same dataset across different LLMs. The output storage layer now branches results by the chat model used, so running the same dataset with `rnj-1:8b-cloud`, `lfm2.5:8b`, `qwen3:8b`, or any other model produces separate, non-overlapping artifacts.

## How it works

The `chat_model` field was added to:
- `ExtractionRunState` — captured when a run state is created
- `ExtractionRunResult` — captured when a final validated result is saved

`FileSystemExtractionOutputRepository` uses this value to branch the output directory:

```
.runtime/output/{data_package_id}/
  {chat_model}/
    extraction_result.json
    extraction_run_state.json
    extraction_warnings.json
```

When `chat_model` is `None`, results fall back to the legacy flat path:
```
.runtime/output/{data_package_id}/
  extraction_result.json
```

This means:
- Running package `30089342` with `rnj-1:8b-cloud` stores results under `30089342/rnj-1:8b-cloud/`
- Switching the runtime model to `qwen3:8b` and re-running `30089342` stores results under `30089342/qwen3:8b/`
- Both sets of results coexist and can be loaded independently based on whichever model is currently configured

## Loading behavior

`_load_run_state_or_none`, `_load_result_or_none`, and `get_extraction_result` all pass the currently configured `chat_model` to the repository. This ensures the API always loads the result that matches the active model.

## Evaluation workflow

To compare models on the same dataset:

1. Set model A via `.env` or runtime PATCH
2. Run extraction on the dataset
3. Set model B
4. Re-run extraction on the same dataset (use `force_rerun=true` or the staged API with `target_stage: "complete"`)
5. Inspect both result directories under `.runtime/output/{package_id}/`

## Future work

- Add an explicit "compare results for package X across models" API/UI view
- Capture per-model metrics (token usage, duration, schema validation score) in a summary index
- Tag evaluation runs with model metadata for the thesis evaluation table
