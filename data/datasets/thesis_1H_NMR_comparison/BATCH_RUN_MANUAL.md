# Thesis 1H NMR Batch Run Manual

This manual explains how to run the thesis 1H NMR comparison batch with
`scripts/thesis_nmr_batch.py`.

The script calls the public SIMONE API endpoints stage by stage. It does not
call backend internals. It uploads each dataset, switches the runtime model,
runs orientation, chunking, evidence extraction, profile construction, and
vocabulary grounding, then copies the runtime artifacts into a stable result
folder.

## Prerequisites

Run all commands from the repository root.

On Windows PowerShell:

```powershell
.\simone.bat status
.\simone.bat dev
```

Verify that the API is reachable:

```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/health" -UseBasicParsing -TimeoutSec 10
```

The default runner expects these files:

```text
data/datasets/thesis_1H_NMR_comparison/1H_NMR_raw_bruker_10.zip
data/datasets/thesis_1H_NMR_comparison/1H_NMR_chemspectra_export.zip
```

The expected package ids are:

```text
1H_NMR_raw_bruker_10.zip       2e38a6a6
1H_NMR_chemspectra_export.zip  7ac2f7cf
```

## Run Matrix

The default run matrix contains two datasets and four configurations.

Datasets:

```text
1H_NMR_raw_bruker_10
1H_NMR_chemspectra_export
```

Configurations:

```text
fixed_1024__gemma3_12b_cloud
fixed_128__gemma3_12b_cloud
semantic_1024_128_t95_bw2__gemma3_12b_cloud
fixed_1024__gemma3_27b_cloud
```

This produces eight runs in total.

## Dry Run

Before running the batch, print the selected matrix and output folders:

```powershell
backend\.venv\Scripts\python.exe scripts\thesis_nmr_batch.py --dry-run
```

Use a fixed batch name when results must be grouped under a known folder:

```powershell
backend\.venv\Scripts\python.exe scripts\thesis_nmr_batch.py `
  --batch-timestamp 20260703-final `
  --dry-run
```

## Full Batch

Run the full matrix:

```powershell
backend\.venv\Scripts\python.exe scripts\thesis_nmr_batch.py `
  --batch-timestamp 20260703-final `
  --stage-timeout-minutes 240 `
  --poll-interval-seconds 10
```

Use a new `--batch-timestamp` for each independent final batch. Existing result
folders are protected by default.

If a folder already exists and should be replaced, add `--overwrite`:

```powershell
backend\.venv\Scripts\python.exe scripts\thesis_nmr_batch.py `
  --batch-timestamp 20260703-final `
  --stage-timeout-minutes 240 `
  --poll-interval-seconds 10 `
  --overwrite
```

## Single Run

Run one dataset and one configuration:

```powershell
backend\.venv\Scripts\python.exe scripts\thesis_nmr_batch.py `
  --batch-timestamp 20260703-final `
  --only-dataset 2e38a6a6 `
  --only-config fixed_1024__gemma3_12b_cloud `
  --stage-timeout-minutes 240 `
  --poll-interval-seconds 10
```

`--only-dataset` accepts the dataset slug, ZIP filename, or package id.

Examples:

```text
1H_NMR_raw_bruker_10
1H_NMR_raw_bruker_10.zip
2e38a6a6
```

## Output Layout

Results are written to:

```text
data/datasets/thesis_1H_NMR_comparison/results/{batch_timestamp}/{dataset_slug}/{config_id}/
```

Important files in each run folder:

```text
run_manifest.json
config.json
source.zip
grounded_final_draft.json
generated_reconstructed_draft.json
generated_attribute_draft.json
generated_core_draft.json
requirement_report.json
vocab_queries.json
grounded_validation.json
token_usage.json
api_token_usage.json
extraction_run_state.json
runtime_artifacts/
```

`run_manifest.json` is the primary run ledger. It records the stage order,
request parameters, start and finish timestamps, durations, final status, copied
runtime artifacts, and errors.

The stage order is:

```text
upload
runtime_model_switch
orientation
chunking
evidence_context
profile_construction
grounding
artifact_export
```

## Status Checks

Show the saved manifest state for a batch:

```powershell
$root = Resolve-Path 'data\datasets\thesis_1H_NMR_comparison\results\20260703-final'
Get-ChildItem $root -Recurse -Filter run_manifest.json | ForEach-Object {
  $rel = $_.FullName.Substring($root.Path.Length + 1).Replace('\run_manifest.json', '')
  $j = Get-Content $_.FullName -Raw | ConvertFrom-Json
  [pscustomobject]@{
    run = $rel
    status = $j.status
    orientation = $j.stages.orientation.final_status
    chunking = $j.stages.chunking.final_status
    evidence = $j.stages.evidence_context.final_status
    profile = $j.stages.profile_construction.final_status
    grounding = $j.stages.grounding.final_status
    export = $j.stages.artifact_export.final_status
  }
} | Sort-Object run | Format-Table -AutoSize
```

Show active API tasks:

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/v1/tasks' -TimeoutSec 10 |
  ConvertTo-Json -Depth 5
```

Check evidence progress for a fixed-token run:

```powershell
Invoke-RestMethod `
  -Uri 'http://127.0.0.1:8000/api/v1/extraction/stages/evidence/2e38a6a6/progress?chunking_strategy=fixed_tokens&chat_model=gemma3%3A12b-cloud' `
  -TimeoutSec 10 |
  Select-Object status, @{n='stage';e={$_.progress.stage}}, @{n='processed';e={$_.progress.processed_chunks}}, @{n='total';e={$_.progress.total_chunks}} |
  Format-List
```

Check grounding progress:

```powershell
Invoke-RestMethod `
  -Uri 'http://127.0.0.1:8000/api/v1/extraction/stages/grounding/2e38a6a6/progress?chunking_strategy=fixed_tokens&chat_model=gemma3%3A12b-cloud' `
  -TimeoutSec 10 |
  ConvertTo-Json -Depth 4
```

## Pause After The Current Run

The script itself runs the selected plan serially. For a normal foreground run,
press `Ctrl+C` only after the current run has completed and the next run has not
started yet.

For safer pausing, run each remaining configuration one at a time with
`--only-dataset` and `--only-config`. This avoids stopping the script in the
middle of a stage.

Example:

```powershell
backend\.venv\Scripts\python.exe scripts\thesis_nmr_batch.py `
  --batch-timestamp 20260703-final `
  --only-dataset 2e38a6a6 `
  --only-config semantic_1024_128_t95_bw2__gemma3_12b_cloud `
  --stage-timeout-minutes 240 `
  --poll-interval-seconds 10 `
  --overwrite
```

Then start the next selected run manually.

## Resume A Partial Batch

The script does not skip completed runs automatically. To resume a partial
batch, inspect the manifests, identify missing or failed runs, and invoke the
script once per remaining dataset and configuration with the same
`--batch-timestamp`.

Use `--overwrite` only for a run folder that is incomplete, failed, or meant to
be replaced.

Example resume command:

```powershell
backend\.venv\Scripts\python.exe scripts\thesis_nmr_batch.py `
  --batch-timestamp 20260703-final `
  --only-dataset 7ac2f7cf `
  --only-config fixed_128__gemma3_12b_cloud `
  --stage-timeout-minutes 240 `
  --poll-interval-seconds 10 `
  --overwrite
```

## Background Controller For Remaining Runs

If another agent wants a controlled background resume, use a small PowerShell
loop. This lets the agent stop the outer loop after the current run while the
current Python process continues.

```powershell
$ErrorActionPreference = 'Continue'
$root = 'C:\Users\simcl\Documents\GitHub\Semantic Inference Module for Ontology-driven Node Extraction (SIMONE)'
$python = Join-Path $root 'backend\.venv\Scripts\python.exe'
$batch = '20260703-final'
$logRoot = Join-Path $root "data\datasets\thesis_1H_NMR_comparison\results\$batch\_batch_logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$out = Join-Path $logRoot 'resume_stdout.log'
$err = Join-Path $logRoot 'resume_stderr.log'

$runs = @(
  @('2e38a6a6', 'semantic_1024_128_t95_bw2__gemma3_12b_cloud'),
  @('2e38a6a6', 'fixed_1024__gemma3_27b_cloud'),
  @('7ac2f7cf', 'fixed_1024__gemma3_12b_cloud'),
  @('7ac2f7cf', 'fixed_128__gemma3_12b_cloud'),
  @('7ac2f7cf', 'semantic_1024_128_t95_bw2__gemma3_12b_cloud'),
  @('7ac2f7cf', 'fixed_1024__gemma3_27b_cloud')
)

Set-Location $root
foreach ($run in $runs) {
  $dataset = $run[0]
  $config = $run[1]
  Add-Content -Path $out -Value "START $((Get-Date).ToString('o')) dataset=$dataset config=$config"
  & $python 'scripts\thesis_nmr_batch.py' `
    --batch-timestamp $batch `
    --stage-timeout-minutes 240 `
    --poll-interval-seconds 10 `
    --only-dataset $dataset `
    --only-config $config `
    --overwrite >> $out 2>> $err
  $exitCode = $LASTEXITCODE
  Add-Content -Path $out -Value "END $((Get-Date).ToString('o')) dataset=$dataset config=$config exit=$exitCode"
  if ($exitCode -ne 0) {
    Add-Content -Path $err -Value "STOP after failed run dataset=$dataset config=$config exit=$exitCode"
    exit $exitCode
  }
}
```

Start it in a hidden background PowerShell process if needed:

```powershell
Start-Process powershell.exe -WindowStyle Hidden -ArgumentList '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', 'path\to\resume-batch.ps1'
```

To pause after the current configured run, stop only the outer PowerShell
controller process. Do not stop the active Python process if the current run
should finish.

## Reproducibility Notes

Each configuration sets:

```text
generation_seed = 42
```

The runner patches the runtime `chat_model` and `generation_seed` before each
run. It restores the previous runtime values when the selected plan finishes.

The seed can reduce variation, but it does not guarantee identical outputs on
all devices. Differences can still come from model version, backend changes,
Ollama behavior, hardware, vocabulary database contents, and external service
state.

## Moving The Run To Another Device

Before running on another device, verify:

```text
The repository is on the intended commit.
The two ZIP files exist under data/datasets/thesis_1H_NMR_comparison.
The SIMONE API is reachable at http://127.0.0.1:8000.
Neo4j contains the required vocabularies.
The requested Ollama chat models are available.
The backend virtual environment exists or dependencies are installed.
```

Then run:

```powershell
backend\.venv\Scripts\python.exe scripts\thesis_nmr_batch.py --dry-run
```

If the dry run looks correct, start a single smoke run first:

```powershell
backend\.venv\Scripts\python.exe scripts\thesis_nmr_batch.py `
  --batch-timestamp SMOKE-YYYYMMDD `
  --only-dataset 2e38a6a6 `
  --only-config fixed_1024__gemma3_12b_cloud `
  --stage-timeout-minutes 240 `
  --poll-interval-seconds 10
```

After the smoke run, check that these files exist:

```text
run_manifest.json
grounded_final_draft.json
generated_reconstructed_draft.json
api_token_usage.json
runtime_artifacts/
```

Only then start the full batch.

## Failure Handling

If a run fails, keep its result folder. The partial `run_manifest.json` and
`runtime_artifacts/` folder are useful for diagnosis.

Common causes:

```text
API is not running.
Neo4j or Ollama is not reachable.
The configured model is missing.
The stage timeout was too short.
The target result folder already exists without --overwrite.
The uploaded package id does not match the expected id.
```

After fixing the cause, rerun only the failed configuration with the same
`--batch-timestamp` and `--overwrite`.

