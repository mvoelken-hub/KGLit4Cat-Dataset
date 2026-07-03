from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "thesis_nmr_batch.py"
SPEC = importlib.util.spec_from_file_location("thesis_nmr_batch", SCRIPT_PATH)
assert SPEC is not None
thesis_nmr_batch = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = thesis_nmr_batch
SPEC.loader.exec_module(thesis_nmr_batch)


def test_batch_plan_expands_two_datasets_and_four_configs(tmp_path):
    plan = thesis_nmr_batch.build_run_plan(
        batch_timestamp="20260703-120000",
        results_dir=tmp_path,
    )

    assert len(plan) == 8
    assert {run.dataset.expected_package_id for run in plan} == {"2e38a6a6", "7ac2f7cf"}
    assert {run.config.config_id for run in plan} == {
        "fixed_1024__gemma3_12b_cloud",
        "fixed_128__gemma3_12b_cloud",
        "semantic_1024_128_t95_bw2__gemma3_12b_cloud",
        "fixed_1024__gemma3_27b_cloud",
    }


def test_batch_plan_result_paths_include_timestamp_dataset_and_config(tmp_path):
    plan = thesis_nmr_batch.build_run_plan(
        batch_timestamp="20260703-120000",
        results_dir=tmp_path,
        only_dataset="2e38a6a6",
        only_config="fixed_128__gemma3_12b_cloud",
    )

    assert len(plan) == 1
    assert plan[0].result_dir == (
        tmp_path
        / "20260703-120000"
        / "1H_NMR_raw_bruker_10"
        / "fixed_128__gemma3_12b_cloud"
    )


def test_fixed_128_config_uses_128_as_target_and_maximum():
    config = next(
        item
        for item in thesis_nmr_batch.RUN_CONFIGS
        if item.config_id == "fixed_128__gemma3_12b_cloud"
    )

    assert config.chunking_strategy == "fixed_tokens"
    assert config.fixed_tokens_per_chunk == 128
    assert config.min_tokens_per_chunk == 128
    assert config.max_tokens_per_chunk == 128
    assert config.generation_seed == 42


def test_stage_order_contains_required_timestamped_stages():
    assert thesis_nmr_batch.stage_names() == [
        "upload",
        "runtime_model_switch",
        "orientation",
        "chunking",
        "evidence_context",
        "profile_construction",
        "grounding",
        "artifact_export",
    ]


def test_safe_path_component_matches_runtime_model_directory():
    assert thesis_nmr_batch.safe_path_component("gemma3:12b-cloud") == "gemma3_12b-cloud"
