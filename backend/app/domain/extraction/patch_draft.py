from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from app.domain.datasources import ContentChunk
from app.domain.extraction.agents import DEFAULT_OUTPUT_RETRIES, prompted_json_output
from app.domain.extraction.artifacts import InitialContext
from app.domain.profiles import ProfileManifest


PATCH_DRAFT_INSTRUCTIONS = (
    "You extract structured metadata from document chunks and return JSON "
    "merge patches for a current DCAT dataset draft. Return only partial "
    "updates supported by the current chunk batch. Do not rewrite unchanged "
    "fields. Avoid duplicate information. Focus on data-generating activity, "
    "device or instrument details, entities, distributions, file roles, and "
    "qualitative or quantitative attributes extracted from the chunks. For "
    "objects in lists, include an id field when possible. Use ids from the "
    "current draft to update existing objects. Do not use null to delete "
    "existing values; null values are ignored by the merge function."
)


class MergePatch(BaseModel):
    patch: dict[str, Any] = Field(default_factory=dict)


class PatchDraftPrerequisiteError(Exception):
    """Raised when patch extraction is requested before prerequisites exist."""


class ChunkingRequiredError(Exception):
    """Raised when patch extraction is requested before chunking completed."""


@dataclass
class PatchDraftDeps:
    initial_context: InitialContext
    current_draft: dict[str, Any]
    chunk_batch: list[ContentChunk]
    batch_no: int
    total_batches: int
    document_file_path: str
    profile_manifest: ProfileManifest
    profile_json_schema: dict[str, Any]


@dataclass
class PatchRecord:
    file_name: str
    patch: dict[str, Any]


@dataclass
class PatchDraftResult:
    draft: dict[str, Any]
    patches: list[PatchRecord]


def create_patch_draft_agent(
    *,
    model: Any,
    output_retries: int = DEFAULT_OUTPUT_RETRIES,
) -> Agent[PatchDraftDeps, MergePatch]:
    agent = Agent(
        model,
        deps_type=PatchDraftDeps,
        output_type=prompted_json_output(
            MergePatch,
            name="MergePatch",
            description="RFC7396-style partial update for a current DCAT dataset draft.",
        ),
        instructions=PATCH_DRAFT_INSTRUCTIONS,
        model_settings={
            "temperature": 0.0,
            "seed": 42,
        },
        output_retries=output_retries,
    )

    @agent.instructions
    def add_patch_extraction_context(ctx: RunContext[PatchDraftDeps]) -> str:
        chunk_texts = ""
        for index, chunk in enumerate(ctx.deps.chunk_batch, start=1):
            chunk_texts += (
                f"Chunk {index}/{len(ctx.deps.chunk_batch)}:\n"
                f"{_chunk_json_for_prompt(chunk)}\n\n"
            )

        return (
            f"DCAT schema profile:\n{ctx.deps.profile_manifest.identifier}\n\n"
            f"Target class:\n{ctx.deps.profile_manifest.target_class}\n\n"
            "Profile JSON Schema:\n"
            f"{json.dumps(ctx.deps.profile_json_schema, indent=2)}\n\n"
            f"Document file path:\n{ctx.deps.document_file_path}\n\n"
            "InitialContext JSON:\n"
            f"{ctx.deps.initial_context.model_dump_json(indent=2)}\n\n"
            "Current DCAT dataset draft JSON:\n"
            f"{json.dumps(ctx.deps.current_draft, indent=2)}\n\n"
            f"Chunk batch {ctx.deps.batch_no}/{ctx.deps.total_batches}:\n"
            f"{chunk_texts}\n\n"
            "Return only a JSON object like {\"patch\": {...}} where patch "
            "contains the merge patch updates supported by this chunk batch."
        )

    return agent


async def extract_merge_patch(
    *,
    initial_context: InitialContext,
    draft: dict[str, Any],
    document_file_path: str,
    chunk_batch: list[ContentChunk],
    batch_no: int,
    total_batches: int,
    profile_manifest: ProfileManifest,
    profile_json_schema: dict[str, Any],
    model: Any,
) -> dict[str, Any]:
    agent = create_patch_draft_agent(model=model)
    deps = PatchDraftDeps(
        initial_context=initial_context,
        current_draft=draft,
        chunk_batch=chunk_batch,
        batch_no=batch_no,
        total_batches=total_batches,
        document_file_path=document_file_path,
        profile_manifest=profile_manifest,
        profile_json_schema=profile_json_schema,
    )
    result = await agent.run(
        (
            f"Extract the merge patch for chunk batch {batch_no}/{total_batches}. "
            "Return only the MergePatch JSON object. Do not include unchanged fields."
        ),
        deps=deps,
    )
    return result.output.patch


async def patch_draft_from_content_chunks(
    *,
    initial_context: InitialContext,
    initial_draft: dict[str, Any],
    content_chunks_by_file: list[list[ContentChunk]],
    profile_manifest: ProfileManifest,
    profile_json_schema: dict[str, Any],
    model: Any,
    num_chunks_per_turn: int,
) -> PatchDraftResult:
    draft = copy.deepcopy(initial_draft)
    patches: list[PatchRecord] = []

    for document_index, chunks in enumerate(content_chunks_by_file, start=1):
        if not chunks:
            continue

        effective_batch_size = max(1, min(num_chunks_per_turn, len(chunks)))
        total_batches = (len(chunks) + effective_batch_size - 1) // effective_batch_size
        document_file_path = chunks[0].file_path

        for batch_index, start_index in enumerate(
            range(0, len(chunks), effective_batch_size),
            start=1,
        ):
            chunk_batch = chunks[start_index : start_index + effective_batch_size]
            patch = await extract_merge_patch(
                initial_context=initial_context,
                draft=draft,
                document_file_path=document_file_path,
                chunk_batch=chunk_batch,
                batch_no=batch_index,
                total_batches=total_batches,
                profile_manifest=profile_manifest,
                profile_json_schema=profile_json_schema,
                model=model,
            )
            draft = apply_merge_patch(draft, patch)
            patches.append(
                PatchRecord(
                    file_name=(
                        f"{document_index}_{ContentChunk.get_chunk_group_id_from_file_path(document_file_path)}"
                        f"_patch_{batch_index}_{total_batches}.json"
                    ),
                    patch=patch,
                )
            )

    return PatchDraftResult(draft=draft, patches=patches)


def apply_merge_patch(
    draft: dict[str, Any],
    patch: dict[str, Any],
) -> dict[str, Any]:
    draft_copy = copy.deepcopy(draft)
    patch_copy = copy.deepcopy(patch)
    return deep_merge(draft_copy, patch_copy)


def deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if value is None:
            continue

        dst_value = dst.get(key)
        if isinstance(value, dict) and isinstance(dst_value, dict):
            deep_merge(dst_value, value)
        elif isinstance(value, list) and isinstance(dst_value, list):
            dst[key] = _merge_lists(dst_value, value)
        else:
            dst[key] = value

    return dst


def _merge_lists(dst_value: list[Any], patch_value: list[Any]) -> list[Any]:
    if not patch_value:
        return dst_value
    if not dst_value:
        return patch_value

    if all(isinstance(item, dict) for item in patch_value) and all(
        isinstance(item, dict) for item in dst_value
    ):
        return _merge_list_of_dicts_by_id(dst_value, patch_value)

    if all(isinstance(item, str) for item in patch_value) and all(
        isinstance(item, str) for item in dst_value
    ):
        return _merge_list_of_strings(dst_value, patch_value)

    return patch_value


def _merge_list_of_dicts_by_id(
    dst_value: list[dict[str, Any]],
    patch_value: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for item in dst_value:
        item.setdefault("id", _new_patch_id())

    dst_items_by_id = {item["id"]: item for item in dst_value}
    for item in patch_value:
        item.setdefault("id", _new_patch_id())
        item_id = item["id"]
        if item_id in dst_items_by_id:
            deep_merge(dst_items_by_id[item_id], item)
        else:
            dst_value.append(item)

    return dst_value


def _merge_list_of_strings(
    dst_value: list[str],
    patch_value: list[str],
) -> list[str]:
    merged = list(dst_value)
    for item in patch_value:
        if item not in merged:
            merged.append(item)
    return merged


def _new_patch_id() -> str:
    return str(uuid4())[:8]


def _chunk_json_for_prompt(chunk: ContentChunk) -> str:
    return chunk.model_dump_json(
        exclude={"embedding"},
    )
