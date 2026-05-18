from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent, RunContext

from app.domain.datasources import DataPackage
from app.domain.extraction.agents import DEFAULT_OUTPUT_RETRIES, prompted_json_output
from app.domain.extraction.artifacts import InitialContext


INITIAL_CONTEXT_INSTRUCTIONS = (
    "You are a scientific data archivist. Analyze research artifact bundles "
    "and extract structured context metadata. Be precise about instrument or "
    "device names, analytical techniques, sample identifiers, compound names, "
    "file roles, metadata provenance, keywords, and uncertainty. Use the "
    "available tools to inspect the file tree and read the most informative "
    "files before producing the final InitialContext."
)


@dataclass
class InitialContextDeps:
    data_package: DataPackage
    max_files_to_read: int = 12
    max_chars_per_file: int = 3000
    files_read: set[str] = field(default_factory=set)


def create_initial_context_agent(
    *,
    model: Any,
    output_retries: int = DEFAULT_OUTPUT_RETRIES,
    tool_retries: int | None = 3,
) -> Agent[InitialContextDeps, InitialContext]:
    agent = Agent(
        model,
        deps_type=InitialContextDeps,
        output_type=prompted_json_output(
            InitialContext,
            name="InitialContext",
            description="Structured context metadata extracted from a research artifact bundle.",
        ),
        instructions=INITIAL_CONTEXT_INSTRUCTIONS,
        model_settings={
            "temperature": 0.0,
            "seed": 42,
        },
        output_retries=output_retries,
        tool_retries=tool_retries,
    )

    @agent.tool
    def list_dataset_files(ctx: RunContext[InitialContextDeps]) -> list[str]:
        """Return all file paths contained in the current data package."""
        return list_initial_context_dataset_files(ctx.deps)

    @agent.tool
    def read_file_content(
        ctx: RunContext[InitialContextDeps],
        file_path: str,
        start_char_idx: int = 0,
        max_chars: int = 3000,
    ) -> str:
        """Read extracted text content from a file path in the data package."""
        return read_initial_context_file_content(
            ctx.deps,
            file_path=file_path,
            start_char_idx=start_char_idx,
            max_chars=max_chars,
        )

    return agent


async def extract_initial_context_from_data_package(
    *,
    data_package: DataPackage,
    model: Any,
    max_files_to_read: int = 12,
    max_chars_per_file: int = 3000,
) -> InitialContext:
    agent = create_initial_context_agent(model=model)
    deps = InitialContextDeps(
        data_package=data_package,
        max_files_to_read=max_files_to_read,
        max_chars_per_file=max_chars_per_file,
    )
    result = await agent.run(
        _initial_context_prompt(data_package, max_files_to_read, max_chars_per_file),
        deps=deps,
    )
    return result.output


def list_initial_context_dataset_files(deps: InitialContextDeps) -> list[str]:
    return deps.data_package.get_file_path_list()


def read_initial_context_file_content(
    deps: InitialContextDeps,
    *,
    file_path: str,
    start_char_idx: int = 0,
    max_chars: int = 3000,
) -> str:
    if file_path not in deps.files_read:
        if len(deps.files_read) >= deps.max_files_to_read:
            return (
                "[File read limit reached; use already inspected files or "
                "increase max_files_to_read.]"
            )
        deps.files_read.add(file_path)

    file_entry = deps.data_package.get_file_entry(file_path)
    content = file_entry.get_extracted_content()

    start = max(0, start_char_idx)
    effective_max_chars = max(0, min(max_chars, deps.max_chars_per_file))
    end = min(len(content), start + effective_max_chars)
    truncated = end < len(content)
    return content[start:end] + ("...[truncated]" if truncated else "")


def _initial_context_prompt(
    data_package: DataPackage,
    max_files_to_read: int,
    max_chars_per_file: int,
) -> str:
    return (
        "Analyze the research artifact archive and extract an InitialContext. "
        "Start by listing all files, then read the most informative files "
        "such as README files, metadata tables, instrument exports, report PDFs, "
        "and file headers. "
        f"The data package name is '{data_package.file_name}' and it contains "
        f"{len(data_package.files)} files. Read up to {max_files_to_read} files "
        f"and up to {max_chars_per_file} characters per file. Fill every output "
        "field with concise evidence-grounded values. Use empty lists when no "
        "entities, relationships, metadata sources, or keywords can be identified."
    )
