from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic_ai import Agent, RunContext

from app.domain.datasources import DataPackage
from app.domain.extraction.agents import DEFAULT_OUTPUT_RETRIES, prompted_json_output
from app.domain.extraction.artifacts import InitialContext
from app.domain.extraction.token_budget import BudgetedUsage, TokenBudget


INITIAL_CONTEXT_INSTRUCTIONS = (
    "You are a scientific data archivist. Extract a compact InitialContext "
    "from research artifact bundles. Use the output field descriptions as the "
    "contract, ground values in file evidence, and prefer null or empty lists "
    "when support is weak. Use the available tools to inspect the file tree "
    "and read the most informative files before producing the final output."
)


@dataclass
class InitialContextDeps:
    data_package: DataPackage
    max_files_to_read: int = 12
    max_chars_per_file: int = 3000
    files_read: set[str] = field(default_factory=set)


TokenUsageCallback = Callable[[str, Any, int], None]


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
    token_budget: TokenBudget | None = None,
    on_token_usage: TokenUsageCallback | None = None,
) -> InitialContext:
    agent = create_initial_context_agent(model=model)
    deps = InitialContextDeps(
        data_package=data_package,
        max_files_to_read=max_files_to_read,
        max_chars_per_file=max_chars_per_file,
    )
    prompt = _initial_context_prompt(data_package, max_files_to_read, max_chars_per_file)
    estimated_input_tokens = token_budget.estimate_text_tokens(prompt) if token_budget else 0
    result = await agent.run(
        prompt,
        deps=deps,
    )
    if on_token_usage is not None:
        usage = (
            BudgetedUsage(
                result.usage,
                token_budget.metadata(estimated_input_tokens=estimated_input_tokens),
            )
            if token_budget
            else result.usage
        )
        on_token_usage("initial_context", usage, 1)
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
        "Analyze the research artifact archive and extract a compact "
        "InitialContext. "
        "Start by listing all files, then read the most informative files "
        "such as README files, metadata tables, instrument exports, report PDFs, "
        "and file headers. "
        f"The data package name is '{data_package.file_name}' and it contains "
        f"{len(data_package.files)} files. Read up to {max_files_to_read} files "
        f"and up to {max_chars_per_file} characters per file. Follow the field "
        "descriptions, keep values concise, and avoid unsupported inferences."
    )
