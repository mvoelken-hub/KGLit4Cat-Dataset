from enum import Enum
from typing import Awaitable, Callable, Literal
import numpy as np
from hashlib import sha256

from dataclasses import dataclass
from pydantic import BaseModel, Field, computed_field

from app.domain.datasources.datasource import (
    FileEntry
)

from app.domain.datasources.text_quality import (
    TextQualityConfig,
    TextQualityDecision,
    DecisionKind,
    classify_text_line
)

from app.domain.datasources.errors import (
    EmbeddingDistanceCalcError
)

from app.domain.token_budget import PromptTokenBudgeter

_DEFAULT_MAX_TOKENS_PER_CHUNK = 1024
_DEFAULT_MIN_TOKENS_PER_CHUNK = 128
_DEFAULT_MAX_FILTERED_LINE_GAP = 32


class ChunkingStrategy(str, Enum):
    SEMANTIC = "semantic"
    FIXED_TOKENS = "fixed_tokens"


_DEFAULT_CHUNKING_STRATEGY = ChunkingStrategy.SEMANTIC
_DEFAULT_FIXED_TOKENS_PER_CHUNK = 1024


@dataclass
class FilteredLine:
    text: str
    line_idx: int

@dataclass
class BufferWindow:
    line: FilteredLine
    combined: str
    embedding: list[float] | None = None
    distance_next: float | None = None


@dataclass
class ChunkSegment:
    chunk: "ContentChunk"
    barrier_before: bool = False


class ChunkPostProcessingMetadata(BaseModel):
    source_chunk_count: int = Field(1, ge=1)
    operations: list[str] = Field(default_factory=list)


class ContentChunk(BaseModel):
    content: str
    data_package_id: str
    file_path: str
    start_idx: int = Field(..., ge=0, description="Start line index of the chunk in the original file")
    end_idx: int = Field(..., ge=0, description="End line index of the chunk in the original file")
    filtered_line_indices: list[int] = Field(default_factory=list, description="Original line indices that passed the text quality filter and are included in this chunk")
    summary: str | None = None
    embedding: list[float] | None = None
    post_processing: ChunkPostProcessingMetadata = Field(default_factory=ChunkPostProcessingMetadata)

    @computed_field
    @property
    def chunk_group_id(self) -> str:
        return self.get_chunk_group_id_from_file_path(self.file_path)

    @classmethod
    async def create_chunks_for_file_entry(
        cls,
        data_package_id: str,
        file_entry: FileEntry,
        embedding_func: Callable[[list[str]], Awaitable[list[list[float]]]],
        text_classification_func: Callable[[str], TextQualityDecision] = classify_text_line,
        min_lines_for_chunking: int = 100,
        buffer_window_size: int = 1,
        embedding_batch_size: int = 32,
        semantic_chunking_threshold: float = 95.0,
        protected_line_indices: list[int] | None = None,
        max_tokens_per_chunk: int = _DEFAULT_MAX_TOKENS_PER_CHUNK,
        min_tokens_per_chunk: int = _DEFAULT_MIN_TOKENS_PER_CHUNK,
        max_filtered_line_gap: int = _DEFAULT_MAX_FILTERED_LINE_GAP,
        token_budgeter: PromptTokenBudgeter | None = None,
        chunking_strategy: ChunkingStrategy | Literal["semantic", "fixed_tokens"] = _DEFAULT_CHUNKING_STRATEGY,
        fixed_tokens_per_chunk: int = _DEFAULT_FIXED_TOKENS_PER_CHUNK,
    ) -> list["ContentChunk"]:
        chunking_strategy = ChunkingStrategy(chunking_strategy) if isinstance(chunking_strategy, str) else chunking_strategy
        token_budgeter = token_budgeter or PromptTokenBudgeter()

        chunk_list: list[ContentChunk] = []

        lines: list[str] = file_entry.get_extracted_content().splitlines(keepends=True)

        filtered_lines: list[FilteredLine] = [
            FilteredLine(text=line, line_idx=i) for i, line in enumerate(lines)
            if text_classification_func(line).kind == DecisionKind.KEEP
        ]

        # Ensure explicitly protected lines are always present, even if the
        # classifier would have dropped them.  They are inserted in original
        # order and deduplicated by line_idx.
        if protected_line_indices:
            protected = {
                line.line_idx: line
                for line in filtered_lines
            }
            for i in protected_line_indices:
                if 0 <= i < len(lines) and i not in protected:
                    protected[i] = FilteredLine(text=lines[i], line_idx=i)
            filtered_lines = [
                protected[i] for i in sorted(protected)
            ]

        if not filtered_lines:
            return []

        if chunking_strategy == ChunkingStrategy.FIXED_TOKENS:
            fixed_chunks = cls._build_fixed_token_chunks(
                filtered_lines,
                data_package_id=data_package_id,
                file_path=file_entry.file_path,
                fixed_tokens_per_chunk=fixed_tokens_per_chunk,
                token_budgeter=token_budgeter,
            )
            return cls._post_process_chunks(
                fixed_chunks,
                filtered_lines=filtered_lines,
                data_package_id=data_package_id,
                file_path=file_entry.file_path,
                max_tokens_per_chunk=fixed_tokens_per_chunk,
                min_tokens_per_chunk=min_tokens_per_chunk,
                max_filtered_line_gap=max_filtered_line_gap,
                token_budgeter=token_budgeter,
                boundary_distances={},
            )

        if len(filtered_lines) < min_lines_for_chunking:
            single_chunk = cls(
                content="".join(line.text for line in filtered_lines),
                data_package_id=data_package_id,
                file_path=file_entry.file_path,
                start_idx=filtered_lines[0].line_idx,
                end_idx=filtered_lines[-1].line_idx,
                filtered_line_indices=[line.line_idx for line in filtered_lines]
            )
            bounded = [single_chunk]
            if max_tokens_per_chunk and max_tokens_per_chunk > 0:
                bounded = cls._split_chunks_by_token_budget(
                    [single_chunk],
                    filtered_lines=filtered_lines,
                    data_package_id=data_package_id,
                    file_path=file_entry.file_path,
                    max_tokens_per_chunk=max_tokens_per_chunk,
                    token_budgeter=token_budgeter,
                )
            return cls._post_process_chunks(
                bounded,
                filtered_lines=filtered_lines,
                data_package_id=data_package_id,
                file_path=file_entry.file_path,
                max_tokens_per_chunk=max_tokens_per_chunk,
                min_tokens_per_chunk=min_tokens_per_chunk,
                max_filtered_line_gap=max_filtered_line_gap,
                token_budgeter=token_budgeter,
            )

        combined_lines = combine_lines(
            filtered_lines,
            buffer_size=buffer_window_size
        )

        for i in range(0, len(combined_lines), embedding_batch_size):
            batch = combined_lines[i:i + embedding_batch_size]
            texts_to_embed = [item.combined for item in batch]
            embeddings = await embedding_func(texts_to_embed)
            for item, embedding in zip(batch, embeddings):
                item.embedding = embedding

        distances = attach_cosine_distances(combined_lines)
        boundary_distances = {
            item.line.line_idx: item.distance_next
            for item in combined_lines
            if item.distance_next is not None
        }

        breakpoint_distance_threshold = np.percentile(distances, semantic_chunking_threshold)

        indices_about_threshold = [
            index for index, distance in enumerate(distances)
            if distance >= breakpoint_distance_threshold
        ]

        for i, breakpoint_index in enumerate(indices_about_threshold):

            start_index = 0 if i == 0 else indices_about_threshold[i - 1]
            end_index = (
                breakpoint_index
                if i < len(indices_about_threshold) - 1
                else len(combined_lines) - 1
            )

            group = combined_lines[start_index : end_index + 1]
            combined_text = "".join(item.line.text for item in group)
            chunk_list.append(cls(
                content=combined_text,
                data_package_id=data_package_id,
                file_path=file_entry.file_path,
                start_idx=group[0].line.line_idx,
                end_idx=group[-1].line.line_idx,
                filtered_line_indices=[item.line.line_idx for item in group]
            ))

        bounded_chunks = chunk_list
        if max_tokens_per_chunk and max_tokens_per_chunk > 0:
            bounded_chunks = cls._split_chunks_by_token_budget(
                chunk_list,
                filtered_lines=filtered_lines,
                data_package_id=data_package_id,
                file_path=file_entry.file_path,
                max_tokens_per_chunk=max_tokens_per_chunk,
                token_budgeter=token_budgeter,
            )

        return cls._post_process_chunks(
            bounded_chunks,
            filtered_lines=filtered_lines,
            data_package_id=data_package_id,
            file_path=file_entry.file_path,
            max_tokens_per_chunk=max_tokens_per_chunk,
            min_tokens_per_chunk=min_tokens_per_chunk,
            max_filtered_line_gap=max_filtered_line_gap,
            token_budgeter=token_budgeter,
            boundary_distances=boundary_distances,
        )

    @classmethod
    def _build_fixed_token_chunks(
        cls,
        filtered_lines: list[FilteredLine],
        *,
        data_package_id: str,
        file_path: str,
        fixed_tokens_per_chunk: int,
        token_budgeter: PromptTokenBudgeter,
    ) -> list["ContentChunk"]:
        if not filtered_lines:
            return []
        chunks: list[ContentChunk] = []
        window: list[FilteredLine] = []
        window_tokens = 0
        for line in filtered_lines:
            line_tokens = token_budgeter.count(line.text)
            if window and window_tokens + line_tokens > fixed_tokens_per_chunk:
                chunks.append(cls._chunk_from_filtered_lines(
                    window,
                    data_package_id=data_package_id,
                    file_path=file_path,
                    operations=["fixed_token_split"],
                ))
                window = []
                window_tokens = 0
            window.append(line)
            window_tokens += line_tokens
            if line_tokens > fixed_tokens_per_chunk:
                chunks.append(cls._chunk_from_filtered_lines(
                    window,
                    data_package_id=data_package_id,
                    file_path=file_path,
                    operations=["fixed_token_split"],
                ))
                window = []
                window_tokens = 0
        if window:
            chunks.append(cls._chunk_from_filtered_lines(
                window,
                data_package_id=data_package_id,
                file_path=file_path,
                operations=["fixed_token_split"],
            ))
        return chunks

    @classmethod
    def _split_chunks_by_token_budget(
        cls,
        chunks: list["ContentChunk"],
        *,
        filtered_lines: list[FilteredLine],
        data_package_id: str,
        file_path: str,
        max_tokens_per_chunk: int,
        token_budgeter: PromptTokenBudgeter,
    ) -> list["ContentChunk"]:
        bounded: list[ContentChunk] = []
        for chunk in chunks:
            if token_budgeter.count(chunk.content) <= max_tokens_per_chunk:
                bounded.append(chunk)
                continue
            lines_in_chunk = [
                line
                for line in filtered_lines
                if chunk.start_idx <= line.line_idx <= chunk.end_idx
            ]
            window: list[FilteredLine] = []
            window_tokens = 0
            for line in lines_in_chunk:
                line_tokens = token_budgeter.count(line.text)
                if window and window_tokens + line_tokens > max_tokens_per_chunk:
                    bounded.append(cls._chunk_from_filtered_lines(
                        window,
                        data_package_id=data_package_id,
                        file_path=file_path,
                        operations=["token_cap_split"],
                    ))
                    window = []
                    window_tokens = 0
                window.append(line)
                window_tokens += line_tokens
                if line_tokens > max_tokens_per_chunk:
                    bounded.append(cls._chunk_from_filtered_lines(
                        window,
                        data_package_id=data_package_id,
                        file_path=file_path,
                        operations=["token_cap_split"],
                    ))
                    window = []
                    window_tokens = 0
            if window:
                bounded.append(cls._chunk_from_filtered_lines(
                    window,
                    data_package_id=data_package_id,
                    file_path=file_path,
                    operations=(
                        ["token_cap_split"]
                        if token_budgeter.count(chunk.content) > max_tokens_per_chunk
                        else None
                    ),
                ))
        return bounded

    @classmethod
    def _post_process_chunks(
        cls,
        chunks: list["ContentChunk"],
        *,
        filtered_lines: list[FilteredLine],
        data_package_id: str,
        file_path: str,
        max_tokens_per_chunk: int,
        min_tokens_per_chunk: int,
        max_filtered_line_gap: int,
        token_budgeter: PromptTokenBudgeter,
        boundary_distances: dict[int, float] | None = None,
    ) -> list["ContentChunk"]:
        if not chunks:
            return []
        line_by_index = {line.line_idx: line for line in filtered_lines}
        segments = cls._split_chunks_by_filtered_line_gap(
            chunks,
            line_by_index=line_by_index,
            data_package_id=data_package_id,
            file_path=file_path,
            max_filtered_line_gap=max_filtered_line_gap,
        )
        merged = cls._merge_small_segments(
            segments,
            line_by_index=line_by_index,
            data_package_id=data_package_id,
            file_path=file_path,
            max_tokens_per_chunk=max_tokens_per_chunk,
            min_tokens_per_chunk=min_tokens_per_chunk,
            token_budgeter=token_budgeter,
            boundary_distances=boundary_distances or {},
        )
        return [segment.chunk for segment in merged]

    @classmethod
    def _split_chunks_by_filtered_line_gap(
        cls,
        chunks: list["ContentChunk"],
        *,
        line_by_index: dict[int, FilteredLine],
        data_package_id: str,
        file_path: str,
        max_filtered_line_gap: int,
    ) -> list[ChunkSegment]:
        if max_filtered_line_gap <= 0:
            return [ChunkSegment(chunk=chunk) for chunk in chunks]
        segments: list[ChunkSegment] = []
        for chunk in chunks:
            lines = cls._lines_for_chunk(chunk, line_by_index)
            if not lines:
                segments.append(ChunkSegment(chunk=chunk))
                continue
            groups: list[list[FilteredLine]] = []
            current: list[FilteredLine] = [lines[0]]
            for previous, line in zip(lines, lines[1:]):
                if line.line_idx - previous.line_idx > max_filtered_line_gap:
                    groups.append(current)
                    current = []
                current.append(line)
            groups.append(current)
            if len(groups) == 1:
                segments.append(ChunkSegment(chunk=chunk))
                continue
            for index, group in enumerate(groups):
                segment_chunk = cls._chunk_from_filtered_lines(
                    group,
                    data_package_id=data_package_id,
                    file_path=file_path,
                    source_chunk_count=chunk.post_processing.source_chunk_count,
                    operations=[
                        *chunk.post_processing.operations,
                        "line_gap_split",
                    ],
                )
                segments.append(
                    ChunkSegment(
                        chunk=segment_chunk,
                        barrier_before=index > 0,
                    )
                )
        return segments

    @classmethod
    def _merge_small_segments(
        cls,
        segments: list[ChunkSegment],
        *,
        line_by_index: dict[int, FilteredLine],
        data_package_id: str,
        file_path: str,
        max_tokens_per_chunk: int,
        min_tokens_per_chunk: int,
        token_budgeter: PromptTokenBudgeter,
        boundary_distances: dict[int, float],
    ) -> list[ChunkSegment]:
        if min_tokens_per_chunk <= 0:
            return segments
        index = 0
        while index < len(segments):
            chunk = segments[index].chunk
            if token_budgeter.count(chunk.content) >= min_tokens_per_chunk:
                index += 1
                continue
            candidates = cls._merge_candidates(
                segments,
                index=index,
                line_by_index=line_by_index,
                max_tokens_per_chunk=max_tokens_per_chunk,
                token_budgeter=token_budgeter,
                boundary_distances=boundary_distances,
            )
            if not candidates:
                index += 1
                continue
            _, direction = min(candidates, key=lambda item: item[0])
            left_index = index - 1 if direction == "left" else index
            right_index = index if direction == "left" else index + 1
            merged_chunk = cls._merge_pair(
                segments[left_index].chunk,
                segments[right_index].chunk,
                line_by_index=line_by_index,
                data_package_id=data_package_id,
                file_path=file_path,
            )
            segments[left_index:right_index + 1] = [
                ChunkSegment(
                    chunk=merged_chunk,
                    barrier_before=segments[left_index].barrier_before,
                )
            ]
            index = max(0, left_index - 1)
        return segments

    @classmethod
    def _merge_candidates(
        cls,
        segments: list[ChunkSegment],
        *,
        index: int,
        line_by_index: dict[int, FilteredLine],
        max_tokens_per_chunk: int,
        token_budgeter: PromptTokenBudgeter,
        boundary_distances: dict[int, float],
    ) -> list[tuple[tuple[int, float, int], str]]:
        candidates: list[tuple[tuple[int, float, int], str]] = []
        if index > 0 and not segments[index].barrier_before:
            distance = cls._boundary_distance(
                segments[index - 1].chunk,
                segments[index].chunk,
                boundary_distances,
            )
            if cls._can_merge(
                segments[index - 1].chunk,
                segments[index].chunk,
                line_by_index=line_by_index,
                max_tokens_per_chunk=max_tokens_per_chunk,
                token_budgeter=token_budgeter,
            ):
                candidates.append(((0 if distance is not None else 1, distance or 0.0, 0), "left"))
        if index + 1 < len(segments) and not segments[index + 1].barrier_before:
            distance = cls._boundary_distance(
                segments[index].chunk,
                segments[index + 1].chunk,
                boundary_distances,
            )
            if cls._can_merge(
                segments[index].chunk,
                segments[index + 1].chunk,
                line_by_index=line_by_index,
                max_tokens_per_chunk=max_tokens_per_chunk,
                token_budgeter=token_budgeter,
            ):
                candidates.append(((0 if distance is not None else 1, distance or 0.0, 1), "right"))
        return candidates

    @classmethod
    def _can_merge(
        cls,
        left: "ContentChunk",
        right: "ContentChunk",
        *,
        line_by_index: dict[int, FilteredLine],
        max_tokens_per_chunk: int,
        token_budgeter: PromptTokenBudgeter,
    ) -> bool:
        if max_tokens_per_chunk <= 0:
            return True
        lines = cls._combined_lines_for_chunks(left, right, line_by_index=line_by_index)
        content = "".join(line.text for line in lines)
        return token_budgeter.count(content) <= max_tokens_per_chunk

    @classmethod
    def _merge_pair(
        cls,
        left: "ContentChunk",
        right: "ContentChunk",
        *,
        line_by_index: dict[int, FilteredLine],
        data_package_id: str,
        file_path: str,
    ) -> "ContentChunk":
        operations = [
            *left.post_processing.operations,
            *right.post_processing.operations,
            "min_token_merge",
        ]
        return cls._chunk_from_filtered_lines(
            cls._combined_lines_for_chunks(left, right, line_by_index=line_by_index),
            data_package_id=data_package_id,
            file_path=file_path,
            source_chunk_count=(
                left.post_processing.source_chunk_count
                + right.post_processing.source_chunk_count
            ),
            operations=operations,
        )

    @staticmethod
    def _boundary_distance(
        left: "ContentChunk",
        right: "ContentChunk",
        boundary_distances: dict[int, float],
    ) -> float | None:
        left_indices = left.filtered_line_indices
        right_indices = right.filtered_line_indices
        if not left_indices or not right_indices:
            return None
        boundary_index = left_indices[-1]
        if boundary_index in boundary_distances:
            return boundary_distances[boundary_index]
        if right_indices[0] in boundary_distances:
            return boundary_distances[right_indices[0]]
        return None

    @classmethod
    def _combined_lines_for_chunks(
        cls,
        left: "ContentChunk",
        right: "ContentChunk",
        *,
        line_by_index: dict[int, FilteredLine],
    ) -> list[FilteredLine]:
        indices = sorted(set(left.filtered_line_indices + right.filtered_line_indices))
        return [line_by_index[index] for index in indices if index in line_by_index]

    @staticmethod
    def _lines_for_chunk(
        chunk: "ContentChunk",
        line_by_index: dict[int, FilteredLine],
    ) -> list[FilteredLine]:
        return [
            line_by_index[index]
            for index in chunk.filtered_line_indices
            if index in line_by_index
        ]

    @classmethod
    def _chunk_from_filtered_lines(
        cls,
        lines: list[FilteredLine],
        *,
        data_package_id: str,
        file_path: str,
        source_chunk_count: int = 1,
        operations: list[str] | None = None,
    ) -> "ContentChunk":
        deduped_operations = list(dict.fromkeys(operations or []))
        return cls(
            content="".join(item.text for item in lines),
            data_package_id=data_package_id,
            file_path=file_path,
            start_idx=lines[0].line_idx,
            end_idx=lines[-1].line_idx,
            filtered_line_indices=[item.line_idx for item in lines],
            post_processing=ChunkPostProcessingMetadata(
                source_chunk_count=source_chunk_count,
                operations=deduped_operations,
            ),
        )

    @staticmethod
    def get_chunk_group_id_from_file_path(file_path: str) -> str:
        return sha256(file_path.encode()).hexdigest()[:5]

def combine_lines(lines: list[FilteredLine], buffer_size: int = 1) -> list[BufferWindow]:
    """Combine each line with its neighbours using a sliding buffer window.

    Returns a list of BufferWindow objects.
    """
    combined_lines: list[BufferWindow] = []

    for i, line in enumerate(lines):
        combined_text = ""
        for j in range(i - buffer_size, i):
            if 0 <= j:
                combined_text += lines[j].text

        combined_text += line.text

        for j in range(i + 1, i + buffer_size + 1):
            if j < len(lines):
                combined_text += lines[j].text

        combined_lines.append(BufferWindow(line=line, combined=combined_text))

    return combined_lines

def attach_cosine_distances(
    combined_lines: list[BufferWindow],
) -> list[float]:
    """
    Calculate cosine distances between the embeddings of combined lines and attach them to the BufferWindow objects.
    
    Args:
        - combined_lines: List of BufferWindow objects with embeddings already computed.

    Returns:
        - List of cosine distances between each line and the next line.
    """
    distances: list[float] = []
    for i, item in enumerate(combined_lines[:-1]):
        embedding_current = item.embedding
        embedding_next = combined_lines[i + 1].embedding
        if embedding_current is None or embedding_next is None:
            raise EmbeddingDistanceCalcError(f"Missing embedding for line index {i} or {i + 1}")

        current = np.asarray(embedding_current, dtype=float)
        next_embedding = np.asarray(embedding_next, dtype=float)
        denominator = np.linalg.norm(current) * np.linalg.norm(next_embedding)
        similarity = 0.0 if denominator == 0 else float(np.dot(current, next_embedding) / denominator)
        distance = 1 - similarity
        distances.append(distance)
        item.distance_next = distance

    return distances
