import math
from typing import Awaitable, Callable
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

_ESTIMATED_CHARS_PER_TOKEN = 4
_DEFAULT_MAX_TOKENS_PER_CHUNK = 1024


def _estimated_tokens(text: str) -> int:
    return math.ceil(len(text) / _ESTIMATED_CHARS_PER_TOKEN)

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



class ContentChunk(BaseModel):
    content: str
    data_package_id: str
    file_path: str
    start_idx: int = Field(..., ge=0, description="Start line index of the chunk in the original file")
    end_idx: int = Field(..., ge=0, description="End line index of the chunk in the original file")
    filtered_line_indices: list[int] = Field(default_factory=list, description="Original line indices that passed the text quality filter and are included in this chunk")
    summary: str | None = None
    embedding: list[float] | None = None

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
    ) -> list["ContentChunk"]:
        
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

        if len(filtered_lines) < min_lines_for_chunking:
            single_chunk = cls(
                content="".join(line.text for line in filtered_lines),
                data_package_id=data_package_id,
                file_path=file_entry.file_path,
                start_idx=filtered_lines[0].line_idx,
                end_idx=filtered_lines[-1].line_idx,
                filtered_line_indices=[line.line_idx for line in filtered_lines]
            )
            if max_tokens_per_chunk and max_tokens_per_chunk > 0:
                return cls._split_chunks_by_token_budget(
                    [single_chunk],
                    filtered_lines=filtered_lines,
                    data_package_id=data_package_id,
                    file_path=file_entry.file_path,
                    max_tokens_per_chunk=max_tokens_per_chunk,
                )
            return [single_chunk]
        
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

        if max_tokens_per_chunk and max_tokens_per_chunk > 0:
            return cls._split_chunks_by_token_budget(
                chunk_list,
                filtered_lines=filtered_lines,
                data_package_id=data_package_id,
                file_path=file_entry.file_path,
                max_tokens_per_chunk=max_tokens_per_chunk,
            )

        return chunk_list

    @classmethod
    def _split_chunks_by_token_budget(
        cls,
        chunks: list["ContentChunk"],
        *,
        filtered_lines: list[FilteredLine],
        data_package_id: str,
        file_path: str,
        max_tokens_per_chunk: int,
    ) -> list["ContentChunk"]:
        bounded: list[ContentChunk] = []
        for chunk in chunks:
            if _estimated_tokens(chunk.content) <= max_tokens_per_chunk:
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
                line_tokens = _estimated_tokens(line.text)
                if window and window_tokens + line_tokens > max_tokens_per_chunk:
                    bounded.append(cls._chunk_from_filtered_lines(
                        window,
                        data_package_id=data_package_id,
                        file_path=file_path,
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
                    ))
                    window = []
                    window_tokens = 0
            if window:
                bounded.append(cls._chunk_from_filtered_lines(
                    window,
                    data_package_id=data_package_id,
                    file_path=file_path,
                ))
        return bounded

    @classmethod
    def _chunk_from_filtered_lines(
        cls,
        lines: list[FilteredLine],
        *,
        data_package_id: str,
        file_path: str,
    ) -> "ContentChunk":
        return cls(
            content="".join(item.text for item in lines),
            data_package_id=data_package_id,
            file_path=file_path,
            start_idx=lines[0].line_idx,
            end_idx=lines[-1].line_idx,
            filtered_line_indices=[item.line_idx for item in lines],
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
