"""Unit tests for the chunking domain logic, especially protected_line_indices."""

import unittest
from unittest.mock import AsyncMock

from app.domain.datasources.chunking import ContentChunk, FilteredLine
from app.domain.datasources.datasource import FileEntry
from app.domain.datasources.text_quality import classify_text_line, DecisionKind
from app.domain.token_budget import PromptTokenBudgeter


class FakeFileEntry(FileEntry):
    """Minimal stand-in for FileEntry that only provides extracted content."""

    def __init__(self, content: str):
        super().__init__(
            file_path="test.txt",
            file_name="test.txt",
            file_extension=".txt",
            raw_content=content.encode(),
        )

    def get_extracted_content(self) -> str:
        return self.raw_content.decode()


class WordTokenizerBudgeter(PromptTokenBudgeter):
    def count(self, value: str) -> int:
        return len(value.split())


class ProtectedLineIndicesTests(unittest.IsolatedAsyncioTestCase):
    async def test_protected_indices_keeps_dropped_header(self):
        """When protected_line_indices contains [0], a dropped header is retained."""
        content = (
            "%%%%%%\n"
            "Some normal sentence here.\n"
            "Another normal sentence.\n"
        )
        file_entry = FakeFileEntry(content)

        default_result = classify_text_line("%%%%%%")
        self.assertEqual(default_result.kind, DecisionKind.DROP)

        mock_embed = AsyncMock(return_value=[
            [0.1] * 768,
            [0.2] * 768,
            [0.3] * 768,
        ])

        chunks = await ContentChunk.create_chunks_for_file_entry(
            data_package_id="pkg",
            file_entry=file_entry,
            embedding_func=mock_embed,
            protected_line_indices=[0],
            min_lines_for_chunking=3,
        )

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        # The first line (index 0) must be present in filtered_line_indices
        self.assertIn(0, chunk.filtered_line_indices)
        # The full content should include the protected header
        self.assertIn("%%%%%%", chunk.content)

    async def test_no_protected_indices_does_not_change_behaviour(self):
        """With protected_line_indices=None the header is treated as usual."""
        content = (
            "%%%%%%\n"
            "Some normal sentence here.\n"
            "Another normal sentence.\n"
        )
        file_entry = FakeFileEntry(content)

        mock_embed = AsyncMock(return_value=[
            [0.1] * 768,
            [0.2] * 768,
        ])

        chunks = await ContentChunk.create_chunks_for_file_entry(
            data_package_id="pkg",
            file_entry=file_entry,
            embedding_func=mock_embed,
            protected_line_indices=None,
            min_lines_for_chunking=3,
        )

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertNotIn(0, chunk.filtered_line_indices)
        self.assertNotIn("%%%%%%", chunk.content)

    async def test_protected_indices_does_not_duplicate_already_kept_lines(self):
        """If a protected line is already KEEP, it appears only once."""
        content = (
            "Some normal sentence here.\n"
            "Another normal sentence.\n"
            "Yet another line.\n"
        )
        file_entry = FakeFileEntry(content)

        mock_embed = AsyncMock(return_value=[
            [0.1] * 768,
            [0.2] * 768,
            [0.3] * 768,
        ])

        chunks = await ContentChunk.create_chunks_for_file_entry(
            data_package_id="pkg",
            file_entry=file_entry,
            embedding_func=mock_embed,
            protected_line_indices=[0],
            min_lines_for_chunking=3,
        )

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        # line 0 is already kept, so it should appear exactly once
        self.assertEqual(chunk.filtered_line_indices.count(0), 1)

    async def test_protected_indices_out_of_range_are_ignored(self):
        """Line indices outside the file bounds are silently ignored."""
        content = "Line one.\nLine two.\n"
        file_entry = FakeFileEntry(content)

        mock_embed = AsyncMock(return_value=[
            [0.1] * 768,
            [0.2] * 768,
        ])

        chunks = await ContentChunk.create_chunks_for_file_entry(
            data_package_id="pkg",
            file_entry=file_entry,
            embedding_func=mock_embed,
            protected_line_indices=[0, 5, 99],
            min_lines_for_chunking=2,
        )

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertIn(0, chunk.filtered_line_indices)
        self.assertNotIn(5, chunk.filtered_line_indices)
        self.assertNotIn(99, chunk.filtered_line_indices)

    async def test_fixed_tokens_strategy_splits_by_line_budget(self):
        content = "".join(
            f"line {index} alpha beta gamma delta epsilon\n"
            for index in range(10)
        )
        file_entry = FakeFileEntry(content)

        chunks = await ContentChunk.create_chunks_for_file_entry(
            data_package_id="pkg",
            file_entry=file_entry,
            embedding_func=AsyncMock(return_value=[]),
            chunking_strategy="fixed_tokens",
            fixed_tokens_per_chunk=15,
            min_tokens_per_chunk=1,
            min_lines_for_chunking=1,
            max_tokens_per_chunk=100,
            token_budgeter=WordTokenizerBudgeter(),
        )

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.content.split()), 15)
            self.assertTrue(chunk.content.endswith("\n") or chunk.content == "")

    async def test_fixed_tokens_strategy_never_calls_embedding_func(self):
        content = "one two three four five\nsix seven eight nine ten\n"
        file_entry = FakeFileEntry(content)
        mock_embed = AsyncMock(return_value=[])

        await ContentChunk.create_chunks_for_file_entry(
            data_package_id="pkg",
            file_entry=file_entry,
            embedding_func=mock_embed,
            chunking_strategy="fixed_tokens",
            fixed_tokens_per_chunk=10,
            min_lines_for_chunking=1,
            max_tokens_per_chunk=100,
            token_budgeter=WordTokenizerBudgeter(),
        )

        mock_embed.assert_not_awaited()

    async def test_fixed_tokens_strategy_respects_max_token_cap(self):
        content = "".join(
            f"line {index} alpha beta gamma\n"
            for index in range(6)
        )
        file_entry = FakeFileEntry(content)

        chunks = await ContentChunk.create_chunks_for_file_entry(
            data_package_id="pkg",
            file_entry=file_entry,
            embedding_func=AsyncMock(return_value=[]),
            chunking_strategy="fixed_tokens",
            fixed_tokens_per_chunk=50,
            min_tokens_per_chunk=1,
            max_tokens_per_chunk=10,
            token_budgeter=WordTokenizerBudgeter(),
        )

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.content.split()), 10)

    async def test_default_semantic_strategy_ignores_fixed_token_size(self):
        content = "line 0 alpha beta\nline 1 gamma delta\nline 2 epsilon zeta\n"
        file_entry = FakeFileEntry(content)
        call_count = 0
        async def embed(texts):
            nonlocal call_count
            call_count += 1
            return [[0.1] * 4 for _ in texts]

        chunks = await ContentChunk.create_chunks_for_file_entry(
            data_package_id="pkg",
            file_entry=file_entry,
            embedding_func=embed,
            chunking_strategy="semantic",
            fixed_tokens_per_chunk=3,
            min_lines_for_chunking=1,
            max_tokens_per_chunk=100,
            token_budgeter=WordTokenizerBudgeter(),
        )

        self.assertEqual(len(chunks), 1)
        self.assertGreater(call_count, 0)

    async def test_token_cap_splits_oversized_chunk_after_semantic_chunking(self):
        content = "".join(
            f"line {index} " + ("sample metadata words " * 8) + "\n"
            for index in range(12)
        )
        file_entry = FakeFileEntry(content)

        async def embed(texts):
            return [[0.1] * 4 for _ in texts]

        chunks = await ContentChunk.create_chunks_for_file_entry(
            data_package_id="pkg",
            file_entry=file_entry,
            embedding_func=embed,
            min_lines_for_chunking=3,
            max_tokens_per_chunk=60,
        )

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual((len(chunk.content) + 3) // 4, 60)

    async def test_token_cap_uses_configured_token_budgeter(self):
        class FakeTokenizerBudgeter(PromptTokenBudgeter):
            def count(self, value: str) -> int:
                return len(value.split())

        content = "".join(
            f"line {index} alpha beta gamma delta epsilon\n"
            for index in range(8)
        )
        file_entry = FakeFileEntry(content)

        async def embed(texts):
            return [[0.1] * 4 for _ in texts]

        chunks = await ContentChunk.create_chunks_for_file_entry(
            data_package_id="pkg",
            file_entry=file_entry,
            embedding_func=embed,
            min_lines_for_chunking=3,
            max_tokens_per_chunk=12,
            token_budgeter=FakeTokenizerBudgeter(),
        )

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.content.split()), 12)

    async def test_single_oversized_line_does_not_crash_token_cap(self):
        content = ("sample metadata words " * 30) + "\n"
        file_entry = FakeFileEntry(content)

        chunks = await ContentChunk.create_chunks_for_file_entry(
            data_package_id="pkg",
            file_entry=file_entry,
            embedding_func=AsyncMock(return_value=[]),
            min_lines_for_chunking=3,
            max_tokens_per_chunk=60,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content, content)

    async def test_post_processing_splits_large_filtered_line_gap(self):
        content = "first metadata line\n" + ("\n" * 39) + "second metadata line\n"
        file_entry = FakeFileEntry(content)

        chunks = await ContentChunk.create_chunks_for_file_entry(
            data_package_id="pkg",
            file_entry=file_entry,
            embedding_func=AsyncMock(return_value=[]),
            min_lines_for_chunking=100,
            max_tokens_per_chunk=100,
            token_budgeter=WordTokenizerBudgeter(),
        )

        self.assertEqual([(chunk.start_idx, chunk.end_idx) for chunk in chunks], [(0, 0), (40, 40)])
        for chunk in chunks:
            self.assertIn("line_gap_split", chunk.post_processing.operations)

    async def test_post_processing_does_not_split_gap_at_threshold(self):
        content = "first metadata line\n" + ("\n" * 31) + "second metadata line\n"
        file_entry = FakeFileEntry(content)

        chunks = await ContentChunk.create_chunks_for_file_entry(
            data_package_id="pkg",
            file_entry=file_entry,
            embedding_func=AsyncMock(return_value=[]),
            min_lines_for_chunking=100,
            max_tokens_per_chunk=100,
            token_budgeter=WordTokenizerBudgeter(),
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].filtered_line_indices, [0, 32])
        self.assertNotIn("line_gap_split", chunks[0].post_processing.operations)

    def test_post_processing_merges_small_chunk_with_lowest_distance_neighbor(self):
        lines = [
            FilteredLine("left alpha beta gamma delta\n", 0),
            FilteredLine("tiny\n", 1),
            FilteredLine("right alpha beta gamma delta\n", 2),
        ]
        chunks = [
            ContentChunk._chunk_from_filtered_lines([lines[0]], data_package_id="pkg", file_path="test.txt"),
            ContentChunk._chunk_from_filtered_lines([lines[1]], data_package_id="pkg", file_path="test.txt"),
            ContentChunk._chunk_from_filtered_lines([lines[2]], data_package_id="pkg", file_path="test.txt"),
        ]

        processed = ContentChunk._post_process_chunks(
            chunks,
            filtered_lines=lines,
            data_package_id="pkg",
            file_path="test.txt",
            max_tokens_per_chunk=20,
            min_tokens_per_chunk=5,
            max_filtered_line_gap=32,
            token_budgeter=WordTokenizerBudgeter(),
            boundary_distances={0: 0.9, 1: 0.1},
        )

        self.assertEqual([chunk.filtered_line_indices for chunk in processed], [[0], [1, 2]])
        self.assertIn("min_token_merge", processed[1].post_processing.operations)
        self.assertEqual(processed[1].post_processing.source_chunk_count, 2)

    def test_post_processing_falls_back_to_previous_neighbor_without_distances(self):
        lines = [
            FilteredLine("left alpha beta gamma delta\n", 0),
            FilteredLine("tiny\n", 1),
            FilteredLine("right alpha beta gamma delta\n", 2),
        ]
        chunks = [
            ContentChunk._chunk_from_filtered_lines([lines[0]], data_package_id="pkg", file_path="test.txt"),
            ContentChunk._chunk_from_filtered_lines([lines[1]], data_package_id="pkg", file_path="test.txt"),
            ContentChunk._chunk_from_filtered_lines([lines[2]], data_package_id="pkg", file_path="test.txt"),
        ]

        processed = ContentChunk._post_process_chunks(
            chunks,
            filtered_lines=lines,
            data_package_id="pkg",
            file_path="test.txt",
            max_tokens_per_chunk=20,
            min_tokens_per_chunk=5,
            max_filtered_line_gap=32,
            token_budgeter=WordTokenizerBudgeter(),
        )

        self.assertEqual([chunk.filtered_line_indices for chunk in processed], [[0, 1], [2]])

    def test_post_processing_never_merges_across_line_gap_barrier(self):
        lines = [
            FilteredLine("left alpha beta gamma\n", 0),
            FilteredLine("tiny\n", 40),
        ]
        chunk = ContentChunk._chunk_from_filtered_lines(lines, data_package_id="pkg", file_path="test.txt")

        processed = ContentChunk._post_process_chunks(
            [chunk],
            filtered_lines=lines,
            data_package_id="pkg",
            file_path="test.txt",
            max_tokens_per_chunk=20,
            min_tokens_per_chunk=5,
            max_filtered_line_gap=32,
            token_budgeter=WordTokenizerBudgeter(),
        )

        self.assertEqual([chunk.filtered_line_indices for chunk in processed], [[0], [40]])
        self.assertTrue(all("min_token_merge" not in chunk.post_processing.operations for chunk in processed))

    def test_post_processing_never_exceeds_max_tokens_when_merging(self):
        lines = [
            FilteredLine("one two three four\n", 0),
            FilteredLine("five six seven eight\n", 1),
        ]
        chunks = [
            ContentChunk._chunk_from_filtered_lines([lines[0]], data_package_id="pkg", file_path="test.txt"),
            ContentChunk._chunk_from_filtered_lines([lines[1]], data_package_id="pkg", file_path="test.txt"),
        ]

        processed = ContentChunk._post_process_chunks(
            chunks,
            filtered_lines=lines,
            data_package_id="pkg",
            file_path="test.txt",
            max_tokens_per_chunk=6,
            min_tokens_per_chunk=5,
            max_filtered_line_gap=32,
            token_budgeter=WordTokenizerBudgeter(),
            boundary_distances={0: 0.0},
        )

        self.assertEqual([chunk.filtered_line_indices for chunk in processed], [[0], [1]])

    def test_post_processing_leaves_unmergeable_tiny_chunk_intact(self):
        lines = [FilteredLine("tiny\n", 0)]
        chunk = ContentChunk._chunk_from_filtered_lines(lines, data_package_id="pkg", file_path="test.txt")

        processed = ContentChunk._post_process_chunks(
            [chunk],
            filtered_lines=lines,
            data_package_id="pkg",
            file_path="test.txt",
            max_tokens_per_chunk=20,
            min_tokens_per_chunk=5,
            max_filtered_line_gap=32,
            token_budgeter=WordTokenizerBudgeter(),
        )

        self.assertEqual(len(processed), 1)
        self.assertEqual(processed[0].filtered_line_indices, [0])
        self.assertEqual(processed[0].content, "tiny\n")
        self.assertEqual(processed[0].post_processing.source_chunk_count, 1)


if __name__ == "__main__":
    unittest.main()
