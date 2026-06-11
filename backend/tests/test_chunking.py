"""Unit tests for the chunking domain logic, especially protected_line_indices."""

import unittest
from unittest.mock import AsyncMock

from app.domain.datasources.chunking import ContentChunk, FilteredLine
from app.domain.datasources.datasource import FileEntry
from app.domain.datasources.text_quality import classify_text_line, DecisionKind


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


if __name__ == "__main__":
    unittest.main()
