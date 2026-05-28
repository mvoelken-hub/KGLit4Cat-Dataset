import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from infra.filesystem_extraction_output_repository import FileSystemExtractionOutputRepository


class FileSystemExtractionOutputRepositoryTests(unittest.TestCase):
    def test_write_json_replaces_existing_file_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.json"
            path.write_text('{"value": "old"}', encoding="utf-8")

            FileSystemExtractionOutputRepository._write_json_file(path, {"value": "new"})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": "new"})
            self.assertEqual(list(path.parent.glob(".payload.json.*.tmp")), [])

    def test_failed_replace_keeps_previous_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.json"
            path.write_text('{"value": "old"}', encoding="utf-8")

            with patch("infra.filesystem_extraction_output_repository.os.replace", side_effect=OSError("locked")):
                with self.assertRaises(OSError):
                    FileSystemExtractionOutputRepository._write_json_file(path, {"value": "new"})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": "old"})
            self.assertEqual(list(path.parent.glob(".payload.json.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
