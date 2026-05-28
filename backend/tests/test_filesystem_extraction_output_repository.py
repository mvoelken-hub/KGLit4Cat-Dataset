import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, call

from infra.filesystem_extraction_output_repository import (
    FileSystemExtractionOutputRepository,
    _atomic_replace,
)


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


class AtomicReplaceTests(unittest.TestCase):
    def test_atomic_replace_retries_on_permission_error(self):
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "src.tmp"
            dst = Path(directory) / "dst.json"
            src.write_text("{}", encoding="utf-8")

            with patch("infra.filesystem_extraction_output_repository.os.replace") as mock_replace:
                mock_replace.side_effect = [
                    PermissionError("locked"),
                    None,  # succeeds on 2nd attempt
                ]
                with patch("infra.filesystem_extraction_output_repository.time.sleep") as mock_sleep:
                    _atomic_replace(src, dst)

            self.assertEqual(mock_replace.call_count, 2)
            mock_sleep.assert_called_once_with(0.15)

    def test_atomic_replace_raises_after_exhausted_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "src.tmp"
            dst = Path(directory) / "dst.json"
            src.write_text("{}", encoding="utf-8")

            with patch("infra.filesystem_extraction_output_repository.os.replace", side_effect=PermissionError("locked")):
                with patch("infra.filesystem_extraction_output_repository.time.sleep"):
                    with self.assertRaises(PermissionError):
                        _atomic_replace(src, dst, retries=3)

    def test_atomic_replace_succeeds_immediately(self):
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "src.tmp"
            dst = Path(directory) / "dst.json"
            src.write_text('{"ok": true}', encoding="utf-8")

            _atomic_replace(src, dst)

            self.assertTrue(dst.exists())
            self.assertEqual(json.loads(dst.read_text(encoding="utf-8")), {"ok": True})


if __name__ == "__main__":
    unittest.main()
