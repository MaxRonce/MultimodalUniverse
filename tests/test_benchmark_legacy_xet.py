"""Offline checks for benchmark selection; no Hugging Face requests."""
from pathlib import Path
import tempfile
import unittest
from scripts.benchmark_legacy_xet import select_batch


class SelectionTests(unittest.TestCase):
    def test_skips_finished_files_and_preserves_curl_partials(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            finished = directory / "done.parquet"
            with finished.open("wb") as stream:
                stream.truncate(512_000_000)  # sparse test file, not 512 MB of writes
            (directory / "curl.parquet.curl.incomplete").write_bytes(b"x")
            plan = {"files": [
                {"path": "done.parquet", "size": 512_000_000},
                {"path": "curl.parquet", "size": 512_000_000},
                {"path": "new-a.parquet", "size": 512_000_000},
                {"path": "new-b.parquet", "size": 512_000_000},
            ]}
            entries, size = select_batch(directory, plan, 512_000_000, 1)
            self.assertEqual([e["path"] for e in entries], ["new-a.parquet", "new-b.parquet"])
            self.assertEqual(size, 1_024_000_000)
            self.assertEqual((directory / "curl.parquet.curl.incomplete").read_bytes(), b"x")

    def test_selects_two_waves_for_the_requested_worker_count(self):
        with tempfile.TemporaryDirectory() as root:
            plan = {"files": [{"path": f"{i}.parquet", "size": 300_000_000} for i in range(10)]}
            entries, size = select_batch(Path(root), plan, 500_000_000, 2)
            self.assertEqual(len(entries), 4)
            self.assertEqual(size, 1_200_000_000)

    def test_rejects_paths_outside_the_dataset(self):
        with tempfile.TemporaryDirectory() as root:
            plan = {"files": [{"path": "../escape.parquet", "size": 512_000_000}]}
            with self.assertRaises(ValueError):
                select_batch(Path(root), plan, 500_000_000, 1)


if __name__ == "__main__":
    unittest.main()
