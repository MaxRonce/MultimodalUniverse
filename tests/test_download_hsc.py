"""Offline checks: python -m unittest discover -s tests -v."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import download_hsc as hsc


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "data"
        self.directory = self.root / hsc.DATASETS["wide"]
        self.api = Mock()
        self.api.whoami.return_value = {"name": "test-account"}
        self.api.dataset_info.return_value = SimpleNamespace(
            sha="a" * 40,
            siblings=[
                SimpleNamespace(rfilename="collection.properties", size=3),
                SimpleNamespace(rfilename="catalog_10arcs/hats.properties", size=4),
            ],
        )

    def run_main(self, *options):
        with patch("sys.argv", ["download_hsc.py", "--root", str(self.root),
                                "--dataset", "wide", *options]), \
                patch.object(hsc, "HfApi", return_value=self.api), \
                redirect_stdout(io.StringIO()):
            hsc.main()

    def test_dry_run_authenticates_without_downloading_or_creating_files(self):
        with patch.object(hsc, "snapshot_download") as transfer:
            self.run_main("--dry-run")
        self.api.whoami.assert_called_once()
        transfer.assert_not_called()
        self.assertFalse(self.root.exists())

    def test_failed_transfer_then_restart_keeps_original_commit_and_margins(self):
        with patch.object(hsc, "snapshot_download", side_effect=OSError("network")):
            with self.assertRaises(OSError):
                self.run_main("--attempts", "1")
        manifest = json.loads((self.directory / hsc.MANIFEST).read_text())
        self.assertEqual(manifest["revision"], "a" * 40)
        self.api.dataset_info.reset_mock()
        self.api.dataset_info.return_value.sha = "b" * 40

        def finish(**kwargs):
            self.assertTrue(kwargs["token"])
            self.assertEqual(kwargs["revision"], "a" * 40)
            for entry in manifest["files"]:
                target = kwargs["local_dir"] / entry["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"x" * entry["size"])

        with patch.object(hsc, "snapshot_download", side_effect=finish):
            self.run_main()
        self.api.dataset_info.assert_not_called()
        self.assertTrue((self.directory / "catalog_10arcs/hats.properties").exists())

    def test_verify_only_is_offline_and_rejects_a_truncated_file(self):
        plan = hsc.get_plan(self.api, hsc.DATASETS["wide"], self.directory)
        hsc.save_plan(self.directory, plan)
        for entry in plan["files"]:
            target = self.directory / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x" * entry["size"])
        self.api.reset_mock()
        with patch.object(hsc, "snapshot_download") as transfer:
            self.run_main("--verify-only")
            (self.directory / "collection.properties").write_bytes(b"x")
            with self.assertRaisesRegex(RuntimeError, "Verification echouee"):
                self.run_main("--verify-only")
        self.api.whoami.assert_not_called()
        self.api.dataset_info.assert_not_called()
        transfer.assert_not_called()

    def test_transient_error_is_retried_with_same_download_parameters(self):
        with patch.object(hsc, "snapshot_download", side_effect=[OSError("network"), None]) as transfer, \
                patch.object(hsc.time, "sleep") as sleep, \
                redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()):
            hsc.download({"repo_id": "test/repo", "revision": "a" * 40},
                         self.directory, workers=4, attempts=2)
        self.assertEqual(transfer.call_count, 2)
        self.assertEqual(transfer.call_args_list[0], transfer.call_args_list[1])
        sleep.assert_called_once_with(15)

    def test_download_does_not_report_success_for_an_incomplete_snapshot(self):
        with patch.object(hsc, "snapshot_download"):
            with self.assertRaisesRegex(RuntimeError, "fichiers absents"):
                self.run_main()


if __name__ == "__main__":
    unittest.main()
