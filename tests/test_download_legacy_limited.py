"""Offline tests for capped transfers and bounded concurrency."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import download_legacy_limited as dl


class Bar:
    lock = threading.RLock()
    instances = []

    def __init__(self, total, initial=0, **kwargs):
        self.n = initial
        self.total = total
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    @staticmethod
    def get_lock():
        return Bar.lock

    def update(self, size):
        self.n += size

    def refresh(self):
        pass

    def write(self, message):
        pass

    def set_postfix_str(self, *args, **kwargs):
        pass


class LimitedTests(unittest.TestCase):
    def test_resume_counts_only_new_bytes_and_passes_curl_cap(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "sample.parquet"
            partial = Path(str(target) + ".curl.incomplete")
            partial.write_bytes(b"abcd")
            child = Mock()

            def finish(timeout=None):
                partial.write_bytes(b"abcdefghijkl")
                return 0

            child.wait.side_effect = finish
            progress = Bar(total=12, initial=4)
            with patch.object(dl, "HfApi"), patch.object(dl, "get_token", return_value="fake-token"), \
                    patch.object(dl.subprocess, "Popen", return_value=child) as launch:
                dl.download("curl", {"revision": "a" * 40}, {"path": "sample.parquet", "size": 12},
                            target, progress, 4, rate=50_000_000)
            command = launch.call_args.args[0]
            self.assertEqual(command[command.index("--limit-rate") + 1], "50000000")
            self.assertEqual(progress.n, 12)
            self.assertEqual(target.read_bytes(), b"abcdefghijkl")
            self.assertFalse(partial.exists())
            launch.assert_called_once()

    def test_cancellation_does_not_start_another_transfer(self):
        cancelled = threading.Event()
        cancelled.set()
        with tempfile.TemporaryDirectory() as root, patch.object(dl.subprocess, "Popen") as launch:
            with self.assertRaises(InterruptedError):
                dl.download("curl", {}, {"size": 12}, Path(root) / "sample", Bar(12), 0,
                            cancelled=cancelled)
        launch.assert_not_called()

    def test_four_workers_are_bounded_and_share_the_total_cap(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / dl.REPO
            directory.mkdir(parents=True)
            files = [{"path": f"part-{i}", "size": 1} for i in range(8)]
            (directory / ".hf-download-manifest.json").write_text(json.dumps({
                "repo_id": dl.REPO, "revision": "a" * 40, "files": files
            }))
            active = maximum = 0
            rates = []
            counter_lock = threading.Lock()

            def fake_download(curl, plan, entry, target, progress, initial, rate, cancelled):
                nonlocal active, maximum
                with counter_lock:
                    active += 1
                    maximum = max(maximum, active)
                    rates.append(rate)
                try:
                    time.sleep(0.05)
                    target.write_bytes(b"x")
                    with Bar.get_lock():
                        progress.update(1)
                finally:
                    with counter_lock:
                        active -= 1

            args = SimpleNamespace(root=Path(root), run=True, workers=4, unlimited=False)
            with patch.object(dl.shutil, "which", return_value="curl"), \
                    patch.object(dl, "tqdm", Bar), \
                    patch.object(dl, "download", side_effect=fake_download), \
                    redirect_stdout(io.StringIO()):
                dl.run(args)
            self.assertEqual(maximum, 4)
            self.assertEqual(set(rates), {50_000_000})
            self.assertEqual(len(rates), 8)
            self.assertEqual(Bar.instances[-1].n, 8)
            self.assertTrue(all((directory / f["path"]).is_file() for f in files))


    def test_resolver_url_changes_but_revision_and_path_stay_pinned(self):
        from urllib.parse import parse_qs, urlsplit

        plan = {"revision": "a" * 40}
        entry = {"path": "dataset/Npix=123.parquet"}
        first = urlsplit(dl.transfer_url(plan, entry))
        second = urlsplit(dl.transfer_url(plan, entry))
        self.assertEqual(first.path, second.path)
        self.assertIn(plan["revision"], first.path)
        self.assertEqual(parse_qs(first.query)["download"], ["true"])
        self.assertNotEqual(first.query, second.query)

    def test_retry_gets_fresh_url_and_keeps_resume_and_cap(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "sample.parquet"
            partial = Path(str(target) + ".curl.incomplete")
            partial.write_bytes(b"abcd")
            failed, succeeded = Mock(), Mock()
            failed.wait.return_value = 22

            def finish(timeout=None):
                partial.write_bytes(b"abcdefghijkl")
                return 0

            succeeded.wait.side_effect = finish
            cancelled = Mock()
            cancelled.is_set.return_value = False
            cancelled.wait.return_value = False
            with patch.object(dl, "HfApi"), \
                    patch.object(dl, "get_token", return_value="fake-token"), \
                    patch.object(dl.subprocess, "Popen", side_effect=[failed, succeeded]) as launch:
                dl.download("curl", {"revision": "a" * 40},
                            {"path": "sample.parquet", "size": 12},
                            target, Bar(12, initial=4), 4,
                            rate=50_000_000, cancelled=cancelled)
            commands = [call.args[0] for call in launch.call_args_list]
            self.assertNotEqual(commands[0][-1], commands[1][-1])
            for command in commands:
                self.assertIn("--continue-at", command)
                self.assertEqual(command[command.index("--limit-rate") + 1], "50000000")
                self.assertNotIn("fake-token", " ".join(command))
            self.assertEqual(target.read_bytes(), b"abcdefghijkl")

    def test_keep_going_records_failure_and_downloads_other_files(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / dl.REPO
            directory.mkdir(parents=True)
            files = [{"path": "bad.parquet", "size": 1},
                     {"path": "good.parquet", "size": 1}]
            (directory / ".hf-download-manifest.json").write_text(json.dumps({
                "repo_id": dl.REPO, "revision": "a" * 40, "files": files
            }))
            visited = []

            def fake_download(curl, plan, entry, target, progress, initial, rate, cancelled):
                visited.append(entry["path"])
                if entry["path"] == "bad.parquet":
                    raise dl.TransferFailed("HTTP error")
                target.write_bytes(b"x")
                progress.update(1)

            args = SimpleNamespace(root=Path(root), run=True, workers=1,
                                   unlimited=False, keep_going=True)
            with patch.object(dl.shutil, "which", return_value="curl"), \
                    patch.object(dl, "tqdm", Bar), \
                    patch.object(dl, "download", side_effect=fake_download), \
                    redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "encore incomplet"):
                    dl.run(args)
            self.assertEqual(visited, ["bad.parquet", "good.parquet"])
            self.assertTrue((directory / "good.parquet").is_file())
            self.assertFalse((directory / "bad.parquet").exists())
            failure = json.loads((directory / ".curl-failures.json").read_text())
            self.assertEqual([f["path"] for f in failure["files"]], ["bad.parquet"])
            self.assertEqual(Bar.instances[-1].n, 1)


if __name__ == "__main__":
    unittest.main()
