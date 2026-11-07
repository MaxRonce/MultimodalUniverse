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


if __name__ == "__main__":
    unittest.main()
